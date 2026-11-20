"""Draft collection with generation-time signals read from the commitment forward pass.

Generation is the unchanged D-029 sampler (``sampling.monotone.monotone_unmask``). It
receives prompt IDs and public response lengths only. A :class:`SignalObserver` is
passed as its observer; the sampler ignores the observer's return value, hands it a
clone of the token state and the detached aligned logits, and calls it after the
commit decisions of that forward pass are fixed. So reference targets, which only the
observer holds, cannot reach model inputs, candidate choice, commit order, RNG use or
stopping.

For every committed token the observer takes, from the SAME forward pass:

- ``log_c_gen``: log softmax over the FULL output vocabulary (151,936 rows on
  Open-dCoder, temperature 1, float32) at the committed token. The observer rebuilds
  the sampler's exact row tensor (``logits[editable & masked].float()``) and the same
  ``log_softmax``, so the value must equal the sampler's own ranking score bitwise;
  ``rank_score`` stores the sampler's value and every mismatch is counted;
- ``rank_score``: the exact sort key of ``monotone_unmask``. The sampler ranks masked
  positions by ``-score`` with ``score = log_softmax(rows)[candidate]`` in float32 (a LOG
  probability; ties at exact float32 equality go to the smaller position). It never
  computes ``exp``. In exact arithmetic ranking by ``c = exp(score)`` gives the same order,
  but float32 ``exp`` is not injective (distinct float32 logs can share one float32
  probability, including both underflowing to 0), so a probability-ranked sampler would
  break ties differently. Hence ``rank_score`` is asserted equal to ``log_c_gen``, not to
  ``c``; consumers needing ``c`` take ``exp(log_c_gen)`` in float64 and never re-rank with it.
  D-029's decision text said "probability"; the implemented sampler (unchanged) ranks by
  the log; see D-042. Do not change generation to make a different assertion pass;
- ``log_q_gen``: the same row at the reference target token (analysis-only);
- ``raw_max_log_prob``: the largest full-vocabulary log probability in that row
  (diagnostic: ties and saturation, and whether a forbidden token outranked the commit).

``policy_log_prob`` is the sampler's restricted-policy log probability (softmax over the
allowed token set only), the distribution whose greedy argmax picks the candidate. It is
stored separately and is not ``c``.

Records keep target-independent fields under ``generation`` and target-dependent
fields under ``analysis``; Gate D compares ``generation`` (plus calls) only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from ..data.checker import check_response
from ..data.format import TokenizedFormat
from ..data.tasks import Example
from ..evaluation.generation import ONE_TOKEN_PER_CALL, public_response_length
from ..models.adapter import DiffusionAdapter
from ..sampling.monotone import monotone_unmask, reveal_schedule

DRAFT_SCHEMA = "dlmc.draft.v1"
PROBABILITY_DEFINITION = (
    "log_c_gen / log_q_gen: float32 log_softmax over the full, unrestricted output vocabulary "
    "(temperature 1) of the aligned logits at the commitment forward pass; policy_log_prob: log softmax "
    "over the D-029 allowed token set (renormalized), stored separately; rank_score: the sampler's exact "
    "sort key, the float32 raw LOG probability of the selected candidate (not exp of it), which equals "
    "log_c_gen bitwise; within one call commits follow (-rank_score, position); c = exp(log_c_gen) in float64")
# Tolerance for c + q <= 1 at wrong tokens, in log space (float32 rounding of two log probabilities).
CQ_LOG_TOL = 1e-5


def calls_for(budget: str, length: int) -> int:
    return length if budget == ONE_TOKEN_PER_CALL else int(budget)


def budget_feasible(budget: str, length: int) -> bool:
    return 1 <= calls_for(budget, length) <= length


@dataclass
class _Commit:
    call: int
    order: int
    mask_fraction_before: float
    token_id: int
    log_c: float
    log_q: float
    rank_score: float
    policy_logp: float
    raw_max: float


class SignalObserver:
    """Reads committed-token and reference-target log probabilities from each forward pass.

    It never returns anything and never modifies its inputs. ``keep_states_for_calls``
    keeps CPU copies of the pre-commit token state for those calls (preflight recompute).
    """

    def __init__(self, starts: Sequence[int], lengths: Sequence[int], references: Sequence[Sequence[int]],
                 mask_id: int, keep_states_for_calls: Sequence[int] = ()):
        if not len(starts) == len(lengths) == len(references):
            raise ValueError("one start, length and reference per example")
        for L, ref in zip(lengths, references):
            if len(ref) != L:
                raise ValueError("reference length differs from the public response length")
        self.starts = [int(s) for s in starts]
        self.lengths = [int(n) for n in lengths]
        self._refs = [[int(t) for t in r] for r in references]
        self.mask_id = int(mask_id)
        self.commits: list[list[_Commit | None]] = [[None] * n for n in self.lengths]
        self._order = [0] * len(self.lengths)
        self.rank_score_mismatches = 0
        self.calls_seen: list[int] = []
        self.keep = set(int(c) for c in keep_states_for_calls)
        self.saved_states: dict[int, torch.Tensor] = {}

    def editable(self, ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(ids.shape[1], device=ids.device)[None, :]
        st = torch.tensor(self.starts, device=ids.device)[:, None]
        ln = torch.tensor(self.lengths, device=ids.device)[:, None]
        return (pos >= st) & (pos < st + ln)

    def __call__(self, call: int, ids: torch.Tensor, logits: torch.Tensor, events) -> None:
        self.calls_seen.append(int(call))
        if call in self.keep:
            self.saved_states[int(call)] = ids.detach().cpu().clone()
        masked = self.editable(ids) & (ids == self.mask_id)
        # The sampler's own row tensor and normalisation (sampling/monotone.py), so values match bitwise.
        rows = logits[masked].float()
        lsm = torch.log_softmax(rows, dim=-1)
        where = masked.nonzero().cpu().tolist()
        index = {(b, p): r for r, (b, p) in enumerate(where)}
        remaining = [0] * len(self.lengths)
        for b, _ in where:
            remaining[b] += 1
        for b, n in enumerate(self.lengths):
            done = sum(c is not None for c in self.commits[b])
            if remaining[b] != n - done:
                raise AssertionError(f"example {b}: {remaining[b]} masks at call {call}, expected {n - done}")
        if not events:
            return
        r_idx, tok, tgt = [], [], []
        for ev in events:
            key = (ev.example, self.starts[ev.example] + ev.position)
            if key not in index:
                raise AssertionError(f"commit at {key} was not a masked position of this forward pass")
            r_idx.append(index[key])
            tok.append(int(ev.token_id))
            tgt.append(self._refs[ev.example][ev.position])
        dev = lsm.device
        sel = lsm[torch.tensor(r_idx, device=dev)]
        log_c = sel.gather(1, torch.tensor(tok, device=dev)[:, None]).squeeze(1).cpu().tolist()
        log_q = sel.gather(1, torch.tensor(tgt, device=dev)[:, None]).squeeze(1).cpu().tolist()
        raw_max = sel.max(dim=1).values.cpu().tolist()
        for k, ev in enumerate(events):
            b, p = ev.example, ev.position
            if self.commits[b][p] is not None:
                raise AssertionError(f"example {b} position {p} committed twice")
            if log_c[k] != ev.raw_logp:
                self.rank_score_mismatches += 1
            self.commits[b][p] = _Commit(call=int(call), order=self._order[b],
                                         mask_fraction_before=float(ev.mask_fraction_before),
                                         token_id=tok[k], log_c=float(log_c[k]), log_q=float(log_q[k]),
                                         rank_score=float(ev.raw_logp), policy_logp=float(ev.policy_logp),
                                         raw_max=float(raw_max[k]))
            self._order[b] += 1


def _rng_state(device: torch.device) -> tuple:
    state = [torch.get_rng_state()]
    if device.type == "cuda":
        state.extend(torch.cuda.get_rng_state_all())
    return tuple(state)


@dataclass
class BatchResult:
    records: list[dict]
    batched_forward_calls: int
    example_model_calls: int
    rank_score_mismatches: int
    rng_unchanged: bool
    observer: SignalObserver


def collect_batch(adapter: DiffusionAdapter, examples: Sequence[Example], budget: str, allowed: torch.Tensor,
                  tf: TokenizedFormat, keep_states_for_calls: Sequence[int] = ()) -> BatchResult:
    """Generate one batch (all examples share the response length and the call count) and build records."""
    budget = str(budget)
    lengths = [public_response_length(ex, tf.fmt.slot_len) for ex in examples]
    if len(set(lengths)) != 1:
        raise ValueError("a collection batch must share one public response length")
    L = lengths[0]
    if not budget_feasible(budget, L):
        raise ValueError(f"budget {budget} is infeasible for L={L} (D-029: 1 <= T <= L)")
    T = calls_for(budget, L)
    starts = [len(ex.prompt_ids) for ex in examples]
    obs = SignalObserver(starts, lengths, [ex.response_ids for ex in examples], adapter.tokens.mask_token_id,
                         keep_states_for_calls)
    rng_before = _rng_state(adapter.device)
    res = monotone_unmask(adapter, [ex.prompt_ids for ex in examples], lengths, T, allowed, observer=obs)
    rng_unchanged = all(torch.equal(a, b) for a, b in zip(rng_before, _rng_state(adapter.device)))
    if obs.calls_seen != list(range(1, T + 1)):
        raise AssertionError(f"observer saw calls {obs.calls_seen}, expected 1..{T}")
    records = []
    for b, ex in enumerate(examples):
        commits = obs.commits[b]
        if any(c is None for c in commits):
            raise AssertionError(f"{ex.problem_id}: a position has no commit event")
        draft = res.responses[b]
        if [c.token_id for c in commits] != draft:
            raise AssertionError(f"{ex.problem_id}: commit events disagree with the final draft")
        chk = check_response(draft, ex, tf)
        records.append({
            "schema": DRAFT_SCHEMA,
            "split": ex.split, "problem_id": ex.problem_id, "family": ex.family,
            "length": int(ex.setting["length"]), "operand_digits": int(ex.setting["operand_digits"]),
            "setting": dict(ex.setting), "budget": budget, "calls": int(res.calls_per_example[b]),
            "response_length": L, "prompt_ids": list(ex.prompt_ids),
            "generation": {
                "draft_ids": list(draft),
                "schedule": list(res.schedule[b]),
                "commit_call": [c.call for c in commits],
                "commit_order": [c.order for c in commits],
                "mask_fraction_before": [c.mask_fraction_before for c in commits],
                "log_c_gen": [c.log_c for c in commits],
                "rank_score": [c.rank_score for c in commits],
                "policy_log_prob": [c.policy_logp for c in commits],
                "raw_max_log_prob": [c.raw_max for c in commits],
            },
            "analysis": {
                "reference_ids": list(ex.response_ids),
                "values": list(ex.values),
                "log_q_gen": [c.log_q for c in commits],
                "token_correct": list(chk.token_correct),
                "parsed_values": list(chk.parsed_values),
                "slot_status": list(chk.slot_status),
                "value_correct": list(chk.value_correct),
                "final_correct": bool(chk.final_correct),
                "trace_correct": bool(chk.trace_correct),
                "well_formed": bool(chk.well_formed),
                "n_token_errors": chk.n_tokens - sum(chk.token_correct),
                "n_value_errors": len(chk.value_correct) - sum(chk.value_correct),
            },
        })
    return BatchResult(records=records, batched_forward_calls=res.batched_forward_calls,
                       example_model_calls=sum(res.calls_per_example), rank_score_mismatches=obs.rank_score_mismatches,
                       rng_unchanged=rng_unchanged, observer=obs)


def generation_view(rec: dict) -> dict:
    """Everything a reference target must not influence."""
    return {"problem_id": rec["problem_id"], "budget": rec["budget"], "calls": rec["calls"],
            "prompt_ids": rec["prompt_ids"], "generation": rec["generation"]}


def _is_float32(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x) and float(torch.tensor(x, dtype=torch.float32).item()) == x


def check_record(rec: dict, example: Example, tf: TokenizedFormat) -> list[str]:
    """Invariants of one stored draft record against its dataset example. Returns problems (empty = ok)."""
    bad: list[str] = []
    g, a = rec.get("generation", {}), rec.get("analysis", {})
    pid = rec.get("problem_id")
    if rec.get("schema") != DRAFT_SCHEMA:
        return [f"{pid}: schema {rec.get('schema')!r}"]
    if pid != example.problem_id or rec["split"] != example.split or rec["family"] != example.family:
        bad.append(f"{pid}: identity differs from the dataset example")
    L = public_response_length(example, tf.fmt.slot_len)
    if rec["response_length"] != L or rec["prompt_ids"] != list(example.prompt_ids):
        bad.append(f"{pid}: prompt or public length differs from the dataset")
    if a.get("reference_ids") != list(example.response_ids) or a.get("values") != list(example.values):
        bad.append(f"{pid}: stored reference differs from the dataset example")
    T = calls_for(rec["budget"], L)
    if rec["calls"] != T:
        bad.append(f"{pid}: calls {rec['calls']} != {T}")
    arrays = {k: g.get(k) for k in ("draft_ids", "commit_call", "commit_order", "mask_fraction_before", "log_c_gen",
                                    "rank_score", "policy_log_prob", "raw_max_log_prob")}
    arrays["log_q_gen"] = a.get("log_q_gen")
    arrays["token_correct"] = a.get("token_correct")
    if any(not isinstance(v, list) or len(v) != L for v in arrays.values()):
        return bad + [f"{pid}: per-token arrays missing or not of length {L}"]
    schedule = reveal_schedule(L, T)
    if g.get("schedule") != schedule:
        bad.append(f"{pid}: schedule differs from D-029")
    k_before = [0]
    for n in schedule:
        k_before.append(k_before[-1] + n)
    for s in range(1, T + 1):
        if sum(c == s for c in g["commit_call"]) != schedule[s - 1]:
            bad.append(f"{pid}: call {s} commits differ from the schedule")
            break
    if sorted(g["commit_order"]) != list(range(L)):
        bad.append(f"{pid}: commit_order is not a permutation")
    else:
        by_order = sorted(range(L), key=lambda p: g["commit_order"][p])
        if [g["commit_call"][p] for p in by_order] != sorted(g["commit_call"]):
            bad.append(f"{pid}: commit order contradicts commit calls")
    for p in range(L):
        s = g["commit_call"][p]
        if not 1 <= s <= T:
            bad.append(f"{pid}: commit call {s} out of range")
            break
        if g["mask_fraction_before"][p] != (L - k_before[s - 1]) / L:
            bad.append(f"{pid}: mask fraction at position {p} differs from the schedule")
            break
    for name in ("log_c_gen", "log_q_gen", "rank_score", "policy_log_prob", "raw_max_log_prob"):
        if not all(_is_float32(x) and x <= 0.0 for x in arrays[name]):
            bad.append(f"{pid}: {name} has a non-float32, non-finite or positive value")
    if g["rank_score"] != g["log_c_gen"]:
        bad.append(f"{pid}: rank_score differs from log_c_gen (not the same forward pass)")
    elif sorted(g["commit_order"]) == list(range(L)):
        for s in range(1, T + 1):
            ps = [p for p in range(L) if g["commit_call"][p] == s]
            if sorted(ps, key=lambda p: g["commit_order"][p]) != sorted(ps, key=lambda p: (-g["rank_score"][p], p)):
                bad.append(f"{pid}: commit order within call {s} contradicts the sampler's (-rank_score, position) rule")
                break
    if any(c > m for c, m in zip(g["log_c_gen"], g["raw_max_log_prob"])):
        bad.append(f"{pid}: log_c_gen exceeds the row maximum")
    if any(pol < c - CQ_LOG_TOL for pol, c in zip(g["policy_log_prob"], g["log_c_gen"])):
        bad.append(f"{pid}: restricted policy log prob below the full-vocabulary log prob")
    chk = check_response(g["draft_ids"], example, tf)
    expect = {"token_correct": list(chk.token_correct), "parsed_values": list(chk.parsed_values),
              "slot_status": list(chk.slot_status), "value_correct": list(chk.value_correct),
              "final_correct": bool(chk.final_correct), "trace_correct": bool(chk.trace_correct),
              "well_formed": bool(chk.well_formed), "n_token_errors": chk.n_tokens - sum(chk.token_correct),
              "n_value_errors": len(chk.value_correct) - sum(chk.value_correct)}
    for k, v in expect.items():
        if a.get(k) != v:
            bad.append(f"{pid}: checker label {k} differs from a re-check")
    for p in range(L):
        c, q = g["log_c_gen"][p], a["log_q_gen"][p]
        if a["token_correct"][p]:
            if c != q:
                bad.append(f"{pid}: correct token at {p} has log_c != log_q")
                break
        elif max(c, q) + math.log1p(math.exp(-abs(c - q))) > CQ_LOG_TOL:
            bad.append(f"{pid}: wrong token at {p} has c + q > 1")
            break
    return bad
