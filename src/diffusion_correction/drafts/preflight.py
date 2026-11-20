"""Startup gates run inside the full collection job, before any shard is written.

Each configured preflight batch (50 problems from the start of one split/setting, in
file order, which is also the first batch of the corresponding bank shard) runs:

1. **Gate D.** Collection with the true references, with references permuted within
   the batch, and plain generation without an observer. Everything under
   ``generation`` (drafts, commit calls/order, mask fractions, log c, rank and policy
   scores) and the call counts must be identical across the three; the sampler's own
   ranking scores must equal the observer's log c bitwise; target-dependent fields
   must change (else the gate is vacuous), and ``log_q`` must stay equal wherever the
   permuted target equals the true one.
2. **Same-forward recompute.** For saved pre-commit states (first, middle and last
   call), a fresh batched forward of the same state gives log c and log q equal to the
   stored values (exact).
3. **Independent single-example check.** Each example's state re-run alone (B = 1)
   through the raw model with a manual one-position shift; log c / log q within 2e-3
   (D-004's < 1e-3 batch-vs-alone logit bound, doubled for the log-sum-exp term).
4. **Reproduction.** Validation drafts equal the stored records of the job that
   evaluated the generator (3595133), problem by problem (exact).
5. **Record invariants** (``check_record``) and unchanged CPU/CUDA RNG states.

Any failure stops the job before collection. Timings are recorded separately.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch

from ..data.format import TokenizedFormat
from ..data.tasks import Example
from ..env import synchronize
from ..models.alignment import build_layout
from ..sampling.monotone import monotone_unmask
from ..training.data import setting_key
from .bank import now_utc, write_bytes_atomic, write_json_atomic
from .collector import calls_for, check_record, collect_batch, generation_view
from .identity import sha256_file

INDEPENDENT_LOGP_TOL = 2e-3
PREFLIGHT_SCHEMA = "dlmc.bank_preflight.v1"


def preflight_examples(examples: dict[str, list[Example]], spec: dict) -> list[Example]:
    fam, n, dig = spec["setting"].split(":")
    key = (fam, int(n), int(dig))
    exs = [e for e in examples[spec["split"]] if setting_key(e) == key]
    if len(exs) < int(spec["n"]):
        raise ValueError(f"preflight {spec}: only {len(exs)} problems")
    return exs[:int(spec["n"])]


def permuted_references(exs: list[Example]) -> list[Example]:
    """Copies whose reference (response IDs, values, final answer) is the next example's (cyclic)."""
    out = []
    for i, ex in enumerate(exs):
        donor = exs[(i + 1) % len(exs)]
        c = copy.deepcopy(ex)
        c.response_ids = list(donor.response_ids)
        c.values = list(donor.values)
        c.final_answer = donor.final_answer
        out.append(c)
    return out


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def _load_reference(path: str | Path | None) -> dict | None:
    if not path:
        return None
    out = {}
    for line in Path(path).read_text().splitlines():
        if line:
            r = json.loads(line)
            out[(r["problem_id"], str(r["budget"]))] = r["response_ids"]
    return out


def _event_rows(res, call: int):
    """(example, position-in-response) -> (log_c, log_q) for commits made at ``call``."""
    out = {}
    for b, commits in enumerate(res.observer.commits):
        for p, c in enumerate(commits):
            if c.call == call:
                out[(b, p)] = (c.token_id, c.log_c, c.log_q)
    return out


def run_batch_gates(adapter, tf: TokenizedFormat, allowed: torch.Tensor, exs: list[Example], budget: str,
                    reference: dict | None) -> dict:
    budget = str(budget)
    L = len(exs[0].value_names) * tf.fmt.slot_len
    T = calls_for(budget, L)
    keep = sorted({1, (T + 1) // 2, T})
    dev = adapter.device
    timings = {}
    t = time.perf_counter()
    true = collect_batch(adapter, exs, budget, allowed, tf, keep_states_for_calls=keep)
    synchronize(dev)
    timings["collect_true"] = time.perf_counter() - t
    t = time.perf_counter()
    perm_exs = permuted_references(exs)
    perm = collect_batch(adapter, perm_exs, budget, allowed, tf)
    synchronize(dev)
    timings["collect_permuted"] = time.perf_counter() - t
    t = time.perf_counter()
    prompts = [e.prompt_ids for e in exs]
    lengths = [L] * len(exs)
    plain = monotone_unmask(adapter, prompts, lengths, T, allowed)
    synchronize(dev)
    timings["generate_plain"] = time.perf_counter() - t

    checks: dict[str, object] = {}
    gv_true = [_canon(generation_view(r)) for r in true.records]
    gv_perm = [_canon(generation_view(r)) for r in perm.records]
    checks["gate_d_generation_identical_true_vs_permuted"] = gv_true == gv_perm
    checks["gate_d_drafts_identical_with_and_without_observer"] = \
        [r["generation"]["draft_ids"] for r in true.records] == plain.responses
    plain_scores = {(ev.example, ev.position): (ev.call, ev.token_id, ev.raw_logp) for ev in plain.events}
    obs_scores = {(b, p): (c.call, c.token_id, c.log_c) for b, cs in enumerate(true.observer.commits)
                  for p, c in enumerate(cs)}
    checks["gate_d_commits_and_scores_identical_without_observer"] = plain_scores == obs_scores
    changed_q = changed_labels = 0
    q_stable = True
    for rt, rp, et, ep in zip(true.records, perm.records, exs, perm_exs):
        for p in range(L):
            same_target = et.response_ids[p] == ep.response_ids[p]
            qt, qp = rt["analysis"]["log_q_gen"][p], rp["analysis"]["log_q_gen"][p]
            if same_target and qt != qp:
                q_stable = False
            changed_q += qt != qp
        changed_labels += _canon(rt["analysis"]["token_correct"]) != _canon(rp["analysis"]["token_correct"])
    checks["gate_d_target_dependent_fields_change"] = changed_q > 0 and changed_labels > 0
    checks["gate_d_log_q_unchanged_where_target_unchanged"] = q_stable
    checks["rank_score_mismatches"] = true.rank_score_mismatches + perm.rank_score_mismatches
    checks["rng_unchanged"] = true.rng_unchanged and perm.rng_unchanged
    problems = [m for r, e in zip(true.records, exs) for m in check_record(r, e, tf)]
    problems += [m for r, e in zip(perm.records, perm_exs) for m in check_record(r, e, tf)]
    checks["record_invariant_problems"] = problems[:20]

    # Same-forward recompute and independent single-example check on saved pre-commit states.
    t = time.perf_counter()
    mask_id = adapter.tokens.mask_token_id
    layout = build_layout(prompts, [[mask_id] * L for _ in exs], adapter.tokens.pad_token_id).to(dev)
    exact_bad, n_cmp, max_ind = 0, 0, 0.0
    adapter.model.eval()
    with torch.no_grad():
        for s in keep:
            state = true.observer.saved_states[s].to(dev)
            logits = adapter.aligned_logits(layout.with_input_ids(state))
            masked = layout.editable_mask & (state == mask_id)
            lsm = torch.log_softmax(logits[masked].float(), dim=-1)
            row_of = {(b, pos): r for r, (b, pos) in enumerate(masked.nonzero().cpu().tolist())}
            rows = _event_rows(true, s)
            for (b, p), (tok, lc, lq) in rows.items():
                r = row_of[(b, len(prompts[b]) + p)]
                ref = exs[b].response_ids[p]
                if float(lsm[r, tok]) != lc or float(lsm[r, ref]) != lq:
                    exact_bad += 1
                n_cmp += 1
            del logits, lsm
            for b in range(len(exs)):
                n = len(prompts[b]) + L
                one = state[b:b + 1, :n]
                raw = adapter.raw_logits(one, torch.ones_like(one, dtype=torch.bool))
                for (bb, p), (tok, lc, lq) in rows.items():
                    if bb != b:
                        continue
                    pos = len(prompts[b]) + p
                    lp = torch.log_softmax(raw[0, pos - 1].float(), dim=-1)   # manual shift: row pos-1
                    ref = exs[b].response_ids[p]
                    max_ind = max(max_ind, abs(float(lp[tok]) - lc), abs(float(lp[ref]) - lq))
                del raw
    synchronize(dev)
    timings["recompute_and_independent"] = time.perf_counter() - t
    checks["same_forward_recompute_exact_mismatches"] = exact_bad
    checks["same_forward_recompute_compared"] = n_cmp
    checks["independent_single_example_max_abs_logp_diff"] = max_ind
    if reference is not None and exs[0].split == "val":
        miss = [e.problem_id for e in exs if (e.problem_id, budget) not in reference]
        diff = [e.problem_id for e, r in zip(exs, true.records)
                if (e.problem_id, budget) in reference
                and reference[(e.problem_id, budget)] != r["generation"]["draft_ids"]]
        checks["reproduction_missing"] = len(miss)
        checks["reproduction_differing"] = diff[:20]
        checks["reproduction_compared"] = len(exs) - len(miss)
    passed = (all(checks[k] is True for k in checks if k.startswith("gate_d_"))
              and checks["rank_score_mismatches"] == 0 and checks["rng_unchanged"] is True
              and not problems and exact_bad == 0 and n_cmp > 0 and max_ind <= INDEPENDENT_LOGP_TOL
              and checks.get("reproduction_missing", 0) == 0 and not checks.get("reproduction_differing"))
    return {"split": exs[0].split, "setting": ":".join(map(str, setting_key(exs[0]))), "budget": budget,
            "n": len(exs), "calls": T, "saved_calls": keep, "passed": passed, "checks": checks,
            "seconds": {k: round(v, 3) for k, v in timings.items()},
            "batched_forwards": 3 * T + len(keep) * (1 + len(exs))}


def run_preflight(adapter, tf: TokenizedFormat, allowed: torch.Tensor, examples: dict[str, list[Example]],
                  cfg: dict, out_dir: Path, identity_sha256: str, reference_records: str | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "PREFLIGHT_PASSED"
    if marker.exists():
        raise SystemExit(f"{marker} exists; preflight output is never overwritten")
    pf = cfg["preflight"]
    ref_sha = None
    if reference_records:
        ref_sha = sha256_file(Path(reference_records))
        if pf.get("reference_records_sha256") and ref_sha != pf["reference_records_sha256"]:
            raise SystemExit(f"reference records sha256 {ref_sha} != pinned {pf['reference_records_sha256']}")
    elif any(b["split"] == "val" for b in pf["batches"]):
        raise SystemExit("validation preflight batches need --reference-records")
    reference = _load_reference(reference_records)
    t0 = time.perf_counter()
    results = []
    for spec in pf["batches"]:
        r = run_batch_gates(adapter, tf, allowed, preflight_examples(examples, spec), str(spec["budget"]), reference)
        results.append(r)
        print(f"preflight {r['split']} {r['setting']} T={r['budget']}: {'PASS' if r['passed'] else 'FAIL'} "
              f"{json.dumps(r['checks'], default=str)}", flush=True)
        if not r["passed"]:
            break
    passed = len(results) == len(pf["batches"]) and all(r["passed"] for r in results)
    report = {"schema": PREFLIGHT_SCHEMA, "time_utc": now_utc(), "identity_sha256": identity_sha256,
              "passed": passed, "batches": results, "reference_records_sha256": ref_sha,
              "independent_logp_tolerance": INDEPENDENT_LOGP_TOL,
              "preflight_seconds": round(time.perf_counter() - t0, 2),
              "total_batched_forwards": sum(r["batched_forwards"] for r in results),
              "note": ("preflight compute is separate from collection compute; nothing here is a bank "
                       "record; same-forward recomputes and B=1 checks run only on these batches")}
    write_json_atomic(out_dir / "preflight.json", report)
    if passed:
        write_bytes_atomic(marker, f"identity_sha256 {identity_sha256}\n".encode())
    return report
