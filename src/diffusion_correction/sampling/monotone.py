"""Ordinary monotone unmasking sampler (decision D-029).

Start: prompt fixed, every response position ``<M>``. For L editable tokens and T
model calls (1 <= T <= L), after call s the total revealed count is
K_s = ceil(s*L/T); call s reveals K_s - K_{s-1} >= 1 positions. Revealed tokens
never change, and after call T no mask remains (no extra filling call).

Per call and still-masked position: the candidate is the argmax of the aligned
float32 logits over the allowed token set (ties: smallest token ID). Positions are
ranked by the RAW temperature-1 log probability of their candidate (softmax over
the full output vocabulary; ties: smallest position first). The restricted-policy
log probability is stored separately and never used for ranking.

The sampler receives prompt IDs and public response lengths only: no targets, error
labels or correctness feedback. An optional observer sees each forward pass (for
milestone-4 confidence collection) and cannot influence it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch

from ..models.adapter import DiffusionAdapter
from ..models.alignment import build_layout


def reveal_schedule(length: int, calls: int) -> list[int]:
    """New reveals per call: K_s - K_{s-1} with K_s = ceil(s*L/T). Requires 1 <= T <= L."""
    if length < 1:
        raise ValueError("response length must be >= 1")
    if not 1 <= calls <= length:
        raise ValueError(f"infeasible budget: T={calls} calls for L={length} tokens (need 1 <= T <= L)")
    k = [0] + [-(-s * length // calls) for s in range(1, calls + 1)]  # exact integer ceil
    return [k[s] - k[s - 1] for s in range(1, calls + 1)]


def allowed_token_mask(vocab_size: int, tokenizer_size: int, special_ids: Sequence[int],
                       extra_forbidden: Sequence[int] = ()) -> torch.Tensor:
    """Bool [V]: ordinary tokenizer IDs only (no specials, no mask, no rows >= tokenizer size)."""
    allowed = torch.zeros(vocab_size, dtype=torch.bool)
    allowed[:min(tokenizer_size, vocab_size)] = True
    for i in list(special_ids) + list(extra_forbidden):
        if 0 <= i < vocab_size:
            allowed[i] = False
    return allowed


@dataclass
class CommitEvent:
    example: int            # index within the call's batch
    position: int           # response-relative position (0-based)
    token_id: int
    call: int               # 1-based model call that committed it
    mask_fraction_before: float
    raw_logp: float         # log p_raw(candidate), full vocabulary, temperature 1, float32
    policy_logp: float      # log p over the allowed set (restricted policy)


@dataclass
class SampleResult:
    responses: list[list[int]]
    calls_per_example: list[int]
    batched_forward_calls: int
    events: list[CommitEvent]
    traces: list[list[list[int]]] | None = None  # per example: response state after each call
    schedule: list[list[int]] = field(default_factory=list)  # per example: reveals per call
    final_input_ids: torch.Tensor | None = None    # [B, L_total] full padded state after the last call
    initial_input_ids: torch.Tensor | None = None  # [B, L_total] prompt + all-mask response + pad


Observer = Callable[[int, torch.Tensor, torch.Tensor, list[CommitEvent]], None]


@torch.no_grad()
def monotone_unmask(adapter: DiffusionAdapter, prompts: Sequence[Sequence[int]], response_lengths: Sequence[int],
                    calls: int, allowed: torch.Tensor, observer: Observer | None = None,
                    record_traces: bool = False) -> SampleResult:
    """Generate one batch. Every example shares the same number of calls T."""
    B = len(prompts)
    if len(response_lengths) != B:
        raise ValueError("one response length per prompt")
    schedules = [reveal_schedule(int(n), calls) for n in response_lengths]
    tok = adapter.tokens
    mask_id = tok.mask_token_id
    if allowed.shape != (tok.vocab_size,) or allowed.dtype != torch.bool:
        raise ValueError("allowed must be a bool tensor over the output vocabulary")
    if bool(allowed[mask_id]):
        raise ValueError("the mask token must not be an allowed candidate")
    layout = build_layout(prompts, [[mask_id] * int(n) for n in response_lengths], tok.pad_token_id)
    starts = [len(p) for p in prompts]
    device = adapter.device
    layout = layout.to(device)
    allowed_dev = allowed.to(device)
    vocab_ids = torch.arange(tok.vocab_size, device=device)
    ids = layout.input_ids.clone()
    traces = [[] for _ in range(B)] if record_traces else None
    events: list[CommitEvent] = []
    was_training = adapter.model.training
    adapter.model.eval()
    try:
        for s in range(1, calls + 1):
            masked = layout.editable_mask & (ids == mask_id)
            logits = adapter.aligned_logits(layout.with_input_ids(ids))
            rows = logits[masked].float()                                   # [N, V]
            raw_logp = torch.log_softmax(rows, dim=-1)
            restricted = rows.masked_fill(~allowed_dev, float("-inf"))
            best = restricted.max(dim=-1, keepdim=True).values
            cand = torch.where(restricted == best, vocab_ids, tok.vocab_size).min(dim=-1).values
            pol_logp = torch.log_softmax(restricted, dim=-1).gather(-1, cand[:, None]).squeeze(-1)
            score = raw_logp.gather(-1, cand[:, None]).squeeze(-1)
            where = masked.nonzero().cpu().tolist()                         # row-major (b, pos)
            # float32 -> Python float is exact, so equal scores stay equal.
            cand_c, score_c, pol_c = cand.cpu().tolist(), score.cpu().tolist(), pol_logp.cpu().tolist()
            by_example: dict[int, list[int]] = {}
            for r, (b, _) in enumerate(where):
                by_example.setdefault(b, []).append(r)
            call_events = []
            put_b, put_p, put_t = [], [], []
            for b in range(B):
                rs = by_example.get(b, [])
                n_new = schedules[b][s - 1]
                L = int(response_lengths[b])
                if len(rs) < n_new:
                    raise AssertionError(f"example {b}: {len(rs)} masks left, schedule wants {n_new}")
                frac = len(rs) / L
                # Highest raw log prob first; exact float32 ties -> smaller position first.
                order = sorted(rs, key=lambda r: (-score_c[r], where[r][1]))
                for r in order[:n_new]:
                    pos = where[r][1]
                    put_b.append(b)
                    put_p.append(pos)
                    put_t.append(int(cand_c[r]))
                    call_events.append(CommitEvent(example=b, position=pos - starts[b], token_id=int(cand_c[r]),
                                                   call=s, mask_fraction_before=frac, raw_logp=float(score_c[r]),
                                                   policy_logp=float(pol_c[r])))
            if observer is not None:
                observer(s, ids.clone(), logits.detach(), list(call_events))
            new_ids = ids.clone()
            new_ids[torch.tensor(put_b, device=device), torch.tensor(put_p, device=device)] = \
                torch.tensor(put_t, dtype=ids.dtype, device=device)
            ids = new_ids
            events.extend(call_events)
            if traces is not None:
                for b in range(B):
                    traces[b].append(ids[b, starts[b]:starts[b] + int(response_lengths[b])].tolist())
    finally:
        adapter.model.train(was_training)
    if bool((layout.editable_mask & (ids == mask_id)).any()):
        raise AssertionError("mask tokens remain after the final call")
    responses = [ids[b, starts[b]:starts[b] + int(response_lengths[b])].tolist() for b in range(B)]
    return SampleResult(responses=responses, calls_per_example=[calls] * B, batched_forward_calls=calls,
                        events=events, traces=traces, schedule=schedules, final_input_ids=ids.cpu(),
                        initial_input_ids=layout.input_ids.cpu())
