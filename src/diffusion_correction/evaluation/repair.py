"""Repair of an already-complete saved draft.

``repair_drafts`` is the milestone-5 screen metric (D-022, D-044): one greedy
forward, rewriting every editable token. ``repair_drafts_proseco`` is the
milestone-6 path (D-049): the same greedy rule, repeated as fixed-point
iterations up to a call budget. Each iteration is one NFE. The one-forward path
is unchanged.

Metrics use explicit denominators:
- repaired = was wrong and became correct / n_wrong_before
- damaged  = was correct and became wrong / n_correct_before
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..models.adapter import DiffusionAdapter
from ..models.alignment import build_layout
from ..sampling.monotone import allowed_token_mask
from ..sampling.nfe import ForwardMeter
from ..sampling.proseco import DEFAULT_CORRECTOR_STEPS
from ..training.bank_data import BankDraft


def _greedy_response(aligned_logits: torch.Tensor, editable_mask: torch.Tensor,
                     allowed: torch.Tensor) -> list[list[int]]:
    """Argmax over allowed IDs at every editable position; row-major per example."""
    B, L, V = aligned_logits.shape
    if allowed.numel() != V:
        raise ValueError(f"allowed mask length {allowed.numel()} != vocab {V}")
    # Mask disallowed IDs to -inf in float32.
    rows = aligned_logits.float()
    neg = torch.finfo(rows.dtype).min
    rows = rows.masked_fill(~allowed.to(device=rows.device, dtype=torch.bool).view(1, 1, V), neg)
    pred = rows.argmax(dim=-1)  # [B, L]
    out = []
    for b in range(B):
        idx = editable_mask[b].nonzero(as_tuple=True)[0]
        out.append([int(pred[b, i]) for i in idx.tolist()])
    return out


@torch.no_grad()
def repair_drafts(adapter: DiffusionAdapter, drafts: Sequence[BankDraft], *,
                  batch_size: int = 32, allowed: torch.Tensor | None = None) -> list[dict]:
    """Repair each complete draft with one forward. Returns per-draft records."""
    if not drafts:
        return []
    if allowed is None:
        # Prefer the tokenizer-sized allowed set when provenance carries it; else full vocab minus forbidden.
        tok = adapter.tokens
        forbidden = set(tok.forbidden_output_ids) | {tok.mask_token_id, tok.pad_token_id}
        # Open-dCoder: tokenizer size may be < output rows; tiny: equal.
        n_allowed = getattr(adapter, "tokenizer_size", None) or tok.vocab_size
        # For Open-dCoder the adapter does not store tokenizer_size; callers should pass allowed.
        # Fall back: forbid only the TokenSpec forbidden set.
        allowed = torch.ones(tok.vocab_size, dtype=torch.bool)
        for i in forbidden:
            if 0 <= i < tok.vocab_size:
                allowed[i] = False
        if n_allowed < tok.vocab_size:
            allowed[n_allowed:] = False
    device = next(adapter.model.parameters()).device
    allowed = allowed.to(device)
    was = adapter.model.training
    adapter.model.eval()
    records: list[dict] = []
    for start in range(0, len(drafts), batch_size):
        chunk = drafts[start:start + batch_size]
        prompts = [d.prompt_ids for d in chunk]
        draft_ids = [d.draft_ids for d in chunk]
        refs = [d.reference_ids for d in chunk]
        lay = build_layout(prompts, draft_ids, adapter.tokens.pad_token_id)
        # Targets share lengths with drafts by bank construction.
        tgt = build_layout(prompts, refs, adapter.tokens.pad_token_id, total_length=lay.input_ids.shape[1])
        lay_d = lay.to(device)
        logits = adapter.aligned_logits(lay_d)
        repaired_responses = _greedy_response(logits, lay_d.editable_mask, allowed)
        for d, pred, ref in zip(chunk, repaired_responses, refs):
            records.append(_repair_record(d, pred, ref))
    adapter.model.train(was)
    return records


def _write_responses(ids: torch.Tensor, editable: torch.Tensor, responses: list[list[int]]) -> None:
    for b, pred in enumerate(responses):
        pos = editable[b].nonzero(as_tuple=True)[0].tolist()
        if len(pos) != len(pred):
            raise AssertionError("editable count does not match the response")
        for p, tok in zip(pos, pred, strict=True):
            ids[b, p] = tok


def _repair_record(draft: BankDraft, pred: list[int], ref: list[int], calls: int | None = None) -> dict:
    if len(pred) != len(ref) or len(pred) != len(draft.draft_ids):
        raise ValueError(f"{draft.problem_id}: length mismatch pred/ref/draft")
    before_ok = [a == b for a, b in zip(draft.draft_ids, ref)]
    after_ok = [a == b for a, b in zip(pred, ref)]
    n = len(ref)
    n_wrong_before = sum(1 for x in before_ok if not x)
    n_correct_before = n - n_wrong_before
    n_repaired = sum(1 for b0, a0 in zip(before_ok, after_ok) if (not b0) and a0)
    n_damaged = sum(1 for b0, a0 in zip(before_ok, after_ok) if b0 and (not a0))
    n_wrong_after = sum(1 for x in after_ok if not x)
    rec = {
        "problem_id": draft.problem_id, "split": draft.split, "budget": draft.budget,
        "family": draft.family, "length": draft.length, "operand_digits": draft.operand_digits,
        "setting": draft.setting, "response_length": n,
        "n_wrong_before": n_wrong_before, "n_correct_before": n_correct_before,
        "n_repaired": n_repaired, "n_damaged": n_damaged, "n_wrong_after": n_wrong_after,
        "n_correct_after": n - n_wrong_after,
        "draft_final_correct": draft.final_correct, "draft_well_formed": draft.well_formed,
        "token_correct_before": before_ok, "token_correct_after": after_ok,
        "repaired_ids": pred,
    }
    if calls is not None:
        rec["calls"] = calls
    return rec


@torch.no_grad()
def repair_drafts_proseco(adapter: DiffusionAdapter, drafts: Sequence[BankDraft], *,
                          max_calls: int = DEFAULT_CORRECTOR_STEPS, batch_size: int = 32,
                          allowed: torch.Tensor | None = None) -> list[dict]:
    """Multi-step repair. One greedy rewrite of every editable token per call.

    The first iteration uses ``_greedy_response``, the milestone-5 rule, so
    ``max_calls=1`` matches ``repair_drafts`` token for token. Later iterations
    re-forward the rewritten draft. The batch stops when every draft is unchanged,
    and the unused calls are not backfilled. Every draft in a batch is charged
    that batch's forward count (D-049).
    """
    if max_calls < 1:
        raise ValueError("max_calls must be >= 1")
    if not drafts:
        return []
    if allowed is None:
        allowed = make_allowed_for_adapter(adapter)
    device = next(adapter.model.parameters()).device
    allowed = allowed.to(device)
    was = adapter.model.training
    adapter.model.eval()
    records: list[dict] = []
    try:
        for start in range(0, len(drafts), batch_size):
            chunk = list(drafts[start:start + batch_size])
            current = [list(d.draft_ids) for d in chunk]
            lay = build_layout([d.prompt_ids for d in chunk], current, adapter.tokens.pad_token_id).to(device)
            ids = lay.input_ids.clone()
            meter = ForwardMeter(adapter, max_calls=max_calls)
            preds = current
            for step in range(1, max_calls + 1):
                logits = meter.forward(lay.with_input_ids(ids), kind="corrector", step=step)
                if logits is None:
                    break
                preds = _greedy_response(logits, lay.editable_mask, allowed)
                if preds == current:
                    break
                _write_responses(ids, lay.editable_mask, preds)
                current = preds
            for d, pred, ref in zip(chunk, preds, [d.reference_ids for d in chunk], strict=True):
                records.append(_repair_record(d, pred, ref, calls=meter.nfe))
    finally:
        adapter.model.train(was)
    return records


def summarize_repair(records: Sequence[dict]) -> dict:
    """Aggregate repair/damage with denominators. Rates are None when the denom is 0."""
    n = len(records)
    tw = sum(r["n_wrong_before"] for r in records)
    tc = sum(r["n_correct_before"] for r in records)
    repaired = sum(r["n_repaired"] for r in records)
    damaged = sum(r["n_damaged"] for r in records)
    wrong_after = sum(r["n_wrong_after"] for r in records)
    drafts_with_errors = sum(1 for r in records if r["n_wrong_before"] > 0)
    drafts_fully_repaired = sum(1 for r in records
                                if r["n_wrong_before"] > 0 and r["n_wrong_after"] == 0)
    drafts_damaged = sum(1 for r in records if r["n_damaged"] > 0)
    out = {
        "n_drafts": n,
        "n_tokens": tw + tc,
        "n_wrong_before": tw,
        "n_correct_before": tc,
        "n_repaired": repaired,
        "n_damaged": damaged,
        "n_wrong_after": wrong_after,
        "repair_rate": (repaired / tw) if tw else None,
        "damage_rate": (damaged / tc) if tc else None,
        "token_accuracy_after": ((tw + tc - wrong_after) / (tw + tc)) if (tw + tc) else None,
        "drafts_with_errors": drafts_with_errors,
        "drafts_fully_repaired": drafts_fully_repaired,
        "drafts_with_damage": drafts_damaged,
        "fraction_error_drafts_fully_repaired": (
            drafts_fully_repaired / drafts_with_errors) if drafts_with_errors else None,
    }
    if records and "calls" in records[0]:
        total_calls = sum(int(r["calls"]) for r in records)
        out["model_calls_total"] = total_calls
        out["model_calls_per_draft"] = (total_calls / n) if n else None
    return out


def make_allowed_for_adapter(adapter: DiffusionAdapter, tokenizer_size: int | None = None,
                             special_ids: Sequence[int] | None = None) -> torch.Tensor:
    """Allowed-token mask matching the D-029 sampler when tokenizer metadata is available."""
    tok = adapter.tokens
    if tokenizer_size is None:
        tokenizer_size = tok.vocab_size
    special = list(special_ids) if special_ids is not None else list(tok.forbidden_output_ids)
    return allowed_token_mask(tok.vocab_size, tokenizer_size, special,
                              extra_forbidden=(tok.mask_token_id,))
