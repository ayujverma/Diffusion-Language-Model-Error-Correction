"""Minimal repair-of-complete-draft evaluation (milestone 5; D-022).

One forward on an already-complete saved draft. Every editable position is
re-predicted greedily from the aligned logits (allowed vocabulary). Full
correction samplers stay in milestone 6.

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
            if len(pred) != len(ref) or len(pred) != len(d.draft_ids):
                raise ValueError(f"{d.problem_id}: length mismatch pred/ref/draft")
            before_ok = [a == b for a, b in zip(d.draft_ids, ref)]
            after_ok = [a == b for a, b in zip(pred, ref)]
            n = len(ref)
            n_wrong_before = sum(1 for x in before_ok if not x)
            n_correct_before = n - n_wrong_before
            n_repaired = sum(1 for b0, a0 in zip(before_ok, after_ok) if (not b0) and a0)
            n_damaged = sum(1 for b0, a0 in zip(before_ok, after_ok) if b0 and (not a0))
            n_wrong_after = sum(1 for x in after_ok if not x)
            records.append({
                "problem_id": d.problem_id, "split": d.split, "budget": d.budget,
                "family": d.family, "length": d.length, "operand_digits": d.operand_digits,
                "setting": d.setting, "response_length": n,
                "n_wrong_before": n_wrong_before, "n_correct_before": n_correct_before,
                "n_repaired": n_repaired, "n_damaged": n_damaged, "n_wrong_after": n_wrong_after,
                "n_correct_after": n - n_wrong_after,
                "draft_final_correct": d.final_correct, "draft_well_formed": d.well_formed,
                "token_correct_before": before_ok, "token_correct_after": after_ok,
                "repaired_ids": pred,
            })
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
    return {
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


def make_allowed_for_adapter(adapter: DiffusionAdapter, tokenizer_size: int | None = None,
                             special_ids: Sequence[int] | None = None) -> torch.Tensor:
    """Allowed-token mask matching the D-029 sampler when tokenizer metadata is available."""
    tok = adapter.tokens
    if tokenizer_size is None:
        tokenizer_size = tok.vocab_size
    special = list(special_ids) if special_ids is not None else list(tok.forbidden_output_ids)
    return allowed_token_mask(tok.vocab_size, tokenizer_size, special,
                              extra_forbidden=(tok.mask_token_id,))
