"""Correction-example losses on a saved draft.

L = L_mask + lambda * L_error + mu * L_keep
L_error = sum_{i in E} w_i * ell_i / |E|
L_keep  = mean_{i in K} ell_i
where ell_i = -log p_theta(target_i | question, draft) from the aligned current
logits, E are wrong editable draft tokens, and K are already-correct editable draft
tokens. Empty sets give a graph-connected zero; their frequency is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .token_ce import selected_token_ce, zero_with_graph

WeightsArg = torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None


def token_error_masks(draft_ids: torch.Tensor, target_ids: torch.Tensor,
                      editable_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(E, K) masks: editable positions where the draft differs from / equals the target."""
    if draft_ids.shape != target_ids.shape or draft_ids.shape != editable_mask.shape:
        raise ValueError("draft, target and editable mask must share a shape")
    wrong = draft_ids != target_ids
    return editable_mask & wrong, editable_mask & ~wrong


@dataclass
class CorrectionLossOutput:
    l_error: torch.Tensor
    l_keep: torch.Tensor
    n_error: int
    n_keep: int
    error_ce: torch.Tensor           # detached, row-major order of the error mask
    error_weights: torch.Tensor      # detached
    error_weighted_ce: torch.Tensor  # detached, w_i * ell_i
    keep_ce: torch.Tensor            # detached


def correction_losses(aligned_logits: torch.Tensor, targets: torch.Tensor, error_mask: torch.Tensor,
                      keep_mask: torch.Tensor, weights: WeightsArg = None) -> CorrectionLossOutput:
    """``weights``: None (uniform), a 1-D tensor per error token, or a callable that
    receives the detached error CE and returns such a tensor (current-loss rule)."""
    if bool((error_mask & keep_mask).any()):
        raise ValueError("error and keep sets overlap")
    err_ce = selected_token_ce(aligned_logits, targets, error_mask)
    keep_ce = selected_token_ce(aligned_logits, targets, keep_mask)
    n_err, n_keep = int(err_ce.numel()), int(keep_ce.numel())

    if callable(weights):
        weights = weights(err_ce.detach())
    if weights is None:
        w = torch.ones_like(err_ce).detach()
    else:
        if weights.requires_grad:
            raise ValueError("error weights must be detached")
        if weights.dim() != 1 or weights.numel() != n_err:
            raise ValueError(f"expected {n_err} error weights, got shape {tuple(weights.shape)}")
        w = weights.to(device=err_ce.device, dtype=err_ce.dtype)

    weighted = w * err_ce
    l_error = weighted.sum() / n_err if n_err else zero_with_graph(aligned_logits)
    l_keep = keep_ce.sum() / n_keep if n_keep else zero_with_graph(aligned_logits)
    return CorrectionLossOutput(l_error=l_error, l_keep=l_keep, n_error=n_err, n_keep=n_keep,
                                error_ce=err_ce.detach(), error_weights=w.detach(),
                                error_weighted_ce=weighted.detach(), keep_ce=keep_ce.detach())


def combined_objective(l_mask: torch.Tensor, corr: CorrectionLossOutput, lam: float,
                       mu: float) -> tuple[torch.Tensor, dict]:
    total = l_mask + lam * corr.l_error + mu * corr.l_keep
    parts = {"L_mask": float(l_mask.detach()), "L_error": float(corr.l_error.detach()),
             "L_keep": float(corr.l_keep.detach()), "lambda": lam, "mu": mu,
             "n_error": corr.n_error, "n_keep": corr.n_keep,
             "error_set_empty": corr.n_error == 0, "keep_set_empty": corr.n_keep == 0,
             "L_total": float(total.detach())}
    return total, parts
