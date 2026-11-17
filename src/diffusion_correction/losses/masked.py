"""Ordinary masked-denoising loss (the L_mask branch and the common warmup objective).

Contract (docs/decisions.md D-006): mean CE over masked response positions, pooled
over the physical minibatch. No 1/t time weighting and no auxiliary "path" term,
which the Open-dLLM trainer adds; mask rate sampling lives in the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .token_ce import selected_token_ce, zero_with_graph


@dataclass
class MaskedLossOutput:
    loss: torch.Tensor
    n_masked: int
    token_ce: torch.Tensor  # detached, one entry per masked position


def masked_diffusion_loss(aligned_logits: torch.Tensor, targets: torch.Tensor,
                          masked_positions: torch.Tensor, editable_mask: torch.Tensor) -> MaskedLossOutput:
    if bool((masked_positions & ~editable_mask).any()):
        raise ValueError("masked positions must be editable response positions (no prompt/pad)")
    ce = selected_token_ce(aligned_logits, targets, masked_positions)
    n = int(ce.numel())
    loss = ce.sum() / n if n else zero_with_graph(aligned_logits)
    return MaskedLossOutput(loss=loss, n_masked=n, token_ce=ce.detach())
