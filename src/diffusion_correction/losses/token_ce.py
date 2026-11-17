"""Plain-PyTorch unreduced cross-entropy on aligned logits.

No fused/chunked kernel is used. CDLM documents that liger-kernel 0.5.8's fused
linear CE (used by the Open-dLLM/CDLM 0.5B trainers) returns correct per-token
values but does not apply per-token weights in backward. Any future optimized
replacement must pass the value AND gradient tests in tests/test_losses.py.
"""

from __future__ import annotations

import torch

from ..models.alignment import assert_targets_predictable


def selected_token_ce(aligned_logits: torch.Tensor, targets: torch.Tensor,
                      positions: torch.Tensor) -> torch.Tensor:
    """CE ``-log p(target_i)`` for every True position, in row-major order.

    ``aligned_logits``: [B, L, V]; ``targets``: [B, L] long; ``positions``: [B, L] bool.
    log_softmax runs in at least float32 regardless of model dtype.
    """
    if positions.dtype != torch.bool:
        raise TypeError("positions must be a bool mask")
    if aligned_logits.shape[:2] != targets.shape or targets.shape != positions.shape:
        raise ValueError(f"shape mismatch {tuple(aligned_logits.shape)} / {tuple(targets.shape)} / "
                         f"{tuple(positions.shape)}")
    assert_targets_predictable(positions)
    rows = aligned_logits[positions]
    rows = rows.to(torch.promote_types(rows.dtype, torch.float32))
    logp = torch.log_softmax(rows, dim=-1)
    return -logp.gather(-1, targets[positions].unsqueeze(-1)).squeeze(-1)


def zero_with_graph(aligned_logits: torch.Tensor) -> torch.Tensor:
    """A scalar 0 that stays connected to the graph, so ``backward()`` works on empty sets."""
    return aligned_logits.sum() * 0.0
