"""Response-only masking for the ordinary masked-diffusion branch (D-026, D-028).

Per example b: u_b ~ U(0,1), t_b = clamp(u_b, 1/500, 1 - 1/500); each editable
response token is masked independently with probability t_b. Prompt and pad tokens
never change. Zero-mask examples are kept (no resampling, no forced mask).
Draws come from a dedicated CPU generator in a fixed order: for each example in
batch order, one float64 u, then one float64 draw per response position.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..models.alignment import SequenceLayout

T_MIN = 1.0 / 500
T_MAX = 1.0 - 1.0 / 500


@dataclass
class MaskDraw:
    masked_positions: torch.Tensor  # [B, L] bool, CPU; subset of the editable mask
    t: list[float]                  # clamped rate per example
    u: list[float]                  # raw uniform per example (before clamping)
    n_masked: list[int]             # masked tokens per example

    @property
    def total_masked(self) -> int:
        return int(sum(self.n_masked))


def sample_response_masks(layout: SequenceLayout, generator: torch.Generator,
                          t_min: float = T_MIN, t_max: float = T_MAX) -> MaskDraw:
    editable = layout.editable_mask.cpu()
    B, L = editable.shape
    masked = torch.zeros((B, L), dtype=torch.bool)
    ts, us, counts = [], [], []
    for b in range(B):
        pos = editable[b].nonzero().squeeze(-1)
        u = float(torch.rand(1, generator=generator, dtype=torch.float64).item())
        t = min(max(u, t_min), t_max)
        v = torch.rand(int(pos.numel()), generator=generator, dtype=torch.float64)
        chosen = pos[v < t]
        masked[b, chosen] = True
        us.append(u)
        ts.append(t)
        counts.append(int(chosen.numel()))
    return MaskDraw(masked_positions=masked, t=ts, u=us, n_masked=counts)


def apply_masks(input_ids: torch.Tensor, masked_positions: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    out = input_ids.clone()
    out[masked_positions.to(out.device)] = mask_token_id
    return out
