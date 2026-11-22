"""Shared return value for milestone-6 samplers."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SamplerRun:
    responses: list[list[int]]
    calls_per_example: list[int]
    batched_forward_calls: int
    call_kinds: list[str]
    max_calls: int
    masks_remaining: list[int]
    # Returnable response after each completed forward of THIS horizon.
    # Not a smaller-budget run: unmask quotas and D3IM keep-counts depend on the
    # planned horizon (D-049). Empty when the caller did not ask for it.
    trajectory: list[list[list[int]]] = field(default_factory=list)
    final_input_ids: torch.Tensor | None = None
    plan: dict = field(default_factory=dict)
