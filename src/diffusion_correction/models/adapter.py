"""Adapter contract shared by the real backbone and the tiny test backbone.

An adapter owns token-ID facts (mask/pad/EOS/vocab) and exposes one call,
:meth:`DiffusionAdapter.aligned_logits`, which never receives labels. Losses are
computed outside the model from the aligned logits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .alignment import SequenceLayout, align_logits


@dataclass(frozen=True)
class TokenSpec:
    mask_token_id: int
    pad_token_id: int
    eos_token_id: int
    vocab_size: int  # rows in the output layer (may exceed the tokenizer size)
    # Token IDs a sampler may never place in an editable position.
    forbidden_output_ids: tuple[int, ...] = ()


@dataclass
class DiffusionAdapter:
    model: nn.Module  # returns RAW logits [B, L, V] from (input_ids, bool attention_mask)
    tokens: TokenSpec
    provenance: dict = field(default_factory=dict)  # revisions, hashes; written into run records

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    def raw_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask)

    def aligned_logits(self, layout: SequenceLayout) -> torch.Tensor:
        """``[B, L, V]`` logits where row ``i`` is the distribution for token ``i``."""
        return align_logits(self.raw_logits(layout.input_ids, layout.attention_mask))

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
