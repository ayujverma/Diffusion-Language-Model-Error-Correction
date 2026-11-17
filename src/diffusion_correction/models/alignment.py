"""Logit/target alignment and explicit sequence-layout masks.

Open-dCoder keeps the autoregressive shift even though attention is bidirectional:
the raw model output at position i-1 is the distribution for the token at
position i (pinned `generation_utils.py`: ``logits = cat([logits[:, :1], logits[:, :-1]])``;
Open-dLLM trainer: ``labels[..., 1:]`` against ``hidden_states[..., :-1, :]``).

Every consumer (masked loss, correction loss, confidence collection, samplers)
must go through :func:`align_logits` so this convention lives in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

# Position 0 has no predictor under the shift; it must always be a prompt token.
FIRST_PREDICTABLE_POSITION = 1


def align_logits(raw_logits: torch.Tensor) -> torch.Tensor:
    """Re-index raw logits ``[B, L, V]`` so ``aligned[:, i]`` predicts token ``i``.

    ``aligned[:, 0]`` is a constant zero row (uniform distribution, no gradient path).
    Upstream fills it with ``raw[:, 0]`` instead; the value is irrelevant because
    position 0 is never a target, which :func:`assert_targets_predictable` enforces.
    """
    if raw_logits.dim() != 3:
        raise ValueError(f"expected [B, L, V] logits, got shape {tuple(raw_logits.shape)}")
    filler = torch.zeros_like(raw_logits[:, :1])
    return torch.cat([filler, raw_logits[:, :-1]], dim=1)


def assert_targets_predictable(position_mask: torch.Tensor) -> None:
    """Fail loudly if any selected target position has no aligned predictor."""
    if position_mask.dim() != 2:
        raise ValueError(f"expected [B, L] mask, got shape {tuple(position_mask.shape)}")
    if bool(position_mask[:, :FIRST_PREDICTABLE_POSITION].any()):
        raise ValueError("position 0 selected as a target; it has no aligned logit under the shift")


@dataclass(frozen=True)
class SequenceLayout:
    """Right-padded ``[prompt | response | pad]`` batch with explicit masks.

    Masks are built from lengths, never from token-ID equality: in Open-dCoder the
    pad token and EOS are the same ID (151643), so ``input_ids != pad_id`` would
    also hide legitimate EOS tokens.
    """

    input_ids: torch.Tensor  # [B, L] long
    attention_mask: torch.Tensor  # [B, L] bool, True for real (non-pad) tokens
    prompt_mask: torch.Tensor  # [B, L] bool
    response_mask: torch.Tensor  # [B, L] bool
    editable_mask: torch.Tensor  # [B, L] bool; positions a sampler/corrector may change

    @property
    def pad_mask(self) -> torch.Tensor:
        return ~self.attention_mask

    def to(self, device: torch.device | str) -> "SequenceLayout":
        return SequenceLayout(*(t.to(device) for t in (
            self.input_ids, self.attention_mask, self.prompt_mask, self.response_mask, self.editable_mask)))

    def with_input_ids(self, input_ids: torch.Tensor) -> "SequenceLayout":
        if input_ids.shape != self.input_ids.shape:
            raise ValueError("replacement input_ids must keep the layout shape")
        return SequenceLayout(input_ids, self.attention_mask, self.prompt_mask, self.response_mask,
                              self.editable_mask)


def build_layout(
    prompts: Sequence[Sequence[int]],
    responses: Sequence[Sequence[int]],
    pad_token_id: int,
    total_length: int | None = None,
) -> SequenceLayout:
    """Concatenate prompt and response token IDs per example and right-pad.

    The response tokens may be targets, drafts, or mask tokens; the layout does not
    care. Every prompt must be non-empty so that position 0 is never editable.
    """
    if len(prompts) != len(responses):
        raise ValueError(f"{len(prompts)} prompts vs {len(responses)} responses")
    lengths = [len(p) + len(r) for p, r in zip(prompts, responses)]
    L = max(lengths) if total_length is None else total_length
    if any(n > L for n in lengths):
        raise ValueError(f"sequence longer than total_length={L}")
    B = len(prompts)
    input_ids = torch.full((B, L), pad_token_id, dtype=torch.long)
    attention = torch.zeros((B, L), dtype=torch.bool)
    prompt_mask = torch.zeros((B, L), dtype=torch.bool)
    response_mask = torch.zeros((B, L), dtype=torch.bool)
    for b, (p, r) in enumerate(zip(prompts, responses)):
        if len(p) < FIRST_PREDICTABLE_POSITION:
            raise ValueError("prompt must contain at least one token")
        n_p, n_r = len(p), len(r)
        input_ids[b, :n_p] = torch.tensor(list(p), dtype=torch.long)
        input_ids[b, n_p:n_p + n_r] = torch.tensor(list(r), dtype=torch.long)
        attention[b, :n_p + n_r] = True
        prompt_mask[b, :n_p] = True
        response_mask[b, n_p:n_p + n_r] = True
    return SequenceLayout(input_ids, attention, prompt_mask, response_mask, response_mask.clone())
