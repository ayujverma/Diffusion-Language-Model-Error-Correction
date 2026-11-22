"""Greedy allowed-token selection shared by the milestone-6 samplers (D-049).

Ties on the maximal allowed logit take the smallest token ID, matching D-029 and
``torch.argmax``. The ranking score is the float32 full-vocabulary log-softmax of
that candidate (temperature 1), not the restricted-policy probability.
"""

from __future__ import annotations

import torch


def greedy_allowed(aligned_logits: torch.Tensor, allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Argmax over the allowed set and its float32 full-vocabulary log-probability.

    ``aligned_logits`` is ``[B, L, V]``. Returns ``cand`` ``[B, L]`` long and
    ``raw_logp`` ``[B, L]`` float32.
    """
    if aligned_logits.dim() != 3:
        raise ValueError(f"expected [B, L, V] logits, got {tuple(aligned_logits.shape)}")
    rows = aligned_logits.float()
    _b, _l, vocab = rows.shape
    if allowed.shape != (vocab,) or allowed.dtype != torch.bool:
        raise ValueError("allowed must be a bool mask over the output vocabulary")
    allowed_dev = allowed.to(device=rows.device)
    if not bool(allowed_dev.any()):
        raise ValueError("allowed mask is empty")
    restricted = rows.masked_fill(~allowed_dev.view(1, 1, vocab), float("-inf"))
    best = restricted.max(dim=-1, keepdim=True).values
    vocab_ids = torch.arange(vocab, device=rows.device)
    cand = torch.where(restricted == best, vocab_ids, vocab).min(dim=-1).values
    if bool((cand == vocab).any()):
        raise RuntimeError("a row has no allowed token")
    raw_logp = torch.log_softmax(rows, dim=-1).gather(-1, cand.unsqueeze(-1)).squeeze(-1)
    return cand, raw_logp


def token_raw_logp(aligned_logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Float32 full-vocabulary log-probability of the given token at each position."""
    if aligned_logits.dim() != 3:
        raise ValueError(f"expected [B, L, V] logits, got {tuple(aligned_logits.shape)}")
    rows = aligned_logits.float()
    if token_ids.shape != rows.shape[:2]:
        raise ValueError("token_ids must match the logit batch and length")
    vocab = rows.shape[-1]
    if int(token_ids.min()) < 0 or int(token_ids.max()) >= vocab:
        raise ValueError("token id outside the output vocabulary")
    logp = torch.log_softmax(rows, dim=-1)
    return logp.gather(-1, token_ids.long().unsqueeze(-1)).squeeze(-1)
