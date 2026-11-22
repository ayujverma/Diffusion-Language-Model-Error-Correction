"""Family B training losses (milestone 7, D-051).

ProSeCo arms share one online draft. ``z_t`` masks the gold response. The first
forward keeps a gradient for ``L_mask`` and, under stop-grad, writes an allowed-set
argmax at every editable position. The second forward is the corrector. ``a(t) = 1/t``
multiplies the corrector only. SCOPE is added after Q-5 is closed; it is not this file's
ProSeCo path.

These are adaptations to Open-dCoder. They are not reproductions of the papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..models.adapter import DiffusionAdapter
from ..models.alignment import SequenceLayout, assert_targets_predictable
from ..sampling.candidates import greedy_allowed
from .correction import token_error_masks
from .masked import masked_diffusion_loss
from .token_ce import zero_with_graph

def time_weight(t: float) -> float:
    """``a(t) = 1/t`` for a mask rate in ``(0, 1]`` (D-051)."""
    t = float(t)
    if not 0.0 < t <= 1.0:
        raise ValueError(f"mask rate t must lie in (0, 1], got {t}")
    return 1.0 / t


def corrector_log_probs(aligned_logits: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Log-probabilities with the mask id removed from the normalizer.

    Matches the pinned corrector's ``mask_index += -inf`` before ``log_softmax`` on
    every class except the mask id, which is never a target. Dropping that logit
    from the log-sum-exp keeps the mask-column gradient at 0 without a ``-inf``
    backward.
    """
    if aligned_logits.dim() != 3:
        raise ValueError(f"expected [B, L, V] logits, got {tuple(aligned_logits.shape)}")
    vocab = aligned_logits.shape[-1]
    if not 0 <= mask_token_id < vocab:
        raise ValueError("mask id is outside the logit vocabulary")
    rows = aligned_logits.to(torch.promote_types(aligned_logits.dtype, torch.float32))
    keep = torch.ones(vocab, dtype=torch.bool, device=rows.device)
    keep[mask_token_id] = False
    lse = torch.logsumexp(rows[..., keep], dim=-1, keepdim=True)
    return rows - lse


def gathered_corrector_logp(aligned_logits: torch.Tensor, targets: torch.Tensor,
                            positions: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Corrector ``log p(target)`` at ``positions``, row-major. Empty -> length 0."""
    if positions.dtype != torch.bool:
        raise TypeError("positions must be a bool mask")
    if aligned_logits.shape[:2] != targets.shape or targets.shape != positions.shape:
        raise ValueError("logits, targets, and positions must share a batch shape")
    if int(positions.sum()) == 0:
        return aligned_logits.new_zeros(0, dtype=torch.promote_types(aligned_logits.dtype, torch.float32))
    assert_targets_predictable(positions)
    logp = corrector_log_probs(aligned_logits, mask_token_id)
    chosen = targets[positions]
    if bool((chosen == mask_token_id).any()):
        raise ValueError("corrector target is the mask id")
    return logp[positions].gather(-1, chosen.unsqueeze(-1)).squeeze(-1)


def example_scales(positions: torch.Tensor, t_per_example: Sequence[float], like: torch.Tensor) -> torch.Tensor:
    """``a(t_b)`` for each True position, in the same row-major order as the gather."""
    if len(t_per_example) != positions.shape[0]:
        raise ValueError("one mask rate per example")
    weights = [time_weight(t) for t in t_per_example]
    pieces = []
    for b, w in enumerate(weights):
        n = int(positions[b].sum())
        if n:
            pieces.append(torch.full((n,), w, dtype=like.dtype, device=like.device))
    if not pieces:
        return like.new_zeros(0)
    return torch.cat(pieces)


def assemble_draft(context_ids: torch.Tensor, editable: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    """Write the detached candidate at every editable position. Prompt and pad stay."""
    if context_ids.shape != editable.shape or candidate.shape != context_ids.shape:
        raise ValueError("context, editable mask, and candidate must share a shape")
    draft = context_ids.clone()
    draft = torch.where(editable, candidate.detach().to(dtype=draft.dtype), draft)
    return draft


def detached_allowed_argmax(aligned_logits: torch.Tensor, allowed: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop-grad allowed argmax and its full-vocabulary log-probability (D-051, D-029)."""
    with torch.no_grad():
        cand, logp = greedy_allowed(aligned_logits.detach(), allowed)
    if cand.requires_grad or logp.requires_grad:
        raise RuntimeError("draft tokens or log c received a gradient")
    return cand, logp


@dataclass
class CorrectorTerms:
    loss: torch.Tensor
    n_error: int
    n_keep: int
    error_empty: bool
    keep_empty: bool


def _partition(draft_ids: torch.Tensor, targets: torch.Tensor, editable: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    err, keep = token_error_masks(draft_ids, targets, editable)
    return err, keep, int(err.sum()), int(keep.sum())


def _pooled(values: torch.Tensor, scales: torch.Tensor, aligned_logits: torch.Tensor,
            n_err: int, n_keep: int) -> torch.Tensor:
    loss = (scales * values).sum() / int(values.numel())
    if n_err == 0 or n_keep == 0:
        loss = loss + zero_with_graph(aligned_logits)
    return loss


def proseco_basic_loss(aligned_logits: torch.Tensor, targets: torch.Tensor, draft_ids: torch.Tensor,
                       editable: torch.Tensor, t_per_example: Sequence[float],
                       mask_token_id: int) -> CorrectorTerms:
    """One pool of ``a(t) * CE`` over editable positions. ``draft_ids`` only define E/K logs."""
    n = int(editable.sum())
    _, _, n_err, n_keep = _partition(draft_ids, targets, editable)
    if n == 0:
        loss = zero_with_graph(aligned_logits)
    else:
        ce = -gathered_corrector_logp(aligned_logits, targets, editable, mask_token_id)
        loss = _pooled(ce, example_scales(editable, t_per_example, ce), aligned_logits, n_err, n_keep)
    return CorrectorTerms(loss=loss, n_error=n_err, n_keep=n_keep,
                          error_empty=n_err == 0, keep_empty=n_keep == 0)


def proseco_error_prioritized_loss(aligned_logits: torch.Tensor, targets: torch.Tensor,
                                   draft_ids: torch.Tensor, editable: torch.Tensor,
                                   t_per_example: Sequence[float], mask_token_id: int) -> CorrectorTerms:
    """Eq. 11: CE where the draft is wrong, ``-p(gold)`` where it is already right.

    One denominator, the editable count, matching the pinned reduction. The keep
    term is the probability, not the log probability.
    """
    err, _, n_err, n_keep = _partition(draft_ids, targets, editable)
    n = int(editable.sum())
    if n == 0:
        loss = zero_with_graph(aligned_logits)
    else:
        logp = gathered_corrector_logp(aligned_logits, targets, editable, mask_token_id)
        ce = -logp
        prob = logp.exp()
        term = torch.where(err[editable], ce, -prob)
        loss = _pooled(term, example_scales(editable, t_per_example, term), aligned_logits, n_err, n_keep)
    return CorrectorTerms(loss=loss, n_error=n_err, n_keep=n_keep,
                          error_empty=n_err == 0, keep_empty=n_keep == 0)


@dataclass
class ProsecoStep:
    loss: torch.Tensor
    l_mask: torch.Tensor
    l_corr: torch.Tensor
    draft_ids: torch.Tensor
    z_ids: torch.Tensor
    first_tokens: torch.Tensor
    log_c: torch.Tensor
    parts: dict
    logits1: torch.Tensor
    logits2: torch.Tensor


def proseco_training_step(adapter: DiffusionAdapter, layout: SequenceLayout,
                          masked_positions: torch.Tensor, t_per_example: Sequence[float],
                          allowed: torch.Tensor, *, targets: torch.Tensor | None = None,
                          arm: str = "proseco_basic") -> ProsecoStep:
    """Two forwards. ``layout.input_ids`` is the context (gold, before masking).

    ``targets`` defaults to that context. A different target tensor changes the loss
    and does not change ``z_t`` or the draft.
    """
    if arm not in ("proseco_basic", "proseco_error_prioritized"):
        raise ValueError(f"arm {arm!r} is not implemented on this step yet")
    if masked_positions.shape != layout.input_ids.shape:
        raise ValueError("mask shape must match the layout")
    if bool((masked_positions & ~layout.editable_mask).any()):
        raise ValueError("masks must be editable response positions")
    if targets is None:
        targets = layout.input_ids
    mask_id = adapter.tokens.mask_token_id
    z_ids = layout.input_ids.clone()
    z_ids = z_ids.masked_fill(masked_positions, mask_id)
    logits1 = adapter.aligned_logits(layout.with_input_ids(z_ids))
    l_mask_out = masked_diffusion_loss(logits1, targets, masked_positions, layout.editable_mask)
    cand, log_c_map = detached_allowed_argmax(logits1, allowed)
    draft = assemble_draft(z_ids, layout.editable_mask, cand)
    logits2 = adapter.aligned_logits(layout.with_input_ids(draft))
    if arm == "proseco_basic":
        terms = proseco_basic_loss(logits2, targets, draft, layout.editable_mask, t_per_example, mask_id)
    else:
        terms = proseco_error_prioritized_loss(logits2, targets, draft, layout.editable_mask,
                                              t_per_example, mask_id)
    loss = l_mask_out.loss + terms.loss
    parts = {
        "arm": arm,
        "L_mask": float(l_mask_out.loss.detach()),
        "L_corr": float(terms.loss.detach()),
        "n_masked": l_mask_out.n_masked,
        "n_error": terms.n_error,
        "n_keep": terms.n_keep,
        "error_set_empty": terms.error_empty,
        "keep_set_empty": terms.keep_empty,
        "a_t": [time_weight(t) for t in t_per_example],
        "time_weight": "1/t",
    }
    return ProsecoStep(loss=loss, l_mask=l_mask_out.loss, l_corr=terms.loss, draft_ids=draft.detach(),
                       z_ids=z_ids.detach(), first_tokens=cand.detach(), log_c=log_c_map.detach(),
                       parts=parts, logits1=logits1, logits2=logits2)
