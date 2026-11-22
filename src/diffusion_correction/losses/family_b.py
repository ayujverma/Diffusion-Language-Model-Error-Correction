"""Family B training losses (milestone 7, D-051).

ProSeCo arms share one online draft. ``z_t`` masks the gold response. The first
forward keeps a gradient for ``L_mask`` and, under stop-grad, writes an allowed-set
argmax at every editable position. The second forward is the corrector. ``a(t) = 1/t``
multiplies the corrector only. SCOPE (D-052) is a separate step in this file:
its draft pass is ``no_grad``, and it does not use ``a(t)`` or rank weights.

These are adaptations to Open-dCoder. They are not reproductions of the papers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from ..models.adapter import DiffusionAdapter
from ..models.alignment import SequenceLayout, assert_targets_predictable
from ..sampling.candidates import greedy_allowed
from .correction import token_error_masks
from .masked import masked_diffusion_loss
from .token_ce import selected_token_ce, zero_with_graph
from .weighting import DEFAULT_STAGE_EDGES, compute_error_weights, stage_bin

PROSECO_ARMS = ("proseco_basic", "proseco_error_prioritized", "proseco_gen_conf")


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
    error_weights: torch.Tensor | None = None
    weight_diagnostics: dict | None = None


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


def error_confidence_groups(log_c: torch.Tensor, error_mask: torch.Tensor,
                            t_per_example: Sequence[float],
                            stage_edges: Sequence[float] = DEFAULT_STAGE_EDGES
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Stop-grad ``log c`` of the committed token, and the mask-rate bin of its example.

    Online steps have no bank budget. The group is that bin inside the minibatch (D-051).
    """
    scores, rates = [], []
    for b in range(error_mask.shape[0]):
        idx = error_mask[b].nonzero(as_tuple=False).flatten()
        n = int(idx.numel())
        if n == 0:
            continue
        scores.append(log_c[b, idx].detach().reshape(n).double().cpu())
        rates.append(torch.full((n,), float(t_per_example[b]), dtype=torch.float64))
    if not scores:
        return torch.zeros(0, dtype=torch.float64), torch.zeros(0, dtype=torch.long)
    return torch.cat(scores), stage_bin(torch.cat(rates), tuple(float(x) for x in stage_edges))


def proseco_gen_conf_loss(aligned_logits: torch.Tensor, targets: torch.Tensor, draft_ids: torch.Tensor,
                          editable: torch.Tensor, t_per_example: Sequence[float], mask_token_id: int,
                          log_c: torch.Tensor, *, beta: float = 0.5,
                          stage_edges: Sequence[float] = DEFAULT_STAGE_EDGES) -> CorrectorTerms:
    """D-046: CE on errors and keeps, errors rank-weighted by stop-grad ``log c``.

    ``log_c`` is ``[B, L]`` from the draft-construction forward. It is not ``q`` and
    not a probability from ``aligned_logits`` (the corrector forward).
    """
    err, keep, n_err, n_keep = _partition(draft_ids, targets, editable)
    diag: dict = {"n_errors": n_err, "error_set_empty": n_err == 0, "beta": beta,
                  "groups": "mask_rate_bin"}
    weights = torch.zeros(0, dtype=torch.float32)
    if n_err == 0:
        l_error = zero_with_graph(aligned_logits)
    else:
        ce = -gathered_corrector_logp(aligned_logits, targets, err, mask_token_id)
        scores, groups = error_confidence_groups(log_c, err, t_per_example, stage_edges)
        ranked = compute_error_weights("high_conf", groups, beta, log_c=scores)
        if ranked.weights.requires_grad:
            raise RuntimeError("rank weights received a gradient")
        weights = ranked.weights.detach()
        w = weights.to(device=ce.device, dtype=ce.dtype)
        l_error = (example_scales(err, t_per_example, ce) * w * ce).sum() / n_err
        diag.update(ranked.diagnostics)
    if n_keep == 0:
        l_keep = zero_with_graph(aligned_logits)
    else:
        ce_k = -gathered_corrector_logp(aligned_logits, targets, keep, mask_token_id)
        l_keep = (example_scales(keep, t_per_example, ce_k) * ce_k).sum() / n_keep
    return CorrectorTerms(loss=l_error + l_keep, n_error=n_err, n_keep=n_keep,
                          error_empty=n_err == 0, keep_empty=n_keep == 0,
                          error_weights=weights, weight_diagnostics=diag)


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
                          arm: str = "proseco_basic", beta: float = 0.5,
                          stage_edges: Sequence[float] = DEFAULT_STAGE_EDGES) -> ProsecoStep:
    """Two forwards. ``layout.input_ids`` is the context (gold, before masking).

    ``targets`` defaults to that context. A different target tensor changes the loss
    and does not change ``z_t`` or the draft.
    """
    if arm not in PROSECO_ARMS:
        raise ValueError(f"arm {arm!r} is not a ProSeCo training arm")
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
    elif arm == "proseco_error_prioritized":
        terms = proseco_error_prioritized_loss(logits2, targets, draft, layout.editable_mask,
                                              t_per_example, mask_id)
    else:
        terms = proseco_gen_conf_loss(logits2, targets, draft, layout.editable_mask, t_per_example,
                                      mask_id, log_c_map, beta=beta, stage_edges=stage_edges)
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
        "weight_diagnostics": terms.weight_diagnostics,
    }
    return ProsecoStep(loss=loss, l_mask=l_mask_out.loss, l_corr=terms.loss, draft_ids=draft.detach(),
                       z_ids=z_ids.detach(), first_tokens=cand.detach(), log_c=log_c_map.detach(),
                       parts=parts, logits1=logits1, logits2=logits2)


# --- SCOPE training (D-052). Not a ProSeCo arm and not D3IM. ---

SCOPE_RHO = 0.3
SCOPE_SAMPLE_TEMPERATURE = 1.5
SCOPE_MIX = 0.5


def sample_beta22(generator: torch.Generator) -> float:
    """Beta(2, 2) as a ratio of two Gamma(2, 1) draws. Four uniforms, this generator only."""
    def gamma2() -> torch.Tensor:
        u1 = torch.rand((), generator=generator, dtype=torch.float64).clamp_min(1e-300)
        u2 = torch.rand((), generator=generator, dtype=torch.float64).clamp_min(1e-300)
        return -torch.log(u1) - torch.log(u2)

    g1, g2 = gamma2(), gamma2()
    return float((g1 / (g1 + g2)).item())


def clamp_scope_rate(raw: float, low: float = 0.1, high: float = 0.9) -> float:
    return min(max(float(raw), low), high)


def mix_is_mdm(generator: torch.Generator, probability: float = SCOPE_MIX) -> bool:
    """One Bernoulli from ``generator``. Does not read Python's ``random``."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("MDM mix probability must lie in [0, 1]")
    u = float(torch.rand((), generator=generator, dtype=torch.float64).item())
    return u < probability


@dataclass
class ScopeMask:
    masked_positions: torch.Tensor
    rates: list[float]
    raw_rates: list[float]
    n_masked: list[int]

    @property
    def total_masked(self) -> int:
        return int(sum(self.n_masked))


def sample_scope_masks(layout: SequenceLayout, generator: torch.Generator) -> ScopeMask:
    """Per example: one Beta(2, 2) rate, clamped, then a Bernoulli on each response token."""
    editable = layout.editable_mask.cpu()
    masked = torch.zeros_like(editable)
    rates, raws, counts = [], [], []
    for b in range(editable.shape[0]):
        pos = editable[b].nonzero().flatten()
        raw = sample_beta22(generator)
        rate = clamp_scope_rate(raw)
        draw = torch.rand(int(pos.numel()), generator=generator, dtype=torch.float64)
        chosen = pos[draw < rate]
        masked[b, chosen] = True
        raws.append(raw)
        rates.append(rate)
        counts.append(int(chosen.numel()))
    return ScopeMask(masked_positions=masked, rates=rates, raw_rates=raws, n_masked=counts)


def commit_top(confidence: torch.Tensor, masked: torch.Tensor, rho: float = SCOPE_RHO) -> torch.Tensor:
    """Highest confidence first among masked positions. Ties: smallest position."""
    if confidence.shape != masked.shape:
        raise ValueError("confidence and mask must share a shape")
    committed = torch.zeros_like(masked)
    for b in range(masked.shape[0]):
        idx = masked[b].nonzero().flatten()
        n = int(idx.numel())
        if n == 0:
            continue
        k = min(n, max(1, math.floor(rho * n)))
        scores = confidence[b].detach()
        order = sorted(range(n), key=lambda i: (-float(scores[int(idx[i])]), int(idx[i])))
        take = [int(idx[i]) for i in order[:k]]
        committed[b, take] = True
    return committed


def scope_loss_positions(masked: torch.Tensor, committed: torch.Tensor) -> torch.Tensor:
    """Committed this step union still masked. Clean visible tokens are not in ``masked``."""
    if bool((committed & ~masked).any()):
        raise ValueError("a commit fell outside the masked positions")
    return committed | (masked & ~committed)


def scope_context_ids(gold_ids: torch.Tensor, masked: torch.Tensor, committed: torch.Tensor,
                      samples: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Visible gold stays. Committed positions take ``samples``. The rest of the mask stays masked."""
    out = gold_ids.clone()
    samples = samples.to(device=out.device)
    committed = committed.to(device=out.device)
    masked = masked.to(device=out.device)
    out[masked & ~committed] = mask_token_id
    out[committed] = samples[committed].to(dtype=out.dtype)
    return out


def sample_allowed_tokens(aligned_logits: torch.Tensor, positions: torch.Tensor, allowed: torch.Tensor,
                          generator: torch.Generator, temperature: float = SCOPE_SAMPLE_TEMPERATURE
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample at ``positions`` and return temperature-1 probabilities of those ids.

    The categorical uses ``temperature`` (1.5). The returned confidence does not:
    it is the temperature-1 full-vocabulary probability of the sampled id (D-052).
    """
    if temperature <= 0:
        raise ValueError("sample temperature must be positive")
    rows = aligned_logits.detach().float()
    scaled = rows / float(temperature)
    allowed_dev = allowed.to(device=scaled.device)
    scaled = scaled.masked_fill(~allowed_dev.view(1, 1, -1), float("-inf"))
    probs = torch.softmax(scaled, dim=-1).cpu()
    full = torch.softmax(rows, dim=-1).cpu()
    ids = torch.zeros(rows.shape[:2], dtype=torch.long)
    conf = torch.zeros(rows.shape[:2], dtype=torch.float32)
    for b, pos in positions.detach().cpu().nonzero(as_tuple=False).tolist():
        choice = int(torch.multinomial(probs[b, pos], 1, generator=generator).item())
        ids[b, pos] = choice
        conf[b, pos] = full[b, pos, choice]
    return ids, conf


@dataclass
class ScopeStep:
    loss: torch.Tensor
    logits1: torch.Tensor
    logits2: torch.Tensor
    samples: torch.Tensor
    confidence: torch.Tensor
    committed: torch.Tensor
    loss_positions: torch.Tensor
    context_ids: torch.Tensor
    parts: dict


def scope_training_step(adapter: DiffusionAdapter, layout: SequenceLayout, masked_positions: torch.Tensor,
                        allowed: torch.Tensor, sample_generator: torch.Generator, *,
                        targets: torch.Tensor | None = None, temperature: float = SCOPE_SAMPLE_TEMPERATURE,
                        rho: float = SCOPE_RHO) -> ScopeStep:
    """``no_grad`` draft, then CE on committed ∪ still-masked. Clean context is excluded."""
    if bool((masked_positions & ~layout.editable_mask).any()):
        raise ValueError("SCOPE masks must be editable response positions")
    if targets is None:
        targets = layout.input_ids
    mask_id = adapter.tokens.mask_token_id
    z_ids = layout.input_ids.masked_fill(masked_positions, mask_id)
    with torch.no_grad():
        logits1 = adapter.aligned_logits(layout.with_input_ids(z_ids))
        samples, confidence = sample_allowed_tokens(logits1, masked_positions, allowed, sample_generator,
                                                    temperature)
        committed = commit_top(confidence, masked_positions, rho)
    loss_pos = scope_loss_positions(masked_positions, committed)
    context = scope_context_ids(layout.input_ids, masked_positions, committed, samples, mask_id)
    logits2 = adapter.aligned_logits(layout.with_input_ids(context))
    if int(loss_pos.sum()) == 0:
        loss = zero_with_graph(logits2)
        n = 0
    else:
        ce = selected_token_ce(logits2, targets, loss_pos)
        n = int(ce.numel())
        loss = ce.sum() / n
    visible = layout.editable_mask & ~masked_positions
    parts = {
        "arm": "scope",
        "n_masked": int(masked_positions.sum()),
        "n_committed": int(committed.sum()),
        "n_loss_positions": n,
        "n_clean_visible": int(visible.sum()),
        "clean_excluded": True,
        "draft_pass": "no_grad",
        "confidence": "temperature_1_probability_of_sampled_token",
    }
    return ScopeStep(loss=loss, logits1=logits1, logits2=logits2, samples=samples.detach(),
                     confidence=confidence.detach(), committed=committed, loss_positions=loss_pos,
                     context_ids=context.detach(), parts=parts)
