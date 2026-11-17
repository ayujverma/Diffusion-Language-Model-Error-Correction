"""Rank-based error weights shared by every weighting arm.

For the n errors of one (decoding budget, generation-stage bin) group in the
physical minibatch: midranks 1..n (ties averaged), r = (rank - 0.5) / n,
w_raw = 1 + beta * (2r - 1), w = w_raw / mean_group(w_raw).
Midranks have mean (n + 1) / 2, so the mean of r is exactly 0.5 and the
normalisation only removes floating-point error. Singleton or fully tied groups get
weight 1. beta = 0 gives the uniform rule. Weights never carry gradients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

DEFAULT_STAGE_EDGES: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)  # [0,.2) [.2,.4) [.4,.6) [.6,.8) [.8,1]


class WeightRule(str, Enum):
    UNIFORM = "uniform"
    HIGH_CONF = "high_conf"            # rank saved log c upward
    LOW_CONF = "low_conf"              # reverse of HIGH_CONF ranks
    SHUFFLED_CONF = "shuffled_conf"    # saved log c permuted within groups, then ranked upward
    GEN_DIFFICULTY = "gen_difficulty"  # rank saved -log q upward
    CURRENT_LOSS = "current_loss"      # rank detached current CE upward


def stage_bin(mask_fraction: torch.Tensor, edges: tuple[float, ...] = DEFAULT_STAGE_EDGES) -> torch.Tensor:
    """Bin index for the mask fraction immediately before a token's commitment."""
    mf = mask_fraction.detach().double().cpu()
    if bool(((mf < 0) | (mf > 1) | torch.isnan(mf)).any()):
        raise ValueError("mask fractions must lie in [0, 1]")
    return torch.bucketize(mf, torch.tensor(edges, dtype=torch.float64), right=True)


def make_group_ids(budgets: torch.Tensor, mask_fractions: torch.Tensor,
                   edges: tuple[float, ...] = DEFAULT_STAGE_EDGES) -> torch.Tensor:
    """Compact group IDs for (decoding budget, stage bin) pairs."""
    bins = stage_bin(mask_fractions, edges)
    pairs = torch.stack([budgets.detach().long().cpu(), bins], dim=1)
    _, inverse = torch.unique(pairs, dim=0, return_inverse=True)
    return inverse


def midranks_within_groups(scores: torch.Tensor, groups: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Average ranks (1-based, ties share the mean rank) within each group, and group sizes."""
    s = scores.detach().double().cpu()
    g = groups.detach().long().cpu()
    if s.shape != g.shape or s.dim() != 1:
        raise ValueError("scores and groups must be matching 1-D tensors")
    if not bool(torch.isfinite(s).all()):
        raise ValueError("weighting scores must be finite")
    ranks = torch.empty_like(s)
    sizes = torch.empty_like(s)
    for gid in torch.unique(g):
        idx = (g == gid).nonzero(as_tuple=True)[0]
        uniq, inverse, counts = torch.unique(s[idx], sorted=True, return_inverse=True, return_counts=True)
        upper = torch.cumsum(counts, 0).double()
        mid = upper - (counts.double() - 1) / 2
        ranks[idx] = mid[inverse]
        sizes[idx] = float(idx.numel())
    return ranks, sizes


def rank_weights(scores: torch.Tensor, groups: torch.Tensor, beta: float, reverse: bool = False) -> torch.Tensor:
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1) so weights stay positive")
    ranks, n = midranks_within_groups(scores, groups)
    if reverse:
        ranks = n + 1 - ranks
    r = (ranks - 0.5) / n
    w_raw = 1.0 + beta * (2.0 * r - 1.0)
    g = groups.detach().long().cpu()
    w = torch.empty_like(w_raw)
    for gid in torch.unique(g):
        m = g == gid
        w[m] = w_raw[m] / w_raw[m].mean()
    return w


@dataclass
class WeightResult:
    weights: torch.Tensor  # float32, no grad, one per error token in row-major order
    rule: WeightRule
    beta: float
    diagnostics: dict = field(default_factory=dict)


def _check_and_describe(w: torch.Tensor, groups: torch.Tensor, beta: float, tol: float = 1e-6) -> dict:
    if w.requires_grad:
        raise AssertionError("error weights must not require grad")
    if not bool(torch.isfinite(w).all()):
        raise AssertionError("non-finite error weight")
    g = groups.detach().long().cpu()
    wd = w.detach().double().cpu()
    sizes = []
    for gid in torch.unique(g):
        m = g == gid
        sizes.append(int(m.sum()))
        if abs(float(wd[m].mean()) - 1.0) > tol:
            raise AssertionError(f"group {int(gid)} mean weight {float(wd[m].mean())} != 1")
    if wd.numel() and (float(wd.min()) < 1 - beta - tol or float(wd.max()) > 1 + beta + tol):
        raise AssertionError(f"weights outside [{1 - beta}, {1 + beta}]")
    n = int(wd.numel())
    return {
        "n_errors": n,
        "n_groups": len(sizes),
        "group_sizes": sizes,
        "n_singleton_groups": sum(1 for s in sizes if s == 1),
        "weight_mean": float(wd.mean()) if n else None,
        "weight_var": float(wd.var(unbiased=False)) if n else None,
        "weight_min": float(wd.min()) if n else None,
        "weight_max": float(wd.max()) if n else None,
        "ess": float(wd.sum() ** 2 / (wd ** 2).sum()) if n else 0.0,
        "has_variation": bool(n and float(wd.max() - wd.min()) > tol),
    }


def compute_error_weights(
    rule: WeightRule | str,
    groups: torch.Tensor,
    beta: float = 0.5,
    *,
    log_c: torch.Tensor | None = None,
    log_q: torch.Tensor | None = None,
    current_ce: torch.Tensor | None = None,
    shuffle_seed: int | None = None,
) -> WeightResult:
    """Weights for the error tokens of one physical minibatch.

    All score tensors are 1-D, aligned with ``groups``, in the row-major order of
    the error mask. ``log_c``/``log_q`` are the SAVED generation-time float32 log
    probabilities of the committed (wrong) token and of the correct target;
    ``current_ce`` is the corrector's current CE and is detached here.
    """
    rule = WeightRule(rule)
    n = int(groups.numel())
    extra: dict = {}

    def need(x, name):
        if x is None:
            raise ValueError(f"rule {rule.value} needs {name}")
        if x.numel() != n:
            raise ValueError(f"{name} has {x.numel()} entries for {n} errors")
        return x.detach()

    if rule is WeightRule.UNIFORM or n == 0:
        w = torch.ones(n, dtype=torch.float64)
    elif rule is WeightRule.HIGH_CONF:
        w = rank_weights(need(log_c, "log_c"), groups, beta)
    elif rule is WeightRule.LOW_CONF:
        w = rank_weights(need(log_c, "log_c"), groups, beta, reverse=True)
    elif rule is WeightRule.GEN_DIFFICULTY:
        w = rank_weights(-need(log_q, "log_q"), groups, beta)
    elif rule is WeightRule.CURRENT_LOSS:
        w = rank_weights(need(current_ce, "current_ce"), groups, beta)
    elif rule is WeightRule.SHUFFLED_CONF:
        if shuffle_seed is None:
            raise ValueError("shuffled_conf needs an explicit shuffle_seed")
        scores = need(log_c, "log_c").double().cpu()
        shuffled = scores.clone()
        gen = torch.Generator().manual_seed(shuffle_seed)
        g = groups.detach().long().cpu()
        for gid in torch.unique(g):
            idx = (g == gid).nonzero(as_tuple=True)[0]
            shuffled[idx] = scores[idx[torch.randperm(idx.numel(), generator=gen)]]
        extra["n_scores_moved"] = int((shuffled != scores).sum())
        w = rank_weights(shuffled, groups, beta)
    else:  # pragma: no cover
        raise ValueError(rule)

    diag = _check_and_describe(w, groups, beta if rule is not WeightRule.UNIFORM else 0.0)
    diag.update(extra)
    return WeightResult(weights=w.float(), rule=rule, beta=beta, diagnostics=diag)
