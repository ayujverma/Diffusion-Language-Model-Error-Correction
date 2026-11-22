"""Token-state edits for the milestone-6 samplers.

Prompt and padding positions are never writable: every edit is gated by the
layout's editable (response) mask. Gold targets are not an input.
"""

from __future__ import annotations

import torch


def slice_responses(ids: torch.Tensor, starts: list[int], length: int) -> list[list[int]]:
    return [ids[b, starts[b]:starts[b] + length].tolist() for b in range(ids.shape[0])]


def masks_remaining(responses: list[list[int]], mask_id: int) -> list[int]:
    return [sum(t == mask_id for t in row) for row in responses]


def fill_masked(ids: torch.Tensor, cands: torch.Tensor, editable: torch.Tensor, mask_id: int) -> torch.Tensor:
    """Copy greedy candidates onto editable positions that are still ``<M>``."""
    masked = editable & (ids == mask_id)
    return torch.where(masked, cands.to(dtype=ids.dtype), ids)


def write_editable(ids: torch.Tensor, cands: torch.Tensor, editable: torch.Tensor) -> torch.Tensor:
    """Replace every editable position. Prompt and pad stay as in ``ids``."""
    return torch.where(editable, cands.to(dtype=ids.dtype), ids)


def same_editable(a: torch.Tensor, b: torch.Tensor, editable: torch.Tensor) -> bool:
    return bool(torch.equal(a[editable], b[editable]))


def commit_topk(ids: torch.Tensor, cands: torch.Tensor, logp: torch.Tensor, editable: torch.Tensor,
                mask_id: int, quota: int) -> torch.Tensor:
    """Reveal ``quota`` still-masked editable positions, highest float32 log-prob first.

    Exact float32 ties take the smallest absolute position. Already-visible tokens,
    the prompt, and padding are not written.
    """
    if quota < 1:
        raise ValueError("commit quota must be >= 1")
    new_ids = ids.clone()
    logp_cpu = logp.detach().float().cpu()
    cand_cpu = cands.detach().cpu()
    for b in range(ids.shape[0]):
        masked = (editable[b] & (ids[b] == mask_id)).nonzero(as_tuple=True)[0].tolist()
        if quota > len(masked):
            raise AssertionError(f"example {b}: quota {quota} exceeds {len(masked)} masked positions")
        order = sorted(masked, key=lambda p: (-logp_cpu[b, p].item(), p))
        for p in order[:quota]:
            new_ids[b, p] = cand_cpu[b, p]
    return new_ids


def finalize_step(main: torch.Tensor, merge_from: torch.Tensor | None, cands: torch.Tensor,
                  logp: torch.Tensor, editable: torch.Tensor, mask_id: int, quota: int) -> torch.Tensor:
    """Returnable main state after a completed forward, without another model call.

    If ``merge_from`` is set, already-visible editable tokens are replaced by that
    temporary draft (a corrector revision). The step's unmask quota is then written
    from ``cands`` into positions that are still masked.
    """
    state = main.clone()
    if merge_from is not None:
        visible = editable & (main != mask_id)
        state = torch.where(visible, merge_from.to(dtype=state.dtype), state)
    return commit_topk(state, cands, logp, editable, mask_id, quota)
