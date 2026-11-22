"""Training-free extra-call control (D-049). Not a ReMDM adaptation.

Q-6 (ReMDM schedule and sigma_t on Open-dCoder) is still open. This control spends
a matched call budget without training new weights:

1. D-029 monotone unmasking for ``T = min(max_calls, response length)`` calls.
   Those calls reveal tokens; they do not revise them.
2. If ``max_calls`` exceeds the length, each remaining call is one revision: the
   editable position whose *current* token has the lowest float32 full-vocabulary
   log-probability is replaced by the greedy allowed candidate from that same
   forward. One position per call. The sequence stays complete (nothing is remasked).

D3IM instead ranks the *new* candidate and can remask visible tokens. A call whose
replacement equals the current token still counts: the model was run.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..models.adapter import DiffusionAdapter
from ..models.alignment import build_layout
from .candidates import greedy_allowed, token_raw_logp
from .monotone import monotone_unmask
from .nfe import ForwardMeter
from .results import SamplerRun
from .state import masks_remaining, slice_responses


@torch.no_grad()
def extra_unmask(adapter: DiffusionAdapter, prompts: Sequence[Sequence[int]], response_lengths: Sequence[int],
                 *, max_calls: int, allowed: torch.Tensor) -> SamplerRun:
    """Spend exactly ``max_calls`` forwards. Equal response lengths required."""
    batch = len(prompts)
    if len(response_lengths) != batch or batch == 0:
        raise ValueError("one non-empty batch of response lengths")
    lengths = [int(n) for n in response_lengths]
    if any(n != lengths[0] for n in lengths):
        raise ValueError("extra_unmask requires equal response lengths; group before calling")
    length = lengths[0]
    if max_calls < 1:
        raise ValueError("max_calls must be >= 1")
    tok = adapter.tokens
    if bool(allowed[tok.mask_token_id]):
        raise ValueError("the mask token must not be an allowed candidate")

    meter = ForwardMeter(adapter, max_calls=max_calls)
    unmask_calls = min(max_calls, length)
    was_training = adapter.model.training
    adapter.model.eval()
    try:
        mono = monotone_unmask(adapter, prompts, lengths, unmask_calls, allowed, meter=meter)
        layout = build_layout(prompts, mono.responses, tok.pad_token_id).to(adapter.device)
        ids = layout.input_ids.clone()
        starts = [len(p) for p in prompts]
        editable = layout.editable_mask
        revision = 0
        while meter.nfe < max_calls:
            revision += 1
            logits = meter.forward(layout.with_input_ids(ids), kind="revision", step=unmask_calls + revision)
            if logits is None:
                break
            if bool((editable & (ids == tok.mask_token_id)).any()):
                raise AssertionError("revision saw a mask; the unmask phase should have finished")
            scores = token_raw_logp(logits, ids)
            cands, _logp = greedy_allowed(logits, allowed)
            _revise_lowest(ids, scores, cands, editable)
    finally:
        adapter.model.train(was_training)

    responses = slice_responses(ids, starts, length)
    left = masks_remaining(responses, tok.mask_token_id)
    if any(left):
        raise AssertionError("extra_unmask left mask tokens")
    if meter.nfe != max_calls:
        raise AssertionError(f"extra_unmask used {meter.nfe} forwards, budget is {max_calls}")
    return SamplerRun(
        responses=responses, calls_per_example=[meter.nfe] * batch, batched_forward_calls=meter.nfe,
        call_kinds=meter.kinds, max_calls=max_calls, masks_remaining=left,
        final_input_ids=ids.detach().cpu(),
        plan={"unmask_calls": unmask_calls, "revision_calls": meter.nfe - unmask_calls})


def _revise_lowest(ids: torch.Tensor, scores: torch.Tensor, cands: torch.Tensor, editable: torch.Tensor) -> None:
    """Replace one editable position per example: lowest current-token log-prob, then smallest index."""
    scores_cpu = scores.detach().float().cpu()
    cand_cpu = cands.detach().cpu()
    for b in range(ids.shape[0]):
        positions = editable[b].nonzero(as_tuple=True)[0].tolist()
        if not positions:
            raise AssertionError("no editable position")
        chosen = min(positions, key=lambda p: (scores_cpu[b, p].item(), p))
        ids[b, chosen] = cand_cpu[b, chosen]
