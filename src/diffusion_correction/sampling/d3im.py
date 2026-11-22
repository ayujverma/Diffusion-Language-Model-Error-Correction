"""D3IM inference, adapted from the rules recorded in source_audit §3.2 (D-049).

Not a numerical reproduction of SCOPE's implementation (none is in the pinned
sources). One forward per step. The new candidate is the allowed-set argmax; its
confidence is that candidate's float32 full-vocabulary softmax probability.
Positions outside the top-K of the active block are remasked, including tokens
that were already visible. The final step of a block keeps the whole block.
Earlier blocks stay frozen. Prompt and padding are immutable.

EOS/pad/mask confidence is excluded by the allowed-token mask (our responses have
no EOS; D-018). Q-5 (SCOPE's training-time commit confidence) is not decided here.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..models.adapter import DiffusionAdapter
from ..models.alignment import build_layout
from .candidates import greedy_allowed
from .nfe import ForwardMeter
from .results import SamplerRun
from .state import masks_remaining, slice_responses


def block_spans(length: int, block_length: int | None) -> list[tuple[int, int]]:
    """Response-relative ``[start, end)`` spans. ``None`` is one block, the whole response."""
    if length < 1:
        raise ValueError("response length must be >= 1")
    if block_length is None:
        return [(0, length)]
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    spans = []
    start = 0
    while start < length:
        spans.append((start, min(length, start + block_length)))
        start += block_length
    return spans


def keep_count(block_len: int, step: int, steps_per_block: int) -> int:
    """Positions kept after local step ``step`` (1-based). The block's last step keeps all."""
    if not 1 <= step <= steps_per_block:
        raise ValueError("step out of range")
    if step == steps_per_block:
        return block_len
    return (block_len * step) // steps_per_block


@torch.no_grad()
def d3im_unmask(adapter: DiffusionAdapter, prompts: Sequence[Sequence[int]], response_lengths: Sequence[int],
                *, max_calls: int, allowed: torch.Tensor, block_length: int | None = None,
                record_trajectory: bool = False) -> SamplerRun:
    """Run a D3IM horizon of ``max_calls`` steps. Each step is one NFE.

    ``max_calls`` must be divisible by the number of blocks. The last step of each
    block fills that block, so a finished run has no mask left. An intermediate
    keep-count may be 0 (the active block is entirely remasked); that forward still
    counts.
    """
    batch = len(prompts)
    if len(response_lengths) != batch or batch == 0:
        raise ValueError("one non-empty batch of response lengths")
    lengths = [int(n) for n in response_lengths]
    if any(n != lengths[0] for n in lengths):
        raise ValueError("d3im_unmask requires equal response lengths; group before calling")
    length = lengths[0]
    if max_calls < 1:
        raise ValueError("max_calls must be >= 1")
    spans = block_spans(length, block_length)
    if max_calls % len(spans) != 0:
        raise ValueError(
            f"D3IM horizon {max_calls} is not divisible by {len(spans)} blocks "
            f"(block_length={block_length})")
    steps_per_block = max_calls // len(spans)
    tok = adapter.tokens
    if allowed.shape != (tok.vocab_size,) or allowed.dtype != torch.bool:
        raise ValueError("allowed must be a bool mask over the output vocabulary")
    if bool(allowed[tok.mask_token_id]):
        raise ValueError("the mask token must not be an allowed candidate")

    meter = ForwardMeter(adapter, max_calls=max_calls)
    layout = build_layout(prompts, [[tok.mask_token_id] * length for _ in prompts], tok.pad_token_id)
    starts = [len(p) for p in prompts]
    device = adapter.device
    layout = layout.to(device)
    ids = layout.input_ids.clone()
    trajectory: list[list[list[int]]] = []
    was_training = adapter.model.training
    adapter.model.eval()
    try:
        for block_i, (lo, hi) in enumerate(spans):
            for local in range(1, steps_per_block + 1):
                step = block_i * steps_per_block + local
                logits = meter.forward(layout.with_input_ids(ids), kind="d3im", step=step)
                if logits is None:
                    raise RuntimeError("D3IM meter refused a planned step")
                cands, logp = greedy_allowed(logits, allowed)
                prob = logp.exp()
                k = keep_count(hi - lo, local, steps_per_block)
                _apply_clean_slate(ids, cands, prob, starts, lo, hi, k, tok.mask_token_id)
                if record_trajectory:
                    trajectory.append(slice_responses(ids, starts, length))
    finally:
        adapter.model.train(was_training)

    responses = slice_responses(ids, starts, length)
    left = masks_remaining(responses, tok.mask_token_id)
    if any(left):
        raise AssertionError("finished D3IM horizon left mask tokens")
    if meter.nfe != max_calls:
        raise AssertionError(f"D3IM used {meter.nfe} forwards, horizon is {max_calls}")
    return SamplerRun(
        responses=responses, calls_per_example=[meter.nfe] * batch, batched_forward_calls=meter.nfe,
        call_kinds=meter.kinds, max_calls=max_calls, masks_remaining=left, trajectory=trajectory,
        final_input_ids=ids.detach().cpu(),
        plan={"blocks": spans, "steps_per_block": steps_per_block, "block_length": block_length})


def _apply_clean_slate(ids: torch.Tensor, cands: torch.Tensor, prob: torch.Tensor, starts: list[int],
                       lo: int, hi: int, k: int, mask_id: int) -> None:
    """Keep the ``k`` highest new-candidate probabilities in the active block; mask the rest."""
    prob_cpu = prob.detach().float().cpu()
    cand_cpu = cands.detach().cpu()
    for b, start in enumerate(starts):
        positions = [start + rel for rel in range(lo, hi)]
        # Float32 probability, then smallest absolute position. Underflow can tie distinct log-probs.
        ranked = sorted(positions, key=lambda p: (-prob_cpu[b, p].item(), p))
        keep = ranked[:k]
        for p in positions:
            ids[b, p] = mask_id
        for p in keep:
            ids[b, p] = cand_cpu[b, p]
