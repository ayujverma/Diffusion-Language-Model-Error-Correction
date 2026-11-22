"""ProSeCo-style correction sampler (D-049). An inference adaptation, not training.

This is the shared correction decoder later checkpoints are evaluated with. It is
not Family A (fixed-bank loss reweighting) and not ``proseco_gen_conf`` (D-046),
which is a milestone-7 training objective.

Adapted from ``llada/generate.py`` at ProSeCo ``43ce881`` (see D-049 for every
difference). The temporary complete draft is the corrector's input. The main state
keeps its masks until the step's unmask commit. Prompt and padding are immutable.
Targets, error masks, and the checker never enter this function.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..models.adapter import DiffusionAdapter
from ..models.alignment import build_layout
from .candidates import greedy_allowed
from .monotone import reveal_schedule
from .nfe import ForwardMeter
from .results import SamplerRun
from .state import fill_masked, finalize_step, masks_remaining, same_editable, slice_responses, write_editable

# Upstream llada/generate.py defaults at the pinned ProSeCo commit. Provisional
# until a validation sweep selects decoder settings (D-049). Not a paper mandate.
DEFAULT_OMEGA = 2
DEFAULT_CORRECTOR_STEPS = 4


def worst_case_nfe(unmask_steps: int, omega: int, corrector_steps: int) -> int:
    """Forwards if every due corrector loop runs all ``corrector_steps`` iterations."""
    if unmask_steps < 0 or omega < 1 or corrector_steps < 0:
        raise ValueError("unmask_steps >= 0, omega >= 1, corrector_steps >= 0")
    return unmask_steps + corrector_steps * (unmask_steps // omega)


def completing_unmask_steps(length: int, max_calls: int, omega: int, corrector_steps: int) -> int:
    """Largest ``t`` in ``1..length`` whose worst-case NFE fits in ``max_calls``, else 0."""
    if length < 1 or max_calls < 1:
        raise ValueError("length and max_calls must be >= 1")
    best = 0
    for t in range(1, length + 1):
        if worst_case_nfe(t, omega, corrector_steps) <= max_calls:
            best = t
        else:
            break
    return best


def _check(length: int, max_calls: int, omega: int, corrector_steps: int) -> None:
    if length < 1:
        raise ValueError("response length must be >= 1")
    if max_calls < 1:
        raise ValueError("max_calls must be >= 1")
    if omega < 1:
        raise ValueError("omega must be >= 1")
    if corrector_steps < 0:
        raise ValueError("corrector_steps must be >= 0")


@torch.no_grad()
def proseco_unmask(adapter: DiffusionAdapter, prompts: Sequence[Sequence[int]], response_lengths: Sequence[int],
                   *, max_calls: int, allowed: torch.Tensor, omega: int = DEFAULT_OMEGA,
                   corrector_steps: int = DEFAULT_CORRECTOR_STEPS,
                   record_trajectory: bool = False) -> SamplerRun:
    """Generate one batch. Every example must share one response length.

    See D-049. With ``corrector_steps=0`` the commits match D-029 monotone unmasking
    at ``T = min(max_calls, length)``. A budget stops inside a corrector loop by
    skipping iterations that would exceed ``max_calls``; the step's unmask commit
    still uses the last completed logits and is not an extra NFE. Early fixed
    points are not backfilled. ``max_calls >= 1`` ends with no mask left.
    """
    batch = len(prompts)
    if len(response_lengths) != batch:
        raise ValueError("one response length per prompt")
    if batch == 0:
        raise ValueError("empty batch")
    lengths = [int(n) for n in response_lengths]
    if any(n != lengths[0] for n in lengths):
        raise ValueError("proseco_unmask requires equal response lengths; group before calling")
    length = lengths[0]
    _check(length, max_calls, omega, corrector_steps)
    tok = adapter.tokens
    if allowed.shape != (tok.vocab_size,) or allowed.dtype != torch.bool:
        raise ValueError("allowed must be a bool mask over the output vocabulary")
    if bool(allowed[tok.mask_token_id]):
        raise ValueError("the mask token must not be an allowed candidate")

    planned = completing_unmask_steps(length, max_calls, omega, corrector_steps)
    # T* == 0: even one fully corrected unmask step does not fit. Run one step that
    # reveals every token, and stop the corrector loop when the budget runs out.
    quotas = reveal_schedule(length, planned) if planned >= 1 else [length]
    meter = ForwardMeter(adapter, max_calls=max_calls)
    layout = build_layout(prompts, [[tok.mask_token_id] * length for _ in prompts], tok.pad_token_id)
    starts = [len(p) for p in prompts]
    device = adapter.device
    layout = layout.to(device)
    main = layout.input_ids.clone()
    editable = layout.editable_mask
    trajectory: list[list[list[int]]] = []
    was_training = adapter.model.training
    adapter.model.eval()
    try:
        for step, quota in enumerate(quotas, start=1):
            logits = meter.forward(layout.with_input_ids(main), kind="predictor", step=step)
            if logits is None:
                break
            cands, logp = greedy_allowed(logits, allowed)
            step_ids = finalize_step(main, None, cands, logp, editable, tok.mask_token_id, quota)
            if record_trajectory:
                trajectory.append(slice_responses(step_ids, starts, length))
            due = corrector_steps > 0 and step % omega == 0
            if due and (meter.max_calls is None or meter.nfe < meter.max_calls):
                temp = fill_masked(main, cands, editable, tok.mask_token_id)
                for _k in range(corrector_steps):
                    clogits = meter.forward(layout.with_input_ids(temp), kind="corrector", step=step)
                    if clogits is None:
                        break
                    cc, clp = greedy_allowed(clogits, allowed)
                    temp_after = write_editable(temp, cc, editable)
                    step_ids = finalize_step(main, temp_after, cc, clp, editable, tok.mask_token_id, quota)
                    if record_trajectory:
                        trajectory.append(slice_responses(step_ids, starts, length))
                    if same_editable(temp, temp_after, editable):
                        break
                    temp = temp_after
            main = step_ids
    finally:
        adapter.model.train(was_training)

    responses = slice_responses(main, starts, length)
    left = masks_remaining(responses, tok.mask_token_id)
    if any(left):
        raise AssertionError("proseco_unmask left mask tokens; the D-049 plan commits every position")
    if meter.nfe > max_calls:
        raise AssertionError("NFE exceeded max_calls")
    if record_trajectory and len(trajectory) != meter.nfe:
        raise AssertionError("trajectory length must equal the number of forwards")
    return SamplerRun(
        responses=responses, calls_per_example=[meter.nfe] * batch, batched_forward_calls=meter.nfe,
        call_kinds=meter.kinds, max_calls=max_calls, masks_remaining=left,
        trajectory=trajectory, final_input_ids=main.detach().cpu(),
        plan={"unmask_steps": len(quotas), "quotas": list(quotas), "omega": omega,
              "corrector_steps": corrector_steps, "completing": planned >= 1,
              "worst_case_nfe": worst_case_nfe(len(quotas), omega, corrector_steps) if planned >= 1 else None})
