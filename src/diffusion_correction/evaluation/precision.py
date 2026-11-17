"""bf16 vs fp32 forward/loss/gradient comparison on a controlled real-data fixture.

The fixture is fixed training examples with masks from a fixed generator. The fp32
reference is computed twice to measure the run-to-run noise floor on the device.
Candidates: bf16 autocast with fp32 master weights, and pure bf16 parameters.

Screening tolerances, declared before running (bf16 unit roundoff 2^-8 ~ 3.9e-3,
accumulated over 24 layers):
- |loss_bf16 - loss_fp32| / loss_fp32 <= 2e-2;
- greedy agreement on masked positions >= 0.98;
- global gradient relative L2 error <= 0.1 and cosine >= 0.995.
Passing them only makes bf16 eligible; it is adopted only if it is also faster.
"""

from __future__ import annotations

import time

import torch

from ..env import synchronize
from ..losses.masked import masked_diffusion_loss
from ..models.open_dcoder import load_open_dcoder
from ..training.data import collate_targets, select_subset
from ..training.masking import apply_masks, sample_response_masks
from ..training.rng import make_generator

TOL = {"loss_rel": 2e-2, "argmax_agreement": 0.98, "grad_rel_l2": 0.1, "grad_cosine": 0.995}


def _run(model_adapter, lay, masked, targets, device, autocast: bool, reps: int = 3):
    model = model_adapter.model
    model.train()
    times = []
    for r in range(reps):
        model.zero_grad(set_to_none=True)
        synchronize(device)
        t0 = time.perf_counter()
        ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16) if autocast else torch.autocast(
            device_type=device.type, enabled=False)
        with ctx:
            logits = model_adapter.aligned_logits(lay)
        out = masked_diffusion_loss(logits, targets, masked, lay.editable_mask)
        out.loss.backward()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    rows = logits.detach()[masked].float()
    grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    return float(out.loss.detach()), rows.cpu(), grads, times


def _compare(ref, cand) -> dict:
    loss_r, rows_r, g_r, _ = ref
    loss_c, rows_c, g_c, _ = cand
    num = sum(float(((g_c[k] - g_r[k]) ** 2).sum()) for k in g_r)
    den = sum(float((g_r[k] ** 2).sum()) for k in g_r)
    dot = sum(float((g_c[k] * g_r[k]).sum()) for k in g_r)
    nc = sum(float((g_c[k] ** 2).sum()) for k in g_r)
    per = sorted(float((g_c[k] - g_r[k]).norm() / g_r[k].norm()) for k in g_r if float(g_r[k].norm()) > 0)
    return {
        "loss": loss_c, "loss_rel_diff": abs(loss_c - loss_r) / abs(loss_r),
        "logit_max_abs_diff": float((rows_c - rows_r).abs().max()),
        "logit_mean_abs_diff": float((rows_c - rows_r).abs().mean()),
        "argmax_agreement": float((rows_c.argmax(-1) == rows_r.argmax(-1)).float().mean()),
        "grad_global_rel_l2": (num / den) ** 0.5, "grad_global_cosine": dot / (den * nc) ** 0.5,
        "grad_per_tensor_rel_l2_median": per[len(per) // 2], "grad_per_tensor_rel_l2_max": per[-1],
        "n_grad_tensors": len(g_r),
    }


def precision_check(device: torch.device, examples, n_examples: int = 8, seed: int = 0,
                    model_dir: str = "runs/hf/open-dcoder-0.5B") -> dict:
    adapter = load_open_dcoder(model_dir, device=device, dtype=torch.float32)
    exs = select_subset(examples, n_examples, seed)
    lay = collate_targets(exs, adapter.tokens.pad_token_id)
    draw = sample_response_masks(lay, make_generator(seed, "precision_fixture"))
    report = {"device": str(device), "n_examples": len(exs), "shape": list(lay.input_ids.shape),
              "n_masked": draw.total_masked, "tolerances": TOL, "candidates": {}}
    lay_d = lay.to(device)
    targets = lay_d.input_ids
    masked = draw.masked_positions.to(device)
    inp = lay_d.with_input_ids(apply_masks(targets, masked, adapter.tokens.mask_token_id))
    ref = _run(adapter, inp, masked, targets, device, autocast=False)
    rep = _run(adapter, inp, masked, targets, device, autocast=False)
    report["fp32_loss"] = ref[0]
    report["fp32_seconds_fwd_bwd"] = ref[3]
    report["fp32_repeat_noise_floor"] = _compare(ref, rep)
    for name, fn in (("bf16_autocast_fp32_master", lambda: _run(adapter, inp, masked, targets, device, True)),):
        try:
            cand = fn()
            c = _compare(ref, cand)
            c["seconds_fwd_bwd"] = cand[3]
            c["supported"] = True
        except Exception as e:  # recorded, not hidden
            c = {"supported": False, "error": f"{type(e).__name__}: {e}"}
        report["candidates"][name] = c
    try:
        adapter.model.to(torch.bfloat16)
        cand = _run(adapter, inp, masked, targets, device, autocast=False)
        c = _compare(ref, cand)
        c["seconds_fwd_bwd"] = cand[3]
        c["supported"] = True
    except Exception as e:
        c = {"supported": False, "error": f"{type(e).__name__}: {e}"}
    report["candidates"]["bf16_parameters"] = c
    for c in report["candidates"].values():
        if c.get("supported"):
            c["passes_screening_tolerances"] = (c["loss_rel_diff"] <= TOL["loss_rel"]
                                                and c["argmax_agreement"] >= TOL["argmax_agreement"]
                                                and c["grad_global_rel_l2"] <= TOL["grad_rel_l2"]
                                                and c["grad_global_cosine"] >= TOL["grad_cosine"])
    return report
