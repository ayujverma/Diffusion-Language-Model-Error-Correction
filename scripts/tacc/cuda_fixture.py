"""Controlled reference fixture for the CUDA pilot (D-033).

``make`` RUNS ON: the Mac, CPU fp32 (deterministic reference). ``check`` RUNS ON: a
Lonestar6 GPU compute node inside scripts/tacc/pilot.slurm. Both load ckpt-0001000
read-only and perform the SAME computation:

1. greedy D-029 generation at T = L for the first problem of each monitoring setting;
2. one training forward on a fixed batch (one train example per setting, D-028 masks
   drawn from a dedicated CPU stream), the pooled masked loss, per-token CE and the
   masked-row logits summary (top-5, logsumexp);
3. backward, pre-clip global and per-tensor gradient norms, clipping at 1.0;
4. one AdamW step from the checkpoint's own optimizer state, per-tensor update norms.

The tolerances below were fixed before any CUDA output existed. ``check`` exits non-zero
when any tolerance fails. A greedy-generation mismatch whose reference margin is below
GEN_NEAR_TIE is reported as a near tie, not as a failure.

    .venv/bin/python scripts/tacc/cuda_fixture.py make \
        --checkpoint runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000 \
        --out runs/reference/cuda_pilot_fixture
    python scripts/tacc/cuda_fixture.py check --device cuda --checkpoint <copy of ckpt-0001000> \
        --fixture runs/reference/cuda_pilot_fixture --out <dir>/fixture_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from diffusion_correction.env import environment_report, resolve_device, synchronize
from diffusion_correction.losses.masked import masked_diffusion_loss
from diffusion_correction.sampling.monotone import monotone_unmask
from diffusion_correction.training.data import collate_targets, load_verified_examples, select_subset, setting_key
from diffusion_correction.training.masking import apply_masks, sample_response_masks
from diffusion_correction.training.model_io import load_trained_adapter
from diffusion_correction.training.rng import make_generator
from diffusion_correction.training.run import load_problem_id_subset, sampler_allowed, task_format
from diffusion_correction.training.trainer import mask_digest

DATA_DIR = "runs/data/pilot-seed20261116"
MANIFEST = "990ecd1dd2341b81bc742fa91189b129877cd9b48b196fbdbc208e8586c81536"
MONITOR = "configs/eval/monitor_val_stage2_first50.json"
MONITOR_SHA = "324d5ca8eb01b3fd47f087472197adb6f687965a3bca6fec6e35c0b113dc3d94"
SEED = 20261117

# Declared before any CUDA result (fp32 parameters, TF32 disabled, CPU fp32 reference).
TOLERANCES = {
    "loss_rel": 1e-4,             # pooled masked loss, relative
    "token_ce_abs": 1e-3,         # per masked token CE, max absolute difference
    "lse_abs": 1e-3,              # logsumexp of each masked row, max absolute difference
    "top5_logit_abs": 1e-2,       # top-5 logit values at the reference's top-5 IDs
    "argmax_margin_floor": 1e-3,  # rows with reference top-2 margin below this may flip
    "grad_norm_rel": 1e-3,        # global pre-clip gradient norm, relative
    "tensor_grad_norm_rel": 1e-2, # per tensor, over tensors with norm >= 1e-3 x the largest
    "tensor_update_norm_rel": 1e-2,  # per-tensor AdamW update norm, same filter
    "gen_near_tie": 1e-3,         # reference top-2 allowed-logit margin at a committed token
}


def compute(checkpoint: str, device: torch.device) -> tuple[dict, dict]:
    tok, tf = task_format("runs/hf/open-dcoder-0.5B", DATA_DIR)
    adapter, model_id = load_trained_adapter(checkpoint, device)
    allowed = sampler_allowed(adapter, tf)
    model = adapter.model
    out: dict = {"model": model_id}
    tens: dict[str, torch.Tensor] = {}

    # 1. greedy generation at T = L, one problem per monitoring setting
    mon, _ = load_problem_id_subset(DATA_DIR, MANIFEST, MONITOR, MONITOR_SHA)
    firsts, seen = [], set()
    for e in mon:
        if setting_key(e) not in seen:
            seen.add(setting_key(e))
            firsts.append(e)
    gen = []
    for e in firsts:
        L = len(e.value_names) * tf.fmt.slot_len
        margins = []
        start = len(e.prompt_ids)
        allowed_dev = allowed.to(device)

        def observer(_s, _ids, logits, events, start=start, margins=margins):
            for ev in events:
                row = logits[ev.example, start + ev.position].float().masked_fill(~allowed_dev, float("-inf"))
                top = torch.topk(row, 2).values
                margins.append(float(top[0] - top[1]))

        res = monotone_unmask(adapter, [e.prompt_ids], [L], L, allowed, observer=observer)
        gen.append({"problem_id": e.problem_id, "setting": list(setting_key(e)), "response_ids": res.responses[0],
                    "min_committed_margin": min(margins)})
    out["generation"] = gen

    # 2-4. one training forward/backward/AdamW step on a fixed batch
    train, _ = load_verified_examples(DATA_DIR, "train", MANIFEST)
    exs = select_subset(train, 18, 0)
    lay = collate_targets(exs, adapter.tokens.pad_token_id)
    draw = sample_response_masks(lay, make_generator(SEED, "cuda_fixture"))
    out["batch"] = {"problem_ids": [e.problem_id for e in exs], "shape": list(lay.input_ids.shape),
                    "n_masked": draw.total_masked, "mask_digest": mask_digest(draw.masked_positions)}
    lay_d = lay.to(device)
    masked = draw.masked_positions.to(device)
    inp = apply_masks(lay_d.input_ids, masked, adapter.tokens.mask_token_id)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    names = {id(p): n for n, p in model.named_parameters()}
    logits = adapter.aligned_logits(lay_d.with_input_ids(inp))
    lo = masked_diffusion_loss(logits, lay_d.input_ids, masked, lay_d.editable_mask)
    rows = logits.detach()[masked].float()
    top = torch.topk(rows, 5, dim=-1)
    tens["token_ce"] = lo.token_ce.float().cpu()
    tens["lse"] = torch.logsumexp(rows, dim=-1).cpu()
    tens["top5_ids"] = top.indices.cpu()
    tens["top5_values"] = top.values.cpu()
    tens["argmax"] = rows.argmax(dim=-1).cpu()
    out["loss"] = float(lo.loss.detach())
    del rows, top
    lo.loss.backward()
    gnorms = torch.stack([p.grad.detach().float().norm() for p in params]).cpu()
    gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    out["grad_norm_pre_clip"] = gn
    tens["tensor_grad_norms"] = gnorms
    state = torch.load(Path(checkpoint) / "optimizer.pt", map_location="cpu", weights_only=False)
    opt = torch.optim.AdamW(params, lr=3e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False)
    opt.load_state_dict(state["optimizer"])
    before = [p.detach().clone() for p in params]
    opt.step()
    synchronize(device)
    tens["tensor_update_norms"] = torch.stack([(p.detach() - b).float().norm() for p, b in zip(params, before)]).cpu()
    out["param_names"] = [names[id(p)] for p in params]
    out["lr_used"] = opt.param_groups[0]["lr"]
    return out, tens


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


def compare(cur: dict, ct: dict, ref: dict, rt: dict) -> dict:
    T = TOLERANCES
    r: dict = {"tolerances": T}
    r["batch_identical"] = cur["batch"] == ref["batch"]
    r["loss"] = {"reference": ref["loss"], "current": cur["loss"], "rel": rel(cur["loss"], ref["loss"])}
    r["token_ce_max_abs"] = float((ct["token_ce"] - rt["token_ce"]).abs().max())
    r["lse_max_abs"] = float((ct["lse"] - rt["lse"]).abs().max())
    same_ids = bool(torch.equal(ct["top5_ids"], rt["top5_ids"]))
    r["top5_ids_identical"] = same_ids
    r["top5_logit_max_abs"] = float((ct["top5_values"] - rt["top5_values"]).abs().max()) if same_ids else None
    margin = rt["top5_values"][:, 0] - rt["top5_values"][:, 1]
    flips = (ct["argmax"] != rt["argmax"])
    r["argmax_rows"] = int(flips.numel())
    r["argmax_flips"] = int(flips.sum())
    r["argmax_flips_above_margin_floor"] = int((flips & (margin >= T["argmax_margin_floor"])).sum())
    r["grad_norm"] = {"reference": ref["grad_norm_pre_clip"], "current": cur["grad_norm_pre_clip"],
                      "rel": rel(cur["grad_norm_pre_clip"], ref["grad_norm_pre_clip"])}

    def per_tensor(key):
        a, b = ct[key].double(), rt[key].double()
        keep = b >= 1e-3 * b.max()
        rr = ((a - b).abs() / b.clamp_min(1e-30))[keep]
        worst = int(torch.nonzero(keep).squeeze(-1)[rr.argmax()])
        return {"max_rel": float(rr.max()), "n_compared": int(keep.sum()), "worst": ref["param_names"][worst]}

    r["tensor_grad_norms"] = per_tensor("tensor_grad_norms")
    r["tensor_update_norms"] = per_tensor("tensor_update_norms")
    r["lr_used"] = cur["lr_used"]
    gens = []
    for c, g in zip(cur["generation"], ref["generation"]):
        same = c["response_ids"] == g["response_ids"]
        gens.append({"problem_id": g["problem_id"], "identical": same,
                     "reference_min_margin": g["min_committed_margin"],
                     "near_tie": (not same) and g["min_committed_margin"] < T["gen_near_tie"]})
    r["generation"] = gens
    checks = {
        "batch_identical": r["batch_identical"],
        "loss_rel": r["loss"]["rel"] <= T["loss_rel"],
        "token_ce_abs": r["token_ce_max_abs"] <= T["token_ce_abs"],
        "lse_abs": r["lse_max_abs"] <= T["lse_abs"],
        "top5": same_ids and r["top5_logit_max_abs"] <= T["top5_logit_abs"],
        "argmax": r["argmax_flips_above_margin_floor"] == 0,
        "grad_norm_rel": r["grad_norm"]["rel"] <= T["grad_norm_rel"],
        "tensor_grad_norm_rel": r["tensor_grad_norms"]["max_rel"] <= T["tensor_grad_norm_rel"],
        "tensor_update_norm_rel": r["tensor_update_norms"]["max_rel"] <= T["tensor_update_norm_rel"],
        "generation": all(g["identical"] or g["near_tie"] for g in gens),
        "lr_is_3e-5": cur["lr_used"] == 3e-5,
    }
    r["checks"] = checks
    r["passed"] = all(checks.values())
    return r


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["make", "check"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fixture", default="runs/reference/cuda_pilot_fixture")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    device = resolve_device(a.device)
    t0 = time.perf_counter()
    cur, tens = compute(a.checkpoint, device)
    cur["seconds"] = round(time.perf_counter() - t0, 1)
    cur["environment"] = environment_report(device)
    if a.mode == "make":
        base = Path(a.out or a.fixture)
        base.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tens, base.with_suffix(".pt"))
        cur["tolerances"] = TOLERANCES
        cur["tensors_sha256"] = hashlib.sha256(base.with_suffix(".pt").read_bytes()).hexdigest()
        base.with_suffix(".json").write_text(json.dumps(cur, indent=2, default=str) + "\n")
        print(json.dumps({k: cur[k] for k in ("loss", "grad_norm_pre_clip", "batch", "seconds")}, indent=2))
        print(f"wrote {base}.pt and {base}.json")
        return 0
    base = Path(a.fixture)
    ref = json.loads(base.with_suffix(".json").read_text())
    if hashlib.sha256(base.with_suffix(".pt").read_bytes()).hexdigest() != ref["tensors_sha256"]:
        raise SystemExit("fixture tensors do not match their recorded sha256")
    if ref["tolerances"] != TOLERANCES:
        raise SystemExit("tolerances differ from the ones recorded with the fixture")
    rt = torch.load(base.with_suffix(".pt"), map_location="cpu", weights_only=True)
    rep = compare(cur, tens, ref, rt)
    rep["environment"] = cur["environment"]
    rep["reference_environment"] = ref["environment"]
    rep["seconds"] = cur["seconds"]
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if "environment" not in k}, indent=2, default=str))
    return 0 if rep["passed"] and not any(math.isnan(x) for x in (rep["loss"]["rel"],)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
