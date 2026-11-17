"""Read-only continuation preflight for a resumable checkpoint (CPU only; no model forward).

RUNS ON: the Mac, and on a Lonestar6 compute node inside scripts/tacc/env_check.slurm
(it loads about 8 GB of tensors, so not on a login node). Nothing is written next to
the checkpoint; the report goes to --out.

Checks: file hashes; resume compatibility of the continuation config; dataset identity
recomputed from the files; base-model identity; model tensors (keys, shapes, dtypes,
distance from the pretrained weights); optimizer and scheduler state; the projected LR
trajectory to max_updates using the trainer's own schedule function; RNG streams; the
exact next batches (example IDs and D-028 mask draws) that a resume must reproduce;
pre-clipping gradient-norm and MPS memory statistics from the original run's metrics.

    python scripts/tacc/preflight_checkpoint.py \
        --checkpoint runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000 \
        --config configs/train/warmup_continuation.yaml \
        --metrics runs/train/warmup-pilot-v1/metrics.jsonl --out runs/preflight/ckpt-0001000.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from diffusion_correction.models.open_dcoder import resolve_token_spec, verify_files
from diffusion_correction.training.checkpoint import verify_checkpoint
from diffusion_correction.training.config import RESUME_MUTABLE, compat_hash, load_config
from diffusion_correction.training.data import EpochSampler, collate_targets, load_verified_examples
from diffusion_correction.training.masking import sample_response_masks
from diffusion_correction.training.rng import STREAMS, make_generator
from diffusion_correction.training.trainer import lr_factor, mask_digest


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="the continuation config that will resume this checkpoint")
    p.add_argument("--metrics", default=None, help="metrics.jsonl of the run that produced the checkpoint")
    p.add_argument("--next-batches", type=int, default=10)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t0 = time.perf_counter()
    ck = Path(a.checkpoint)
    rep: dict = {"checkpoint": str(ck), "checks": {}}
    chk = rep["checks"]

    complete = verify_checkpoint(ck)
    rep["files_sha256"] = complete["files"]
    rep["complete_json_sha256"] = sha((ck / "COMPLETE.json").read_bytes())
    rep["file_bytes"] = {f: (ck / f).stat().st_size for f in complete["files"]} | {
        "COMPLETE.json": (ck / "COMPLETE.json").stat().st_size}
    chk["file_hashes_match_COMPLETE_json"] = True
    state = json.loads((ck / "state.json").read_text())
    rep["counters"] = {k: state[k] for k in ("update", "batches_consumed", "examples_consumed", "skipped_zero_mask",
                                              "train_wall_seconds")}
    rep["sampler"] = {k: state["sampler"][k] for k in ("n", "epoch", "pos", "examples_consumed")} | {
        "perm_sha256": sha(json.dumps(state["sampler"]["perm"]).encode())}
    rep["written_by"] = {"git": state["git"], "versions": state["versions"], "device": state["device"]}
    rep["timers_in_state"] = state.get("timers")
    rep["selection_in_state"] = state.get("selection")

    cfg = load_config(a.config)
    old = state["config"]
    rep["continuation_config"] = a.config
    chk["compat_hash_matches"] = compat_hash(cfg) == state["compat_hash"]
    rep["compat_hash"] = {"checkpoint": state["compat_hash"], "continuation": compat_hash(cfg)}
    rep["immutable_key_differences"] = sorted(k for k in set(old) | set(cfg)
                                              if k not in RESUME_MUTABLE and old.get(k) != cfg.get(k))
    rep["mutable_sections_changed"] = sorted(k for k in RESUME_MUTABLE if old.get(k) != cfg.get(k))
    rep["resolved_training_settings"] = {k: old[k] for k in (
        "seed", "method", "dtype", "autocast", "batch_size", "grad_accum", "optimizer", "scheduler",
        "clip_grad_norm", "mask", "backbone", "model_dir", "data", "init_checkpoint")}

    d = cfg["data"]
    examples, ident = load_verified_examples(d["dir"], d["split"], d["manifest_sha256"], d["subset"])
    chk["data_identity_matches"] = ident.to_dict() == state["data_identity"]
    rep["data_identity"] = ident.to_dict()
    _, vid = load_verified_examples(d["dir"], "val", d["manifest_sha256"])
    rep["val_identity"] = vid.to_dict()

    base = verify_files(Path(cfg["model_dir"]), check_weights=True)
    chk["base_model_files_match"] = base == state["base_model"]["file_sha256"]

    from safetensors.torch import load_file

    m = load_file(str(ck / "model.safetensors"))
    m0 = load_file(str(Path(cfg["model_dir"]) / "model.safetensors"))
    m0.pop("lm_head.weight", None)
    chk["model_keys_match_pretrained"] = set(m) == set(m0)
    chk["model_shapes_match_pretrained"] = all(m[k].shape == m0[k].shape for k in m)
    rep["model"] = {"tensors": len(m), "parameters": sum(v.numel() for v in m.values()),
                    "dtypes": sorted({str(v.dtype) for v in m.values()}),
                    "lm_head_stored": "lm_head.weight" in m,
                    "tensors_differing_from_pretrained": sum(not torch.equal(m[k], m0[k].to(m[k].dtype)) for k in m),
                    "l2_distance_from_pretrained": sum(float(((m[k].double() - m0[k].double()) ** 2).sum())
                                                       for k in m) ** 0.5,
                    "l2_norm": sum(float((m[k].double() ** 2).sum()) for k in m) ** 0.5}
    chk["model_float32"] = rep["model"]["dtypes"] == ["torch.float32"]
    del m0

    opt = torch.load(ck / "optimizer.pt", map_location="cpu", weights_only=False)
    o, sch = opt["optimizer"], opt["scheduler"]
    steps = sorted({float(s["step"]) for s in o["state"].values()})
    st_bytes = sum(t.numel() * t.element_size() for s in o["state"].values() for t in s.values() if torch.is_tensor(t))
    rep["optimizer"] = {"param_groups": [{k: v for k, v in g.items() if k != "params"} for g in o["param_groups"]],
                        "n_param_tensors": sum(len(g["params"]) for g in o["param_groups"]),
                        "n_state_entries": len(o["state"]), "step_values": steps,
                        "state_keys": sorted({k for s in o["state"].values() for k in s}),
                        "state_dtypes": sorted({str(t.dtype) for s in o["state"].values() for t in s.values()
                                                if torch.is_tensor(t)}),
                        "state_bytes": st_bytes,
                        "exp_avg_numel": sum(s["exp_avg"].numel() for s in o["state"].values())}
    rep["scheduler_state"] = {k: v for k, v in sch.items() if k != "lr_lambdas"}
    chk["optimizer_covers_all_tensors"] = len(o["state"]) == rep["optimizer"]["n_param_tensors"] == len(m)
    chk["optimizer_step_equals_update"] = steps == [float(state["update"])]
    chk["scheduler_epoch_equals_update"] = sch["last_epoch"] == state["update"]

    warm = int(old["scheduler"]["warmup_updates"])
    base_lr = float(old["optimizer"]["lr"])
    lr_now = o["param_groups"][0]["lr"]
    limit = int(cfg["limits"]["max_updates"])
    proj = {u: base_lr * lr_factor(u, warm) for u in range(state["update"], limit)}
    rep["learning_rate"] = {
        "schedule": f"{old['scheduler']['name']}: lr = {base_lr} * min(1, (completed + 1) / {warm}); "
                    "no decay horizon, so max_updates does not enter the schedule",
        "lr_stored_in_optimizer": lr_now, "base_lrs": sch.get("base_lrs"), "last_lr": sch.get("_last_lr"),
        "lr_for_next_update": proj[state["update"]],
        "projected_distinct_lrs_to_max_updates": sorted(set(proj.values())), "projected_through_update": limit}
    chk["lr_constant_through_max_updates"] = set(proj.values()) == {lr_now} and lr_now == base_lr

    rng = torch.load(ck / "rng.pt", map_location="cpu", weights_only=False)
    rep["rng"] = {"global_keys": sorted(rng["global"]), "global_device_type": rng["global"].get("device_type"),
                  "streams": {k: {"state_sha256": sha(v.numpy().tobytes()), "bytes": int(v.numel())}
                              for k, v in rng["streams"].items()},
                  "fresh_stream_sha256": {k: sha(make_generator(old["seed"], k).get_state().numpy().tobytes())
                                          for k in STREAMS}}
    chk["all_streams_present"] = set(rng["streams"]) == set(STREAMS)
    chk["streams_advanced_from_seed"] = all(
        rep["rng"]["streams"][k]["state_sha256"] != rep["rng"]["fresh_stream_sha256"][k]
        for k in ("data_order", "train_mask"))
    chk["eval_stream_untouched_by_training"] = (rep["rng"]["streams"]["eval"]["state_sha256"]
                                                 == rep["rng"]["fresh_stream_sha256"]["eval"])

    g_order, g_mask = torch.Generator(), torch.Generator()
    g_order.set_state(rng["streams"]["data_order"])
    g_mask.set_state(rng["streams"]["train_mask"])
    sampler = EpochSampler(len(examples), g_order)
    sampler.load_state_dict(state["sampler"])
    pad = resolve_token_spec(Path(cfg["model_dir"])).pad_token_id
    nxt = []
    for k in range(a.next_batches):
        idx = sampler.next_indices(int(old["batch_size"]))
        exs = [examples[i] for i in idx]
        lay = collate_targets(exs, pad)
        draw = sample_response_masks(lay, g_mask, old["mask"]["t_min"], old["mask"]["t_max"])
        nxt.append({"batch": state["batches_consumed"] + k + 1, "epoch": sampler.epoch, "example_indices": idx,
                    "problem_ids_sha256": sha("\n".join(e.problem_id for e in exs).encode()),
                    "mask_digest": mask_digest(draw.masked_positions), "n_masked": draw.total_masked,
                    "mask_t_sha256": sha(json.dumps(draw.t).encode()), "shape": list(lay.input_ids.shape)})
    rep["expected_next_batches"] = nxt

    if a.metrics:
        recs = [json.loads(x) for x in Path(a.metrics).read_text().splitlines() if x]
        upd = [r for r in recs if "grad_norm" in r]
        gn = [r["grad_norm"] for r in upd]
        clip = float(old["clip_grad_norm"])
        fac = [min(1.0, clip / g) for g in gn]
        last = gn[-100:]
        mem = [r["memory"] for r in upd if "memory" in r]

        def rng_of(key):
            xs = [x[key] for x in mem if key in x]
            return {"min": min(xs), "max": max(xs), "n": len(xs)} if xs else None

        rep["gradient_diagnostics"] = {
            "source": a.metrics, "n_updates": len(gn), "clip_threshold": clip,
            "pre_clip_grad_norm": {"median": statistics.median(gn), "p95": pct(gn, 0.95), "min": min(gn),
                                   "max": max(gn), "last100_median": statistics.median(last),
                                   "last100_p95": pct(last, 0.95)},
            "clip_factor": {"definition": "min(1, clip / pre-clip norm), the scale applied to the gradient",
                            "median": statistics.median(fac), "p05": pct(fac, 0.05), "p95": pct(fac, 0.95),
                            "last100_median": statistics.median(fac[-100:])},
            "fraction_clipped": sum(r["clipped"] for r in upd) / len(upd),
            "not_measured": ["post-clip per-tensor gradient norms", "parameter-update norms per step",
                             "AdamW second-moment statistics over time", "per-layer gradient norms"]}
        rep["mps_memory_from_metrics"] = {
            "definitions": {"mps_current_allocated_gb": "live tensors (allocator)",
                            "mps_driver_allocated_gb": "Metal driver allocation incl. allocator cache",
                            "process_peak_rss_gb": "process peak resident set (unified memory)"},
            "mps_current_allocated_gb": rng_of("mps_current_allocated_gb"),
            "mps_driver_allocated_gb": rng_of("mps_driver_allocated_gb"),
            "process_peak_rss_gb": rng_of("process_peak_rss_gb")}

    rep["passed"] = all(chk.values())
    rep["seconds"] = round(time.perf_counter() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print(json.dumps({"passed": rep["passed"], "checks": chk, "counters": rep["counters"],
                      "learning_rate": rep["learning_rate"], "optimizer": rep["optimizer"]["param_groups"],
                      "gradient_diagnostics": rep.get("gradient_diagnostics")}, indent=2, default=str))
    print(f"wrote {a.out}")
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
