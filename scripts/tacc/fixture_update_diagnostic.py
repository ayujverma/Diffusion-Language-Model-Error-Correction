"""Bounded diagnostic for the CUDA pilot fixture's per-tensor update-norm check (pilot 3592809).

Not a gate: it never writes ENV_CHECK_PASSED or PILOT_PASSED, never changes cuda_fixture.py,
its TOLERANCES or the reference fixture, and never saves a checkpoint.

``run`` repeats steps 2-4 of ``cuda_fixture.compute`` (same 18-example batch, same masks from
the ``cuda_fixture`` CPU stream, same loss, backward, clipping at 1.0 and one fp32 AdamW step
from the checkpoint's optimizer state; generation, which does not touch the step, is skipped)
on one device, once. All training, clipping and optimizer arithmetic stays fp32. It records:
- alignment: parameter names, shapes and order against the reference fixture and the saved
  optimizer state; the optimizer settings actually loaded; step counters; bitwise equality of
  the loaded moments with the file, and their float64 norms;
- per tensor: the native fp32 gradient and update norms computed exactly as the fixture does,
  and float64 norms of the same quantities computed on CPU in fixed chunks from the saved fp32
  values (update = post - pre);
- the raw post-step embedding and its post-clip gradient (fp32), for elementwise comparison;
- on CUDA, allocator counters (allocated/reserved/peaks, alloc retries, OOMs) at each phase.
The pre-step weights are copied to host memory (not cloned on the GPU).

``compare A B`` compares two ``run`` outputs (B is the reference side, as the Mac CPU fixture
was): norms both ways, the embedding update elementwise, and how the difference splits by the
size of the AdamW denominator.

    python scripts/tacc/fixture_update_diagnostic.py run --device cpu \
        --checkpoint runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000 --out runs/fixture_diag/mac_cpu
    python scripts/tacc/fixture_update_diagnostic.py compare A_DIR B_DIR --checkpoint <ckpt-0001000> --out X.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cuda_fixture as F  # noqa: E402  (constants and the reference fixture format)

from diffusion_correction.env import environment_report, resolve_device, synchronize  # noqa: E402
from diffusion_correction.losses.masked import masked_diffusion_loss  # noqa: E402
from diffusion_correction.training.data import collate_targets, load_verified_examples, select_subset  # noqa: E402
from diffusion_correction.training.masking import apply_masks, sample_response_masks  # noqa: E402
from diffusion_correction.training.model_io import load_trained_adapter  # noqa: E402
from diffusion_correction.training.rng import make_generator  # noqa: E402
from diffusion_correction.training.trainer import mask_digest  # noqa: E402

EMBED = "model.embed_tokens.weight"
CHUNK = 1 << 22
MEM_KEYS = ("allocated_bytes.all.current", "allocated_bytes.all.peak", "reserved_bytes.all.current",
            "reserved_bytes.all.peak", "num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free",
            "num_sync_all_streams")


def f64_sq(x: torch.Tensor, y: torch.Tensor | None = None) -> float:
    """Sum of squares of x (or x - y) in float64 on CPU, in fixed chunks, partial sums combined with fsum."""
    a = x.detach().reshape(-1).cpu()
    b = None if y is None else y.detach().reshape(-1).cpu()
    parts = []
    for s in range(0, a.numel(), CHUNK):
        d = a[s:s + CHUNK].double()
        if b is not None:
            d = d - b[s:s + CHUNK].double()
        parts.append(float((d * d).sum()))
    return math.fsum(parts)


def sha256_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def mem(device: torch.device, phase: str) -> dict:
    if device.type != "cuda":
        return {"phase": phase}
    s = torch.cuda.memory_stats(device)
    free, total = torch.cuda.mem_get_info(device)
    return {"phase": phase, **{k: s.get(k) for k in MEM_KEYS}, "device_used_bytes": total - free,
            "device_total_bytes": total}


def per_tensor_rel(a: torch.Tensor, b: torch.Tensor, names: list[str]) -> dict:
    """The fixture's per-tensor comparison (b is the reference; same >= 1e-3 x max filter)."""
    a, b = a.double(), b.double()
    keep = b >= 1e-3 * b.max()
    rr = ((a - b).abs() / b.clamp_min(1e-30))[keep]
    worst = int(torch.nonzero(keep).squeeze(-1)[rr.argmax()])
    return {"max_rel": float(rr.max()), "n_compared": int(keep.sum()), "worst": names[worst],
            "embed_rel": float((a[0] - b[0]).abs() / b[0]) if names[0] == EMBED else None}


def run(checkpoint: str, device: torch.device, out: Path, fixture: Path) -> dict:
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.perf_counter()
    memlog = [mem(device, "start")]
    adapter, model_id = load_trained_adapter(checkpoint, device)
    model = adapter.model
    memlog.append(mem(device, "model_loaded"))

    train, _ = load_verified_examples(F.DATA_DIR, "train", F.MANIFEST)
    exs = select_subset(train, 18, 0)
    lay = collate_targets(exs, adapter.tokens.pad_token_id)
    draw = sample_response_masks(lay, make_generator(F.SEED, "cuda_fixture"))
    batch = {"problem_ids": [e.problem_id for e in exs], "shape": list(lay.input_ids.shape),
             "n_masked": draw.total_masked, "mask_digest": mask_digest(draw.masked_positions)}
    lay_d = lay.to(device)
    masked = draw.masked_positions.to(device)
    inp = apply_masks(lay_d.input_ids, masked, adapter.tokens.mask_token_id)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    pnames = {id(p): n for n, p in model.named_parameters()}
    names = [pnames[id(p)] for p in params]
    logits = adapter.aligned_logits(lay_d.with_input_ids(inp))
    lo = masked_diffusion_loss(logits, lay_d.input_ids, masked, lay_d.editable_mask)
    loss = float(lo.loss.detach())
    del logits
    memlog.append(mem(device, "forward"))
    lo.loss.backward()
    del lo
    memlog.append(mem(device, "backward"))
    grad_native = torch.stack([p.grad.detach().float().norm() for p in params]).cpu()
    grad_f64 = torch.tensor([math.sqrt(f64_sq(p.grad)) for p in params], dtype=torch.float64)  # pre-clip, as native
    gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))

    saved = torch.load(Path(checkpoint) / "optimizer.pt", map_location="cpu", weights_only=False)["optimizer"]
    opt = torch.optim.AdamW(params, lr=3e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False)
    opt.load_state_dict(saved)
    group = {k: v for k, v in opt.param_groups[0].items() if k != "params"}
    file_state = saved["state"]
    align = {
        "n_params": len(params), "n_param_groups": len(opt.param_groups),
        "group_param_indices_are_0_to_n": saved["param_groups"][0]["params"] == list(range(len(params))),
        "state_shapes_match_params": all(tuple(file_state[i]["exp_avg"].shape) == tuple(p.shape)
                                         for i, p in enumerate(params)),
    }
    moments, steps = [], set()
    for i, p in enumerate(params):
        s = opt.state[p]
        steps.add((float(s["step"]), str(s["step"].device), str(s["step"].dtype)))
        ea, eq = s["exp_avg"].detach().cpu(), s["exp_avg_sq"].detach().cpu()
        moments.append({"exp_avg_equal_file": bool(torch.equal(ea, file_state[i]["exp_avg"])),
                        "exp_avg_sq_equal_file": bool(torch.equal(eq, file_state[i]["exp_avg_sq"])),
                        "exp_avg_norm_f64": math.sqrt(f64_sq(ea)), "exp_avg_sq_norm_f64": math.sqrt(f64_sq(eq)),
                        "device": str(s["exp_avg"].device), "dtype": str(s["exp_avg"].dtype)})
        del ea, eq
    del saved, file_state
    memlog.append(mem(device, "optimizer_loaded"))

    before = [p.detach().to("cpu", copy=True) for p in params]
    opt.step()
    synchronize(device)
    memlog.append(mem(device, "optimizer_step"))
    upd_native = []
    for p, b in zip(params, before):  # the fixture's formula, one transient device tensor at a time
        bd = b.to(device)
        upd_native.append((p.detach() - bd).float().norm().cpu())
        del bd
    upd_native = torch.stack(upd_native)
    upd_f64 = torch.tensor([math.sqrt(f64_sq(p, b)) for p, b in zip(params, before)], dtype=torch.float64)
    memlog.append(mem(device, "update_norms"))

    assert names[0] == EMBED, names[0]
    embed_post = params[0].detach().cpu().clone()
    embed_grad = params[0].grad.detach().cpu().clone()
    torch.save({"embed_post": embed_post}, out / "embed_post.pt")
    torch.save({"embed_grad_post_clip": embed_grad}, out / "embed_grad.pt")
    torch.save({"names": names, "grad_norm_native_fp32": grad_native, "grad_norm_f64": grad_f64,
                "update_norm_native_fp32": upd_native, "update_norm_f64": upd_f64}, out / "norms.pt")

    ref = json.loads(fixture.with_suffix(".json").read_text())
    rt = torch.load(fixture.with_suffix(".pt"), map_location="cpu", weights_only=True)
    res = {
        "device": str(device), "checkpoint": checkpoint, "model": model_id, "torch_threads": torch.get_num_threads(),
        "batch": batch, "batch_identical_to_fixture": batch == ref["batch"],
        "loss": loss, "loss_rel_vs_fixture": abs(loss - ref["loss"]) / abs(ref["loss"]),
        "grad_norm_pre_clip": gn,
        "embed_update_norm": {"native_fp32": float(upd_native[0]), "f64": float(upd_f64[0]),
                              "native_rel_to_f64": abs(float(upd_native[0]) - float(upd_f64[0])) / float(upd_f64[0])},
        "max_native_vs_f64_update_norm_rel": float(((upd_native.double() - upd_f64).abs() / upd_f64).max()),
        "max_native_vs_f64_grad_norm_rel": float(((grad_native.double() - grad_f64).abs()
                                                  / grad_f64.clamp_min(1e-30)).max()),
        "names_equal_fixture": names == ref["param_names"],
        "shapes": [list(p.shape) for p in params], "param_dtypes": sorted({str(p.dtype) for p in params}),
        "alignment": align, "optimizer_settings_loaded": {k: (list(v) if isinstance(v, tuple) else v)
                                                          for k, v in group.items()},
        "step_counters": sorted([list(s) for s in steps]),
        "moments_all_equal_file": all(m["exp_avg_equal_file"] and m["exp_avg_sq_equal_file"] for m in moments),
        "moments": moments,
        "embed_pre_sha256": sha256_tensor(before[0]), "embed_post_sha256": sha256_tensor(embed_post),
        "embed_grad_sha256": sha256_tensor(embed_grad),
        "native_vs_fixture": {
            "update_norms_bitwise_equal": bool(torch.equal(upd_native, rt["tensor_update_norms"])),
            "grad_norms_bitwise_equal": bool(torch.equal(grad_native, rt["tensor_grad_norms"])),
            "tensor_update_norms": per_tensor_rel(upd_native, rt["tensor_update_norms"], names),
            "tensor_grad_norms": per_tensor_rel(grad_native, rt["tensor_grad_norms"], names),
            "tolerance_tensor_update_norm_rel": F.TOLERANCES["tensor_update_norm_rel"],
        },
        "memory": memlog, "seconds": round(time.perf_counter() - t0, 1),
        "environment": environment_report(device),
        "files": {n: sha256_file(out / n) for n in ("embed_post.pt", "embed_grad.pt", "norms.pt")},
    }
    (out / "diag.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    return res


def compare(a_dir: Path, b_dir: Path, checkpoint: str) -> dict:
    from safetensors import safe_open

    A, B = (json.loads((d / "diag.json").read_text()) for d in (a_dir, b_dir))
    for d, j in ((a_dir, A), (b_dir, B)):
        for n, h in j["files"].items():
            if sha256_file(d / n) != h:
                raise SystemExit(f"{d / n} does not match its recorded sha256")
    na, nb = (torch.load(d / "norms.pt", weights_only=True) for d in (a_dir, b_dir))
    names = nb["names"]
    r: dict = {"a": str(a_dir), "b_reference": str(b_dir), "a_device": A["device"], "b_device": B["device"],
               "a_torch": A["environment"]["torch"], "b_torch": B["environment"]["torch"]}
    r["alignment_equal"] = {
        "names": na["names"] == nb["names"], "shapes": A["shapes"] == B["shapes"], "batch": A["batch"] == B["batch"],
        "optimizer_settings": A["optimizer_settings_loaded"] == B["optimizer_settings_loaded"],
        "step_values": [s[0] for s in A["step_counters"]] == [s[0] for s in B["step_counters"]],
        "moment_norms": [(m["exp_avg_norm_f64"], m["exp_avg_sq_norm_f64"]) for m in A["moments"]]
                        == [(m["exp_avg_norm_f64"], m["exp_avg_sq_norm_f64"]) for m in B["moments"]],
        "moments_equal_file_both": A["moments_all_equal_file"] and B["moments_all_equal_file"],
        "embed_pre_sha256": A["embed_pre_sha256"] == B["embed_pre_sha256"],
    }
    r["loss_rel"] = abs(A["loss"] - B["loss"]) / abs(B["loss"])
    r["grad_norm_pre_clip_rel"] = abs(A["grad_norm_pre_clip"] - B["grad_norm_pre_clip"]) / B["grad_norm_pre_clip"]
    r["update_norms"] = {"native_fp32": per_tensor_rel(na["update_norm_native_fp32"], nb["update_norm_native_fp32"],
                                                       names),
                         "f64": per_tensor_rel(na["update_norm_f64"], nb["update_norm_f64"], names)}
    r["grad_norms"] = {"native_fp32": per_tensor_rel(na["grad_norm_native_fp32"], nb["grad_norm_native_fp32"], names),
                       "f64": per_tensor_rel(na["grad_norm_f64"], nb["grad_norm_f64"], names)}
    for side, n in (("a", na), ("b", nb)):
        r[f"embed_update_norm_{side}"] = {"native_fp32": float(n["update_norm_native_fp32"][0]),
                                          "f64": float(n["update_norm_f64"][0])}
        r[f"embed_norm_accumulation_rel_{side}"] = (abs(float(n["update_norm_native_fp32"][0])
                                                        - float(n["update_norm_f64"][0])) / float(n["update_norm_f64"][0]))

    # Elementwise, in row chunks: pre from the checkpoint, post and grads from the two runs.
    pa = torch.load(a_dir / "embed_post.pt", weights_only=True)["embed_post"]
    pb = torch.load(b_dir / "embed_post.pt", weights_only=True)["embed_post"]
    ga = torch.load(a_dir / "embed_grad.pt", weights_only=True)["embed_grad_post_clip"]
    gb = torch.load(b_dir / "embed_grad.pt", weights_only=True)["embed_grad_post_clip"]
    with safe_open(str(Path(checkpoint) / "model.safetensors"), "pt") as f:
        pre = f.get_tensor(EMBED)
    if sha256_tensor(pre) != B["embed_pre_sha256"]:
        raise SystemExit("checkpoint embedding differs from the runs' pre-step embedding")
    opt = torch.load(Path(checkpoint) / "optimizer.pt", map_location="cpu", weights_only=False, mmap=True)
    st = opt["optimizer"]["state"][0]
    g0 = opt["optimizer"]["param_groups"][0]
    beta2, eps, step = g0["betas"][1], g0["eps"], float(st["step"]) + 1
    bc2 = 1 - beta2 ** step
    edges = [0, 2, 10, 100, math.inf]  # (sqrt(v_hat) + eps) / eps, from B's gradient
    bk = [{"denominator_over_eps": [lo, hi], "n": 0, "diff_sq": [], "ref_sq": [], "grad_diff_sq": [],
           "grad_ref_sq": []} for lo, hi in zip(edges, edges[1:])]
    diff_sq, ref_sq, a_sq, gdiff_sq, gref_sq, max_abs, n_equal = [], [], [], [], [], 0.0, 0
    max_rel_big, n_big = 0.0, 0
    rows = 8192
    for s in range(0, pre.shape[0], rows):
        p0 = pre[s:s + rows].double()
        da, db = pa[s:s + rows].double() - p0, pb[s:s + rows].double() - p0
        dd = da - db
        diff_sq.append(float((dd * dd).sum()))
        ref_sq.append(float((db * db).sum()))
        a_sq.append(float((da * da).sum()))
        n_equal += int((pa[s:s + rows] == pb[s:s + rows]).sum())
        max_abs = max(max_abs, float(dd.abs().max()))
        big = db.abs() >= 1e-6  # elements with an update of at least 1e-6 (lr is 3e-5)
        if big.any():
            max_rel_big = max(max_rel_big, float((dd.abs()[big] / db.abs()[big]).max()))
            n_big += int(big.sum())
        gA, gB = ga[s:s + rows].double(), gb[s:s + rows].double()
        gd = gA - gB
        gdiff_sq.append(float((gd * gd).sum()))
        gref_sq.append(float((gB * gB).sum()))
        v = beta2 * st["exp_avg_sq"][s:s + rows].double() + (1 - beta2) * gB * gB
        ratio = ((v / bc2).sqrt() + eps) / eps
        for b, (lo, hi) in zip(bk, zip(edges, edges[1:])):
            m = (ratio >= lo) & (ratio < hi)
            b["n"] += int(m.sum())
            b["diff_sq"].append(float((dd[m] ** 2).sum()))
            b["ref_sq"].append(float((db[m] ** 2).sum()))
            b["grad_diff_sq"].append(float((gd[m] ** 2).sum()))
            b["grad_ref_sq"].append(float((gB[m] ** 2).sum()))
    D, R2, A2 = math.fsum(diff_sq), math.fsum(ref_sq), math.fsum(a_sq)
    for b in bk:
        d, rr, gd2, gr2 = (math.fsum(b.pop(k)) for k in ("diff_sq", "ref_sq", "grad_diff_sq", "grad_ref_sq"))
        b.update(share_of_update_diff_sq=d / D if D else 0.0, share_of_reference_update_sq=rr / R2,
                 update_diff_rel_within=math.sqrt(d / rr) if rr else None,
                 grad_diff_rel_within=math.sqrt(gd2 / gr2) if gr2 else None)
    r["embed_elementwise"] = {
        "n_elements": pre.numel(), "post_bitwise_equal_fraction": n_equal / pre.numel(),
        "update_norm_f64_a": math.sqrt(A2), "update_norm_f64_b": math.sqrt(R2),
        "norm_rel_f64": abs(math.sqrt(A2) - math.sqrt(R2)) / math.sqrt(R2),
        "update_diff_norm_rel": math.sqrt(D / R2), "update_diff_max_abs": max_abs,
        "update_diff_max_rel_where_ref_update_ge_1e-6": max_rel_big, "n_ref_update_ge_1e-6": n_big,
        "grad_post_clip_diff_norm_rel": math.sqrt(math.fsum(gdiff_sq) / math.fsum(gref_sq)),
        "by_adam_denominator": bk, "adam_step_used_for_buckets": step,
    }
    tol = F.TOLERANCES["tensor_update_norm_rel"]
    nat, f64 = r["update_norms"]["native_fp32"]["embed_rel"], r["update_norms"]["f64"]["embed_rel"]
    r["verdict"] = {
        "tolerance": tol, "embed_rel_native_fp32_norms": nat, "embed_rel_f64_norms": f64,
        "native_exceeds_tolerance": nat > tol, "f64_exceeds_tolerance": f64 > tol,
        "norm_accumulation_explains_failure": nat > tol >= f64,
        "embed_rel_a_native_vs_b_f64": abs(r["embed_update_norm_a"]["native_fp32"] - r["embed_update_norm_b"]["f64"])
                                       / r["embed_update_norm_b"]["f64"],
        "embed_native_rel_to_f64_a": r["embed_norm_accumulation_rel_a"],
        "embed_native_rel_to_f64_b": r["embed_norm_accumulation_rel_b"],
        "embed_update_elementwise_diff_norm_rel": r["embed_elementwise"]["update_diff_norm_rel"],
        "max_rel_all_tensors_native": r["update_norms"]["native_fp32"]["max_rel"],
        "max_rel_all_tensors_f64": r["update_norms"]["f64"]["max_rel"],
        "note": "diagnostic only; the official gate (cuda_fixture.py, native fp32 norms) is unchanged",
    }
    return r


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="mode", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--checkpoint", required=True)
    pr.add_argument("--device", required=True, choices=["cpu", "cuda"])
    pr.add_argument("--fixture", default="runs/reference/cuda_pilot_fixture")
    pr.add_argument("--out", required=True)
    pc = sub.add_parser("compare")
    pc.add_argument("a")
    pc.add_argument("b_reference")
    pc.add_argument("--checkpoint", required=True)
    pc.add_argument("--out", required=True)
    a = p.parse_args()
    if a.mode == "run":
        res = run(a.checkpoint, resolve_device(a.device), Path(a.out), Path(a.fixture))
        print(json.dumps({k: res[k] for k in ("device", "loss", "grad_norm_pre_clip", "embed_update_norm",
                                              "max_native_vs_f64_update_norm_rel", "max_native_vs_f64_grad_norm_rel",
                                              "batch_identical_to_fixture",
                                              "names_equal_fixture", "moments_all_equal_file", "step_counters",
                                              "optimizer_settings_loaded", "native_vs_fixture", "seconds")},
                         indent=2, default=str))
        return 0
    rep = compare(Path(a.a), Path(a.b_reference), a.checkpoint)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str) + "\n")
    print(json.dumps({k: rep[k] for k in ("alignment_equal", "loss_rel", "update_norms", "verdict")}, indent=2))
    print(json.dumps({k: v for k, v in rep["embed_elementwise"].items() if k != "by_adam_denominator"}, indent=2))
    for b in rep["embed_elementwise"]["by_adam_denominator"]:
        print(b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
