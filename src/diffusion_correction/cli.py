"""Command-line entry point: ``dlmc <command>`` (or ``python -m diffusion_correction.cli``).

Milestone 1 commands: ``doctor`` and ``download-backbone``. Later milestones add
inspect-example, generate-data, warmup, collect-drafts, and the rest.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .env import DTYPES, environment_report, memory_snapshot, resolve_device, synchronize


def _training_gate(adapter, device: torch.device, prompt_ids: list[int], target_ids: list[int],
                   wrong_index: int, wrong_id: int, optimizer: str, lr: float) -> dict:
    """One real forward -> correction loss -> backward -> optimizer step, with checks."""
    from .losses import correction_losses, masked_diffusion_loss, token_error_masks
    from .models.alignment import build_layout

    tokens = adapter.tokens
    draft = list(target_ids)
    draft[wrong_index] = wrong_id
    lay = build_layout([prompt_ids], [draft], tokens.pad_token_id).to(device)
    targets = build_layout([prompt_ids], [target_ids], tokens.pad_token_id).input_ids.to(device)
    E, K = token_error_masks(lay.input_ids, targets, lay.editable_mask)

    masked_in = lay.input_ids.clone()
    mpos = lay.editable_mask.clone()
    mpos[:, lay.editable_mask[0].nonzero()[::2, 0]] = False  # mask every other response token
    masked_in[mpos] = tokens.mask_token_id

    model = adapter.model
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = (torch.optim.AdamW(params, lr=lr, weight_decay=0.0) if optimizer == "adamw"
           else torch.optim.SGD(params, lr=lr))
    watch = model.model.layers[0].self_attn.q_proj.weight
    before = watch.detach().float().clone()

    synchronize(device)
    t0 = time.perf_counter()
    lm = masked_diffusion_loss(adapter.aligned_logits(lay.with_input_ids(masked_in)), targets, mpos,
                               lay.editable_mask)
    corr = correction_losses(adapter.aligned_logits(lay), targets, E, K)
    total = lm.loss + corr.l_error + corr.l_keep
    total.backward()
    grad_norm = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None)))
    finite = all(bool(torch.isfinite(p.grad).all()) for p in params if p.grad is not None)
    opt.step()
    synchronize(device)
    step_s = time.perf_counter() - t0
    delta = float((watch.detach().float() - before).abs().max())
    opt.zero_grad(set_to_none=True)
    model.eval()
    return {
        "sequence_length": int(lay.input_ids.shape[1]), "n_masked": lm.n_masked, "n_error": corr.n_error,
        "n_keep": corr.n_keep, "L_mask": float(lm.loss.detach()), "L_error": float(corr.l_error.detach()),
        "L_keep": float(corr.l_keep.detach()), "grad_norm": grad_norm, "grads_finite": finite,
        "optimizer": optimizer, "lr": lr, "watched_param_max_abs_update": delta,
        "two_forwards_backward_step_seconds": round(step_s, 3),
        "passed": finite and delta > 0 and bool(torch.isfinite(total.detach())),
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    dtype = DTYPES[args.dtype]
    torch.manual_seed(args.seed)
    report = {"command": "doctor", "time_utc": datetime.now(timezone.utc).isoformat(),
              "environment": environment_report(device), "backbone": args.backbone, "dtype": args.dtype,
              "attention": "torch.nn.functional.scaled_dot_product_attention, bidirectional, key-padding mask",
              "kv_cache": False}

    t0 = time.perf_counter()
    if args.backbone == "tiny":
        from .models.tiny import build_tiny_adapter

        adapter = build_tiny_adapter(seed=args.seed, dtype=dtype, device=device)
        prompt, target, wrong_idx, wrong_id = [1, 2, 3], [4, 5, 6, 7], 1, 9
    else:
        from .models.open_dcoder import DEFAULT_LOCAL_DIR, load_open_dcoder, load_tokenizer

        local = Path(args.model_dir or DEFAULT_LOCAL_DIR)
        adapter = load_open_dcoder(local, device=device, dtype=dtype, verify_weights_hash=not args.skip_hash)
        tok = load_tokenizer(local)
        prompt = tok.encode("# Compute a=17+28; b=a*3; c=b-19. Return a,b,c.\nanswer = ").ids
        target = tok.encode("045,135,116").ids
        wrong_idx, wrong_id = 2, tok.encode("6").ids[0]
    synchronize(device)
    report["load_seconds"] = round(time.perf_counter() - t0, 2)
    report["provenance"] = adapter.provenance
    report["tokens"] = adapter.tokens.__dict__
    report["trainable_parameters"] = adapter.trainable_parameter_count()

    # Inference forward (no labels anywhere), fully masked response.
    from .models.alignment import build_layout

    lay = build_layout([prompt], [[adapter.tokens.mask_token_id] * len(target)], adapter.tokens.pad_token_id).to(device)
    with torch.no_grad():
        synchronize(device)
        t1 = time.perf_counter()
        logits = adapter.aligned_logits(lay)
        synchronize(device)
    report["inference_forward"] = {"seconds": round(time.perf_counter() - t1, 3), "shape": list(logits.shape),
                                   "logits_dtype": str(logits.dtype), "finite": bool(torch.isfinite(logits).all())}
    if not args.skip_train_step:
        report["training_gate"] = _training_gate(adapter, device, prompt, target, wrong_idx, wrong_id,
                                                 args.optimizer, args.lr)
    report["memory"] = memory_snapshot(device)
    ok = report["inference_forward"]["finite"] and report.get("training_gate", {"passed": True})["passed"]
    report["status"] = "PASS" if ok else "FAIL"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"doctor-{args.backbone}-{device.type}-{args.dtype}-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {path}")
    return 0 if ok else 1


def cmd_download(args: argparse.Namespace) -> int:
    from .models.open_dcoder import DEFAULT_LOCAL_DIR, download, verify_files

    local = Path(args.model_dir or DEFAULT_LOCAL_DIR)
    download(local, include_weights=not args.no_weights)
    print(json.dumps(verify_files(local, check_weights=not args.no_weights), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dlmc")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="environment report + forward/loss/backward/step gate")
    d.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    d.add_argument("--backbone", default="tiny", choices=["tiny", "open-dcoder"])
    d.add_argument("--dtype", default="float32", choices=sorted(DTYPES))
    d.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    d.add_argument("--lr", type=float, default=1e-5)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--model-dir", default=None)
    d.add_argument("--skip-hash", action="store_true", help="skip the 1.26 GB weights sha256 check")
    d.add_argument("--skip-train-step", action="store_true")
    d.add_argument("--out-dir", default="runs/doctor")
    d.set_defaults(func=cmd_doctor)

    dl = sub.add_parser("download-backbone", help="fetch pinned Open-dCoder files and verify sha256")
    dl.add_argument("--model-dir", default=None)
    dl.add_argument("--no-weights", action="store_true")
    dl.set_defaults(func=cmd_download)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
