"""Command-line entry point: ``dlmc <command>`` (or ``python -m diffusion_correction.cli``).

Commands: ``doctor``, ``download-backbone`` (milestone 1); ``generate-data``,
``verify-data``, ``inspect-example`` (milestone 2). Later milestones add warmup,
collect-drafts, and the rest.
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


def _task_tokenizer(model_dir: str | None):
    """Pinned tokenizer (small files sha256-checked) and facts recorded in manifests."""
    from .models.open_dcoder import DEFAULT_LOCAL_DIR, REPO_ID, REVISION, load_tokenizer, resolve_token_spec, \
        verify_files

    local = Path(model_dir or DEFAULT_LOCAL_DIR)
    hashes = verify_files(local, check_weights=False)
    spec = resolve_token_spec(local)
    facts = {"repo": REPO_ID, "revision": REVISION, "tokenizer_json_sha256": hashes["tokenizer.json"],
             "library": "tokenizers", "mask_token_id": spec.mask_token_id, "pad_token_id": spec.pad_token_id,
             "eos_token_id": spec.eos_token_id}
    return load_tokenizer(local), spec, facts


def cmd_generate_data(args: argparse.Namespace) -> int:
    import hashlib

    import yaml

    from .data.format import TokenizedFormat
    from .data.splits import DataConfig, generate_splits, verify_dataset, write_splits
    from .env import git_state

    raw = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None:
        raw["seed"] = args.seed
    cfg = DataConfig.from_dict(raw)
    tok, _, facts = _task_tokenizer(args.model_dir)
    out = Path(args.out or f"runs/data/{cfg.name}-seed{cfg.seed}")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite to replace it")
    t0 = time.perf_counter()
    splits = generate_splits(cfg, tok)
    tf = TokenizedFormat.from_tokenizer(tok, cfg.response_format())
    written = write_splits(out, cfg, splits, facts, tf)
    gen_s = time.perf_counter() - t0
    report = verify_dataset(out, tok)
    prov = {"command": "generate-data", "config_file": str(args.config),
            "config_file_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "seed": cfg.seed, "time_utc": datetime.now(timezone.utc).isoformat(), "git": git_state(),
            "manifest_sha256": written["manifest_sha256"], "generation_seconds": round(gen_s, 2),
            "verification": report}
    (out / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    summary = {"out_dir": str(out), "manifest_sha256": written["manifest_sha256"],
               "splits": {s: {"n": v["n_examples"], "sha256": v["sha256"]} for s, v in written["splits"].items()},
               "verification": report}
    print(json.dumps(summary, indent=2))
    return 0


def cmd_verify_data(args: argparse.Namespace) -> int:
    import hashlib

    from .data.splits import verify_dataset

    tok, _, _ = _task_tokenizer(args.model_dir)
    report = verify_dataset(Path(args.data_dir), tok)
    report["manifest_sha256"] = hashlib.sha256((Path(args.data_dir) / "manifest.json").read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))
    return 0


def cmd_inspect_example(args: argparse.Namespace) -> int:
    from .data.format import ResponseFormat, TokenizedFormat
    from .data.inspect import inspect_example
    from .data.splits import load_split

    tok, spec, _ = _task_tokenizer(args.model_dir)
    if args.fixture:
        from .data.fixtures import get_fixture
        from .data.tasks import build_example

        prob = get_fixture(args.fixture)
        tf = TokenizedFormat.from_tokenizer(tok, ResponseFormat(prob.spec.value_bound))
        ex = build_example(prob, "fixture", tok, tf, heldout=False)
    else:
        if not args.data_dir:
            raise SystemExit("give --fixture NAME or --data-dir DIR [--split S] [--index I | --problem-id ID]")
        recs = load_split(Path(args.data_dir), args.split)
        if args.problem_id:
            matches = [r for r in recs if r.problem_id == args.problem_id]
            if not matches:
                raise SystemExit(f"problem {args.problem_id} not in {args.split}")
            ex = matches[0]
        else:
            ex = recs[args.index]
        tf = TokenizedFormat.from_tokenizer(tok, ResponseFormat(ex.value_bound))

    draft = None
    if args.draft_values is not None:
        draft = tf.encode_values([int(v) for v in args.draft_values.split(",")])
    elif args.draft_text is not None:
        draft = tok.encode(args.draft_text, add_special_tokens=False).ids
    elif args.draft_ids is not None:
        draft = [int(v) for v in args.draft_ids.split(",")]
    table = inspect_example(ex, tok, tf, spec.pad_token_id, spec.mask_token_id, draft_ids=draft,
                            n_pad=args.pad, prompt_rows=args.prompt_rows)
    print(table.render())
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(table.to_dict(), indent=2))
        print(f"\nwrote {args.json}")
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

    g = sub.add_parser("generate-data", help="generate, write and verify task splits with a hashed manifest")
    g.add_argument("--config", default="configs/data/smoke.yaml")
    g.add_argument("--seed", type=int, default=None, help="override the config seed")
    g.add_argument("--out", default=None, help="default runs/data/<name>-seed<seed>")
    g.add_argument("--overwrite", action="store_true")
    g.add_argument("--model-dir", default=None)
    g.set_defaults(func=cmd_generate_data)

    v = sub.add_parser("verify-data", help="re-check a written dataset (hashes, disjointness, labels, tokens)")
    v.add_argument("--data-dir", required=True)
    v.add_argument("--model-dir", default=None)
    v.set_defaults(func=cmd_verify_data)

    ie = sub.add_parser("inspect-example", help="token/mask/target table for one example")
    ie.add_argument("--fixture", default=None, help="arith_2_plus_3, arith_17_28, arith_negative_zero, "
                                                    "program_basic, program_const_minus_var")
    ie.add_argument("--data-dir", default=None)
    ie.add_argument("--split", default="train")
    ie.add_argument("--index", type=int, default=0)
    ie.add_argument("--problem-id", default=None)
    ie.add_argument("--draft-values", default=None, help="HYPOTHETICAL draft as comma-separated integers")
    ie.add_argument("--draft-text", default=None, help="HYPOTHETICAL draft text, tokenized as is")
    ie.add_argument("--draft-ids", default=None, help="HYPOTHETICAL draft as comma-separated token IDs")
    ie.add_argument("--pad", type=int, default=2, help="pad positions appended to show the pad mask")
    ie.add_argument("--prompt-rows", type=int, default=4, help="trailing prompt rows to show (-1 = all)")
    ie.add_argument("--json", default=None, help="also write the table as JSON to this path")
    ie.add_argument("--model-dir", default=None)
    ie.set_defaults(func=cmd_inspect_example)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
