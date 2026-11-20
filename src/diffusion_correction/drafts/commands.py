"""CLI glue for the milestone-4 bank commands (registered in ``diffusion_correction.cli``)."""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

from .bank import (bank_identity, build_plan, collect_bank, load_bank_config, load_bank_examples, validate_bank,
                   write_json_atomic)
from .identity import generator_identity, sha256_json


def _memory(device):
    import torch

    from ..env import memory_snapshot

    out = memory_snapshot(device)
    if device.type == "cuda":
        st = torch.cuda.memory_stats(device)
        out.update(num_alloc_retries=st.get("num_alloc_retries", 0), num_ooms=st.get("num_ooms", 0))
    return out


class _Stop:
    def __init__(self):
        self.reason = None
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, self._handle)

    def _handle(self, signum, _frame):
        self.reason = signal.Signals(signum).name
        print(f"received {self.reason}; stopping after the current shard", flush=True)

    def __call__(self):
        return self.reason


def _context(args, need_model: bool = True):
    """Config, verified train/val examples, plan, generator identity, environment, model."""
    from ..env import environment_report, resolve_device
    from ..training.run import sampler_allowed, task_format

    cfg = load_bank_config(args.config)
    data_dir = args.data_dir or cfg["data"]["dir"]
    _, tf = task_format(args.model_dir, data_dir)
    examples, idents = load_bank_examples(cfg, data_dir)
    plan = build_plan(cfg, examples, idents, tf.fmt.slot_len)
    ctx = {"cfg": cfg, "tf": tf, "examples": examples, "plan": plan}
    if not need_model:
        return ctx
    gcfg = cfg["generator"]
    print(f"verifying generator checkpoint {args.checkpoint}", flush=True)
    gen = generator_identity(args.checkpoint, gcfg["model_sha256"], int(gcfg["update"]))
    device = resolve_device(args.device)
    env = environment_report(device)
    if device.type == "cuda" and (env["git"].get("dirty") or not env["git"].get("commit")):
        raise SystemExit(f"refusing CUDA collection from a dirty or unknown source tree: {env['git']}")
    from ..training.model_io import load_trained_adapter

    adapter, info = load_trained_adapter(args.checkpoint, device, args.model_dir)
    if info["model_file_sha256"] != gcfg["model_sha256"]:
        raise SystemExit("loaded model sha256 differs from the pinned generator")
    if gen.get("tokenizer_json_sha256") is not None and \
            gen["tokenizer_json_sha256"] != adapter.provenance.get("file_sha256", {}).get("tokenizer.json"):
        raise SystemExit("tokenizer differs from the generator's recorded tokenizer")
    identity = bank_identity(cfg, plan, gen, env, os.environ.get("DLMC_LOCK_SHA256"))
    ctx.update(device=device, env=env, gen=gen, adapter=adapter, allowed=sampler_allowed(adapter, tf),
               identity=identity)
    return ctx


def cmd_bank_preflight(args: argparse.Namespace) -> int:
    from .preflight import run_preflight

    c = _context(args)
    print(json.dumps({"identity_sha256": c["identity"]["identity_sha256"], "plan_totals": c["plan"]["totals"]},
                     indent=2), flush=True)
    rep = run_preflight(c["adapter"], c["tf"], c["allowed"], c["examples"], c["cfg"], Path(args.out),
                        c["identity"]["identity_sha256"], args.reference_records)
    write_json_atomic(Path(args.out) / "identity.json", c["identity"])
    print(f"preflight {'PASSED' if rep['passed'] else 'FAILED'} in {rep['preflight_seconds']}s; wrote {args.out}")
    return 0 if rep["passed"] else 1


def cmd_collect_drafts(args: argparse.Namespace) -> int:
    from .bank import open_bank

    c = _context(args)
    ident = c["identity"]
    if args.preflight_out:
        from .preflight import run_preflight

        rep = run_preflight(c["adapter"], c["tf"], c["allowed"], c["examples"], c["cfg"], Path(args.preflight_out),
                            ident["identity_sha256"], args.reference_records)
        write_json_atomic(Path(args.preflight_out) / "identity.json", ident)
        if not rep["passed"]:
            print(f"PREFLIGHT FAILED; collection not started. See {args.preflight_out}/preflight.json", flush=True)
            return 2
        print(f"PREFLIGHT PASSED in {rep['preflight_seconds']}s", flush=True)
    if args.require_preflight:
        marker = Path(args.require_preflight) / "PREFLIGHT_PASSED"
        if not marker.exists() or marker.read_text().split()[-1] != ident["identity_sha256"]:
            raise SystemExit(f"{marker}: missing or for a different identity; collection does not start")
    root = Path(args.bank_dir)
    open_bank(root, ident, c["plan"], args.resume)
    print(json.dumps({"bank": str(root), "identity_sha256": ident["identity_sha256"],
                      "plan_totals": c["plan"]["totals"]}, indent=2), flush=True)
    status = collect_bank(root, c["plan"], ident, c["adapter"], c["tf"], c["allowed"], c["examples"],
                          stop_at_unix=args.stop_at_unix, max_shards=args.max_shards, should_stop=_Stop(),
                          memory=lambda: _memory(c["device"]))
    print(json.dumps(status, indent=2, default=str))
    return 0 if status["state"] == "collection_complete" else 3


def _stored_identity(root: Path, cfg: dict, plan: dict) -> dict:
    ident = json.loads((root / "IDENTITY.json").read_text())
    if sha256_json(ident["pinned"]) != ident["identity_sha256"]:
        raise SystemExit("IDENTITY.json pinned fields do not match its identity hash")
    if ident["pinned"]["config_sha256"] != sha256_json(cfg) or ident["pinned"]["plan_sha256"] != sha256_json(plan):
        raise SystemExit("bank config or plan recomputed here differs from the bank's IDENTITY.json")
    return ident


def cmd_validate_bank(args: argparse.Namespace) -> int:
    c = _context(args, need_model=False)
    root = Path(args.bank_dir)
    ident = _stored_identity(root, c["cfg"], c["plan"])
    rep = validate_bank(root, c["plan"], ident, c["examples"], c["tf"], finalize=args.finalize,
                        reference_records=args.reference_records)
    print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, indent=2, default=str))
    for p in rep["problems"][:20]:
        print("PROBLEM:", p)
    return 0 if rep["passed"] else 1


def cmd_bank_step0(args: argparse.Namespace) -> int:
    from ..env import environment_report, resolve_device
    from ..training.model_io import load_trained_adapter
    from .step0 import run_step0

    root = Path(args.bank_dir)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    pin = manifest["identity"]["pinned"]
    gen = generator_identity(args.checkpoint, pin["generator_model_sha256"], pin["generator_update"])
    device = resolve_device(args.device)
    adapter, _ = load_trained_adapter(args.checkpoint, device, args.model_dir)
    summary = run_step0(root, adapter, gen, environment_report(device), batch_size=args.batch_size,
                        resume=args.resume, stop_at_unix=args.stop_at_unix, should_stop=_Stop())
    print(json.dumps(summary, indent=2))
    return 0 if summary["state"] == "complete" else 3


def cmd_bank_diagnostics(args: argparse.Namespace) -> int:
    from .diagnostics import write_diagnostics

    d = write_diagnostics(Path(args.bank_dir), Path(args.out) if args.out else None)
    print(json.dumps({"n_drafts": d["n_drafts"], "n_tokens": d["n_tokens"], "has_step0": d["has_step0"]}))
    return 0


def cmd_bank_report(args: argparse.Namespace) -> int:
    from ..models.open_dcoder import load_tokenizer
    from .report import build_report

    c = _context(args, need_model=False)
    tok = load_tokenizer(args.model_dir)
    texts = {(e.split, e.problem_id): {"prompt_text": e.prompt_text, "response_text": e.response_text}
             for v in c["examples"].values() for e in v}
    rep = build_report(Path(args.bank_dir), lambda i: tok.decode([i]), texts,
                       Path(args.out) if args.out else None)
    print(json.dumps({"examples": len(rep["examples"]), "verification": rep["verification"]}, indent=2))
    return 0 if rep["verification"]["passed"] else 1


def register(sub) -> None:
    devices = ["cpu", "mps", "cuda", "auto"]

    def common(p, model=True):
        p.add_argument("--config", required=True, help="bank config YAML (configs/bank/...)")
        p.add_argument("--data-dir", default=None, help="override the config's data dir (same pinned hashes)")
        p.add_argument("--model-dir", default="runs/hf/open-dcoder-0.5B")
        if model:
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--device", default="auto", choices=devices)

    p = sub.add_parser("bank-preflight", help="milestone-4 startup gates (Gate D, recompute, reproduction)")
    common(p)
    p.add_argument("--reference-records", default=None, help="records.jsonl of the generator's validation eval")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_bank_preflight)

    p = sub.add_parser("collect-drafts", help="collect the immutable train/val draft bank (resumable shards)")
    common(p)
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--require-preflight", default=None, help="preflight dir whose PREFLIGHT_PASSED must match")
    p.add_argument("--preflight-out", default=None, help="run the startup gates first, in this process; stop on failure")
    p.add_argument("--reference-records", default=None, help="validation records of the generator (preflight gate)")
    p.add_argument("--stop-at-unix", type=float, default=None)
    p.add_argument("--max-shards", type=int, default=None)
    p.set_defaults(func=cmd_collect_drafts)

    p = sub.add_parser("validate-bank", help="re-check every shard/record; --finalize writes MANIFEST + BANK_COMPLETE")
    common(p, model=False)
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--reference-records", default=None)
    p.set_defaults(func=cmd_validate_bank)

    p = sub.add_parser("bank-step0", help="step-0 correction CE on a finalized bank (diagnostic, resumable)")
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-dir", default="runs/hf/open-dcoder-0.5B")
    p.add_argument("--device", default="auto", choices=devices)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stop-at-unix", type=float, default=None)
    p.set_defaults(func=cmd_bank_step0)

    p = sub.add_parser("bank-diagnostics", help="signal diagnostics of a finalized bank (no model)")
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_bank_diagnostics)

    p = sub.add_parser("bank-report", help="verified examples report (Markdown + HTML) from a finalized bank")
    common(p, model=False)
    p.add_argument("--bank-dir", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_bank_report)
