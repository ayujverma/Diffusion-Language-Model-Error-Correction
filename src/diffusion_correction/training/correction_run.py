"""Assemble a shared-bank correction run from a resolved config (``dlmc train-correction``)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..data.splits import verify_dataset
from ..evaluation.repair import make_allowed_for_adapter
from .correction_config import correction_config_hash
from .correction_trainer import CorrectionTrainer, prepare_bank_drafts
from .model_io import build_training_adapter
from .run import sampler_allowed, task_format
from .trainer import find_resume_checkpoint


def prepare_correction_trainer(cfg: dict, device: torch.device, run_dir: Path,
                               resume: str | None) -> CorrectionTrainer:
    run_dir = Path(run_dir)
    d = cfg["data"]
    data_identity = None
    tf = None
    if d.get("dir") and cfg["backbone"] == "open-dcoder":
        tok, tf = task_format(cfg["model_dir"], d["dir"])
        if d.get("verify"):
            t0 = time.perf_counter()
            report = verify_dataset(Path(d["dir"]), tok)
            print(f"verify-data passed ({report['n_examples']} records, {time.perf_counter() - t0:.1f}s)",
                  flush=True)
            data_identity = {"data_dir": d["dir"], "manifest_sha256": d["manifest_sha256"],
                             "verify": True, "n_examples": report["n_examples"]}
        else:
            data_identity = {"data_dir": d["dir"], "manifest_sha256": d["manifest_sha256"], "verify": False}

    train_drafts, bank_ident, repair_drafts, repair_ident = prepare_bank_drafts(cfg)
    print(json.dumps({"bank": bank_ident.to_dict(), "repair": repair_ident,
                      "n_train_drafts": len(train_drafts),
                      "n_repair_drafts": len(repair_drafts) if repair_drafts else 0}, indent=2), flush=True)

    adapter, info = build_training_adapter(cfg, device)
    if tf is not None:
        allowed = sampler_allowed(adapter, tf)
    else:
        allowed = make_allowed_for_adapter(adapter)
    trainer = CorrectionTrainer(cfg, adapter, info, train_drafts, bank_ident, device, run_dir,
                                repair_drafts_list=repair_drafts, repair_identity=repair_ident,
                                data_identity=data_identity, allowed=allowed)
    ckpt = None
    if resume == "auto":
        ckpt = find_resume_checkpoint(run_dir, cfg["checkpoint"]["durable_dir"], cfg["run_name"])
    elif resume:
        ckpt = Path(resume)
    if ckpt is not None:
        trainer.resume(ckpt)
        (run_dir / f"config.resume-{trainer.update:07d}.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"resumed from {ckpt} at update {trainer.update}", flush=True)
    else:
        if (run_dir / "checkpoints").exists() and any((run_dir / "checkpoints").iterdir()):
            raise SystemExit(f"{run_dir} already has checkpoints; pass --resume auto or a new --run-dir")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.resolved.json").write_text(json.dumps(cfg, indent=2) + "\n")
        (run_dir / "config.sha256").write_text(correction_config_hash(cfg) + "\n")
        (run_dir / "bank_identity.json").write_text(json.dumps(bank_ident.to_dict(), indent=2) + "\n")
        if repair_ident is not None:
            (run_dir / "repair_identity.json").write_text(json.dumps(repair_ident, indent=2) + "\n")
    return trainer
