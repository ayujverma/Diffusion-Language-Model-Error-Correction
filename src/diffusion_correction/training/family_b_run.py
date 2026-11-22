"""Assemble a Family B run (``dlmc train-family-b``).

Online drafts from the train split. No bank. Test splits are not loaded.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..data.splits import verify_dataset
from ..models.adapter import DiffusionAdapter
from .data import load_verified_examples
from .family_b_config import family_b_config_hash
from .family_b_trainer import FamilyBTrainer
from .model_io import build_training_adapter
from .run import sampler_allowed, task_format
from .trainer import find_resume_checkpoint


def allowed_for(adapter: DiffusionAdapter, tf) -> torch.Tensor:
    if tf is not None:
        return sampler_allowed(adapter, tf)
    spec = adapter.tokens
    allowed = torch.ones(spec.vocab_size, dtype=torch.bool)
    for i in set(spec.forbidden_output_ids) | {spec.mask_token_id, spec.pad_token_id, spec.eos_token_id}:
        if 0 <= int(i) < spec.vocab_size:
            allowed[int(i)] = False
    if not bool(allowed.any()):
        raise ValueError("allowed-token mask is empty")
    return allowed


def prepare_family_b_trainer(cfg: dict, device: torch.device, run_dir: Path, resume: str | None,
                             examples=None) -> FamilyBTrainer:
    run_dir = Path(run_dir)
    d = cfg["data"]
    data_identity = None
    tf = None
    if cfg["backbone"] == "open-dcoder":
        tok, tf = task_format(cfg["model_dir"], d["dir"])
        examples, ident = load_verified_examples(d["dir"], "train", d["manifest_sha256"],
                                                 allowed_splits=("train",))
        if d.get("verify"):
            t0 = time.perf_counter()
            report = verify_dataset(Path(d["dir"]), tok)
            print(f"verify-data passed ({report['n_examples']} records, {time.perf_counter() - t0:.1f}s)",
                  flush=True)
        data_identity = ident.to_dict()
    elif examples is None:
        raise ValueError("tiny Family B runs need examples passed in")
    else:
        data_identity = {"split": "train", "n_selected": len(examples), "source": "caller",
                         "backbone": "tiny"}

    adapter, info = build_training_adapter(cfg, device)
    trainer = FamilyBTrainer(cfg, adapter, info, examples, data_identity, device, run_dir,
                             allowed=allowed_for(adapter, tf))
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
        (run_dir / "config.sha256").write_text(family_b_config_hash(cfg) + "\n")
        (run_dir / "data_identity.json").write_text(json.dumps(data_identity, indent=2) + "\n")
    return trainer
