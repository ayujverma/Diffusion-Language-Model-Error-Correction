"""Assemble a real-backbone training run from a resolved config (used by ``dlmc train``)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..data.format import ResponseFormat, TokenizedFormat
from ..data.splits import verify_dataset
from ..sampling.monotone import allowed_token_mask
from .config import config_hash
from .data import load_verified_examples
from .model_io import build_training_adapter
from .trainer import Trainer, find_resume_checkpoint


def task_format(model_dir: str, data_dir: str):
    from ..models.open_dcoder import load_tokenizer, verify_files

    verify_files(Path(model_dir), check_weights=False)
    tok = load_tokenizer(model_dir)
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    tf = TokenizedFormat.from_tokenizer(tok, ResponseFormat(int(manifest["response_format"]["value_bound"])))
    return tok, tf


def sampler_allowed(adapter, tf: TokenizedFormat) -> torch.Tensor:
    return allowed_token_mask(adapter.tokens.vocab_size, tf.tokenizer_size, sorted(tf.special_ids),
                              extra_forbidden=(adapter.tokens.mask_token_id,))


def prepare_trainer(cfg: dict, device: torch.device, run_dir: Path, resume: str | None) -> Trainer:
    run_dir = Path(run_dir)
    d = cfg["data"]
    tok, tf = task_format(cfg["model_dir"], d["dir"])
    examples, ident = load_verified_examples(d["dir"], d["split"], d["manifest_sha256"], d["subset"])
    if d["verify"]:
        t0 = time.perf_counter()
        report = verify_dataset(Path(d["dir"]), tok)
        print(f"verify-data passed ({report['n_examples']} records, {time.perf_counter() - t0:.1f}s)", flush=True)
    ev = cfg["eval"]
    eval_examples, eval_ident = None, None
    if ev["split"] == "val":
        eval_examples, vid = load_verified_examples(d["dir"], "val", d["manifest_sha256"],
                                                    {"n": ev["n_examples"], "seed": ev["subset_seed"]})
        eval_ident = vid.to_dict()
    adapter, info = build_training_adapter(cfg, device)
    trainer = Trainer(cfg, adapter, info, examples, ident, device, run_dir, eval_examples=eval_examples,
                      eval_identity=eval_ident, tf=tf, allowed=sampler_allowed(adapter, tf))
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
        (run_dir / "config.sha256").write_text(config_hash(cfg) + "\n")
    return trainer
