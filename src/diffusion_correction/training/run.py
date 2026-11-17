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


def load_problem_id_subset(data_dir: str, manifest_sha256: str, ids_file: str, ids_sha256: str):
    """Fixed val subset from an explicit problem-ID list; the list's sha256 (over the IDs joined by
    newlines, the same convention as ``selected_problem_ids_sha256``) must match the config."""
    from ..data.splits import sha256_bytes

    ids = json.loads(Path(ids_file).read_text())["problem_ids"]
    got = sha256_bytes("\n".join(ids).encode())
    if got != ids_sha256:
        raise ValueError(f"{ids_file}: problem-ID sha256 {got} != configured {ids_sha256}")
    examples, vid = load_verified_examples(data_dir, "val", manifest_sha256)
    by_id = {e.problem_id: e for e in examples}
    missing = [i for i in ids if i not in by_id]
    if missing or len(set(ids)) != len(ids):
        raise ValueError(f"{ids_file}: {len(missing)} IDs not in val (e.g. {missing[:3]}) or duplicate IDs")
    ident = {**vid.to_dict(), "subset": {"problem_ids_file": str(ids_file), "problem_ids_sha256": ids_sha256},
             "selected_problem_ids_sha256": got, "n_selected": len(ids)}
    return [by_id[i] for i in ids], ident


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
    if ev["split"] == "val" and ev["problem_ids_file"]:
        eval_examples, eval_ident = load_problem_id_subset(d["dir"], d["manifest_sha256"], ev["problem_ids_file"],
                                                           ev["problem_ids_sha256"])
    elif ev["split"] == "val":
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
