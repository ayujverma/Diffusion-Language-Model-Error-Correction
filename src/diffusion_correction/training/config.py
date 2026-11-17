"""Training configuration: defaults, validation, and the resume-compatibility hash.

Values marked ENGINEERING DEFAULT are reasonable starting choices for a 0.5B
fine-tune, not values mandated by any paper (D-030).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "run_name": "unnamed",
    "seed": 0,
    "backbone": "open-dcoder",          # open-dcoder | tiny (software tests only)
    "model_dir": "runs/hf/open-dcoder-0.5B",
    "init_checkpoint": None,            # None = pinned pretrained weights
    "data": {
        "dir": None,
        "split": "train",
        "manifest_sha256": None,        # required for open-dcoder runs
        "subset": None,                 # {"n": 32, "seed": 0}: stratified fixed subset
        "verify": True,                 # full verify_dataset before training
    },
    "method": "full",                   # full | lora
    "lora": {"rank": 16, "alpha": 32.0, "dropout": 0.0,
             "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
             "init_seed": 0},
    "dtype": "float32",                 # parameter dtype (master weights)
    "autocast": None,                   # None | bfloat16 (only after the precision check)
    "batch_size": 16,                   # physical minibatch = loss-normalisation unit
    "grad_accum": 1,                    # D-028: only 1 is supported
    "optimizer": {"name": "adamw", "lr": 3.0e-5, "betas": [0.9, 0.999], "eps": 1.0e-8,
                  "weight_decay": 0.01},
    "scheduler": {"name": "linear_warmup_constant", "warmup_updates": 20},
    "clip_grad_norm": 1.0,
    "mask": {"t_min": 1.0 / 500, "t_max": 1.0 - 1.0 / 500},
    "limits": {"max_updates": 100, "max_wall_seconds": None},
    "checkpoint": {"every_updates": 250, "keep_last": 2, "durable_dir": None},
    "eval": {
        "every_updates": 0,             # 0 = only at the end
        "at_start": False,              # also evaluate before the first update (pretrained baseline)
        "split": "val",                 # val | train_subset (the training examples themselves)
        "n_examples": 256,
        "subset_seed": 0,
        "batch_size": 32,
        "generate_budgets": [],         # e.g. ["L"]: monotone-sampler accuracy at these budgets
        "early_stop": None,             # {"budget": "L", "trace_accuracy": 1.0, "max_masked_loss": 0.05}
    },
    "log_every": 1,
    "memory_every": 50,
    "release_cache_every": 1,  # return cached accelerator memory every N updates; 0 = never
}

# Keys that may change between an interrupted run and its resumption.
RESUME_MUTABLE = ("limits", "checkpoint", "eval", "log_every", "memory_every", "release_cache_every", "run_name")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise ValueError(f"unknown config key {k!r}")
        out[k] = _merge(out[k], v) if isinstance(out[k], dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def resolve_config(raw: dict) -> dict:
    cfg = _merge(DEFAULTS, raw)
    if cfg["grad_accum"] != 1:
        raise ValueError("grad_accum != 1 is not supported: its normalisation and empty-micro-batch rule "
                         "are undefined (D-028)")
    if cfg["method"] not in ("full", "lora"):
        raise ValueError(f"method must be full or lora, got {cfg['method']!r}")
    if cfg["dtype"] not in ("float32", "float64"):
        raise ValueError("parameters must be float32 (or float64 for tiny tests); use autocast for bf16")
    if cfg["autocast"] not in (None, "bfloat16"):
        raise ValueError("autocast must be null or bfloat16")
    if cfg["eval"]["split"] not in ("val", "train_subset"):
        raise ValueError("eval.split must be val or train_subset (test splits are never used here)")
    if cfg["data"]["split"] not in ("train", "val"):
        raise ValueError("training data must come from the train split (val only for checks)")
    if cfg["backbone"] == "open-dcoder" and not cfg["data"]["manifest_sha256"]:
        raise ValueError("data.manifest_sha256 is required for real-backbone runs")
    if cfg["batch_size"] < 1:
        raise ValueError("batch_size must be >= 1")
    return cfg


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return resolve_config(raw)


def compat_hash(cfg: dict) -> str:
    """Hash of every setting that must not change on resume."""
    core = {k: v for k, v in cfg.items() if k not in RESUME_MUTABLE}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
