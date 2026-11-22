"""Family B training configuration (milestone 7).

Four arms share the batch, seed, optimizer, init checkpoint, and update cap.
Only ``arm`` and ``run_name`` differ. There is no draft bank: drafts are built
online from the train split. Test splits are refused.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from ..losses.family_b import PROSECO_ARMS
from ..losses.weighting import DEFAULT_STAGE_EDGES

FAMILY_B_ARMS = (*PROSECO_ARMS, "scope")
REFUSED_SPLITS = ("test_id", "test_long", "test_unseen")

FAMILY_B_DEFAULTS: dict = {
    "run_name": "unnamed-family-b",
    "arm": "proseco_basic",
    "seed": 20261120,
    "backbone": "open-dcoder",
    "model_dir": "runs/hf/open-dcoder-0.5B",
    "init_checkpoint": None,
    "data": {
        "dir": None,
        "manifest_sha256": None,
        "split": "train",
        "verify": True,
    },
    "method": "full",
    "lora": {"rank": 16, "alpha": 32.0, "dropout": 0.0,
             "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
             "init_seed": 0},
    "dtype": "float32",
    "autocast": None,
    "batch_size": 16,
    "grad_accum": 1,
    "optimizer": {"name": "adamw", "lr": 3.0e-5, "betas": [0.9, 0.999], "eps": 1.0e-8,
                  "weight_decay": 0.01},
    "scheduler": {"name": "linear_warmup_constant", "warmup_updates": 20},
    "clip_grad_norm": 1.0,
    "mask": {"t_min": 1.0 / 500, "t_max": 1.0 - 1.0 / 500},
    "corrector": {
        "time_weight": "inv_t",
        "beta": 0.5,
        "stage_edges": list(DEFAULT_STAGE_EDGES),
    },
    "scope": {"mix_probability": 0.5, "sample_temperature": 1.5, "commit_rho": 0.3},
    "limits": {
        "max_updates": 1000,
        "max_wall_seconds": None,
        "max_train_seconds": None,
        "train_seconds_origin_update": None,
        "stop_at_unix": None,
    },
    "checkpoint": {
        "every_updates": 250, "keep_last": 2, "durable_dir": None,
        "protect_updates": [], "protect_every": 0,
    },
    "log_every": 10,
    "memory_every": 50,
    "release_cache_every": 1,
}

FAMILY_B_RESUME_MUTABLE = ("limits", "checkpoint", "log_every", "memory_every",
                           "release_cache_every", "run_name")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise ValueError(f"unknown family-b config key {k!r}")
        out[k] = _merge(out[k], v) if isinstance(out[k], dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def resolve_family_b_config(raw: dict) -> dict:
    cfg = _merge(FAMILY_B_DEFAULTS, raw)
    if cfg["arm"] not in FAMILY_B_ARMS:
        raise ValueError(f"arm must be one of {FAMILY_B_ARMS}, got {cfg['arm']!r}")
    if cfg["grad_accum"] != 1:
        raise ValueError("grad_accum != 1 is not supported (D-028)")
    if cfg["method"] != "full":
        raise ValueError("Family B is full fine-tuning (D-030)")
    if cfg["dtype"] not in ("float32", "float64"):
        raise ValueError("parameters must be float32 (or float64 for tiny tests)")
    if cfg["autocast"] not in (None, "bfloat16"):
        raise ValueError("autocast must be null or bfloat16")
    if cfg["batch_size"] < 1:
        raise ValueError("batch_size must be >= 1")
    split = cfg["data"]["split"]
    if split in REFUSED_SPLITS or split != "train":
        raise ValueError(
            f"train-family-b refuses split {split!r}. Training reads the train split only; "
            f"{REFUSED_SPLITS} are never loaded.")
    if cfg["corrector"]["time_weight"] != "inv_t":
        raise ValueError("corrector.time_weight must be inv_t (a(t) = 1/t, D-051)")
    beta = float(cfg["corrector"]["beta"])
    if not 0.0 <= beta < 1.0:
        raise ValueError("corrector.beta must be in [0, 1)")
    cfg["corrector"]["beta"] = beta
    cfg["corrector"]["stage_edges"] = [float(x) for x in cfg["corrector"]["stage_edges"]]
    mix = float(cfg["scope"]["mix_probability"])
    if not 0.0 <= mix <= 1.0:
        raise ValueError("scope.mix_probability must lie in [0, 1]")
    cfg["scope"]["mix_probability"] = mix
    if float(cfg["scope"]["sample_temperature"]) <= 0:
        raise ValueError("scope.sample_temperature must be positive")
    if not 0.0 < float(cfg["scope"]["commit_rho"]) <= 1.0:
        raise ValueError("scope.commit_rho must lie in (0, 1]")
    if cfg["backbone"] == "open-dcoder":
        if not cfg["data"]["manifest_sha256"]:
            raise ValueError("data.manifest_sha256 is required for real-backbone runs")
        if not cfg["data"]["dir"]:
            raise ValueError("data.dir is required for real-backbone runs")
        if not cfg["init_checkpoint"]:
            raise ValueError("init_checkpoint is required (start from ckpt-0008000, not a Family A arm)")
    lim = cfg["limits"]
    if lim["max_train_seconds"] is not None and lim["train_seconds_origin_update"] is None:
        raise ValueError("limits.max_train_seconds needs limits.train_seconds_origin_update")
    return cfg


def load_family_b_config(path: str | Path, overrides: dict | None = None) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return resolve_family_b_config(raw)


def family_b_compat_hash(cfg: dict) -> str:
    core = {k: v for k, v in cfg.items() if k not in FAMILY_B_RESUME_MUTABLE}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()


def family_b_config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
