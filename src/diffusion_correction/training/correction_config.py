"""Correction-training configuration (milestone 5 shared-bank screen).

Arm YAMLs share every non-weight setting. Only ``weighting.rule`` (and ``run_name``)
differ across the four-way screen plus low/shuffled controls (D-044).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from ..losses.weighting import DEFAULT_STAGE_EDGES, WeightRule

CORRECTION_DEFAULTS: dict = {
    "run_name": "unnamed-correction",
    "seed": 0,
    "backbone": "open-dcoder",
    "model_dir": "runs/hf/open-dcoder-0.5B",
    "init_checkpoint": None,            # required for real arms: ckpt-0008000
    "data": {
        "dir": None,
        "manifest_sha256": None,
        "verify": True,                 # full verify_dataset once at start (data files, not bank)
    },
    "bank": {
        "dir": None,
        "identity_sha256": None,
        "manifest_sha256": None,
        "split": "train",
        "budgets": ["1", "2", "4", "8", "L"],
        "verify_shards": True,
    },
    "weighting": {
        "rule": "uniform",
        "beta": 0.5,
        "stage_edges": list(DEFAULT_STAGE_EDGES),
        "shuffle_seed": 20261120,       # used only by shuffled_conf; set on every arm for shared YAML surface
    },
    "objective": {"lambda": 1.0, "mu": 1.0},
    "method": "full",
    "lora": {"rank": 16, "alpha": 32.0, "dropout": 0.0,
             "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
             "init_seed": 0},
    "dtype": "float32",
    "autocast": None,
    "batch_size": 32,
    "grad_accum": 1,
    "optimizer": {"name": "adamw", "lr": 3.0e-5, "betas": [0.9, 0.999], "eps": 1.0e-8,
                  "weight_decay": 0.01},
    "scheduler": {"name": "linear_warmup_constant", "warmup_updates": 20},
    "clip_grad_norm": 1.0,
    "mask": {"t_min": 1.0 / 500, "t_max": 1.0 - 1.0 / 500},
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
    "eval": {
        "every_updates": 250,
        "at_start": True,
        "batch_size": 32,
        "repair": {
            "split": "val",
            "budget": "4",              # primary comparison budget (D-044)
            "n_examples": None,         # None = all matching drafts
            "subset_seed": 0,
        },
    },
    "log_every": 10,
    "memory_every": 50,
    "release_cache_every": 1,
}

# Keys that may change between an interrupted run and its resumption.
CORRECTION_RESUME_MUTABLE = ("limits", "checkpoint", "eval", "log_every", "memory_every",
                             "release_cache_every", "run_name")


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in out:
            raise ValueError(f"unknown correction config key {k!r}")
        out[k] = _merge(out[k], v) if isinstance(out[k], dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def resolve_correction_config(raw: dict) -> dict:
    cfg = _merge(CORRECTION_DEFAULTS, raw)
    if cfg["grad_accum"] != 1:
        raise ValueError("grad_accum != 1 is not supported (D-028)")
    if cfg["method"] not in ("full", "lora"):
        raise ValueError(f"method must be full or lora, got {cfg['method']!r}")
    if cfg["dtype"] not in ("float32", "float64"):
        raise ValueError("parameters must be float32 (or float64 for tiny tests)")
    if cfg["autocast"] not in (None, "bfloat16"):
        raise ValueError("autocast must be null or bfloat16")
    if cfg["batch_size"] < 1:
        raise ValueError("batch_size must be >= 1")
    bank = cfg["bank"]
    if bank["split"] not in ("train", "val"):
        raise ValueError("bank.split must be train or val")
    if not bank["budgets"]:
        raise ValueError("bank.budgets must be non-empty")
    bank["budgets"] = [str(b) for b in bank["budgets"]]
    w = cfg["weighting"]
    WeightRule(w["rule"])  # validate
    if not 0.0 <= float(w["beta"]) < 1.0:
        raise ValueError("weighting.beta must be in [0, 1)")
    w["stage_edges"] = [float(x) for x in w["stage_edges"]]
    obj = cfg["objective"]
    if float(obj["lambda"]) < 0 or float(obj["mu"]) < 0:
        raise ValueError("objective.lambda and mu must be >= 0")
    if cfg["backbone"] == "open-dcoder":
        if not cfg["data"]["manifest_sha256"]:
            raise ValueError("data.manifest_sha256 is required for real-backbone runs")
        if not cfg["init_checkpoint"]:
            raise ValueError("init_checkpoint is required for correction arms (start from ckpt-0008000)")
        for key in ("dir", "identity_sha256", "manifest_sha256"):
            if not bank[key]:
                raise ValueError(f"bank.{key} is required for real-backbone runs")
    repair = cfg["eval"]["repair"]
    if repair["split"] not in ("train", "val"):
        raise ValueError("eval.repair.split must be train or val")
    repair["budget"] = str(repair["budget"])
    lim = cfg["limits"]
    if lim["max_train_seconds"] is not None and lim["train_seconds_origin_update"] is None:
        raise ValueError("limits.max_train_seconds needs limits.train_seconds_origin_update")
    return cfg


def load_correction_config(path: str | Path, overrides: dict | None = None) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    for dotted, value in (overrides or {}).items():
        node = raw
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return resolve_correction_config(raw)


def correction_compat_hash(cfg: dict) -> str:
    core = {k: v for k, v in cfg.items() if k not in CORRECTION_RESUME_MUTABLE}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()


def correction_config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
