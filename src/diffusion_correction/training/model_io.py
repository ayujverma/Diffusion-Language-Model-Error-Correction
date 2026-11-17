"""Build the trainable adapter from a config, and reload trained checkpoints for evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..env import DTYPES
from ..models.adapter import DiffusionAdapter
from .checkpoint import verify_checkpoint
from .lora import LoRAConfig, apply_lora, expected_lora_parameter_count, load_lora_state_dict

FULL_MODEL_FILE = "model.safetensors"
LORA_FILE = "adapter.safetensors"


def lora_config(cfg: dict) -> LoRAConfig:
    lc = cfg["lora"]
    return LoRAConfig(rank=int(lc["rank"]), alpha=float(lc["alpha"]), dropout=float(lc["dropout"]),
                      targets=tuple(lc["targets"]), init_seed=int(lc["init_seed"]))


def full_model_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """State dict without the tied ``lm_head.weight`` alias (as in the pinned checkpoint)."""
    sd = model.state_dict()
    tied = getattr(getattr(model, "config", None), "tie_word_embeddings", False)
    return {k: v for k, v in sd.items() if not (tied and k == "lm_head.weight")}


def load_full_model_tensors(model: torch.nn.Module, path: Path) -> None:
    from safetensors.torch import load_file

    state = load_file(str(path))
    own = full_model_tensors(model)
    if set(own) != set(state):
        raise ValueError(f"{path}: keys differ from the model (missing={sorted(set(own) - set(state))[:3]}, "
                         f"unexpected={sorted(set(state) - set(own))[:3]})")
    with torch.no_grad():
        for k, v in own.items():
            if v.shape != state[k].shape:
                raise ValueError(f"{path}: {k} shape {tuple(state[k].shape)} != {tuple(v.shape)}")
            v.copy_(state[k].to(device=v.device, dtype=v.dtype))
    if getattr(model.config, "tie_word_embeddings", False):
        assert model.lm_head.weight is model.model.embed_tokens.weight, "embedding tie broken"


def base_adapter(backbone: str, model_dir: str, device: torch.device, dtype_name: str, seed: int = 0
                 ) -> DiffusionAdapter:
    dtype = DTYPES[dtype_name]
    if backbone == "tiny":
        from ..models.tiny import build_tiny_adapter

        return build_tiny_adapter(seed=seed, dtype=dtype, device=device)
    from ..models.open_dcoder import load_open_dcoder

    return load_open_dcoder(Path(model_dir), device=device, dtype=dtype)


def build_training_adapter(cfg: dict, device: torch.device, adapter: DiffusionAdapter | None = None
                           ) -> tuple[DiffusionAdapter, dict]:
    """Pretrained (or ``init_checkpoint``) weights, then LoRA if configured. Returns (adapter, info)."""
    if adapter is None:
        adapter = base_adapter(cfg["backbone"], cfg["model_dir"], device, cfg["dtype"], cfg["seed"])
    info: dict = {"base": adapter.provenance, "init_checkpoint": None}
    if cfg["init_checkpoint"]:
        ck = Path(cfg["init_checkpoint"])
        complete = verify_checkpoint(ck)
        if FULL_MODEL_FILE not in complete["files"]:
            raise ValueError("init_checkpoint must be a full-model checkpoint")
        load_full_model_tensors(adapter.model, ck / FULL_MODEL_FILE)
        info["init_checkpoint"] = {"path": str(ck), "model_sha256": complete["files"][FULL_MODEL_FILE]}
    if cfg["method"] == "lora":
        lc = lora_config(cfg)
        wrapped = apply_lora(adapter.model, lc)
        n = adapter.trainable_parameter_count()
        expected = expected_lora_parameter_count(adapter.model.config, lc)
        if n != expected:
            raise AssertionError(f"LoRA trainable parameters {n} != closed form {expected}")
        info["lora"] = {**lc.to_dict(), "wrapped_modules": len(wrapped)}
    info["trainable_parameters"] = adapter.trainable_parameter_count()
    info["total_parameters"] = sum(p.numel() for p in adapter.model.parameters())
    return adapter, info


def load_trained_adapter(checkpoint: str | Path | None, device: torch.device,
                         model_dir: str = "runs/hf/open-dcoder-0.5B") -> tuple[DiffusionAdapter, dict]:
    """``None`` or ``"pretrained"`` gives the pinned pretrained model. Otherwise rebuild the
    checkpoint's model (base + full weights or LoRA tensors) after verifying its hashes."""
    if checkpoint in (None, "pretrained"):
        adapter = base_adapter("open-dcoder", model_dir, device, "float32")
        return adapter, {"checkpoint": "pretrained", "base": adapter.provenance}
    ck = Path(checkpoint)
    complete = verify_checkpoint(ck)
    state = json.loads((ck / "state.json").read_text())
    cfg = state["config"]
    # Full checkpoints hold every weight; LoRA checkpoints need their base (pinned or init_checkpoint).
    init = cfg["init_checkpoint"] if cfg["method"] == "lora" else None
    adapter, info = build_training_adapter({**cfg, "init_checkpoint": init, "model_dir": model_dir}, device)
    if state["base_model"].get("file_sha256") != adapter.provenance.get("file_sha256"):
        raise ValueError("checkpoint base model differs from the loaded pinned base")
    if cfg["method"] == "full":
        load_full_model_tensors(adapter.model, ck / FULL_MODEL_FILE)
    else:
        from safetensors.torch import load_file

        load_lora_state_dict(adapter.model, load_file(str(ck / LORA_FILE)))
    model_file = FULL_MODEL_FILE if cfg["method"] == "full" else LORA_FILE
    return adapter, {"checkpoint": str(ck), "update": state["update"], "method": cfg["method"],
                     "model_file_sha256": complete["files"][model_file], "data_identity": state["data_identity"],
                     "base": adapter.provenance, "run_name": cfg["run_name"]}
