"""Minimal LoRA for the bidirectional Qwen2 backbone (feasibility comparison).

``y = W x + b + (alpha / r) * B A x`` with A: [r, in], B: [out, r]. A uses the
Kaiming-uniform initialisation of ``nn.Linear`` (a = sqrt(5)) from a seeded generator;
B starts at zero, so the wrapped model initially equals the base model exactly.
The base weights (and embeddings, norms, the tied ``lm_head``) are frozen.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    targets: tuple[str, ...] = DEFAULT_TARGETS
    init_seed: int = 0

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def to_dict(self) -> dict:
        d = asdict(self)
        d["targets"] = list(self.targets)
        d["scaling"] = self.scaling
        return d


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, cfg: LoRAConfig, generator: torch.Generator):
        super().__init__()
        self.base = base
        self.rank, self.scaling = cfg.rank, cfg.scaling
        a = torch.empty(cfg.rank, base.in_features, dtype=torch.float32)
        bound = 1.0 / math.sqrt(base.in_features)  # kaiming_uniform_(a=sqrt(5)) bound for fan_in
        a.uniform_(-bound, bound, generator=generator)
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(a.to(device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, cfg.rank, device=dev, dtype=dt))
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> list[str]:
    """Freeze every parameter, then wrap the target linears. Returns wrapped module names."""
    for p in model.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(cfg.init_seed)
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in cfg.targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, cfg, gen))
                wrapped.append(f"{name}.{child_name}" if name else child_name)
    if not wrapped:
        raise ValueError(f"no target modules {cfg.targets} found")
    return wrapped


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach() for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k}


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    own = lora_state_dict(model)
    if set(own) != set(state):
        missing, extra = sorted(set(own) - set(state))[:3], sorted(set(state) - set(own))[:3]
        raise ValueError(f"LoRA state mismatch: missing={missing} unexpected={extra}")
    model.load_state_dict(state, strict=False)


def expected_lora_parameter_count(model_cfg, cfg: LoRAConfig) -> int:
    """Closed-form count for the Qwen2 shapes, used to cross-check the wrapped model."""
    h, i, d = model_cfg.hidden_size, model_cfg.intermediate_size, model_cfg.head_dim
    kv = model_cfg.num_key_value_heads * d
    q = model_cfg.num_attention_heads * d
    shapes = {"q_proj": (h, q), "k_proj": (h, kv), "v_proj": (h, kv), "o_proj": (q, h),
              "gate_proj": (h, i), "up_proj": (h, i), "down_proj": (i, h)}
    per_layer = sum(cfg.rank * (fi + fo) for name, (fi, fo) in shapes.items() if name in cfg.targets)
    return per_layer * model_cfg.num_hidden_layers
