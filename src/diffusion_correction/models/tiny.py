"""Randomly initialised tiny backbone for software and gradient checks only.

It runs the same :class:`BidirectionalQwen2` code path and the same shift
convention as Open-dCoder. Results on it are never evidence about the pretrained
model.
"""

from __future__ import annotations

import torch

from .adapter import DiffusionAdapter, TokenSpec
from .qwen2_bidir import BidirectionalQwen2, Qwen2Config

# Toy vocabulary: 0-9 digits, then punctuation/specials. IDs are local to this backbone.
TINY_VOCAB = [str(d) for d in range(10)] + [",", "=", "+", "-", "*", ";", " ", "<pad>", "<mask>", "<eos>"]
TINY_TOKENS = TokenSpec(
    mask_token_id=TINY_VOCAB.index("<mask>"),
    pad_token_id=TINY_VOCAB.index("<pad>"),
    eos_token_id=TINY_VOCAB.index("<eos>"),
    vocab_size=len(TINY_VOCAB),
    forbidden_output_ids=(TINY_VOCAB.index("<mask>"), TINY_VOCAB.index("<pad>")),
)


def tiny_config(**overrides) -> Qwen2Config:
    base = dict(vocab_size=len(TINY_VOCAB), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, rms_norm_eps=1e-6, rope_theta=10_000.0,
                tie_word_embeddings=True)
    base.update(overrides)
    return Qwen2Config(**base)


def build_tiny_adapter(seed: int = 0, dtype: torch.dtype = torch.float64,
                       device: str | torch.device = "cpu", **cfg_overrides) -> DiffusionAdapter:
    gen = torch.Generator().manual_seed(seed)
    model = BidirectionalQwen2(tiny_config(**cfg_overrides))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
                continue
            p.copy_(torch.randn(p.shape, generator=gen) * 0.2)
    model = model.to(device=device, dtype=dtype)
    return DiffusionAdapter(model=model, tokens=TINY_TOKENS,
                            provenance={"backbone": "tiny-random", "seed": seed, "dtype": str(dtype)})
