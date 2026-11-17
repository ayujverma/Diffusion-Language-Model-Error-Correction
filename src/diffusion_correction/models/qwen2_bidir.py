# Derived from `modeling_qwen2.py` in https://huggingface.co/fredzzp/open-dcoder-0.5B
# (revision d0d86d5b99960c05258bb1f8265dd91564dbac67), itself derived from the Qwen2
# implementation by the Qwen team, Alibaba Group, The HuggingFace Inc. team and
# Bytedance Ltd. Licensed under the Apache License, Version 2.0.
#
# Changes: a self-contained re-implementation of the same forward computation with
# (1) bidirectional attention as the only mode, (2) an explicit key-padding mask,
# (3) no KV cache, labels, generation mixin, or transformers dependency.
"""Bidirectional Qwen2 backbone used by Open-dCoder and by the tiny test backbone."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf_dict(cls, cfg: dict) -> "Qwen2Config":
        if cfg.get("model_type") != "qwen2":
            raise ValueError(f"unsupported model_type {cfg.get('model_type')!r}")
        if cfg.get("rope_scaling") not in (None, {}):
            raise ValueError("rope_scaling is not implemented")
        if cfg.get("use_sliding_window", False):
            raise ValueError("sliding-window attention is not implemented")
        if cfg.get("hidden_act", "silu") != "silu":
            raise ValueError("only SiLU MLP is implemented")
        if cfg.get("attention_dropout", 0.0) != 0.0:
            raise ValueError("attention dropout is not implemented")
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg["rope_theta"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        h = x.to(torch.float32) if input_dtype in (torch.float16, torch.bfloat16) else x
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * h.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class Qwen2RotaryEmbedding(nn.Module):
    """Default (unscaled) RoPE, computed in float32 and cast to the activation dtype."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freq_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        inv = self.inv_freq.to(device=x.device, dtype=freq_dtype)
        freqs = position_ids.to(freq_dtype)[..., None] * inv  # [B, L, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class Qwen2Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos_, sin_ = cos[:, None], sin[:, None]
        q = q * cos_ + _rotate_half(q) * sin_
        k = k * cos_ + _rotate_half(k) * sin_
        rep = self.num_heads // self.num_kv_heads
        if rep > 1:
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # Bidirectional by construction: is_causal=False and the mask only removes pad keys.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        return self.o_proj(out.transpose(1, 2).reshape(B, L, self.num_heads * self.head_dim))


class Qwen2MLP(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.self_attn = Qwen2Attention(cfg)
        self.mlp = Qwen2MLP(cfg)
        self.input_layernorm = Qwen2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen2Model(nn.Module):
    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Qwen2DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = Qwen2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(cfg.head_dim, cfg.rope_theta)


def key_padding_attn_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """``[B, L]`` bool (True = real token) -> SDPA boolean mask ``[B, 1, L, L]``.

    Every query row may attend to every real key, in both directions. Pad queries
    also attend to real keys, so no row is fully masked (which would give NaNs).
    """
    if attention_mask.dtype != torch.bool:
        raise TypeError("attention_mask must be bool (True = real token)")
    if not bool(attention_mask.any(dim=-1).all()):
        raise ValueError("every sequence needs at least one real token")
    B, L = attention_mask.shape
    return attention_mask[:, None, None, :].expand(B, 1, L, L)


class BidirectionalQwen2(nn.Module):
    """Returns RAW logits (shift not applied); use ``alignment.align_logits`` next."""

    def __init__(self, cfg: Qwen2Config):
        super().__init__()
        self.config = cfg
        self.model = Qwen2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.model.embed_tokens(input_ids)
        position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        cos, sin = self.model.rotary_emb(x, position_ids)
        mask = key_padding_attn_mask(attention_mask)
        for layer in self.model.layers:
            x = layer(x, cos, sin, mask)
        logits = self.lm_head(self.model.norm(x))
        return logits.to(torch.promote_types(logits.dtype, torch.float32))
