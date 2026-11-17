"""Named, reproducible RNG streams (D-028) and global RNG capture for checkpoints.

Each stream is a CPU ``torch.Generator`` seeded from sha256(run seed, stream name),
so draws do not depend on the device and one stream never advances another.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

STREAMS = ("data_order", "train_mask", "eval")


def stream_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}|{name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def make_generator(seed: int, name: str) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(stream_seed(seed, name))
    return g


def capture_global_rng(device: torch.device) -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state()
    elif device.type == "mps":
        state["torch_mps"] = torch.mps.get_rng_state()
    state["device_type"] = device.type
    return state


def restore_global_rng(state: dict, device: torch.device) -> bool:
    """Restore global RNGs; returns False if the accelerator RNG could not be restored (other device)."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("device_type") != device.type:
        return False
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_cuda"])
    elif device.type == "mps":
        torch.mps.set_rng_state(state["torch_mps"])
    return True
