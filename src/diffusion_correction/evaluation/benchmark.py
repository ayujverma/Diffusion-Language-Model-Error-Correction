"""Training-step timing and memory pilot (D-022 early runtime estimate).

Each configuration loads a FRESH pretrained model, runs ``warmup_steps`` counted
optimizer updates (these allocate the AdamW state), then times ``timed_steps``
updates with the accelerator synchronized before and after each one.

Memory definitions:
- ``mps_current_allocated_gb``: bytes held by live tensors (MPS allocator).
- ``mps_driver_allocated_gb``: total Metal driver allocation for the process,
  including the allocator's cache. The caching allocator keeps its high-water mark,
  so the maximum over steps approximates peak device memory.
- ``cuda_peak_allocated_gb`` / ``cuda_peak_reserved_gb``: torch peak counters.
- ``process_peak_rss_gb``: peak resident set size of the process (unified memory on
  Apple silicon counts GPU buffers here too).
- ``analytic``: parameter, gradient and AdamW-state bytes computed from counts.
"""

from __future__ import annotations

import copy
import statistics
import tempfile
import time
from pathlib import Path

import torch

from ..env import memory_snapshot, synchronize
from ..training.data import load_verified_examples, setting_key
from ..training.model_io import build_training_adapter
from ..training.trainer import Trainer


def _mem(device: torch.device) -> dict:
    m = memory_snapshot(device)
    if device.type == "cuda":
        m["cuda_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    return m


def longest_setting_indices(examples) -> list[int]:
    """Indices of the setting with the longest prompt+response, longest first."""
    by: dict[tuple, list[int]] = {}
    for i, ex in enumerate(examples):
        by.setdefault(setting_key(ex), []).append(i)
    key = max(by, key=lambda k: max(len(examples[i].prompt_ids) + len(examples[i].response_ids) for i in by[k]))
    return sorted(by[key], key=lambda i: -(len(examples[i].prompt_ids) + len(examples[i].response_ids)))


def run_benchmark(cfg: dict, device: torch.device, *, method: str, batch_size: int, lengths: str,
                  warmup_steps: int, timed_steps: int, autocast: str | None = None) -> dict:
    if timed_steps > 20:
        raise ValueError("at most 20 timed updates per configuration (milestone-3 pilot limit)")
    cfg = copy.deepcopy(cfg)
    cfg.update(method=method, batch_size=batch_size, autocast=autocast)
    cfg["checkpoint"]["every_updates"] = 0
    d = cfg["data"]
    examples, ident = load_verified_examples(d["dir"], d["split"], d["manifest_sha256"], None)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    adapter, info = build_training_adapter(cfg, device)
    synchronize(device)
    load_s = time.perf_counter() - t0
    mem_loaded = _mem(device)
    with tempfile.TemporaryDirectory() as tmp:
        tr = Trainer(cfg, adapter, info, examples, ident, device, Path(tmp))
        order = longest_setting_indices(examples) if lengths == "longest" else None
        steps, skipped, k = [], 0, 0
        mem_after_first = None
        while len(steps) < warmup_steps + timed_steps:
            if order is not None:
                idx = [order[(k * batch_size + j) % len(order)] for j in range(batch_size)]
                k += 1
            else:
                idx = tr.sampler.next_indices(batch_size)
            synchronize(device)
            t1 = time.perf_counter()
            rec = tr.train_step(idx)
            synchronize(device)
            e2e = time.perf_counter() - t1
            if rec.get("event") == "skipped_zero_mask":
                skipped += 1
                continue
            rec["end_to_end_seconds"] = e2e
            rec["memory"] = _mem(device)
            if mem_after_first is None:
                mem_after_first = rec["memory"]
            steps.append(rec)
        timed = steps[warmup_steps:]
    sec = [r["step_seconds"] for r in timed]
    e2e = [r["end_to_end_seconds"] for r in timed]
    real = [r["real_tokens"] for r in timed]
    padded = [r["shape"][0] * r["shape"][1] for r in timed]
    trainable = info["trainable_parameters"]
    total = info["total_parameters"]
    bpe = 4  # float32 master weights, grads and AdamW moments
    drv = [r["memory"].get("mps_driver_allocated_gb") for r in timed if "mps_driver_allocated_gb" in r["memory"]]
    out = {
        "method": method, "autocast": autocast, "param_dtype": cfg["dtype"], "device": str(device),
        "device_note": "CPU fallback (not an accelerator measurement)" if device.type == "cpu" else "accelerator",
        "batch_size": batch_size, "lengths": lengths, "warmup_steps_counted": warmup_steps,
        "timed_updates": len(timed), "skipped_zero_mask_batches": skipped,
        "trainable_parameters": trainable, "total_parameters": total, "load_seconds": round(load_s, 2),
        "batch_shapes": sorted({tuple(r["shape"]) for r in timed}),
        "real_tokens_per_update_mean": statistics.mean(real), "padded_tokens_per_update_mean": statistics.mean(padded),
        "masked_tokens_per_update_mean": statistics.mean(r["n_masked"] for r in timed),
        "seconds_per_update_mean": statistics.mean(sec), "seconds_per_update_median": statistics.median(sec),
        "seconds_per_update_min": min(sec), "seconds_per_update_max": max(sec),
        "end_to_end_seconds_per_update_mean": statistics.mean(e2e),
        "examples_per_second": batch_size / statistics.mean(e2e),
        "real_tokens_per_second": statistics.mean(real) / statistics.mean(e2e),
        "loss_first_timed": timed[0]["loss"], "loss_last_timed": timed[-1]["loss"],
        "memory_after_load": mem_loaded, "memory_after_first_update": mem_after_first,
        "memory_last_update": timed[-1]["memory"],
        "mps_driver_allocated_gb_max_over_timed": max(drv) if drv else None,
        "analytic_gb": {"parameters": round(total * bpe / 1e9, 3), "gradients": round(trainable * bpe / 1e9, 3),
                        "adamw_state": round(2 * trainable * bpe / 1e9, 3)},
        "timing_boundary": "step_seconds: synchronized forward+loss+backward+clip+AdamW+scheduler; "
                           "end_to_end adds collation and CPU mask sampling",
    }
    del tr, adapter
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return out
