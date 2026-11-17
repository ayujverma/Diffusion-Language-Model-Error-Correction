"""Device resolution and environment/provenance capture for run records."""

from __future__ import annotations

import os
import platform
import resource
import subprocess
import sys
from pathlib import Path

import torch

DTYPES = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16, "float16": torch.float16}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available")
    device = torch.device(name)
    configure_precision(device)
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def release_cached_memory(device: torch.device) -> None:
    """MPS-only workaround (D-032): return cached allocator blocks to the driver.

    With varying padded shapes the MPS cache grew until macOS swapped. The CUDA caching
    allocator reuses blocks, so on CUDA (and CPU) this does nothing.
    """
    if device.type == "mps":
        torch.mps.empty_cache()


def configure_precision(device: torch.device) -> dict:
    """Pin float32 matmul/convolution precision on CUDA (no TF32, D-033) and return the settings."""
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    return precision_report(device)


def precision_report(device: torch.device) -> dict:
    out = {"float32_matmul_precision": torch.get_float32_matmul_precision()}
    if device.type == "cuda":
        out.update(cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                   cuda_matmul_allow_fp16_reduced_precision=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                   sdpa_enabled={"flash": torch.backends.cuda.flash_sdp_enabled(),
                                 "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                                 "math": torch.backends.cuda.math_sdp_enabled(),
                                 "cudnn": torch.backends.cuda.cudnn_sdp_enabled()})
    return out


def memory_snapshot(device: torch.device) -> dict:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    out = {"process_peak_rss_gb": round(rss_bytes / 1e9, 3)}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        out["cuda_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 3)
        out["cuda_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        out["cuda_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 3)
        out["cuda_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
        out["cuda_device_used_gb"] = round((total - free) / 1e9, 3)  # all processes and contexts
        out["cuda_total_gb"] = round(total / 1e9, 3)
    elif device.type == "mps":
        out["mps_current_allocated_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 3)
        out["mps_driver_allocated_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 3)
        out["mps_recommended_max_gb"] = round(torch.mps.recommended_max_memory() / 1e9, 3)
    return out


def git_state(repo: Path = Path(".")) -> dict:
    def run(*args):
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def environment_report(device: torch.device) -> dict:
    import numpy

    try:
        import transformers
        tf_version = transformers.__version__
    except ImportError:
        tf_version = None
    import tokenizers

    report = {
        "os": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
        "torch": torch.__version__, "numpy": numpy.__version__, "tokenizers": tokenizers.__version__,
        "transformers_installed": tf_version, "transformers_used_by_model": False,
        "device": str(device), "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(), "git": git_state(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        report.update(cuda_device=torch.cuda.get_device_name(device), cuda_runtime=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version(), cuda_capability=f"{props.major}.{props.minor}",
                      cuda_total_memory_gb=round(props.total_memory / 1e9, 3),
                      cuda_device_count=torch.cuda.device_count(), cuda_arch_list=torch.cuda.get_arch_list())
    report["precision"] = precision_report(device)
    report["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    report["hostname"] = platform.node()
    return report
