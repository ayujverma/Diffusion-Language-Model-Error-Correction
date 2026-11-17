"""Device resolution and environment/provenance capture for run records."""

from __future__ import annotations

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
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device: torch.device) -> dict:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = rss if sys.platform == "darwin" else rss * 1024
    out = {"process_peak_rss_gb": round(rss_bytes / 1e9, 3)}
    if device.type == "cuda":
        out["cuda_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        out["cuda_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 3)
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
        report["cuda_device"] = torch.cuda.get_device_name(0)
    return report
