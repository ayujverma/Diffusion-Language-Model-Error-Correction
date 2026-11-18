"""Pilot 3592809 diagnostic tooling: the common float64 norm, the fixture's per-tensor filter, the
pilot memory summary, and the diagnostic job's separation from the official gates."""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TACC = ROOT / "scripts" / "tacc"
sys.path.insert(0, str(TACC))
import fixture_update_diagnostic as D  # noqa: E402


def test_f64_norm_is_chunk_independent_and_exact(monkeypatch):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1000, 37, generator=g) * 1e-4
    x[0, :5] = 3.0  # a few large entries next to many tiny ones (the embedding update's shape)
    y = torch.randn(1000, 37, generator=g) * 1e-4
    exact = math.fsum(float(v) ** 2 for v in (x.double() - y.double()).reshape(-1))
    import cuda_fixture as F
    for chunk in (7, 4096, 1 << 22):
        monkeypatch.setattr(F, "NORM_CHUNK", chunk)
        assert D.f64_sq(x, y) == pytest.approx(exact, rel=1e-13)
        assert D.f64_sq(x) == pytest.approx(math.fsum(float(v) ** 2 for v in x.double().reshape(-1)), rel=1e-13)


def test_per_tensor_rel_uses_the_fixture_filter():
    names = [D.EMBED, "b", "c"]
    ref = torch.tensor([1.0, 1.0, 1e-4], dtype=torch.float32)  # c is below 1e-3 x max: not compared
    cur = torch.tensor([1.02, 1.0, 5e-4], dtype=torch.float32)
    r = D.per_tensor_rel(cur, ref, names)
    assert r["n_compared"] == 2 and r["worst"] == D.EMBED
    assert r["max_rel"] == pytest.approx(0.02, rel=1e-6) and r["embed_rel"] == pytest.approx(0.02, rel=1e-6)


def write_pilot(R, warn_line=None):
    (R / "runs" / "U").mkdir(parents=True)
    (R / "steps.txt").write_text("fixture 1 60\n")
    recs = [{"update": 1001 + k, "memory": {"cuda_allocated_gb": 8.0, "cuda_peak_allocated_gb": 20.0 + k,
                                            "cuda_reserved_gb": 24.0, "cuda_peak_reserved_gb": 30.0 + k,
                                            "cuda_device_used_gb": 31.5 + k, "cuda_total_gb": 42.4}}
            for k in range(3)]
    (R / "runs/U/metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (R / "pilot_report.json").write_text(json.dumps({"measurements": {"peak_memory_U": 32.0}}))
    (R / "U.log").write_text("update 1001\n" + (warn_line + "\n" if warn_line else "") + "done\n")


def summarize(R, *extra):
    out = R / "summary.json"
    r = subprocess.run([sys.executable, str(TACC / "summarize_pilot_memory.py"), "--pilot-dir", str(R),
                        "--out", str(out), *extra], capture_output=True, text=True)
    return r, (json.loads(out.read_text()) if out.exists() else None)


def test_memory_summary_separates_allocated_reserved_and_device(tmp_path):
    line = "UserWarning: CUDA allocator: cudaMalloc retries observed"
    write_pilot(tmp_path, line)
    r, s = summarize(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    u = s["runs"]["U"]
    assert u["cuda_peak_reserved_gb"]["max"] == 32.0 == s["peak_memory_U_recomputed"] == s["peak_memory_U_in_report"]
    assert u["cuda_peak_allocated_gb"]["max"] == 22.0 and u["peak_reserved_minus_peak_allocated_gb"] == 10.0
    assert u["device_used_minus_peak_reserved_gb"] == 1.5
    hits = s["logs"][str(tmp_path / "U.log")]["lines"]
    assert [h["line"] for h in hits] == [2] and hits[0]["text"] == line
    assert s["any_retry_line"] and not s["any_out_of_memory_line"] and not s["allocator_counters_recorded_by_pilot"]


def test_memory_summary_refuses_a_non_pilot_dir(tmp_path):
    r, s = summarize(tmp_path)
    assert r.returncode != 0 and "not a pilot report directory" in r.stderr and s is None


def test_diagnostic_job_is_separate_from_the_gates():
    job = (TACC / "fixture_diagnostic.slurm").read_text()
    code = "\n".join(line for line in job.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"PILOT_PASSED|ENV_CHECK_PASSED|cuda_fixture\.py|dlmc\b.*train|--resume", code)
    assert "--no-requeue" in job and "verify_origin" in code and "DLMC_ORIGIN" in code
    assert code.count("run --device") == 2  # one disposable update per TACC backend
    diag = (TACC / "fixture_update_diagnostic.py").read_text()
    assert not re.search(r"PILOT_PASSED\"|ENV_CHECK_PASSED\"|save_checkpoint|TOLERANCES\[[^]]+\]\s*=", diag)
