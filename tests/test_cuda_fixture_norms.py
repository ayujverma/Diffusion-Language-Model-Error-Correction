"""D-036: chunked CPU float64 per-tensor norms for the CUDA pilot fixture; tolerances unchanged."""

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "tacc"))
import cuda_fixture as F  # noqa: E402


def test_tolerances_keep_the_one_percent_per_tensor_gates():
    assert F.TOLERANCES["tensor_update_norm_rel"] == 1e-2
    assert F.TOLERANCES["tensor_grad_norm_rel"] == 1e-2
    assert F.TOLERANCES["grad_norm_rel"] == 1e-3
    assert F.NORM_MEASUREMENT == "chunked_cpu_float64_l2"
    assert F.DEFAULT_FIXTURE.endswith("cuda_pilot_fixture_f64norms")
    assert F.LEGACY_FIXTURE.endswith("cuda_pilot_fixture")
    assert F.LEGACY_FIXTURE != F.DEFAULT_FIXTURE


def test_chunked_l2_matches_fsum_reference_and_is_chunk_independent(monkeypatch):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(50_000, 17, generator=g) * 1e-4
    x[0, :8] = torch.tensor([3.0, -2.5, 1.25, 0.0, 4.0, -1.0, 0.5, 2.0])
    y = torch.randn(50_000, 17, generator=g) * 1e-4
    exact = math.fsum(float(v) ** 2 for v in (x.double() - y.double()).reshape(-1))
    exact_x = math.fsum(float(v) ** 2 for v in x.double().reshape(-1))
    for chunk in (7, 1024, 1 << 22):
        monkeypatch.setattr(F, "NORM_CHUNK", chunk)
        assert F.chunked_l2_sq_f64(x, y) == pytest.approx(exact, rel=1e-13)
        assert F.chunked_l2_norm_f64(x, y) == pytest.approx(math.sqrt(exact), rel=1e-13)
        assert F.chunked_l2_sq_f64(x) == pytest.approx(exact_x, rel=1e-13)


def test_chunked_l2_on_embedding_scale_tensor():
    """~8M elements (same order of reduction issues as the 136M embedding, cheaper in CI)."""
    g = torch.Generator().manual_seed(1)
    n, d = 8192, 1024  # 8,388,608
    pre = torch.randn(n, d, generator=g)
    post = pre + 1e-5 * torch.randn(n, d, generator=g)
    got = F.chunked_l2_norm_f64(post, pre)
    ref = math.sqrt(math.fsum(float(v) ** 2 for v in (post.double() - pre.double()).reshape(-1)))
    assert got == pytest.approx(ref, rel=1e-12)
    native = float((post - pre).float().norm())
    # Native fp32 may differ; the fixture gate uses the float64 path, not native.
    assert math.isfinite(native) and got > 0


def test_make_refuses_legacy_path_and_overwrite(tmp_path):
    legacy = ROOT / F.LEGACY_FIXTURE
    assert legacy.with_suffix(".pt").exists() and legacy.with_suffix(".json").exists()
    r = subprocess.run([sys.executable, str(ROOT / "scripts/tacc/cuda_fixture.py"), "make",
                        "--checkpoint", "unused", "--out", str(legacy)],
                       capture_output=True, text=True, cwd=str(ROOT))
    # Fails before loading the checkpoint (path guard).
    assert r.returncode != 0 and "historical" in (r.stdout + r.stderr)

    out = tmp_path / "fixture"
    out.with_suffix(".json").write_text("{}")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/tacc/cuda_fixture.py"), "make",
                        "--checkpoint", "unused", "--out", str(out)],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode != 0 and "overwrite" in (r.stdout + r.stderr)


def test_check_rejects_legacy_fixture_norm_measurement():
    """New check code must not silently compare f64 norms to the historical fp32 fixture."""
    legacy = ROOT / F.LEGACY_FIXTURE
    ref = json.loads(legacy.with_suffix(".json").read_text())
    assert ref.get("norm_measurement") != F.NORM_MEASUREMENT
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts/tacc/cuda_fixture.py"), "check",
         "--checkpoint", "unused", "--fixture", str(legacy)],
        capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode != 0 and "norm_measurement" in (r.stdout + r.stderr)
    assert "D-036" in (r.stdout + r.stderr)


def test_pilot_points_at_the_f64_fixture():
    text = (ROOT / "scripts/tacc/pilot.slurm").read_text()
    assert "--fixture runs/reference/cuda_pilot_fixture_f64norms" in text
    assert F.TOLERANCES["tensor_update_norm_rel"] == 1e-2


def test_legacy_fixture_hashes_unchanged():
    pt = ROOT / "runs/reference/cuda_pilot_fixture.pt"
    js = ROOT / "runs/reference/cuda_pilot_fixture.json"
    assert hashlib.sha256(pt.read_bytes()).hexdigest() == (
        "b5dc70811254bdf4394499693836e5cf6e38dc0b0f38e31af11872e6a8ce3e74")
    assert hashlib.sha256(js.read_bytes()).hexdigest() == (
        "0f3c48557274ac74e862686ab56d13c5cc6669c35da5ea5cd7caa7db56a6fd24")
