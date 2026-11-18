"""Lonestar6 environment guards (D-034): the driver check, the binding of the venv and every pass marker to
the lock file, and the consistency of the TACC lock with tacc_env.sh and the Mac lock. No TACC access needed:
tacc_env.sh is sourced with a fake $WORK/$SCRATCH and a fake nvidia-smi."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TACC = ROOT / "scripts" / "tacc"
LOCK_REL = "scripts/tacc/requirements-tacc-lock.txt"


def setting(name):
    return re.search(rf"^{name}=(\S+)", (TACC / "tacc_env.sh").read_text(), re.M).group(1)


@pytest.fixture
def ls6(tmp_path):
    work, scratch, fake = tmp_path / "work", tmp_path / "scratch", tmp_path / "bin"
    repo = work / "dlmc" / "repo"
    (repo / "scripts" / "tacc").mkdir(parents=True)
    (repo / LOCK_REL).write_bytes((ROOT / LOCK_REL).read_bytes())
    scratch.mkdir()
    fake.mkdir()
    (fake / "nvidia-smi").write_text('#!/bin/bash\ncase "$*" in\n  *index*) echo 0;;\n'
                                     '  *driver_version*) echo "$FAKE_DRIVER";;\n'
                                     '  *name*) echo "NVIDIA A100-PCIE-40GB";;\n  *) echo fake-nvidia-smi;;\nesac\n')
    (fake / "nvidia-smi").chmod(0o755)

    def run(script, driver="570.195.03"):
        env = {"PATH": f"{fake}:/usr/bin:/bin:/usr/sbin:/sbin", "WORK": str(work), "SCRATCH": str(scratch),
               "TACC_SYSTEM": "ls6", "FAKE_DRIVER": driver, "HOME": str(tmp_path)}
        r = subprocess.run(["bash", "-c", f"source {TACC / 'tacc_env.sh'}; {script}"], env=env,
                           capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    run.work, run.repo = work, repo
    return run


@pytest.mark.parametrize("driver,ok", [("570.195.03", True), ("560.35.05", True), ("560.35.04", False),
                                       ("555.42.06", False), ("580.65.06", True), ("garbage", False)])
def test_driver_guard(ls6, driver, ok):
    rc, out = ls6("check_gpu", driver)
    assert (rc == 0) == ok, out
    if not ok:
        assert "ERROR" in out


def test_build_and_markers_are_bound_to_the_lock(ls6):
    rc, sha = ls6("lock_sha")
    sha = sha.strip()
    assert rc == 0 and len(sha) == 64

    rc, out = ls6("check_env_build")
    assert rc != 0 and "BUILD.txt missing" in out
    env = ls6.work / "dlmc" / "env"
    env.mkdir()
    old = "0" * 64  # an environment built from another lock (e.g. the torch 2.15.0 one)
    (env / "BUILD.txt").write_text(f"torch 2.15.0\n{old}  {LOCK_REL}\n")
    rc, out = ls6("check_env_build")
    assert rc != 0 and "different" in out
    (env / "BUILD.txt").write_text(f"torch {setting('DLMC_TORCH_VERSION')}\n{sha}  {LOCK_REL}\n")
    assert ls6("check_env_build")[0] == 0

    marker = ls6.work / "ENV_CHECK_PASSED"
    rc, out = ls6(f"require_marker {marker} test")
    assert rc != 0 and "not found" in out
    marker.write_text("2026-11-18T00:00:00Z\n")  # a marker without a lock identity is not accepted
    assert ls6(f"require_marker {marker} test")[0] != 0
    marker.write_text(f"2026-11-18T00:00:00Z\nlock_sha256 {old}\n")
    rc, out = ls6(f"require_marker {marker} test")
    assert rc != 0 and "fresh check" in out
    marker.write_text(f"2026-11-18T00:00:00Z\nlock_sha256 {sha}\n")
    assert ls6(f"require_marker {marker} test")[0] == 0


def parse_lock(path):
    pins, hashes, current = {}, {}, None
    for line in path.read_text().splitlines():
        if line and not line.startswith((" ", "#")):
            m = re.match(r"([A-Za-z0-9_.-]+)(==| @ )(\S+)", line)
            current = m.group(1).lower().replace("_", "-")
            pins[current] = m.group(3)
            hashes[current] = []
        elif line.strip().startswith("--hash=sha256:"):
            hashes[current].append(line.strip().removeprefix("--hash=sha256:").rstrip(" \\"))
    return pins, hashes


def test_tacc_lock_is_the_cuda_12_6_build_and_keeps_mac_pins():
    pins, hashes = parse_lock(ROOT / LOCK_REL)
    assert all(hashes.values())
    url = pins.pop("torch")
    version = setting("DLMC_TORCH_VERSION")
    assert url.startswith("https://download-r2.pytorch.org/whl/cu126/torch-" + version.replace("+", "%2B"))
    assert "-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=" in url
    assert hashes["torch"] == [url.split("#sha256=")[1]]
    assert setting("DLMC_CUDA_RUNTIME") == "12.6" and version.endswith("+cu126")
    # no CUDA 13 components survive, and every NVIDIA library is a CUDA 12 build at most 12.6
    assert not [n for n in pins if n.endswith("-cu13") or n in {"nvidia-cublas", "nvidia-cuda-runtime"}]
    nvidia = {n: v for n, v in pins.items() if n.startswith("nvidia-")}
    assert nvidia and all(n.endswith("-cu12") for n in nvidia)
    assert pins["cuda-toolkit"] == "12.6.3" and pins["nvidia-nvjitlink-cu12"] == "12.6.85"
    assert pins["nvidia-cudnn-cu12"] == "9.10.2.21" and setting("DLMC_CUDNN_VERSION") == "91002"

    mac = dict(re.match(r"([A-Za-z0-9_.-]+)==(\S+)", l).groups()
               for l in (ROOT / "requirements-lock.txt").read_text().split())
    mac = {k.lower().replace("_", "-"): v for k, v in mac.items()}
    shared = set(mac) & set(pins)
    assert {"numpy", "safetensors", "tokenizers", "pyyaml", "huggingface-hub", "pytest", "setuptools"} <= shared
    assert {n: pins[n] for n in shared} == {n: mac[n] for n in shared}
    constraints = (TACC / "constraints-from-mac-lock.txt").read_text().splitlines()
    assert constraints == [l for l in (ROOT / "requirements-lock.txt").read_text().splitlines()
                           if not l.startswith("torch==")]
