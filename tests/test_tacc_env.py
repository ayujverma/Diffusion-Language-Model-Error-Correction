"""Lonestar6 launcher guards (D-034, D-035): the driver check, single-GPU selection and inventory on full
gpu-a100 nodes, the full-node partition, the binding of the venv and every pass marker to the lock file, and
the consistency of the TACC lock with tacc_env.sh and the Mac lock. No TACC access needed: tacc_env.sh is
sourced with a fake $WORK/$SCRATCH and a fake three-GPU nvidia-smi."""

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
    (fake / "nvidia-smi").write_text(FAKE_NVIDIA_SMI)
    (fake / "nvidia-smi").chmod(0o755)

    def run(script, driver="570.195.03", **extra):
        env = {"PATH": f"{fake}:/usr/bin:/bin:/usr/sbin:/sbin", "WORK": str(work), "SCRATCH": str(scratch),
               "TACC_SYSTEM": "ls6", "FAKE_DRIVER": driver, "HOME": str(tmp_path), **extra}
        r = subprocess.run(["bash", "-c", f"source {TACC / 'tacc_env.sh'}; {script}"], env=env,
                           capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    run.work, run.repo, run.tmp = work, repo, tmp_path
    return run


# A full gpu-a100 node: three A100s. Like the real tool, it ignores CUDA_VISIBLE_DEVICES.
FAKE_NVIDIA_SMI = r'''#!/bin/bash
idx=(0 1 2); bus=(00000000:06:00.0 00000000:41:00.0 00000000:81:00.0); uuid=(GPU-aaaa GPU-bbbb GPU-cccc)
sel=""; q=""
for a in "$@"; do
  if [[ "$prev" == -i ]]; then sel=$a; fi
  case "$a" in --query-gpu=*) q=${a#--query-gpu=};; esac
  prev=$a
done
[[ -n "$q" ]] || { echo fake-nvidia-smi; exit 0; }
found=0
for k in 0 1 2; do
  [[ -z "$sel" || "$sel" == "${idx[$k]}" || "$sel" == "${uuid[$k]}" ]] || continue
  found=1; out=()
  IFS=, read -ra fs <<< "$q"
  for f in "${fs[@]}"; do case "$f" in
    index) out+=("${idx[$k]}");; pci.bus_id) out+=("${bus[$k]}");; uuid) out+=("${uuid[$k]}");;
    name) out+=("NVIDIA A100-PCIE-40GB");; memory.total) out+=("40960 MiB");; driver_version) out+=("$FAKE_DRIVER");;
  esac; done
  (IFS=,; echo "${out[*]}" | sed 's/,/, /g')
done
[[ $found == 1 ]] || { echo "No devices were found" >&2; exit 6; }
'''


@pytest.mark.parametrize("driver,ok", [("570.195.03", True), ("560.35.05", True), ("560.35.04", False),
                                       ("555.42.06", False), ("580.65.06", True), ("garbage", False)])
def test_driver_guard(ls6, driver, ok):
    rc, out = ls6(f"select_gpu {ls6.tmp}/inv.txt >/dev/null; check_gpu", driver)
    assert (rc == 0) == ok, out
    if not ok:
        assert "ERROR" in out


def test_check_gpu_requires_a_selected_device(ls6):
    rc, out = ls6("check_gpu")
    assert rc != 0 and "select_gpu must run" in out


@pytest.mark.parametrize("assigned,selected", [(None, "0"), ("0", "0"), ("2", "2"), ("1,2", "1"),
                                               ("GPU-cccc", "GPU-cccc")])
def test_select_gpu_one_device_and_inventory(ls6, assigned, selected):
    rec = ls6.tmp / "logs" / "inv.txt"
    extra = {} if assigned is None else {"CUDA_VISIBLE_DEVICES": assigned}
    rc, out = ls6(f'select_gpu {rec} >/dev/null; echo "CVD=$CUDA_VISIBLE_DEVICES ORDER=$CUDA_DEVICE_ORDER"; '
                  'bash -c \'echo "child CVD=$CUDA_VISIBLE_DEVICES"\'', **extra)
    assert rc == 0, out
    assert f"CVD={selected} ORDER=PCI_BUS_ID" in out and f"child CVD={selected}" in out  # exported to children
    text = rec.read_text()
    assert "physical GPUs (nvidia-smi; index,pci.bus_id,uuid,name,memory.total,driver_version): 3" in text
    assert all(u in text for u in ("GPU-aaaa", "GPU-bbbb", "GPU-cccc"))  # full inventory, not just the visible one
    how = ("selection: device 0 (no scheduler-provided" if assigned is None
           else f"selection: first device of the scheduler-provided CUDA_VISIBLE_DEVICES={assigned}")
    assert how in text
    sel_line = next(line for line in text.splitlines() if line.startswith("selected: "))
    assert selected in sel_line and sel_line.count("A100") == 1


@pytest.mark.parametrize("extra,msg", [({"CUDA_VISIBLE_DEVICES": ""}, "no GPU assigned"),
                                       ({"CUDA_VISIBLE_DEVICES": "NoDevFiles"}, "no GPU assigned"),
                                       ({"CUDA_VISIBLE_DEVICES": "5"}, "not in this node"),
                                       ({"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}, "expected PCI_BUS_ID")])
def test_select_gpu_refusals(ls6, extra, msg):
    rc, out = ls6(f"select_gpu {ls6.tmp}/inv.txt", **extra)
    assert rc != 0 and msg in out, out
    assert ls6(f"select_gpu {ls6.tmp}/a.txt >/dev/null; select_gpu {ls6.tmp}/b.txt")[0] != 0  # only once per job


def test_require_job_needs_the_full_node_partition(ls6):
    assert ls6("require_job")[0] != 0  # no Slurm job
    rc, out = ls6("require_job", SLURM_JOB_ID="1", SLURM_JOB_PARTITION="gpu-a100-small")
    assert rc != 0 and "gpu-a100" in out
    assert ls6("require_job", SLURM_JOB_ID="1", SLURM_JOB_PARTITION="gpu-a100")[0] == 0


GPU_SLURM = ["env_check.slurm", "pilot.slurm", "continuation.slurm", "final_eval.slurm", "fixture_diagnostic.slurm",
             "bank_collect.slurm", "call_budget_eval.slurm", "family_b_screen.slurm", "family_b_eval.slurm"]


@pytest.mark.parametrize("name", GPU_SLURM)
def test_gpu_jobs_target_one_gpu_on_a_full_node(name):
    text = (TACC / name).read_text()
    head = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert "#SBATCH -p gpu-a100" in head and "#SBATCH -N 1" in head and "#SBATCH -n 1" in head
    assert not re.search(r"--gres|--gpus|srun|torchrun|torch\.distributed|DistributedDataParallel", text)
    lines = text.splitlines()
    sel = [i for i, line in enumerate(lines) if line.startswith("select_gpu ")]
    assert len(sel) == 1, "select_gpu must be called once, at top level, outside a pipe"
    assert "|" not in lines[sel[0]]
    first_python = min(i for i, line in enumerate(lines)
                       if re.search(r'"\$DLMC_(PY|CLI)"|verify_origin|check_torch_gpu|\bcheck_gpu\b', line)
                       and not line.lstrip().startswith("#"))
    assert sel[0] < first_python and lines.index("require_job") < sel[0]


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
