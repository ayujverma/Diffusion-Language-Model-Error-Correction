#!/bin/bash
#SBATCH -J dlmc-envcheck
#SBATCH -p gpu-a100-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o runs/slurm/%x-%j.out
#SBATCH --no-requeue
# RUNS ON: a Lonestar6 GPU compute node. Submit from $WORK/dlmc/repo:
#   sbatch -A <allocation> scripts/tacc/env_check.slurm
# Checks the node, driver, torch/CUDA build, precision settings, the CPU test suite, the dataset,
# a real-backbone forward/backward/step on CUDA, and that the transferred checkpoint 1000 is
# byte- and state-identical to the Mac (preflight comparison). No training.
# Writes $WORK/dlmc/retained/env_check/<jobid>/ and ENV_CHECK_PASSED there only if all pass.
source "$SLURM_SUBMIT_DIR/scripts/tacc/tacc_env.sh"
require_job
enter_repo
OUT=$DLMC_RETAINED/env_check/$SLURM_JOB_ID
mkdir -p "$OUT"
echo "job $SLURM_JOB_ID account ${SLURM_JOB_ACCOUNT:-?} partition ${SLURM_JOB_PARTITION:-?} node $(hostname)"
scontrol show job "$SLURM_JOB_ID" > "$OUT/scontrol.txt" || true

check_gpu 2>&1 | tee "$OUT/gpu.txt"
nvidia-smi -q > "$OUT/nvidia-smi-q.txt"
lscpu > "$OUT/lscpu.txt"
{ module list 2>&1 || true; echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"; } > "$OUT/modules.txt"
cp "$DLMC_ROOT/env/freeze.txt" "$DLMC_ROOT/env/BUILD.txt" "$OUT/"

DLMC_TORCH_VERSION=$DLMC_TORCH_VERSION DLMC_CUDA_RUNTIME=$DLMC_CUDA_RUNTIME DLMC_CUDNN_VERSION=$DLMC_CUDNN_VERSION \
"$DLMC_PY" - "$OUT/torch_check.json" <<'EOF'
import json, os, sys
import torch
from diffusion_correction.env import environment_report, resolve_device

dev = resolve_device("cuda")          # also applies the D-033 precision settings (TF32 off)
rep = environment_report(dev)
props = torch.cuda.get_device_properties(0)
a = torch.randn(512, 512, dtype=torch.float64)
b = torch.randn(512, 512, dtype=torch.float64)
gpu = (a.float().to(dev) @ b.float().to(dev)).double().cpu()
mm_rel = float((gpu - a @ b).norm() / (a @ b).norm())
checks = {
    "torch_version": torch.__version__ == os.environ["DLMC_TORCH_VERSION"],
    "cuda_runtime": torch.version.cuda == os.environ["DLMC_CUDA_RUNTIME"],
    "cudnn_version": torch.backends.cudnn.version() == int(os.environ["DLMC_CUDNN_VERSION"]),
    "cuda_available": torch.cuda.is_available(),
    "one_visible_gpu": torch.cuda.device_count() == 1,
    "a100": "A100" in props.name,
    "compute_capability_8_0": (props.major, props.minor) == (8, 0),
    "sm_80_in_arch_list": "sm_80" in torch.cuda.get_arch_list(),
    "memory_at_least_39_gb": props.total_memory >= 39e9,
    "tf32_matmul_off": torch.backends.cuda.matmul.allow_tf32 is False,
    "tf32_cudnn_off": torch.backends.cudnn.allow_tf32 is False,
    "fp32_matmul_precision_highest": torch.get_float32_matmul_precision() == "highest",
    "fp32_matmul_matches_cpu_float64_1e-6": mm_rel < 1e-6,
}
out = {"checks": checks, "passed": all(checks.values()), "fp32_matmul_rel_error_vs_float64": mm_rel,
       "device_name": props.name, "total_memory_bytes": props.total_memory, "environment": rep}
open(sys.argv[1], "w").write(json.dumps(out, indent=2, default=str) + "\n")
print(json.dumps(out, indent=2, default=str))
sys.exit(0 if out["passed"] else 1)
EOF

"$DLMC_PY" -m pytest -q -p no:cacheprovider 2>&1 | tee "$OUT/pytest.txt"
"$DLMC_CLI" verify-data --data-dir "$DLMC_DATA_DIR" 2>&1 | tee "$OUT/verify_data.txt"
"$DLMC_CLI" doctor --device cuda --backbone open-dcoder --dtype float32 --optimizer adamw \
    --out-dir "$OUT/doctor" 2>&1 | tee "$OUT/doctor.txt"

verify_origin 2>&1 | tee "$OUT/origin_hashes.txt"
"$DLMC_PY" scripts/tacc/preflight_checkpoint.py --checkpoint "$DLMC_ORIGIN_REL" --config "$DLMC_CONT_CONFIG" \
    --metrics runs/train/warmup-pilot-v1/metrics.jsonl --out "$OUT/preflight.json" > "$OUT/preflight.log" 2>&1 \
    || { tail -30 "$OUT/preflight.log"; die "preflight failed"; }
"$DLMC_PY" scripts/tacc/compare_preflight.py "$DLMC_MAC_PREFLIGHT" "$OUT/preflight.json" \
    | tee "$OUT/preflight_vs_mac.txt"

{
    date -u +%FT%TZ
    echo "commit $(git rev-parse HEAD)"
    echo "torch $DLMC_TORCH_VERSION"
    echo "lock_sha256 $(lock_sha)"
} > "$OUT/ENV_CHECK_PASSED"
echo "ENV CHECK PASSED: $OUT"
