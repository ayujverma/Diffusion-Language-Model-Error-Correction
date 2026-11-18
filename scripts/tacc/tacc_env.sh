# Shared settings for every scripts/tacc script on Lonestar6. Source it; do not execute it.
# Fails with a clear message when something required is missing.
# Layout (docs/warmup_continuation.md, "TACC layout"):
#   $WORK/dlmc/transfer   files copied from the Mac (read-only after verification); keeps ckpt-0001000
#   $WORK/dlmc/repo       git clone of the transferred bundle; runs/{hf,data,...} link into transfer/
#   $WORK/dlmc/retained   durable checkpoints and logs (continuation, pilot copy-timing, final eval)
#   $SCRATCH/dlmc/venv    Python 3.14.7 environment from requirements-tacc-lock.txt (TACC: virtual
#                         environments go on $SCRATCH, not $WORK; rebuild with setup_env.sh if purged)
#   $SCRATCH/dlmc         job working directories (purged after 10 days without access)

set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "${WORK:-}" && -n "${SCRATCH:-}" ]] || die "WORK/SCRATCH are not set: these scripts run on Lonestar6 only"
if [[ "${TACC_SYSTEM:-}" != ls6 && "$(hostname -f 2>/dev/null || hostname)" != *ls6* ]]; then
    die "not a Lonestar6 node (TACC_SYSTEM='${TACC_SYSTEM:-}', hostname '$(hostname)')"
fi

DLMC_ROOT=${DLMC_ROOT:-$WORK/dlmc}
DLMC_TRANSFER=$DLMC_ROOT/transfer
DLMC_REPO=$DLMC_ROOT/repo
DLMC_RETAINED=$DLMC_ROOT/retained
DLMC_SCRATCH=$SCRATCH/dlmc
DLMC_VENV=$DLMC_SCRATCH/venv
DLMC_PY=$DLMC_VENV/bin/python
DLMC_CLI=$DLMC_VENV/bin/dlmc

# Pinned identities (verified on the Mac; docs/warmup_continuation.md, "Transfer manifest").
DLMC_DATA_DIR=runs/data/pilot-seed20261116
DLMC_MANIFEST_SHA256=990ecd1dd2341b81bc742fa91189b129877cd9b48b196fbdbc208e8586c81536
DLMC_ORIGIN_REL=runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000
DLMC_ORIGIN=$DLMC_TRANSFER/$DLMC_ORIGIN_REL
DLMC_ORIGIN_MODEL_SHA256=f48abe457d483b19b854e1fdf6be4bb294c702f949f9d76d7c44a2274b0d5a6a
DLMC_ORIGIN_OPTIMIZER_SHA256=ae55b39481da056d6b4180cd0197a732efecaece7b17c0d0eab52aa0fbfa99af
DLMC_ORIGIN_COMPLETE_SHA256=145dea5b78ac29e7a618574402e333a9a510a1617289de7f0ee72a8297452001
DLMC_ORIGIN_METRICS=$DLMC_TRANSFER/runs/train/warmup-pilot-v1/metrics.jsonl
DLMC_MAC_PREFLIGHT=$DLMC_TRANSFER/runs/preflight/ckpt-0001000-mac.json
DLMC_CONT_CONFIG=configs/train/warmup_continuation.yaml
DLMC_RUN_NAME=warmup-pilot-v1-cont
DLMC_CONT_RUN_DIR=$DLMC_SCRATCH/runs/$DLMC_RUN_NAME
DLMC_CONT_DURABLE=$DLMC_RETAINED/$DLMC_RUN_NAME          # trainer writes <durable_dir>/<run_name>/checkpoints
DLMC_STAGE2_SETTINGS=arithmetic:2:1,arithmetic:6:3,arithmetic:4:1,arithmetic:2:2,arithmetic:4:2,programs:2:1,programs:6:3,programs:2:2,programs:2:3,programs:4:2
DLMC_UV_VERSION=0.13.9
DLMC_UV_SHA256=a7fbaff2b9a271d85c1bdb540941090fd1ce011797b4abeb48e4dcd39c2471d5
DLMC_PYTHON_VERSION=3.14.7
# D-034 fallback build (the CUDA 13.2 build of torch 2.15.0 needs driver R580; ls6 has R570).
DLMC_LOCK=scripts/tacc/requirements-tacc-lock.txt
DLMC_TORCH_VERSION=2.14.1+cu126
DLMC_CUDA_RUNTIME=12.6
DLMC_CUDNN_VERSION=91002                                    # nvidia-cudnn-cu12 9.10.2.21
DLMC_MIN_DRIVER=560.35.05                                   # CUDA 12.6 Update 3 toolkit minimum (NVIDIA release notes)

export UV_CACHE_DIR=$DLMC_SCRATCH/uv-cache
export UV_PYTHON_INSTALL_DIR=$DLMC_SCRATCH/python
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1                                     # all model files are local and hash-checked

# Model execution happens only inside Slurm jobs on compute nodes.
require_job() {
    [[ -n "${SLURM_JOB_ID:-}" ]] || die "this step runs models: submit it with sbatch (never run it on a login node)"
}

# Jobs must be submitted from $DLMC_REPO with an unmodified checkout.
enter_repo() {
    [[ -d "$DLMC_REPO/.git" ]] || die "$DLMC_REPO is not a git checkout: run scripts/tacc/install_transfer.sh first"
    cd "$DLMC_REPO"
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && "$(cd "$SLURM_SUBMIT_DIR" && pwd -P)" != "$(pwd -P)" ]]; then
        die "submit from $DLMC_REPO (submitted from $SLURM_SUBMIT_DIR)"
    fi
    [[ -z "$(git status --porcelain --untracked-files=no)" ]] || die "tracked files in $DLMC_REPO are modified"
    [[ -x "$DLMC_PY" ]] || die "$DLMC_PY missing: run scripts/tacc/setup_env.sh first"
    check_env_build
    echo "repo $(pwd -P) commit $(git rev-parse HEAD) lock_sha256 $(lock_sha)"
}

lock_sha() { sha256sum "$DLMC_REPO/$DLMC_LOCK" | cut -d' ' -f1; }

# The environment must have been built by setup_env.sh from the lock in this checkout.
check_env_build() {
    [[ -f "$DLMC_ROOT/env/BUILD.txt" ]] || die "$DLMC_ROOT/env/BUILD.txt missing: run scripts/tacc/setup_env.sh"
    grep -q "^$(lock_sha)  $DLMC_LOCK\$" "$DLMC_ROOT/env/BUILD.txt" || die "$DLMC_VENV was built from a different \
lock than $DLMC_LOCK in this checkout: move the old venv and $DLMC_ROOT/env aside and rebuild with setup_env.sh"
}

# A pass marker (ENV_CHECK_PASSED, or the copy a pilot keeps) must name the current lock.
require_marker() {  # require_marker <file> <description>
    [[ -f "$1" ]] || die "$2: $1 not found"
    grep -qx "lock_sha256 $(lock_sha)" "$1" || die "$2: $1 was not produced with the current environment lock \
($DLMC_LOCK sha256 $(lock_sha)); a fresh check with this environment is required"
}

version_ge() { [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" == "$2" ]]; }

# Record the GPU and refuse anything that is not one NVIDIA A100 with a new-enough driver.
check_gpu() {
    command -v nvidia-smi >/dev/null || die "nvidia-smi not found: not a GPU node"
    nvidia-smi
    local n driver name
    n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "visible GPUs: $n; first: $name; driver $driver; CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
    [[ "$name" == *A100* ]] || die "expected an A100, found '$name'"
    [[ "$driver" =~ ^[0-9]+(\.[0-9]+)+$ ]] || die "cannot parse driver version '$driver'"
    version_ge "$driver" "$DLMC_MIN_DRIVER" || die "driver $driver < $DLMC_MIN_DRIVER: torch $DLMC_TORCH_VERSION \
(CUDA $DLMC_CUDA_RUNTIME) cannot run. Stop; a different build needs an explicit decision (docs/warmup_continuation.md, \
'Driver too old')"
}

# sha256 of a file equals an expected value.
check_sha() {
    local got
    got=$(sha256sum "$1" | cut -d' ' -f1)
    [[ "$got" == "$2" ]] || die "sha256 mismatch for $1: $got != $2"
    echo "sha256 ok $1"
}

verify_origin() {
    check_sha "$DLMC_ORIGIN/COMPLETE.json" "$DLMC_ORIGIN_COMPLETE_SHA256"
    check_sha "$DLMC_ORIGIN/model.safetensors" "$DLMC_ORIGIN_MODEL_SHA256"
    check_sha "$DLMC_ORIGIN/optimizer.pt" "$DLMC_ORIGIN_OPTIMIZER_SHA256"
    "$DLMC_PY" -c "import sys; from diffusion_correction.training.checkpoint import verify_checkpoint; \
verify_checkpoint(sys.argv[1]); print('checkpoint verified', sys.argv[1])" "$DLMC_ORIGIN"
}
