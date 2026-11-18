#!/bin/bash
#SBATCH -J dlmc-pilot
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o runs/slurm/%x-%j.out
#SBATCH --no-requeue
# RUNS ON: a Lonestar6 GPU compute node. Submit from $WORK/dlmc/repo after a passed env check:
#   DLMC_ENV_CHECK=$WORK/dlmc/retained/env_check/<jobid> sbatch -A <allocation> scripts/tacc/pilot.slurm
#
# CUDA pilot (docs/warmup_continuation.md, "Pilot"). Every training step runs on a disposable
# copy of checkpoint 1000; nothing here feeds the continuation, which reloads the original.
#   fixture  cuda_fixture.py check: forward/loss/logits/gradients/one AdamW step and greedy T=L
#            generation vs the Mac CPU fixture, tolerances declared before any CUDA result
#   bench    dlmc benchmark full,32,longest: 3 counted warmup steps + 20 synchronized timed updates
#   M        baseline monitoring evaluation at update 1000 (cost, CUDA baseline), no training
#   U, U2    uninterrupted 1000 -> 1010 (U2 repeats U: CUDA run-to-run noise floor);
#            U also copies its checkpoints to $WORK to time durable copies
#   R        1000 -> 1005, process exits, resume from R's ckpt-0001005 -> 1010
# Pilot training work: 23 benchmark updates (fresh pretrained weights), 1 fixture step,
# 10 + 10 + 10 updates in U, U2, R (copies of ckpt-1000). None of it counts toward the
# continuation's update counter or training-time cap.
source "$SLURM_SUBMIT_DIR/scripts/tacc/tacc_env.sh"
require_job
enter_repo
JOB_START=$(date +%s)
[[ -n "${DLMC_ENV_CHECK:-}" ]] \
    || die "set DLMC_ENV_CHECK to a passed environment-check directory ($DLMC_RETAINED/env_check/<jobid>)"
require_marker "$DLMC_ENV_CHECK/ENV_CHECK_PASSED" "DLMC_ENV_CHECK"
P=$DLMC_SCRATCH/pilot/$SLURM_JOB_ID          # checkpoints (large, purgeable)
R=$DLMC_RETAINED/pilot/$SLURM_JOB_ID         # reports and run logs (small, kept)
mkdir -p "$P" "$R"
cp "$DLMC_ENV_CHECK/ENV_CHECK_PASSED" "$R/ENV_CHECK_PASSED.used"
select_gpu "$R/gpu_inventory.txt"
check_gpu 2>&1 | tee "$R/gpu.txt"
check_torch_gpu 2>&1 | tee -a "$R/gpu.txt"

set +e
step() {  # step <name> <command...>: run, log, record the exit status, keep going
    local name=$1; shift
    echo "=== $name: $*"
    local t0=$(date +%s)
    "$@" 2>&1 | tee "$R/$name.log"
    local rc=${PIPESTATUS[0]}
    echo "$name $rc $(( $(date +%s) - t0 ))" >> "$R/steps.txt"
    echo "=== $name exit $rc"
    return "$rc"
}

step origin_before verify_origin
COPY=$P/copy/ckpt-0001000
step copy bash -c "mkdir -p '$P/copy' && cp -r '$DLMC_ORIGIN' '$COPY' && chmod -R u+w '$COPY' && \
'$DLMC_PY' -c 'import sys; from diffusion_correction.training.checkpoint import verify_checkpoint as v; \
print(v(sys.argv[1]))' '$COPY'" || die "could not make the disposable copy"

step fixture "$DLMC_PY" scripts/tacc/cuda_fixture.py check --checkpoint "$COPY" --device cuda \
    --out "$R/fixture_check.json"
step bench "$DLMC_CLI" benchmark --config "$DLMC_CONT_CONFIG" --device cuda --configs full,32,longest \
    --warmup-steps 3 --timed-steps 20 --out "$R/benchmark.json"

NOCAP=(--set limits.max_train_seconds=null --set limits.train_seconds_origin_update=null)
NOEVAL=(--set eval.every_updates=0 --set eval.selection=null --set "eval.generate_budgets=[]"
        --set eval.write_records=false)
CK5=(--set checkpoint.every_updates=5 --set checkpoint.keep_last=10 --set "checkpoint.protect_updates=[]"
     --set checkpoint.protect_every=0)
LOG=(--set log_every=1 --set memory_every=1)
TRAIN=("$DLMC_CLI" train --config "$DLMC_CONT_CONFIG" --device cuda)

step M "${TRAIN[@]}" --run-dir "$P/M" --resume "$COPY" --max-updates 1000 "${NOCAP[@]}"
step U "${TRAIN[@]}" --run-dir "$P/U" --resume "$COPY" --max-updates 1010 "${NOCAP[@]}" "${NOEVAL[@]}" \
    "${CK5[@]}" "${LOG[@]}" --durable-dir "$R/durable"
step U2 "${TRAIN[@]}" --run-dir "$P/U2" --resume "$COPY" --max-updates 1010 "${NOCAP[@]}" "${NOEVAL[@]}" \
    "${CK5[@]}" "${LOG[@]}"
step R1 "${TRAIN[@]}" --run-dir "$P/R" --resume "$COPY" --max-updates 1005 "${NOCAP[@]}" "${NOEVAL[@]}" \
    "${CK5[@]}" "${LOG[@]}"
step R2 "${TRAIN[@]}" --run-dir "$P/R" --resume "$P/R/checkpoints/ckpt-0001005" --max-updates 1010 \
    "${NOCAP[@]}" "${NOEVAL[@]}" "${CK5[@]}" "${LOG[@]}"

TOL=(--checkpoint ckpt-0001010 --loss-rtol 1e-4 --param-atol 1e-5 --divergence-rtol 1e-3
     --reference-model "$COPY/model.safetensors")
step compare_U_R "$DLMC_CLI" compare-runs "$P/U" "$P/R" "${TOL[@]}" --out "$R/compare_U_R.json"
step compare_U_U2 "$DLMC_CLI" compare-runs "$P/U" "$P/U2" "${TOL[@]}" --out "$R/compare_U_U2.json"
for run in U U2 R; do
    step "expected_$run" "$DLMC_PY" scripts/tacc/check_expected_batches.py --preflight "$DLMC_MAC_PREFLIGHT" \
        --metrics "$P/$run/metrics.jsonl" --lr 3e-05 --out "$R/expected_$run.json"
done
step origin_after verify_origin

mkdir -p "$R/runs"
for run in M U U2 R; do
    rsync -a --exclude checkpoints "$P/$run" "$R/runs/"
done
set -e
"$DLMC_PY" scripts/tacc/pilot_report.py --report-dir "$R" --job-seconds $(( $(date +%s) - JOB_START )) \
    --su-per-node-hour "$DLMC_SU_PER_NODE_HOUR"
