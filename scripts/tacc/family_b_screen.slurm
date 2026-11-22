#!/bin/bash
#SBATCH -J dlmc-m7
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o runs/slurm/%x-%A_%a.out
#SBATCH --signal=B:USR1@900
#SBATCH --no-requeue
#SBATCH --array=0-3
# NOT SUBMITTED. The user submits this. Account: sbatch -A ASC26098 scripts/tacc/family_b_screen.slurm
# Does not train Family A. Does not read test_id, test_long, or test_unseen.
# One task per Family B arm, same batch, seed (20261120), update cap (1000), and init
# checkpoint 8000. ProSeCo arms run two forwards with gradient (the shape that OOMed at
# batch 32 on an A100 40GB). SCOPE runs one gradient forward, or one no_grad draft plus
# one gradient forward. The batch stays 16 for every arm; do not shrink a single task.
# Smoke a handful of updates first:
#   DLMC_MAX_UPDATES=2 sbatch -A ASC26098 --array=0-3 scripts/tacc/family_b_screen.slurm
# Resume one arm only: DLMC_M7_RESUME=1 sbatch -A ASC26098 --array=<index> scripts/tacc/family_b_screen.slurm
source "$SLURM_SUBMIT_DIR/scripts/tacc/tacc_env.sh"
require_job
enter_repo
select_gpu "$DLMC_RETAINED/family_b/logs/gpu_inventory-$SLURM_JOB_ID.txt"
check_gpu
check_torch_gpu

ARMS=(proseco_basic proseco_error_prioritized proseco_gen_conf scope)
IDX=${SLURM_ARRAY_TASK_ID:?array task id required}
(( IDX >= 0 && IDX < ${#ARMS[@]} )) || die "array index $IDX out of range 0..$((${#ARMS[@]} - 1))"
ARM=${ARMS[$IDX]}
CONFIG=configs/correction/m7_${ARM}.yaml
[[ -f "$CONFIG" ]] || die "missing $CONFIG"

RUN_NAME=$("$DLMC_PY" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['run_name'])")
"$DLMC_PY" scripts/tacc/generator_identity.py --check --checkpoint "$DLMC_GEN" \
    --model-sha256 "$DLMC_GEN_MODEL_SHA256" --update "$DLMC_GEN_UPDATE" \
    --eval-records "$DLMC_GEN_EVAL_RECORDS" --eval-records-sha256 "$DLMC_GEN_EVAL_RECORDS_SHA256" \
    --out "$DLMC_GEN_IDENTITY" || die "generator identity check failed"

JOBDIR=$DLMC_RETAINED/family_b/$RUN_NAME.jobs/job-$SLURM_JOB_ID
mkdir -p "$JOBDIR" "$DLMC_SCRATCH/runs/family_b" "$DLMC_RETAINED/family_b/logs" runs/slurm
echo "arm=$ARM config=$CONFIG run=$RUN_NAME init=$DLMC_GEN jobdir=$JOBDIR"

MARGIN=${DLMC_STOP_MARGIN_SECONDS:-1800}
# Array parent job IDs can return one EndTime line per task; take a single line (D-048).
END=$(squeue -h -j "$SLURM_JOB_ID" -o %e | sort -u | head -n 1)
[[ -n "$END" ]] || die "squeue returned no EndTime for job $SLURM_JOB_ID"
END_UNIX=$(date -d "$END" +%s) || die "cannot parse job end time '$END'"
STOP_AT=$(( END_UNIX - MARGIN ))
(( STOP_AT > $(date +%s) + 600 )) || die "less than 10 min before the stop deadline; raise -t"
echo "job ends $END; training deadline $(date -d @"$STOP_AT") (margin ${MARGIN}s)"

RUN_DIR=$DLMC_SCRATCH/runs/family_b/$RUN_NAME
DURABLE=$DLMC_RETAINED/family_b
RESUME_ARGS=()
if [[ "${DLMC_M7_RESUME:-0}" == 1 ]]; then
    RESUME_ARGS=(--resume auto)
    echo "resuming arm $ARM from local or durable latest"
else
    [[ ! -e "$RUN_DIR/checkpoints" ]] || die "$RUN_DIR already has checkpoints; set DLMC_M7_RESUME=1"
fi

copy_logs() { cp "runs/slurm/dlmc-m7-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$JOBDIR/" 2>/dev/null || true; }
trap copy_logs EXIT

run_step() {
    "$@" &
    local pid=$! rc=1 rc2
    # shellcheck disable=SC2064
    trap "echo 'batch shell: forwarding USR1 to $pid'; kill -USR1 $pid 2>/dev/null || true" USR1 TERM
    while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done
    wait "$pid" 2>/dev/null; rc2=$?
    [[ $rc2 -ne 127 ]] && rc=$rc2
    trap - USR1 TERM
    return "$rc"
}

# Same batch for every arm. Override only if 16 OOMs, and use that value for the whole array.
BATCH=${DLMC_M7_BATCH:-16}
UPDATE_ARGS=()
if [[ -n "${DLMC_MAX_UPDATES:-}" ]]; then
    UPDATE_ARGS=(--max-updates "$DLMC_MAX_UPDATES")
    echo "smoke cap DLMC_MAX_UPDATES=$DLMC_MAX_UPDATES"
fi

RC=0
run_step "$DLMC_CLI" train-family-b --config "$CONFIG" --device cuda \
    --run-dir "$RUN_DIR" --durable-dir "$DURABLE" \
    --init-checkpoint "$DLMC_GEN" --stop-at-unix "$STOP_AT" \
    --set "batch_size=${BATCH}" \
    "${UPDATE_ARGS[@]}" "${RESUME_ARGS[@]}" || RC=$?

mkdir -p "$DURABLE/$RUN_NAME"
for f in summary.json config.resolved.json data_identity.json metrics.jsonl events.jsonl; do
    [[ -f "$RUN_DIR/$f" ]] && cp -f "$RUN_DIR/$f" "$DURABLE/$RUN_NAME/"
done
cp -f "$RUN_DIR/summary.json" "$JOBDIR/" 2>/dev/null || true

if [[ $RC -ne 0 ]]; then
    die "train-family-b failed for arm $ARM (exit $RC); nothing is retried automatically"
fi
echo "ARM COMPLETE $ARM updates=$( "$DLMC_PY" -c "import json; print(json.load(open('$RUN_DIR/summary.json'))['update'])" )"
exit 0
