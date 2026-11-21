#!/bin/bash
#SBATCH -J dlmc-m5
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o runs/slurm/%x-%A_%a.out
#SBATCH --signal=B:USR1@900
#SBATCH --no-requeue
#SBATCH --array=0-5
# RUNS ON: Lonestar6 GPU node. Submit from $WORK/dlmc/repo (docs/handoff.md, D-044):
#   sbatch -A ASC26098 scripts/tacc/correction_screen.slurm
#   resume one arm: DLMC_CORR_RESUME=1 DLMC_CORR_ARRAY=<index> sbatch -A ASC26098 \
#       --array=<index> scripts/tacc/correction_screen.slurm
#
# One array task per weighting arm. No separate pilot/trial job. No identity-changing retries.
# Arms (index): 0 uniform, 1 high_conf, 2 gen_difficulty, 3 current_loss, 4 low_conf, 5 shuffled_conf
source "$SLURM_SUBMIT_DIR/scripts/tacc/tacc_env.sh"
require_job
enter_repo
select_gpu "$DLMC_RETAINED/correction/logs/gpu_inventory-$SLURM_JOB_ID.txt"
check_gpu
check_torch_gpu

ARMS=(uniform high_conf gen_difficulty current_loss low_conf shuffled_conf)
IDX=${SLURM_ARRAY_TASK_ID:?array task id required}
(( IDX >= 0 && IDX < ${#ARMS[@]} )) || die "array index $IDX out of range 0..$((${#ARMS[@]} - 1))"
ARM=${ARMS[$IDX]}
CONFIG=configs/correction/m5_${ARM}.yaml
[[ -f "$CONFIG" ]] || die "missing $CONFIG"

RUN_NAME=$("$DLMC_PY" -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['run_name'])")
BANK=$DLMC_BANK_ROOT/m4-ckpt8000-v1
[[ -f "$BANK/BANK_COMPLETE" ]] || die "bank not finalized: $BANK"
"$DLMC_PY" scripts/tacc/generator_identity.py --check --checkpoint "$DLMC_GEN" \
    --model-sha256 "$DLMC_GEN_MODEL_SHA256" --update "$DLMC_GEN_UPDATE" \
    --eval-records "$DLMC_GEN_EVAL_RECORDS" --eval-records-sha256 "$DLMC_GEN_EVAL_RECORDS_SHA256" \
    --out "$DLMC_GEN_IDENTITY" || die "generator identity check failed"

JOBDIR=$DLMC_RETAINED/correction/$RUN_NAME.jobs/job-$SLURM_JOB_ID
mkdir -p "$JOBDIR" "$DLMC_SCRATCH/runs/correction" runs/slurm
echo "arm=$ARM config=$CONFIG run=$RUN_NAME bank=$BANK jobdir=$JOBDIR"

MARGIN=${DLMC_STOP_MARGIN_SECONDS:-1800}
# Array parent job IDs can return one EndTime line per task; take a single line (D-048).
END=$(squeue -h -j "$SLURM_JOB_ID" -o %e | sort -u | head -n 1)
[[ -n "$END" ]] || die "squeue returned no EndTime for job $SLURM_JOB_ID"
END_UNIX=$(date -d "$END" +%s) || die "cannot parse job end time '$END'"
STOP_AT=$(( END_UNIX - MARGIN ))
(( STOP_AT > $(date +%s) + 600 )) || die "less than 10 min before the stop deadline; raise -t"
echo "job ends $END; training deadline $(date -d @"$STOP_AT") (margin ${MARGIN}s)"

RUN_DIR=$DLMC_SCRATCH/runs/correction/$RUN_NAME
DURABLE=$DLMC_RETAINED/correction
RESUME_ARGS=()
if [[ "${DLMC_CORR_RESUME:-0}" == 1 ]]; then
    RESUME_ARGS=(--resume auto)
    echo "resuming arm $ARM from local or durable latest"
else
    [[ ! -e "$RUN_DIR/checkpoints" ]] || die "$RUN_DIR already has checkpoints; set DLMC_CORR_RESUME=1"
fi

copy_logs() { cp "runs/slurm/dlmc-m5-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "$JOBDIR/" 2>/dev/null || true; }
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

# Dual forward (L_mask + correction) OOM'd at batch 32 on A100 40GB (job 3600168).
# Default 16; override with DLMC_CORR_BATCH if needed (identical across arms).
BATCH=${DLMC_CORR_BATCH:-16}

RC=0
run_step "$DLMC_CLI" train-correction --config "$CONFIG" --device cuda \
    --run-dir "$RUN_DIR" --durable-dir "$DURABLE" --bank-dir "$BANK" \
    --init-checkpoint "$DLMC_GEN" --stop-at-unix "$STOP_AT" \
    --set "batch_size=${BATCH}" "eval.batch_size=${BATCH}" \
    "${RESUME_ARGS[@]}" || RC=$?

mkdir -p "$DURABLE/$RUN_NAME"
for f in summary.json config.resolved.json bank_identity.json repair_identity.json metrics.jsonl evals.jsonl events.jsonl; do
    [[ -f "$RUN_DIR/$f" ]] && cp -f "$RUN_DIR/$f" "$DURABLE/$RUN_NAME/"
done
"$DLMC_PY" scripts/tacc/correction_status.py "$RUN_DIR" --durable "$DURABLE/$RUN_NAME" \
    > "$JOBDIR/status.json" || true
cp -f "$RUN_DIR/summary.json" "$JOBDIR/" 2>/dev/null || true

if [[ $RC -ne 0 ]]; then
    die "train-correction failed for arm $ARM (exit $RC); nothing is retried automatically"
fi
echo "ARM COMPLETE $ARM updates=$( "$DLMC_PY" -c "import json; print(json.load(open('$RUN_DIR/summary.json'))['update'])" )"
exit 0
