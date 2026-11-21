#!/bin/bash
#SBATCH -J dlmc-cont
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -o runs/slurm/%x-%j.out
#SBATCH --signal=B:USR1@900
#SBATCH --no-requeue
# RUNS ON: a Lonestar6 GPU compute node. Submit from $WORK/dlmc/repo (docs/warmup_continuation.md):
#   DLMC_RESUME_FROM=<checkpoint dir> DLMC_PILOT=<passed pilot report dir> \
#   DLMC_BACKUP_RECORD=<verified Ranch backup record> sbatch -A <allocation> [-t HH:MM:SS] scripts/tacc/continuation.slurm
#
# Continues the common warmup from an explicitly named checkpoint (first job: the original
# checkpoint 1000 in $WORK/dlmc/transfer, never a pilot copy) with configs/train/warmup_continuation.yaml:
# at most 10,000 total updates, 7,200 s of training-update time counted from update 1000 across
# all jobs, early stopping after 3 scheduled evaluations without a strictly higher score.
# One attempt per submission: no retries, no requeue. The trainer stops itself before the Slurm
# limit (--stop-at-unix = job end - DLMC_STOP_MARGIN_SECONDS), saves, copies to $WORK and exits;
# SIGUSR1 (15 min before the end, or scancel --signal=USR1 --batch) does the same.
source "$SLURM_SUBMIT_DIR/scripts/tacc/tacc_env.sh"
require_job
enter_repo
[[ -n "${DLMC_RESUME_FROM:-}" ]] || die "set DLMC_RESUME_FROM to the checkpoint to resume (first job: $DLMC_ORIGIN)"
[[ -f "${DLMC_PILOT:-/nonexistent}/PILOT_PASSED" ]] || die "set DLMC_PILOT to a pilot report directory containing PILOT_PASSED"
require_marker "$DLMC_PILOT/ENV_CHECK_PASSED.used" "DLMC_PILOT (the pilot must have run with the current environment)"
grep -q "^BACKUP VERIFIED" "${DLMC_BACKUP_RECORD:-/nonexistent}" 2>/dev/null \
    || die "set DLMC_BACKUP_RECORD to the Ranch backup record of checkpoint 1000 (backup_to_ranch.sh output)"
MARGIN=${DLMC_STOP_MARGIN_SECONDS:-1500}
select_gpu "$DLMC_CONT_DURABLE/logs/gpu_inventory-$SLURM_JOB_ID.txt"
check_gpu
check_torch_gpu
verify_origin

STATUS=(scripts/tacc/continuation_status.py --run-dir "$DLMC_CONT_RUN_DIR" --durable "$DLMC_CONT_DURABLE"
        --origin "$DLMC_ORIGIN")
"$DLMC_PY" "${STATUS[@]}" --check-resume "$DLMC_RESUME_FROM"
RESUME=$DLMC_RESUME_FROM
if [[ "$(readlink -f "$DLMC_RESUME_FROM")" == "$(readlink -f "$DLMC_ORIGIN")" ]]; then
    RESUME=$DLMC_ORIGIN_REL      # the path recorded with the checkpoint's data and model identities
fi

# Array parent job IDs can return one EndTime line per task; take a single line (D-048).
END=$(squeue -h -j "$SLURM_JOB_ID" -o %e | sort -u | head -n 1)
[[ -n "$END" ]] || die "squeue returned no EndTime for job $SLURM_JOB_ID"
END_UNIX=$(date -d "$END" +%s) || die "cannot parse job end time '$END'"
STOP_AT=$(( END_UNIX - MARGIN ))
(( STOP_AT > $(date +%s) + 1800 )) || die "less than 30 min of training time before the stop deadline; raise -t"
echo "job ends $END; trainer deadline $(date -d @"$STOP_AT") (margin ${MARGIN}s)"

mkdir -p "$DLMC_CONT_RUN_DIR" "$DLMC_CONT_DURABLE/logs"
copy_logs() {
    rsync -a --exclude checkpoints "$DLMC_CONT_RUN_DIR/" "$DLMC_CONT_DURABLE/logs/" || echo "WARNING: log copy failed"
    cp "runs/slurm/dlmc-cont-$SLURM_JOB_ID.out" "$DLMC_CONT_DURABLE/logs/" 2>/dev/null || true
}
trap copy_logs EXIT

"$DLMC_CLI" train --config "$DLMC_CONT_CONFIG" --device cuda --run-dir "$DLMC_CONT_RUN_DIR" \
    --resume "$RESUME" --durable-dir "$DLMC_RETAINED" --stop-at-unix "$STOP_AT" &
PID=$!
trap 'echo "batch shell: forwarding USR1 to $PID"; kill -USR1 $PID 2>/dev/null || true' USR1 TERM
set +e
RC=1
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; RC=$?; done
wait "$PID" 2>/dev/null; RC2=$?
[[ $RC2 -ne 127 ]] && RC=$RC2
set -e
echo "trainer exit status $RC"
"$DLMC_PY" "${STATUS[@]}" || true
exit "$RC"
