#!/bin/bash
#SBATCH -J dlmc-bank
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -o runs/slurm/%x-%j.out
#SBATCH --signal=B:USR1@900
#SBATCH --no-requeue
# RUNS ON: a Lonestar6 GPU compute node. Submit from $WORK/dlmc/repo (docs/milestone4_bank.md):
#   DLMC_BANK_CONFIG=configs/bank/<approved config>.yaml sbatch -A <allocation> [-t HH:MM:SS] scripts/tacc/bank_collect.slurm
#   resume a stopped collection: add DLMC_BANK_RESUME=1
#
# One job, no trial job and no retries:
#   1. GPU selection and checks; the generator's files against the identity recorded on the login
#      node (generator_identity.py); the generator's validation records against their pinned hash.
#   2. In one process: the startup gates (Gate D, same-forward recompute, independent single-example
#      check, reproduction of job 3595133's drafts, record invariants, RNG unchanged). On any failure
#      it stops before writing a shard; the preflight report stays in the job directory.
#   3. Collection of every planned shard not yet complete, until done or the deadline
#      (job end - DLMC_STOP_MARGIN_SECONDS); SIGUSR1 (15 min before the end) stops after the current shard.
#   4. When collection is complete: validate + finalize (MANIFEST.json, BANK_COMPLETE), step-0 signals,
#      diagnostics, examples report, and an integrity re-check of the finalized bank.
# Train and validation only; the test splits are never loaded.
source "$SLURM_SUBMIT_DIR/scripts/tacc/tacc_env.sh"
require_job
enter_repo
[[ -n "${DLMC_BANK_CONFIG:-}" && -f "$DLMC_BANK_CONFIG" ]] \
    || die "set DLMC_BANK_CONFIG to the approved bank config (configs/bank/...yaml)"
MARGIN=${DLMC_STOP_MARGIN_SECONDS:-2700}
select_gpu "$DLMC_BANK_ROOT/logs/gpu_inventory-$SLURM_JOB_ID.txt"
check_gpu
check_torch_gpu

NAME=$("$DLMC_PY" -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$DLMC_BANK_CONFIG")
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad bank name '$NAME'"
BANK=$DLMC_BANK_ROOT/$NAME
JOBDIR=$DLMC_BANK_ROOT/$NAME.jobs/job-$SLURM_JOB_ID
mkdir -p "$JOBDIR"
echo "bank $NAME -> $BANK; job records $JOBDIR; config $DLMC_BANK_CONFIG sha256 $(sha256sum "$DLMC_BANK_CONFIG" | cut -d' ' -f1)"
copy_logs() { cp "runs/slurm/dlmc-bank-$SLURM_JOB_ID.out" "$JOBDIR/" 2>/dev/null || true; }
trap copy_logs EXIT

RESUME=()
if [[ "${DLMC_BANK_RESUME:-0}" == 1 ]]; then
    [[ -f "$BANK/IDENTITY.json" ]] || die "DLMC_BANK_RESUME=1 but $BANK has no IDENTITY.json"
    RESUME=(--resume)
else
    [[ ! -e "$BANK" ]] || die "$BANK exists; set DLMC_BANK_RESUME=1 to continue it (never overwritten)"
fi

"$DLMC_PY" scripts/tacc/generator_identity.py --check --checkpoint "$DLMC_GEN" --model-sha256 "$DLMC_GEN_MODEL_SHA256" \
    --update "$DLMC_GEN_UPDATE" --eval-records "$DLMC_GEN_EVAL_RECORDS" \
    --eval-records-sha256 "$DLMC_GEN_EVAL_RECORDS_SHA256" --out "$DLMC_GEN_IDENTITY" \
    || die "generator identity check failed"

END=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
END_UNIX=$(date -d "$END" +%s) || die "cannot parse job end time '$END'"
STOP_AT=$(( END_UNIX - MARGIN ))
(( STOP_AT > $(date +%s) + 1800 )) || die "less than 30 min of collection time before the stop deadline; raise -t"
echo "job ends $END; collection deadline $(date -d @"$STOP_AT") (margin ${MARGIN}s)"

# Run a CLI step in the background so SIGUSR1/SIGTERM reach it; return its exit status.
# Call it as `run_step ... || RC=$?` (errexit is suspended inside the || list).
run_step() {
    "$@" &
    local pid=$! rc=1 rc2
    # shellcheck disable=SC2064  # expand $pid now
    trap "echo 'batch shell: forwarding USR1 to $pid'; kill -USR1 $pid 2>/dev/null || true" USR1 TERM
    while kill -0 "$pid" 2>/dev/null; do wait "$pid"; rc=$?; done
    wait "$pid" 2>/dev/null; rc2=$?
    [[ $rc2 -ne 127 ]] && rc=$rc2
    trap - USR1 TERM
    return "$rc"
}

COMMON=(--config "$DLMC_BANK_CONFIG" --data-dir "$DLMC_DATA_DIR")
if [[ -f "$BANK/BANK_COMPLETE" ]]; then
    echo "bank already finalized; skipping preflight, collection and validation"
else
    RC=0
    run_step "$DLMC_CLI" collect-drafts "${COMMON[@]}" --checkpoint "$DLMC_GEN" --device cuda --bank-dir "$BANK" \
        "${RESUME[@]}" --preflight-out "$JOBDIR/preflight" --reference-records "$DLMC_GEN_EVAL_RECORDS" \
        --stop-at-unix "$STOP_AT" || RC=$?
    "$DLMC_PY" scripts/tacc/bank_status.py "$BANK" > "$JOBDIR/status-after-collection.json" || true
    if [[ $RC -eq 2 ]]; then
        die "preflight failed: no shard was written; report in $JOBDIR/preflight/preflight.json"
    elif [[ $RC -eq 3 ]]; then
        echo "COLLECTION STOPPED before completion (deadline or signal); verified shards are kept."
        echo "Resume: DLMC_BANK_CONFIG=$DLMC_BANK_CONFIG DLMC_BANK_RESUME=1 sbatch -A <allocation> scripts/tacc/bank_collect.slurm"
        exit 3
    elif [[ $RC -ne 0 ]]; then
        die "collection failed (exit $RC); see this log and $BANK/status.json; nothing is retried"
    fi
    "$DLMC_CLI" validate-bank "${COMMON[@]}" --bank-dir "$BANK" --finalize \
        --reference-records "$DLMC_GEN_EVAL_RECORDS" || die "bank validation failed; see $BANK/validation.json"
fi
"$DLMC_PY" scripts/tacc/bank_status.py "$BANK" --verify || die "finalized bank does not verify"

# Derived diagnostics (separately re-runnable; failures here do not touch the bank).
STEP0_RC=0
run_step "$DLMC_CLI" bank-step0 --bank-dir "$BANK" --checkpoint "$DLMC_GEN" --device cuda --resume \
    --stop-at-unix $(( END_UNIX - 600 )) || STEP0_RC=$?
[[ $STEP0_RC -eq 0 ]] || echo "WARNING: step-0 signals incomplete (exit $STEP0_RC); re-run with DLMC_BANK_RESUME=1"
"$DLMC_CLI" bank-diagnostics --bank-dir "$BANK" || echo "WARNING: diagnostics failed"
"$DLMC_CLI" bank-report "${COMMON[@]}" --bank-dir "$BANK" || echo "WARNING: examples report failed verification"
"$DLMC_PY" scripts/tacc/bank_status.py "$BANK" | tee "$JOBDIR/status-final.json"
echo "BANK JOB DONE $NAME"
