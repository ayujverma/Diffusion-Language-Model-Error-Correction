#!/bin/bash
#SBATCH -J dlmc-m6-calls
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o runs/slurm/%x-%j.out
#SBATCH --no-requeue
# USER SUBMITS THIS. Milestone 6 has not been run on TACC. This script does not train.
# Val only. Do not retarget it at a test split or use the output to pick a test budget.
#
# Smoke (first 32 val problems, monotone vs proseco, budget 4):
#   sbatch -A ASC26098 scripts/tacc/call_budget_eval.slurm
# Family A consumer (no retraining), one of:
#   uniform high_conf gen_difficulty current_loss low_conf shuffled_conf
#   DLMC_M6_ARM=uniform sbatch -A ASC26098 scripts/tacc/call_budget_eval.slurm
# Or set DLMC_M6_CKPT to any checkpoint directory (generator or a corrector).
# Full val curve (the user decides the time limit; 02:00:00 may be too small):
#   DLMC_M6_N=all DLMC_M6_BUDGETS=1,2,4,8 \
#   DLMC_M6_METHODS=monotone,proseco,d3im,extra_unmask \
#   sbatch -A ASC26098 scripts/tacc/call_budget_eval.slurm
# omega / corrector steps default to the D-049 provisional values (2 and 4).
source "$SLURM_SUBMIT_DIR/scripts/tacc/tacc_env.sh"
require_job
enter_repo
select_gpu "$DLMC_RETAINED/m6/logs/gpu_inventory-calls-$SLURM_JOB_ID.txt"
check_gpu
check_torch_gpu

ARMS=(uniform high_conf gen_difficulty current_loss low_conf shuffled_conf)
if [[ -n "${DLMC_M6_CKPT:-}" ]]; then
    CKPT=$DLMC_M6_CKPT
elif [[ -n "${DLMC_M6_ARM:-}" ]]; then
    ok=0
    for a in "${ARMS[@]}"; do
        [[ "$DLMC_M6_ARM" == "$a" ]] && ok=1
    done
    [[ "$ok" == 1 ]] || die "DLMC_M6_ARM must be one of: ${ARMS[*]}"
    CKPT=$DLMC_RETAINED/correction/m5-${DLMC_M6_ARM}-v1/checkpoints/ckpt-0001000
else
    die "set DLMC_M6_CKPT or DLMC_M6_ARM (Family A ckpt-0001000; this script does not train)"
fi
[[ -d "$CKPT" && -f "$CKPT/COMPLETE.json" ]] || die "checkpoint missing or incomplete: $CKPT"

METHODS=${DLMC_M6_METHODS:-monotone,proseco}
BUDGETS=${DLMC_M6_BUDGETS:-4}
N=${DLMC_M6_N:-32}
OMEGA=${DLMC_M6_OMEGA:-2}
STEPS=${DLMC_M6_CORRECTOR_STEPS:-4}
OUT=$DLMC_RETAINED/m6/calls/${SLURM_JOB_ID}
mkdir -p "$OUT" "$DLMC_RETAINED/m6/logs"
NARGS=()
if [[ "$N" != "all" ]]; then
    NARGS=(--n-examples "$N")
fi
echo "m6 calls ckpt=$CKPT methods=$METHODS budgets=$BUDGETS n=$N out=$OUT"
"$DLMC_CLI" evaluate-calls --checkpoint "$CKPT" --data-dir "$DLMC_DATA_DIR" \
    --manifest-sha256 "$DLMC_MANIFEST_SHA256" --split val \
    --method "$METHODS" --budgets "$BUDGETS" --omega "$OMEGA" --corrector-steps "$STEPS" \
    --model-dir runs/hf/open-dcoder-0.5B --device cuda --batch-size 32 \
    "${NARGS[@]}" --out "$OUT" \
    || die "evaluate-calls failed"
echo "M6 CALLS COMPLETE (software run; not a selected primary budget; not a test split)"
exit 0
