#!/bin/bash
#SBATCH -J dlmc-m7-calls
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o runs/slurm/%x-%j.out
#SBATCH --no-requeue
# NOT SUBMITTED. The user submits this. Account: sbatch -A ASC26098 scripts/tacc/family_b_eval.slurm
# Does not train Family A. Does not read test_id, test_long, or test_unseen. Split is val.
# omega and S are the D-049 defaults (2 and 4). They are not a selected budget.
# Budgets default to 1,2,4,8. This list is not a selected primary budget.
# DLMC_M6_N=32 is a smoke; set DLMC_M6_N=all for the full val split. The user chooses.
#
# Family B checkpoint (ckpt-0001000 after the 1000-update screen):
#   DLMC_M7_ARM=proseco_basic sbatch -A ASC26098 scripts/tacc/family_b_eval.slurm
#   arms: proseco_basic proseco_error_prioritized proseco_gen_conf scope
# Family A consumer, no retraining:
#   DLMC_M6_ARM=uniform sbatch -A ASC26098 scripts/tacc/family_b_eval.slurm
# Or one explicit checkpoint (a smoke ckpt, the generator, or either family):
#   DLMC_M6_CKPT=/path/to/ckpt sbatch -A ASC26098 scripts/tacc/family_b_eval.slurm
source "$SLURM_SUBMIT_DIR/scripts/tacc/tacc_env.sh"
require_job
enter_repo
select_gpu "$DLMC_RETAINED/family_b/logs/gpu_inventory-calls-$SLURM_JOB_ID.txt"
check_gpu
check_torch_gpu

M7_ARMS=(proseco_basic proseco_error_prioritized proseco_gen_conf scope)
M5_ARMS=(uniform high_conf gen_difficulty current_loss low_conf shuffled_conf)
if [[ -n "${DLMC_M6_CKPT:-}" ]]; then
    CKPT=$DLMC_M6_CKPT
    TAG=explicit
elif [[ -n "${DLMC_M7_ARM:-}" ]]; then
    ok=0
    for a in "${M7_ARMS[@]}"; do
        [[ "$DLMC_M7_ARM" == "$a" ]] && ok=1
    done
    [[ "$ok" == 1 ]] || die "DLMC_M7_ARM must be one of: ${M7_ARMS[*]}"
    CKPT=$DLMC_RETAINED/family_b/m7-${DLMC_M7_ARM}-v1/checkpoints/ckpt-0001000
    TAG=m7-$DLMC_M7_ARM
elif [[ -n "${DLMC_M6_ARM:-}" ]]; then
    ok=0
    for a in "${M5_ARMS[@]}"; do
        [[ "$DLMC_M6_ARM" == "$a" ]] && ok=1
    done
    [[ "$ok" == 1 ]] || die "DLMC_M6_ARM must be one of: ${M5_ARMS[*]}"
    CKPT=$DLMC_RETAINED/correction/m5-${DLMC_M6_ARM}-v1/checkpoints/ckpt-0001000
    TAG=m5-$DLMC_M6_ARM
else
    die "set DLMC_M7_ARM, DLMC_M6_ARM, or DLMC_M6_CKPT (this script does not train)"
fi
[[ -d "$CKPT" && -f "$CKPT/COMPLETE.json" ]] || die "checkpoint missing or incomplete: $CKPT"

METHODS=${DLMC_M6_METHODS:-monotone,proseco,d3im}
BUDGETS=${DLMC_M6_BUDGETS:-1,2,4,8}
N=${DLMC_M6_N:-32}
OMEGA=${DLMC_M6_OMEGA:-2}
STEPS=${DLMC_M6_CORRECTOR_STEPS:-4}
OUT=$DLMC_RETAINED/family_b/calls/${TAG}-${SLURM_JOB_ID}
mkdir -p "$OUT" "$DLMC_RETAINED/family_b/logs"
NARGS=()
if [[ "$N" != "all" ]]; then
    NARGS=(--n-examples "$N")
fi
echo "m7 calls ckpt=$CKPT methods=$METHODS budgets=$BUDGETS n=$N omega=$OMEGA S=$STEPS out=$OUT"
echo "budgets are not a selected primary budget; split is val"
"$DLMC_CLI" evaluate-calls --checkpoint "$CKPT" --data-dir "$DLMC_DATA_DIR" \
    --manifest-sha256 "$DLMC_MANIFEST_SHA256" --split val \
    --method "$METHODS" --budgets "$BUDGETS" --omega "$OMEGA" --corrector-steps "$STEPS" \
    --model-dir runs/hf/open-dcoder-0.5B --device cuda --batch-size 32 \
    "${NARGS[@]}" --out "$OUT" \
    || die "evaluate-calls failed"
echo "M7 CALLS COMPLETE (val only; not a selected primary budget; not a test split; not Family A training)"
exit 0
