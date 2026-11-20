#!/bin/bash
#SBATCH -J dlmc-m5-repair
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -o runs/slurm/%x-%A_%a.out
#SBATCH --no-requeue
#SBATCH --array=0-5
# RUNS ON: Lonestar6 GPU node. After the screen completes, evaluate repair on each arm's
# final checkpoint against identical val bank drafts at the primary budget (D-044: T=4).
#   sbatch -A ASC26098 scripts/tacc/repair_eval.slurm
# Optional: DLMC_CORR_CKPT=/path/to/ckpt-NNNNNNN to override the selected checkpoint per arm
# (same relative path under each arm's durable run); default = durable latest.
source "$SLURM_SUBMIT_DIR/scripts/tacc/tacc_env.sh"
require_job
enter_repo
select_gpu "$DLMC_RETAINED/correction/logs/gpu_inventory-repair-$SLURM_JOB_ID.txt"
check_gpu
check_torch_gpu

ARMS=(uniform high_conf gen_difficulty current_loss low_conf shuffled_conf)
IDX=${SLURM_ARRAY_TASK_ID:?array task id required}
ARM=${ARMS[$IDX]}
RUN_NAME=m5-${ARM}-v1
BANK=$DLMC_BANK_ROOT/m4-ckpt8000-v1
DURABLE=$DLMC_RETAINED/correction/$RUN_NAME
IDENT=e3e810bef987753ebdb8c9cf002de0f113eb9c4e5c61843094893ca5b22203ea
MANIFEST=c5901617bbc6467483d1a91d8679c881dc055d62bbe993aa03fe181092e71ac4
BUDGET=${DLMC_REPAIR_BUDGET:-4}

if [[ -n "${DLMC_CORR_CKPT:-}" ]]; then
    CKPT=$DURABLE/checkpoints/$DLMC_CORR_CKPT
else
    [[ -f "$DURABLE/checkpoints/latest" ]] || die "no durable latest for $RUN_NAME under $DURABLE/checkpoints"
    CKPT=$DURABLE/checkpoints/$(cat "$DURABLE/checkpoints/latest")
fi
[[ -d "$CKPT" && -f "$CKPT/COMPLETE.json" ]] || die "checkpoint missing or incomplete: $CKPT"

OUT=$DLMC_RETAINED/correction/repair/$RUN_NAME
mkdir -p "$OUT" runs/slurm
echo "repair arm=$ARM ckpt=$CKPT budget=$BUDGET out=$OUT"

"$DLMC_CLI" evaluate-repair --checkpoint "$CKPT" --bank-dir "$BANK" \
    --identity-sha256 "$IDENT" --manifest-sha256 "$MANIFEST" \
    --split val --budget "$BUDGET" --verify-shards \
    --data-dir "$DLMC_DATA_DIR" --model-dir runs/hf/open-dcoder-0.5B \
    --device cuda --batch-size 32 --out "$OUT" \
    || die "evaluate-repair failed for $ARM"
echo "REPAIR COMPLETE $ARM"
exit 0
