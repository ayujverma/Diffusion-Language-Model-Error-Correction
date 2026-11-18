#!/bin/bash
#SBATCH -J dlmc-finaleval
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o runs/slurm/%x-%j.out
#SBATCH --no-requeue
# RUNS ON: a Lonestar6 GPU compute node, only after the continuation has stopped by one of its
# rules (max updates, early-stopping patience, training-time cap). Submit from $WORK/dlmc/repo:
#   sbatch -A <allocation> scripts/tacc/final_eval.slurm
# Full stage-2 calibration protocol (validation only; 10 settings x 200 problems x budgets
# 1,2,4,8,L, batch 50) for the original checkpoint 1000 and the selected best checkpoint on the
# same CUDA setup, then compare_calibrations.py. Test splits are never loaded (dlmc calibrate refuses them).
source "$SLURM_SUBMIT_DIR/scripts/tacc/tacc_env.sh"
require_job
enter_repo
select_gpu "$DLMC_RETAINED/final_eval/gpu_inventory-$SLURM_JOB_ID.txt"
check_gpu
check_torch_gpu
verify_origin
BEST=$("$DLMC_PY" scripts/tacc/continuation_status.py --run-dir "$DLMC_CONT_RUN_DIR" --durable "$DLMC_CONT_DURABLE" \
       --origin "$DLMC_ORIGIN" --best) || die "no verified best checkpoint (is the continuation finished?)"
echo "selected checkpoint: $BEST"
OUT=$DLMC_RETAINED/final_eval
[[ ! -e "$OUT/comparison" ]] || die "$OUT/comparison exists; not overwriting"
mkdir -p "$OUT"

calib() {  # calib <checkpoint> <out>
    [[ -e "$2/summary.json" ]] && { echo "exists, reusing: $2"; return 0; }
    [[ ! -e "$2" ]] || die "$2 exists without summary.json (interrupted run); move it aside"
    "$DLMC_CLI" calibrate --checkpoint "$1" --data-dir "$DLMC_DATA_DIR" --manifest-sha256 "$DLMC_MANIFEST_SHA256" \
        --split val --settings "$DLMC_STAGE2_SETTINGS" --n-per-setting 200 --budgets 1,2,4,8,L --batch-size 50 \
        --trace-examples 2 --trace-calls 4 --device cuda --out "$2"
}
calib "$DLMC_ORIGIN_REL" "$OUT/ckpt-0001000-cuda"
if [[ "$(readlink -f "$BEST")" == "$(readlink -f "$DLMC_ORIGIN")" ]]; then
    echo "the selected checkpoint is checkpoint 1000: no later checkpoint scored strictly higher"
    SEL=$OUT/ckpt-0001000-cuda
else
    SEL=$OUT/selected-$(basename "$BEST")-cuda
    calib "$BEST" "$SEL"
fi
"$DLMC_PY" scripts/tacc/compare_calibrations.py "$OUT/ckpt-0001000-cuda" "$SEL" --out "$OUT/comparison" \
    --mps-base runs/calibration/stage2-warmup1000
