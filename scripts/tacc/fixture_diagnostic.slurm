#!/bin/bash
#SBATCH -J dlmc-fixture-diag
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:45:00
#SBATCH -o runs/slurm/%x-%j.out
#SBATCH --no-requeue
# RUNS ON: a Lonestar6 GPU compute node. Submit from $WORK/dlmc/repo:
#   DLMC_MAC_DIAG=<verified Mac diagnostic dir> DLMC_PILOT_DIR=$WORK/dlmc/retained/pilot/<jobid> \
#       sbatch -A <allocation> scripts/tacc/fixture_diagnostic.slurm
#
# Diagnostic only (docs/warmup_continuation.md, "Fixture update-norm diagnostic"). It writes no
# ENV_CHECK_PASSED or PILOT_PASSED, does not run cuda_fixture.py, replaces no reference, saves no
# checkpoint and does not count toward the continuation. One disposable fp32 update from the
# original checkpoint 1000 (read-only, hashes checked before and after) per backend: TACC CPU,
# then TACC CUDA; then comparisons with each other and with the Mac CPU run, and a summary of
# the pilot's memory records.
source "$SLURM_SUBMIT_DIR/scripts/tacc/tacc_env.sh"
require_job
enter_repo
[[ -n "${DLMC_MAC_DIAG:-}" && -f "$DLMC_MAC_DIAG/diag.json" && -f "$DLMC_MAC_DIAG/SHA256SUMS" ]] \
    || die "set DLMC_MAC_DIAG to the transferred Mac CPU diagnostic directory (diag.json, SHA256SUMS)"
[[ -n "${DLMC_PILOT_DIR:-}" && -f "$DLMC_PILOT_DIR/steps.txt" ]] \
    || die "set DLMC_PILOT_DIR to the retained pilot report directory ($DLMC_RETAINED/pilot/<jobid>)"
(cd "$DLMC_MAC_DIAG" && sha256sum -c SHA256SUMS) || die "$DLMC_MAC_DIAG does not match its SHA256SUMS"
D=$DLMC_RETAINED/fixture_diag/$SLURM_JOB_ID
[[ ! -e "$D" ]] || die "$D already exists"
mkdir -p "$D"
select_gpu "$D/gpu_inventory.txt"
check_gpu 2>&1 | tee "$D/gpu.txt"
check_torch_gpu 2>&1 | tee -a "$D/gpu.txt"
verify_origin 2>&1 | tee "$D/origin_before.txt"

DIAG=("$DLMC_PY" scripts/tacc/fixture_update_diagnostic.py)
"${DIAG[@]}" run --device cpu --checkpoint "$DLMC_ORIGIN" --out "$D/tacc_cpu" 2>&1 | tee "$D/tacc_cpu.log"
"${DIAG[@]}" run --device cuda --checkpoint "$DLMC_ORIGIN" --out "$D/tacc_cuda" 2>&1 | tee "$D/tacc_cuda.log"
verify_origin 2>&1 | tee "$D/origin_after.txt"
compare() {  # compare <name> <run dir> <reference run dir>
    "${DIAG[@]}" compare "$2" "$3" --checkpoint "$DLMC_ORIGIN" --out "$D/compare_$1.json" 2>&1 \
        | tee "$D/compare_$1.log"
}
compare tacc_cuda_vs_mac "$D/tacc_cuda" "$DLMC_MAC_DIAG"
compare tacc_cpu_vs_mac "$D/tacc_cpu" "$DLMC_MAC_DIAG"
compare tacc_cuda_vs_tacc_cpu "$D/tacc_cuda" "$D/tacc_cpu"
PLOG=()
for f in runs/slurm/dlmc-pilot-"$(basename "$DLMC_PILOT_DIR")".{out,err}; do [[ -f $f ]] && PLOG+=(--log "$f"); done
"$DLMC_PY" scripts/tacc/summarize_pilot_memory.py --pilot-dir "$DLMC_PILOT_DIR" "${PLOG[@]}" \
    --out "$D/memory_summary.json" 2>&1 | tee "$D/memory_summary.log"
echo "diagnostic outputs in $D (diagnostic only: no pass marker written)"
