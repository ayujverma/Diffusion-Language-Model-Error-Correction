#!/bin/bash
# RUNS ON: a Lonestar6 login node (file operations only; no model execution).
# Verifies the transferred files against the Mac manifest, makes them read-only, and links
# them into the repository at the same relative paths the code and checkpoints record.
#   bash $WORK/dlmc/repo/scripts/tacc/install_transfer.sh
source "$(dirname "$(readlink -f "$0")")/tacc_env.sh"

[[ -f "$DLMC_TRANSFER/runs/tacc_transfer/MANIFEST.sha256" ]] || die "no manifest under $DLMC_TRANSFER"
cd "$DLMC_TRANSFER"
echo "verifying $(wc -l < runs/tacc_transfer/MANIFEST.sha256) files (about 7.2 GB)..."
sha256sum --quiet --strict -c runs/tacc_transfer/MANIFEST.sha256 || die "transfer verification failed"
echo "TRANSFER VERIFIED"

read -r COMMIT BRANCH < runs/tacc_transfer/COMMIT
[[ -d "$DLMC_REPO/.git" ]] || die "clone first: git clone -b $BRANCH $DLMC_TRANSFER/runs/tacc_transfer/dlmc.bundle $DLMC_REPO"
HEAD=$(git -C "$DLMC_REPO" rev-parse HEAD)
[[ "$HEAD" == "$COMMIT" ]] || echo "NOTE: repo HEAD $HEAD differs from the transferred commit $COMMIT (newer code bundle?)"

chmod -R a-w "$DLMC_TRANSFER/runs"
echo "made $DLMC_TRANSFER/runs read-only"

link() {  # link <repo-relative path>
    local dst="$DLMC_REPO/$1" src="$DLMC_TRANSFER/$1"
    [[ -e "$src" ]] || die "missing $src"
    mkdir -p "$(dirname "$dst")"
    if [[ -L "$dst" ]]; then
        [[ "$(readlink "$dst")" == "$src" ]] || die "$dst links to $(readlink "$dst"), expected $src"
    elif [[ -e "$dst" ]]; then
        die "$dst exists and is not a link; not touching it"
    else
        ln -s "$src" "$dst"
    fi
    echo "linked $1"
}
link runs/hf/open-dcoder-0.5B
link runs/data/pilot-seed20261116
link runs/train/warmup-pilot-v1
link runs/calibration/stage2-warmup1000
link runs/calibration/screen-warmup1000
link runs/reference/cuda_pilot_fixture.pt
link runs/reference/cuda_pilot_fixture.json
link runs/reference/cuda_pilot_fixture_f64norms.pt
link runs/reference/cuda_pilot_fixture_f64norms.json
link runs/preflight/ckpt-0001000-mac.json

mkdir -p "$DLMC_REPO/runs/slurm" "$DLMC_RETAINED" "$DLMC_SCRATCH/runs" "$DLMC_SCRATCH/pilot"
{
    echo "verified $(date -u +%FT%TZ) on $(hostname)"
    echo "transfer commit $COMMIT ($BRANCH); repo HEAD $HEAD"
    sha256sum runs/tacc_transfer/MANIFEST.sha256
} > "$DLMC_ROOT/TRANSFER_VERIFIED.txt"
echo "wrote $DLMC_ROOT/TRANSFER_VERIFIED.txt"
