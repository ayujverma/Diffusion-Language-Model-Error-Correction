#!/bin/bash
# RUNS ON: the Mac, from the repository root. Builds the minimum transfer set for Lonestar6:
# a git bundle of the committed code, the exact file list, and a sha256 manifest. Nothing is
# copied or modified outside runs/tacc_transfer/.
#
#   scripts/tacc/stage_transfer.sh
#   DLMC_REMOTE=<user>@ls6.tacc.utexas.edu DLMC_REMOTE_DIR=/work/<nnnnn>/<user>/ls6/dlmc/transfer \
#       scripts/tacc/stage_transfer.sh        # also prints the copy command
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

cd "$(git rev-parse --show-toplevel)"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || die "commit or stash tracked changes first"
COMMIT=$(git rev-parse HEAD)
BRANCH=$(git branch --show-current)
[[ -n "$BRANCH" ]] || die "detached HEAD: check out a branch"

STAGE=runs/tacc_transfer
mkdir -p "$STAGE"
[[ ! -e "$STAGE/MANIFEST.sha256" ]] || die "$STAGE/MANIFEST.sha256 exists; move $STAGE aside to re-stage"

CK=runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000
check() { [[ "$(shasum -a 256 "$1" | cut -d' ' -f1)" == "$2" ]] || die "sha256 mismatch: $1"; }
check $CK/COMPLETE.json 145dea5b78ac29e7a618574402e333a9a510a1617289de7f0ee72a8297452001
check $CK/model.safetensors f48abe457d483b19b854e1fdf6be4bb294c702f949f9d76d7c44a2274b0d5a6a
check $CK/optimizer.pt ae55b39481da056d6b4180cd0197a732efecaece7b17c0d0eab52aa0fbfa99af
check runs/data/pilot-seed20261116/manifest.json 990ecd1dd2341b81bc742fa91189b129877cd9b48b196fbdbc208e8586c81536

{
    for f in COMPLETE.json model.safetensors optimizer.pt rng.pt state.json; do echo "$CK/$f"; done
    for f in config.resolved.json config.sha256 metrics.jsonl evals.jsonl events.jsonl summary.json; do
        echo "runs/train/warmup-pilot-v1/$f"
    done
    for f in README.md __init__.py added_tokens.json config.json generation_config.json generation_utils.py \
             merges.txt model.safetensors modeling_qwen2.py special_tokens_map.json tokenizer.json \
             tokenizer_config.json vocab.json; do
        echo "runs/hf/open-dcoder-0.5B/$f"
    done
    for f in manifest.json provenance.json train.jsonl val.jsonl test_id.jsonl test_long.jsonl test_unseen.jsonl \
             tokenizer_reference_check.json tokenizer_reference_run.log; do
        echo "runs/data/pilot-seed20261116/$f"
    done
    for f in summary.json summary.md records.jsonl traces.txt; do echo "runs/calibration/stage2-warmup1000/$f"; done
    echo runs/calibration/screen-warmup1000/stage2_selection.txt
    echo runs/reference/cuda_pilot_fixture.pt
    echo runs/reference/cuda_pilot_fixture.json
    echo runs/reference/cuda_pilot_fixture_f64norms.pt
    echo runs/reference/cuda_pilot_fixture_f64norms.json
    echo runs/preflight/ckpt-0001000-mac.json
} > "$STAGE/FILES.data.txt"
while read -r f; do [[ -f "$f" ]] || die "missing $f"; done < "$STAGE/FILES.data.txt"

git bundle create "$STAGE/dlmc.bundle" "$BRANCH"
git bundle verify "$STAGE/dlmc.bundle" >/dev/null
echo "$COMMIT $BRANCH" > "$STAGE/COMMIT"

{ cat "$STAGE/FILES.data.txt"; echo "$STAGE/dlmc.bundle"; echo "$STAGE/COMMIT"; } > "$STAGE/FILES.txt"
echo "hashing $(wc -l < "$STAGE/FILES.txt") files..."
while read -r f; do shasum -a 256 "$f"; done < "$STAGE/FILES.txt" > "$STAGE/MANIFEST.sha256.tmp"
while read -r f; do stat -f '%z %N' "$f"; done < "$STAGE/FILES.txt" > "$STAGE/SIZES.txt"
mv "$STAGE/MANIFEST.sha256.tmp" "$STAGE/MANIFEST.sha256"
TOTAL=$(awk '{s+=$1} END {print s}' "$STAGE/SIZES.txt")
echo "$STAGE/MANIFEST.sha256" >> "$STAGE/FILES.txt"
echo "commit $COMMIT ($BRANCH); $(wc -l < "$STAGE/FILES.txt") files; $TOTAL bytes"

if [[ -z "${DLMC_REMOTE:-}" || -z "${DLMC_REMOTE_DIR:-}" ]]; then
    die "staged, but DLMC_REMOTE (user@ls6.tacc.utexas.edu) and DLMC_REMOTE_DIR (\$WORK/dlmc/transfer on ls6, \
from 'echo \$WORK' after logging in) are not set, so no copy command is printed"
fi
cat <<EOF

Copy (one ssh session; re-run the same command if it is interrupted):
  COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -cf - -T $STAGE/FILES.txt \\
    | ssh $DLMC_REMOTE "mkdir -p $DLMC_REMOTE_DIR && tar -xf - -C $DLMC_REMOTE_DIR"
Then on an ls6 login node (docs/warmup_continuation.md, step 3):
  cd $DLMC_REMOTE_DIR && sha256sum --quiet -c runs/tacc_transfer/MANIFEST.sha256 && echo TRANSFER VERIFIED
  git clone -b $BRANCH $DLMC_REMOTE_DIR/runs/tacc_transfer/dlmc.bundle $DLMC_REMOTE_DIR/../repo
  bash $DLMC_REMOTE_DIR/../repo/scripts/tacc/install_transfer.sh
EOF
