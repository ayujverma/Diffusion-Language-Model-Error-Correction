#!/bin/bash
# RUNS ON: a Lonestar6 login node (file transfer only). Archives verified files to Ranch and
# verifies the archived copy. Prints "BACKUP VERIFIED" and writes a record only after the sha256
# computed on Ranch matches the local one.
#
#   bash scripts/tacc/backup_to_ranch.sh ckpt1000                 # checkpoint 1000 + its run logs, data,
#                                                                 # calibration, fixture, manifest, code bundle
#   bash scripts/tacc/backup_to_ranch.sh ckpt8000                 # milestone-4 generator + continuation logs,
#                                                                 # final-eval records, generator identity
#   bash scripts/tacc/backup_to_ranch.sh bank:<bank name>         # a finalized draft bank (re-verified first)
#   bash scripts/tacc/backup_to_ranch.sh correction:m5            # milestone-5 screen + repair under retained/correction
#   bash scripts/tacc/backup_to_ranch.sh <name> <path under $WORK/dlmc> [...]   # e.g. later retained checkpoints
#
# Needs $ARCHIVER and $ARCHIVE (set by TACC on login nodes) and a Ranch allocation.
# The Hugging Face backbone is not archived: it is public and pinned by sha256 in the manifest.
source "$(dirname "$(readlink -f "$0")")/tacc_env.sh"
[[ -n "${ARCHIVER:-}" && -n "${ARCHIVE:-}" ]] || die "ARCHIVER/ARCHIVE are not set; ask TACC for Ranch access"
NAME=${1:?usage: backup_to_ranch.sh <name> [paths relative to \$WORK/dlmc]}
shift
if [[ "$NAME" == ckpt1000 && $# -eq 0 ]]; then
    set -- transfer/runs/train/warmup-pilot-v1 transfer/runs/data transfer/runs/calibration \
           transfer/runs/reference transfer/runs/preflight transfer/runs/tacc_transfer
    cd "$DLMC_TRANSFER"
    sha256sum --quiet --strict -c runs/tacc_transfer/MANIFEST.sha256 || die "transfer no longer verifies"
elif [[ "$NAME" == ckpt8000 && $# -eq 0 ]]; then
    # The milestone-4 generator (D-038), the continuation's logs and evaluation records, and its identity.
    cd "$DLMC_REPO"
    "$DLMC_PY" scripts/tacc/generator_identity.py --check --checkpoint "$DLMC_GEN" \
        --model-sha256 "$DLMC_GEN_MODEL_SHA256" --update "$DLMC_GEN_UPDATE" --eval-records "$DLMC_GEN_EVAL_RECORDS" \
        --eval-records-sha256 "$DLMC_GEN_EVAL_RECORDS_SHA256" --out "$DLMC_GEN_IDENTITY" \
        || die "checkpoint 8000 does not verify against its recorded identity"
    set -- retained/warmup-pilot-v1-cont/checkpoints/ckpt-0008000 retained/warmup-pilot-v1-cont/logs \
           retained/final_eval retained/generator
elif [[ "$NAME" == bank:* && $# -eq 0 ]]; then
    BANK_NAME=${NAME#bank:}
    NAME=bank-$BANK_NAME
    cd "$DLMC_REPO"
    "$DLMC_PY" scripts/tacc/bank_status.py "$DLMC_BANK_ROOT/$BANK_NAME" --verify || die "bank does not verify"
    set -- "retained/bank/$BANK_NAME" "retained/bank/$BANK_NAME.jobs"
elif [[ "$NAME" == correction:m5 && $# -eq 0 ]]; then
    # Milestone-5 screen outputs (all arms + repair records). Re-verify is left to the user.
    cd "$DLMC_REPO"
    [[ -d "$DLMC_RETAINED/correction" ]] || die "missing $DLMC_RETAINED/correction"
    set -- retained/correction
fi
(( $# > 0 )) || die "no paths given"
cd "$DLMC_ROOT"
for p in "$@"; do [[ -e "$p" ]] || die "missing $DLMC_ROOT/$p"; done

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TAR=dlmc-$NAME-$STAMP.tar
STAGE=$DLMC_SCRATCH/ranch-staging
mkdir -p "$STAGE" "$DLMC_RETAINED/backups"
tar -cf "$STAGE/$TAR" "$@"
tar -tvf "$STAGE/$TAR" > "$DLMC_RETAINED/backups/$TAR.list"
LOCAL_SHA=$(sha256sum "$STAGE/$TAR" | cut -d' ' -f1)
SIZE=$(stat -c %s "$STAGE/$TAR")
echo "archive $TAR: $SIZE bytes, sha256 $LOCAL_SHA"

ssh "$ARCHIVER" "mkdir -p '$ARCHIVE/dlmc'"
scp "$STAGE/$TAR" "$ARCHIVER:$ARCHIVE/dlmc/$TAR"
REMOTE_SHA=$(ssh "$ARCHIVER" "sha256sum '$ARCHIVE/dlmc/$TAR'" | cut -d' ' -f1)
REC=$DLMC_RETAINED/backups/$TAR.record
if [[ "$REMOTE_SHA" != "$LOCAL_SHA" ]]; then
    echo "BACKUP FAILED: Ranch sha256 '$REMOTE_SHA' != local $LOCAL_SHA" | tee "$REC.failed"
    exit 1
fi
{
    echo "BACKUP VERIFIED $(date -u +%FT%TZ)"
    echo "ranch $ARCHIVER:$ARCHIVE/dlmc/$TAR"
    echo "bytes $SIZE"
    echo "sha256 $LOCAL_SHA (computed locally and on Ranch)"
    echo "contents $*"
} > "$REC"
cat "$REC"
echo "record: $REC  (staging copy $STAGE/$TAR can be removed by you once you are satisfied)"
