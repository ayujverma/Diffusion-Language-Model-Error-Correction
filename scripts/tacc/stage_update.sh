#!/bin/bash
# RUNS ON: the Mac, from the repository root. Builds a code-only update for the Lonestar6 checkout:
# an incremental git bundle from the commit already on TACC (default: the transferred commit in
# runs/tacc_transfer/COMMIT) to HEAD, with its sha256. The data, checkpoint and the original
# transfer are not touched or restaged. Writes only runs/tacc_update/<base>..<head>/.
#
#   scripts/tacc/stage_update.sh
#   DLMC_REMOTE=<user>@ls6.tacc.utexas.edu DLMC_REMOTE_ROOT=<$WORK on ls6>/dlmc \
#       scripts/tacc/stage_update.sh          # also prints the copy and apply commands
set -euo pipefail
die() { echo "ERROR: $*" >&2; exit 1; }

cd "$(git rev-parse --show-toplevel)"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || die "commit or stash tracked changes first"
BRANCH=$(git branch --show-current)
[[ -n "$BRANCH" ]] || die "detached HEAD: check out a branch"
HEAD=$(git rev-parse HEAD)
if [[ -n "${DLMC_UPDATE_BASE:-}" ]]; then
    BASE=$(git rev-parse --verify "$DLMC_UPDATE_BASE^{commit}")
else
    [[ -f runs/tacc_transfer/COMMIT ]] || die "set DLMC_UPDATE_BASE to the commit checked out on TACC"
    read -r BASE _ < runs/tacc_transfer/COMMIT
fi
[[ "$BASE" != "$HEAD" ]] || die "HEAD is the base commit $BASE: nothing to update"
git merge-base --is-ancestor "$BASE" "$HEAD" || die "$BASE is not an ancestor of HEAD: no fast-forward update"

NAME=${BASE:0:7}..${HEAD:0:7}
STAGE=runs/tacc_update/$NAME
[[ ! -e "$STAGE" ]] || die "$STAGE exists; move it aside to re-stage"
mkdir -p "$STAGE"
git bundle create "$STAGE/dlmc-update.bundle" "$BASE..$BRANCH"
git bundle verify "$STAGE/dlmc-update.bundle" >/dev/null
echo "$BASE $HEAD $BRANCH" > "$STAGE/COMMITS"
(cd "$STAGE" && shasum -a 256 dlmc-update.bundle COMMITS > SHA256SUMS)
git diff --stat "$BASE" "$HEAD" > "$STAGE/DIFFSTAT.txt"
echo "update $BASE -> $HEAD ($BRANCH), $(git rev-list --count "$BASE..$HEAD") commits, \
$(stat -f %z "$STAGE/dlmc-update.bundle") bytes"
cat "$STAGE/SHA256SUMS"

if [[ -z "${DLMC_REMOTE:-}" || -z "${DLMC_REMOTE_ROOT:-}" ]]; then
    die "staged, but DLMC_REMOTE (user@ls6.tacc.utexas.edu) and DLMC_REMOTE_ROOT (\$WORK/dlmc on ls6, \
from 'echo \$WORK' after logging in) are not set, so no copy command is printed"
fi
U=$DLMC_REMOTE_ROOT/updates/$NAME
REPO=$DLMC_REMOTE_ROOT/repo
cat <<EOF

Copy (one ssh session):
  COPYFILE_DISABLE=1 tar --no-mac-metadata --no-xattrs -cf - -C runs/tacc_update $NAME \\
    | ssh $DLMC_REMOTE "mkdir -p $DLMC_REMOTE_ROOT/updates && tar -xf - -C $DLMC_REMOTE_ROOT/updates"
Then on an ls6 login node (docs/warmup_continuation.md, section 11):
  cd $U && sha256sum --strict -c SHA256SUMS
  test "\$(git -C $REPO rev-parse HEAD)" = $BASE
  test -z "\$(git -C $REPO status --porcelain --untracked-files=no)"
  git -C $REPO bundle verify $U/dlmc-update.bundle
  git -C $REPO fetch $U/dlmc-update.bundle $BRANCH:refs/remotes/update/$BRANCH
  git -C $REPO merge --ff-only $HEAD
  test "\$(git -C $REPO rev-parse HEAD)" = $HEAD && echo "UPDATE APPLIED $HEAD"
EOF
