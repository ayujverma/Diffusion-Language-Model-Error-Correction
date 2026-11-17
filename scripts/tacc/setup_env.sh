#!/bin/bash
# RUNS ON: Lonestar6, in a cheap interactive compute session (TACC installs environments from
# compute nodes): idev -p vm-small -N 1 -n 1 -t 00:30:00. Downloads and installs only; nothing
# imports torch here. Builds $SCRATCH/dlmc/venv: uv $DLMC_UV_VERSION (sha256-checked), CPython
# $DLMC_PYTHON_VERSION, and every package from scripts/tacc/requirements-tacc-lock.txt with
# --require-hashes. The GPU-side checks are in env_check.slurm.
#   bash $WORK/dlmc/repo/scripts/tacc/setup_env.sh
source "$(dirname "$(readlink -f "$0")")/tacc_env.sh"
cd "$DLMC_REPO"
LOCK=scripts/tacc/requirements-tacc-lock.txt
[[ -f "$LOCK" ]] || die "missing $LOCK"
[[ ! -e "$DLMC_VENV" ]] || die "$DLMC_VENV already exists; move it aside yourself to rebuild (nothing is deleted)"

TOOLS=$DLMC_ROOT/tools/uv-$DLMC_UV_VERSION
UV=$TOOLS/uv
if [[ ! -x "$UV" ]]; then
    mkdir -p "$TOOLS"
    TGZ=$TOOLS/uv-x86_64-unknown-linux-gnu.tar.gz
    curl -fsSL -o "$TGZ" \
        "https://github.com/astral-sh/uv/releases/download/$DLMC_UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz"
    check_sha "$TGZ" "$DLMC_UV_SHA256"
    tar -xzf "$TGZ" -C "$TOOLS" --strip-components=1
fi
"$UV" --version

mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"
"$UV" python install "$DLMC_PYTHON_VERSION"
"$UV" venv --python "$DLMC_PYTHON_VERSION" --python-preference only-managed "$DLMC_VENV"
"$UV" pip install --python "$DLMC_PY" --index-url https://pypi.org/simple \
    --require-hashes --no-deps -r "$LOCK"
# The project itself, built with the locked setuptools (no network build isolation).
"$UV" pip install --python "$DLMC_PY" --no-deps --no-build-isolation --no-index -e .

mkdir -p "$DLMC_ROOT/env"
"$UV" pip freeze --python "$DLMC_PY" > "$DLMC_ROOT/env/freeze.txt"
"$DLMC_PY" -c "import sys, platform; print(sys.version); print(platform.platform())" \
    | tee "$DLMC_ROOT/env/python.txt"
{
    echo "built $(date -u +%FT%TZ) on $(hostname)"
    echo "uv $("$UV" --version)"
    echo "repo commit $(git rev-parse HEAD)"
    sha256sum "$LOCK"
} > "$DLMC_ROOT/env/BUILD.txt"
cat "$DLMC_ROOT/env/BUILD.txt"
echo "environment ready; next: sbatch -A \$DLMC_ALLOCATION scripts/tacc/env_check.slurm (from $DLMC_REPO)"
