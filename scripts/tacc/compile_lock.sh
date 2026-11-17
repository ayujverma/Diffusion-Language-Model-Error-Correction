#!/bin/bash
# RUNS ON: the Mac (or any machine with uv and network access). Not needed on TACC.
# Resolves the Lonestar6 lock file for linux x86_64 / CPython 3.14.7 with hashes.
set -euo pipefail
cd "$(dirname "$0")"
UV=${UV:-uv}
"$UV" pip compile requirements-tacc.in \
    --python-platform x86_64-manylinux_2_28 --python-version 3.14 \
    --generate-hashes --no-header --index-url https://pypi.org/simple \
    -o requirements-tacc-lock.txt
echo "wrote $(pwd)/requirements-tacc-lock.txt"
