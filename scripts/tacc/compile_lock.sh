#!/bin/bash
# RUNS ON: the Mac (or any machine with uv and network access). Not needed on TACC.
# Resolves the Lonestar6 lock file for linux x86_64 / CPython 3.14.7 with hashes.
# constraints-from-mac-lock.txt is the Mac lock without its torch pin (TACC uses the D-034
# torch build); every other shared package keeps its Mac version.
set -euo pipefail
cd "$(dirname "$0")"
UV=${UV:-uv}
grep -v '^torch==' ../../requirements-lock.txt > constraints-from-mac-lock.txt
[[ $(wc -l < constraints-from-mac-lock.txt) -eq $(( $(wc -l < ../../requirements-lock.txt) - 1 )) ]] \
    || { echo "ERROR: expected exactly one torch== line in requirements-lock.txt" >&2; exit 1; }
"$UV" pip compile requirements-tacc.in \
    --python-platform x86_64-manylinux_2_28 --python-version 3.14 \
    --generate-hashes --no-header --index-url https://pypi.org/simple \
    -o requirements-tacc-lock.txt
echo "wrote $(pwd)/requirements-tacc-lock.txt"
