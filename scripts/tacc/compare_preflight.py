"""Compare a Lonestar6 preflight report with the Mac one for the same checkpoint.

Everything except elapsed time must match: hashes, counters, sampler and RNG stream states,
optimizer/scheduler state, learning-rate projection and the predicted next batches. Floats
(e.g. the float64 distance from the pretrained weights) may differ by CPU summation order
only, so they are compared with a relative tolerance of 1e-9.

Usage: python scripts/tacc/compare_preflight.py MAC.json TACC.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

IGNORED = {"seconds"}
FLOAT_RTOL = 1e-9


def diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k in IGNORED:
                continue
            if k not in a or k not in b:
                out.append(f"{path}/{k}: present on one side only")
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if math.isclose(a, b, rel_tol=FLOAT_RTOL, abs_tol=0.0) else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def main() -> int:
    mac, tacc = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    problems = diff(mac, tacc)
    if not tacc.get("passed"):
        problems.insert(0, "TACC preflight did not pass its own checks")
    for p in problems:
        print(p)
    print("PREFLIGHT MATCHES MAC:", "PASS" if not problems else f"FAIL ({len(problems)} differences)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
