"""Check that a run's batches match the preflight's predicted next batches (device-independent).

The preflight (scripts/tacc/preflight_checkpoint.py) replays the checkpoint's CPU data-order and
mask streams. Example order, problem IDs, mask positions and mask rates come only from those
streams, so a CUDA run resumed from the same checkpoint must reproduce them exactly.

Usage: python scripts/tacc/check_expected_batches.py --preflight PRE.json --metrics RUN/metrics.jsonl [--lr 3e-5]
Exit status 0 only if every predicted batch is present and identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--preflight", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--lr", type=float, default=None, help="expected learning rate for these batches")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    expected = json.loads(Path(a.preflight).read_text())["expected_next_batches"]
    recs = {r["batch"]: r for r in (json.loads(x) for x in Path(a.metrics).read_text().splitlines() if x)}
    rows, ok = [], True
    for e in expected:
        r = recs.get(e["batch"])
        if r is None:
            rows.append({"batch": e["batch"], "present": False})
            ok = False
            continue
        got = {"epoch": r["epoch"], "example_indices": r["example_indices"],
               "problem_ids_sha256": sha("\n".join(r["problem_ids"]).encode()),
               "mask_digest": r["mask_digest"], "n_masked": r["n_masked"],
               "mask_t_sha256": sha(json.dumps(r["mask_t"]).encode()), "shape": r["shape"]}
        bad = sorted(k for k in got if got[k] != e[k])
        if a.lr is not None and r.get("lr") != a.lr:
            bad.append("lr")
        ok = ok and not bad
        rows.append({"batch": e["batch"], "present": True, "mismatched_fields": bad,
                     "mask_digest": r["mask_digest"], "n_masked": r["n_masked"], "lr": r.get("lr"),
                     "loss": r.get("loss")})
    rep = {"preflight": a.preflight, "metrics": a.metrics, "n_expected": len(expected), "batches": rows,
           "passed": ok}
    text = json.dumps(rep, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    print("EXPECTED BATCHES:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
