"""Apply the D-031 stage-2 selection rule to a stage-1 calibration screen.

Usage: .venv/bin/python scripts/select_calibration_settings.py runs/calibration/<screen>/summary.json

Prints the selected settings with the reason for each and the `--settings` argument for
`dlmc calibrate`. Uses only the screen's final-answer accuracy.
"""

from __future__ import annotations

import json
import sys

BAND = (0.40, 0.85)
CENTRE = 0.625
SELECT_BUDGETS = ("1", "2", "4", "8")
PER_FAMILY = 5


def select(rows: list[dict]) -> list[dict]:
    acc: dict[tuple, dict[str, float]] = {}
    for r in rows:
        acc.setdefault((r["family"], r["length"], r["operand_digits"]), {})[r["budget"]] = r["final_answer_accuracy"]
    chosen = []
    for fam in sorted({k[0] for k in acc}):
        keys = sorted(k for k in acc if k[0] == fam)
        picked: list[tuple] = []
        reasons: dict[tuple, str] = {}
        for k, why in (((fam, min(k[1] for k in keys), min(k[2] for k in keys)), "easiest"),
                       ((fam, max(k[1] for k in keys), max(k[2] for k in keys)), "hardest")):
            if k in acc and k not in picked:
                picked.append(k)
                reasons[k] = why
        band = []
        for k in keys:
            if k in picked:
                continue
            dists = [abs(acc[k][b] - CENTRE) for b in SELECT_BUDGETS
                     if b in acc[k] and acc[k][b] is not None and BAND[0] <= acc[k][b] <= BAND[1]]
            if dists:
                band.append((min(dists), k[1], k[2], k))
        free = PER_FAMILY - len(picked)
        for _, _, _, k in sorted(band):
            if len(picked) >= PER_FAMILY:
                break
            picked.append(k)
            reasons[k] = "in band"
        if len(band) < free:
            mid = (fam, sorted({k[1] for k in keys})[len({k[1] for k in keys}) // 2],
                   sorted({k[2] for k in keys})[len({k[2] for k in keys}) // 2])
            if mid in acc and mid not in picked:
                picked.append(mid)
                reasons[mid] = "middle (fewer in-band settings than free slots)"
        for k in picked:
            chosen.append({"family": k[0], "length": k[1], "operand_digits": k[2], "reason": reasons[k],
                           "screen_final_accuracy": acc[k]})
    return chosen


def main() -> int:
    summary = json.loads(open(sys.argv[1]).read())
    chosen = select(summary["rows"])
    for c in chosen:
        print(json.dumps(c))
    print("--settings " + ",".join(f"{c['family']}:{c['length']}:{c['operand_digits']}" for c in chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
