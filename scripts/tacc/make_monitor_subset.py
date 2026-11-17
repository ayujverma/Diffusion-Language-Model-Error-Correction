"""Fix the D-033 monitoring subset from the existing stage-2 calibration IDs (RUNS ON: the Mac).

The subset is the first N problems, in val-file order, of each of the 10 settings that
D-031 stage 2 selected; stage 2 itself took the first 200 per setting in file order, so
the subset is a prefix of each setting's stage-2 IDs. The script asserts that.

    .venv/bin/python scripts/tacc/make_monitor_subset.py --n-per-setting 50 \
        --stage2 runs/calibration/stage2-warmup1000/summary.json \
        --out configs/eval/monitor_val_stage2_first50.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from diffusion_correction.training.data import load_verified_examples, select_per_setting, setting_key


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage2", default="runs/calibration/stage2-warmup1000/summary.json")
    p.add_argument("--n-per-setting", type=int, default=50)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    raw = Path(a.stage2).read_bytes()
    s2 = json.loads(raw)
    meta = s2["meta"]
    ed = meta["eval_data"]
    settings = []
    for s in meta["settings"].split(","):
        fam, n, d = s.split(":")
        settings.append((fam, int(n), int(d)))
    examples, ident = load_verified_examples(ed["data_dir"], "val", ed["manifest_sha256"])
    chosen = select_per_setting(examples, a.n_per_setting, set(settings))
    stage2_ids = set(meta["selected_problem_ids"])
    assert all(e.problem_id in stage2_ids for e in chosen), "monitoring IDs must come from stage 2"
    counts: dict[tuple, int] = {}
    for e in chosen:
        counts[setting_key(e)] = counts.get(setting_key(e), 0) + 1
    assert set(counts) == set(settings) and set(counts.values()) == {a.n_per_setting}
    ids = [e.problem_id for e in chosen]
    rows = {(r["family"], r["length"], r["operand_digits"]): r for r in s2["rows"] if r["budget"] == "L"}
    mps_seconds = sum(rows[k]["model_seconds"] * a.n_per_setting / rows[k]["n_examples"] for k in settings)
    out = {
        "purpose": "D-033 monitoring subset for the warmup continuation (validation only)",
        "rule": f"first {a.n_per_setting} val problems in file order of each D-031 stage-2 setting",
        "source_stage2_summary": a.stage2,
        "source_stage2_summary_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_sha256": ident.manifest_sha256, "val_split_sha256": ident.split_sha256,
        "settings": [{"family": f, "length": n, "operand_digits": d, "n": counts[(f, n, d)]} for f, n, d in settings],
        "n_problems": len(ids),
        "problem_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "estimated_cost": {
            "budget": "L",
            "mps_model_seconds_scaled_from_stage2": round(mps_seconds, 1),
            "note": "stage-2 MPS T=L model seconds scaled to this subset; CUDA cost is measured by the pilot",
        },
        "problem_ids": ids,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "problem_ids"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
