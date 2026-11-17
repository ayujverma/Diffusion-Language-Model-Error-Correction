"""Compare two `dlmc calibrate` outputs: checkpoint 1000 vs the selected checkpoint (same protocol).

Checks that both runs used the same split, settings, problems, budgets and sampler, then reports
per setting and budget: final-answer, value, token and trace accuracy with denominators,
malformed outputs, model calls and model/wall seconds; plus, per budget, the D-033 aggregate
(equal weight per family, equal weight per setting within a family). Optionally compares the
CUDA checkpoint-1000 run with the MPS stage-2 run (cross-device agreement, reported only).

Usage: python scripts/tacc/compare_calibrations.py BASE_DIR SELECTED_DIR --out OUT_DIR [--mps-base DIR]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

PROTOCOL_KEYS = ("split", "n_per_setting", "settings", "n_examples", "selected_problem_ids", "budgets", "sampler",
                 "batch_size", "checkpoint_trained_on_manifest")


def load(d: Path) -> tuple[dict, dict, list[dict]]:
    s = json.loads((d / "summary.json").read_text())
    rows = {(r["family"], r["length"], r["operand_digits"], str(r["budget"])): r for r in s["rows"]}
    recs = [json.loads(x) for x in (d / "records.jsonl").read_text().splitlines() if x]
    return s, rows, recs


def aggregate(rows: dict, budget: str) -> float:
    fams: dict[str, list[float]] = defaultdict(list)
    for (fam, _l, _d, b), r in rows.items():
        if b == budget:
            fams[fam].append(r["final_answer_accuracy"])
    return sum(sum(v) / len(v) for v in fams.values()) / len(fams)


def agreement(ra: list[dict], rb: list[dict]) -> dict:
    key = lambda r: (r["problem_id"], str(r["budget"]))  # noqa: E731
    a, b = {key(r): r for r in ra}, {key(r): r for r in rb}
    if set(a) != set(b):
        raise SystemExit("record sets differ")
    same = sum(a[k]["response_ids"] == b[k]["response_ids"] for k in a)
    return {"records": len(a), "identical_responses": same, "identical_fraction": same / len(a),
            "final_correct_changed": sum(a[k]["final_correct"] != b[k]["final_correct"] for k in a)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("selected")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mps-base", default=None)
    a = ap.parse_args()
    sa, ra, reca = load(Path(a.base))
    sb, rb, recb = load(Path(a.selected))
    mismatched = [k for k in PROTOCOL_KEYS if sa["meta"].get(k) != sb["meta"].get(k)]
    if mismatched or set(ra) != set(rb):
        raise SystemExit(f"protocol differs between runs: {mismatched or 'cells'}")
    ids = lambda recs: sorted({(r["problem_id"], str(r["budget"])) for r in recs})  # noqa: E731
    if ids(reca) != ids(recb):
        raise SystemExit("evaluated problems differ between runs")

    fields = ("n_examples", "final_correct", "final_answer_accuracy", "value_correct", "n_values", "value_accuracy",
              "token_correct", "n_tokens", "token_accuracy", "trace_correct", "trace_accuracy", "malformed_examples",
              "malformed_example_rate", "model_calls_total", "model_seconds", "wall_seconds")
    cells = [{"family": k[0], "length": k[1], "operand_digits": k[2], "budget": k[3],
              "base": {f: ra[k][f] for f in fields}, "selected": {f: rb[k][f] for f in fields},
              "final_answer_accuracy_diff": rb[k]["final_answer_accuracy"] - ra[k]["final_answer_accuracy"]}
             for k in sorted(ra)]
    budgets = sa["budgets"]
    agg = {b: {"base": aggregate(ra, b), "selected": aggregate(rb, b)} for b in budgets}
    rep = {"base": {"dir": a.base, "model": sa["meta"]["model"], "device": sa["meta"]["device"],
                    "total_wall_seconds": sa["total_wall_seconds"]},
           "selected": {"dir": a.selected, "model": sb["meta"]["model"], "device": sb["meta"]["device"],
                        "total_wall_seconds": sb["total_wall_seconds"]},
           "protocol": {k: sa["meta"].get(k) for k in PROTOCOL_KEYS if k != "selected_problem_ids"},
           "aggregate_final_answer_accuracy_equal_family_then_setting": agg,
           "base_vs_selected_response_agreement": agreement(reca, recb), "cells": cells}
    if a.mps_base:
        sm, rm, recm = load(Path(a.mps_base))
        if any(sm["meta"].get(k) != sa["meta"].get(k) for k in ("selected_problem_ids", "budgets", "settings")):
            raise SystemExit("MPS run used a different protocol")
        rep["cross_device_ckpt1000_mps_vs_cuda"] = {
            **agreement(recm, reca),
            "final_answer_accuracy_max_abs_diff": max(abs(rm[k]["final_answer_accuracy"] -
                                                          ra[k]["final_answer_accuracy"]) for k in ra)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")

    sel_update = sb["meta"]["model"].get("update")
    lines = [f"# Final validation comparison: checkpoint 1000 vs selected checkpoint {sel_update}", "",
             "Validation split only; same problems, budgets and sampler for both. Accuracy = correct / n.", "",
             "| budget | aggregate final (1000) | aggregate final (selected) |", "| --- | --- | --- |"]
    lines += [f"| {b} | {agg[b]['base']:.3f} | {agg[b]['selected']:.3f} |" for b in budgets]
    lines += ["", "| family | len | digits | T | final 1000 | final sel | value 1000 | value sel | token 1000 | "
                  "token sel | trace 1000 | trace sel | malformed 1000 | malformed sel | calls | model s 1000 | "
                  "model s sel |", "|" + " --- |" * 17]
    for c in cells:
        x, y = c["base"], c["selected"]
        lines.append(
            f"| {c['family']} | {c['length']} | {c['operand_digits']} | {c['budget']} | "
            f"{x['final_correct']}/{x['n_examples']} | {y['final_correct']}/{y['n_examples']} | "
            f"{x['value_correct']}/{x['n_values']} | {y['value_correct']}/{y['n_values']} | "
            f"{x['token_correct']}/{x['n_tokens']} | {y['token_correct']}/{y['n_tokens']} | "
            f"{x['trace_correct']}/{x['n_examples']} | {y['trace_correct']}/{y['n_examples']} | "
            f"{x['malformed_examples']} | {y['malformed_examples']} | {x['model_calls_total']} | "
            f"{x['model_seconds']:.1f} | {y['model_seconds']:.1f} |")
    lines += ["", f"Total wall seconds: 1000 {sa['total_wall_seconds']}, selected {sb['total_wall_seconds']}."]
    if "cross_device_ckpt1000_mps_vs_cuda" in rep:
        lines += ["", f"Checkpoint 1000, MPS stage 2 vs CUDA: {json.dumps(rep['cross_device_ckpt1000_mps_vs_cuda'])}"]
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
