"""Mutation check for the D-033 continuation rules (same harness as mutation_check_m3.py).

Usage: .venv/bin/python scripts/mutation_check_d033.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mutation_check_m3 import main  # noqa: E402

TESTS = ["tests/test_continuation.py"]
MUTANTS = [
    ("ties count as an improvement", "training/trainer.py",
     "improved = s[\"best_score\"] is None or score > s[\"best_score\"]",
     "improved = s[\"best_score\"] is None or score >= s[\"best_score\"]"),
    ("resume resets the training-update timer", "training/trainer.py",
     "self.timers = dict(state.get(\"timers\") or {**self.timers, \"origin_update\": self.update})",
     "self.timers = {**self.timers, \"origin_update\": self.update}"),
    ("resume resets the early-stopping state", "training/trainer.py",
     "        self.selection = state.get(\"selection\")\n", "        self.selection = None\n"),
    ("evaluation time is charged to the training timer", "training/trainer.py",
     "        self.timers[\"eval_seconds\"] += time.perf_counter() - t0",
     "        self.timers[\"train_update_seconds\"] += 1e6"),
    ("monitoring score pools examples instead of weighting settings", "evaluation/generation.py",
     "    per_family = {f: sum(v) / len(v) for f, v in sorted(fams.items())}",
     "    per_family = {f: sum(r[\"final_correct\"] for r in rows if r[\"family\"] == f) / "
     "sum(r[\"n_examples\"] for r in rows if r[\"family\"] == f) for f in sorted(fams)}"),
    ("deadline ignored", "training/trainer.py",
     "        if limits[\"stop_at_unix\"] is not None and time.time() >= limits[\"stop_at_unix\"]:",
     "        if False:"),
    ("best checkpoint not protected from pruning", "training/trainer.py",
     "            names.add(f\"ckpt-{self.selection['best_update']:07d}\")", "            pass"),
]

if __name__ == "__main__":
    raise SystemExit(main(MUTANTS, TESTS))
