"""Summarise a CUDA pilot (pilot.slurm): gates, measurements, and continuation cost estimates.

Gates (all must pass; declared in docs/warmup_continuation.md before any CUDA result):
  every pilot step exited 0; fixture check passed; compare-runs U vs R passed (exact example
  order/masks/mask rates/learning rates, loss rtol 1e-4, parameter atol 1e-5, divergence/update
  rtol 1e-3); U, U2 and R reproduce the Mac-predicted batches 1001-1010; the baseline
  monitoring evaluation produced a selection score on the pinned subset; checkpoint 1000 was
  byte-identical before and after.
U vs U2 (two identical CUDA runs) is reported as the run-to-run noise floor, not gated.

Writes <report-dir>/pilot_report.json and pilot_report.md, and PILOT_PASSED only if every gate passes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

MAX_TRAIN_SECONDS = 7200
ORIGIN, MAX_UPDATES, EVAL_EVERY, CKPT_EVERY = 1000, 10000, 1000, 500
CKPT_GB = 5.93
MPS_MONITOR_TL_SECONDS = 190.9      # configs/eval/monitor_val_stage2_first50.json estimate (MPS, T=L)
MPS_STAGE2_SECONDS = 1404.17        # runs/calibration/stage2-warmup1000 (MPS, 5 budgets, 2,000 problems)
STARTUP_SECONDS = 180
STOP_MARGIN_SECONDS = 1500


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x] if p.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--job-seconds", type=float, required=True)
    ap.add_argument("--su-per-node-hour", type=float, default=1.5,
                    help="gpu-a100-small charge rate; confirm against the current Lonestar6 user guide")
    a = ap.parse_args()
    R = Path(a.report_dir)

    steps = {}
    for line in (R / "steps.txt").read_text().splitlines():
        name, rc, sec = line.split()
        steps[name] = {"exit": int(rc), "seconds": int(sec)}
    fixture, bench = load(R / "fixture_check.json"), load(R / "benchmark.json")
    cur, cuu2 = load(R / "compare_U_R.json"), load(R / "compare_U_U2.json")
    expected = {r: load(R / f"expected_{r}.json") for r in ("U", "U2", "R")}
    m_sum, u_sum = load(R / "runs/M/summary.json"), load(R / "runs/U/summary.json")
    m_sel = load(R / "runs/M/selection.json")
    m_evals = jsonl(R / "runs/M/evals.jsonl")
    u_metrics = [r for r in jsonl(R / "runs/U/metrics.jsonl") if "loss" in r]

    gates = {
        "all_steps_exit_0": bool(steps) and all(s["exit"] == 0 for s in steps.values()),
        "fixture_check": bool(fixture and fixture.get("passed")),
        "resume_U_vs_R": bool(cur and cur.get("passed")),
        "expected_batches_U_U2_R": all(e and e.get("passed") for e in expected.values()),
        "baseline_monitoring_eval": bool(m_sel and [h["update"] for h in m_sel["history"]] == [ORIGIN]),
        "checkpoint_1000_unchanged": steps.get("origin_before", {}).get("exit") == 0
                                     and steps.get("origin_after", {}).get("exit") == 0,
    }
    passed = all(gates.values())

    # ---- measurements
    iter_s = None
    if len(u_metrics) >= 3:
        tus = [r["train_update_seconds"] for r in u_metrics]
        iter_s = statistics.median(b - a_ for a_, b in zip(tus[1:], tus[2:]))   # skip the first (warm-up) update
    step_s = statistics.median(r["step_seconds"] for r in u_metrics[1:]) if len(u_metrics) > 1 else None
    eval_s = m_sum["timers"]["eval_seconds"] if m_sum else None
    n_saves_u = len([e for e in jsonl(R / "runs/U/events.jsonl") if e.get("event") == "checkpoint_saved"])
    save_copy_s = u_sum["timers"]["checkpoint_seconds"] / n_saves_u if u_sum and n_saves_u else None
    meas = {
        "seconds_per_training_iteration_U_median": iter_s,
        "seconds_per_update_synchronized_U_median": step_s,
        "benchmark": None if not bench else {k: bench["results"][0].get(k) if "results" in bench else bench.get(k)
                                             for k in ("batch_shapes", "seconds_per_update_median",
                                                       "end_to_end_seconds_per_update_mean", "examples_per_second",
                                                       "real_tokens_per_second", "memory_after_load",
                                                       "memory_last_update", "optimizer_state_allocated_gb",
                                                       "optimizer_state_devices", "analytic_gb",
                                                       "warmup_steps_counted", "timed_updates")},
        "peak_memory_U": max((r.get("memory", {}).get("cuda_peak_reserved_gb") or 0) for r in u_metrics)
        if u_metrics else None,
        "baseline_monitoring_eval_seconds": eval_s,
        "baseline_selection_score": m_sel["history"][0]["score"] if m_sel and m_sel["history"] else None,
        "baseline_by_setting": (m_evals[-1].get("generation_by_setting") if m_evals else None),
        "baseline_masked_loss": m_evals[-1].get("masked_loss") if m_evals else None,
        "checkpoint_save_plus_durable_copy_seconds": save_copy_s,
        "fixture": None if not fixture else {k: fixture.get(k) for k in ("passed", "checks", "loss", "grad_norm")},
        "resume_U_vs_R": cur, "noise_floor_U_vs_U2": cuu2,
        "pilot_job_seconds": a.job_seconds,
        "pilot_su": round(max(a.job_seconds / 3600, 0.25) * a.su_per_node_hour, 3),  # 15-min minimum charge
        "step_seconds": steps,
    }

    # ---- provisional estimates (no early stop assumed: the upper bound)
    est = None
    if iter_s and eval_s and save_copy_s:
        n_more = min(MAX_UPDATES - ORIGIN, math.floor(MAX_TRAIN_SECONDS / iter_s))
        n_evals = 1 + n_more // EVAL_EVERY
        n_saves = n_more // CKPT_EVERY + (1 if n_more % CKPT_EVERY else 0)
        final_eval = 1 if n_more % EVAL_EVERY else 0
        work = (STARTUP_SECONDS + n_more * iter_s + (n_evals + final_eval) * eval_s + n_saves * save_copy_s)
        limit_h = math.ceil((work + STOP_MARGIN_SECONDS) * 1.15 / 1800) / 2
        retained = len({1500} | set(range(2000, ORIGIN + n_more + 1, 1000))) + 2
        stage2_s = MPS_STAGE2_SECONDS * eval_s / MPS_MONITOR_TL_SECONDS
        est = {
            "label": "PROVISIONAL: from 10-update pilot measurements, assumes no early stop",
            "additional_updates_within_2h_training_cap": n_more,
            "final_update_if_cap_binds": ORIGIN + n_more,
            "scheduled_evaluations_incl_baseline": n_evals, "checkpoint_saves": n_saves,
            "estimated_job_work_seconds": round(work),
            "recommended_slurm_time_limit_hours": limit_h,
            "continuation_su_upper_bound": round(limit_h * a.su_per_node_hour, 2),
            "retained_checkpoints_max": retained,
            "retained_storage_gb_max_per_copy": round(retained * CKPT_GB, 1),
            "final_eval_seconds_per_checkpoint_estimate": round(stage2_s),
            "final_eval_su_estimate_two_checkpoints": round((2 * stage2_s + 600) / 3600 * a.su_per_node_hour, 2),
            "su_rate_per_node_hour": a.su_per_node_hour,
        }
    rep = {"passed": passed, "gates": gates, "measurements": meas, "estimates": est}
    (R / "pilot_report.json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    lines = [f"# CUDA pilot report ({R.name})", "", f"**{'PASSED' if passed else 'FAILED'}**", "", "| gate | result |",
             "| --- | --- |"] + [f"| {k} | {'pass' if v else 'FAIL'} |" for k, v in gates.items()]
    if cur:
        lines += ["", f"U vs R: loss max rel {cur.get('loss_max_rel_diff')}, param max abs "
                      f"{cur.get('param_max_abs_diff')}, divergence/update {cur.get('divergence_rel_to_update')}"]
    if cuu2:
        lines += [f"U vs U2 (noise floor): loss max rel {cuu2.get('loss_max_rel_diff')}, param max abs "
                  f"{cuu2.get('param_max_abs_diff')}, divergence/update {cuu2.get('divergence_rel_to_update')}"]
    lines += ["", "```json", json.dumps({"measurements": {k: v for k, v in meas.items()
                                                        if k not in ("resume_U_vs_R", "noise_floor_U_vs_U2",
                                                                     "baseline_by_setting")},
                                         "estimates": est}, indent=2, default=str), "```"]
    (R / "pilot_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if passed:
        (R / "PILOT_PASSED").write_text("all gates passed\n")
    print("PILOT:", "PASSED" if passed else "FAILED", R)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
