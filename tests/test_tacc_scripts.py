"""TACC helper scripts on real tiny-backbone D-033 checkpoints: the resume guard and best-checkpoint
lookup (continuation_status.py) and the pilot report's gates and estimate arithmetic."""

import json
import subprocess
import sys
from pathlib import Path

from test_continuation import make_pre_d033, scripted
from test_trainer import make_trainer, tiny_cfg

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "scripts" / "tacc" / "continuation_status.py"


def status(tmp, origin, *extra):
    info = json.loads((origin / "COMPLETE.json").read_text())
    r = subprocess.run([sys.executable, str(STATUS), "--run-dir", str(tmp / "run"),
                        "--durable", str(tmp / "keep" / "t"), "--origin", str(origin), "--origin-update", "4",
                        "--origin-model-sha256", info["files"]["model.safetensors"], *extra],
                       capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def cont_cfg(max_updates):
    return tiny_cfg(eval={"every_updates": 2, "generate_budgets": ["L"],
                          "selection": {"budget": "L", "patience": 2}},
                    checkpoint={"every_updates": 2, "keep_last": 1, "durable_dir": None},
                    limits={"max_updates": max_updates, "max_train_seconds": 1e6, "train_seconds_origin_update": 4})


def test_resume_guard_and_best_checkpoint(tmp_path):
    first = make_trainer(tiny_cfg(limits={"max_updates": 4}, checkpoint={"every_updates": 4}), tmp_path / "w")
    first.train()
    origin = tmp_path / "w" / "checkpoints" / "ckpt-0000004"
    make_pre_d033(origin)
    assert status(tmp_path, origin, "--check-resume", str(origin))[0] == 0

    scores = {4: 0.1, 6: 0.5, 8: 0.2}  # job 1 gets a SIGUSR1-style stop request during the evaluation at 6
    c = cont_cfg(20)
    c["checkpoint"]["durable_dir"] = str(tmp_path / "keep")
    job1 = scripted(c, tmp_path / "run", scores)
    orig_eval = job1.evaluate

    def evaluate(scheduled=None):
        ev = orig_eval(scheduled)
        if job1.update == 6:
            job1.stop_signal = "SIGUSR1"
        return ev

    job1.evaluate = evaluate
    job1.resume(origin)
    assert job1.train()["stop_reason"] == "signal_SIGUSR1" and job1.update == 6

    rc, out = status(tmp_path, origin, "--check-resume", str(origin))
    assert rc != 0 and "already has checkpoints up to 6" in out
    rc, out = status(tmp_path, origin, "--best")
    assert rc != 0 and "not finished" in out
    ck6 = tmp_path / "keep" / "t" / "checkpoints" / "ckpt-0000006"
    assert status(tmp_path, origin, "--check-resume", str(ck6))[0] == 0

    c8 = cont_cfg(8)
    c8["checkpoint"]["durable_dir"] = str(tmp_path / "keep")
    job2 = scripted(c8, tmp_path / "run", scores)
    job2.resume(ck6)
    assert job2.train()["stop_reason"] == "max_updates" and job2.update == 8

    rc, out = status(tmp_path, origin, "--check-resume", str(ck6))
    assert rc != 0 and "would roll back" in out
    ck8 = tmp_path / "keep" / "t" / "checkpoints" / "ckpt-0000008"
    rc, out = status(tmp_path, origin, "--check-resume", str(ck8))
    assert rc != 0 and "already stopped (max_updates)" in out
    rc, out = status(tmp_path, origin, "--best")
    assert rc == 0 and out == str(ck6)  # best = 6 (0.5); 8 scored 0.2

    (ck6 / "rng.pt").write_bytes(b"corrupt")
    rc, out = status(tmp_path, origin, "--best")
    assert rc != 0 and "sha256" in out


def test_pilot_report_gates_and_estimates(tmp_path):
    R = tmp_path
    names = ["origin_before", "copy", "fixture", "bench", "M", "U", "U2", "R1", "R2", "compare_U_R",
             "compare_U_U2", "expected_U", "expected_U2", "expected_R", "origin_after"]
    (R / "steps.txt").write_text("".join(f"{n} 0 10\n" for n in names))
    ok = {"passed": True}
    for n in ("compare_U_R", "compare_U_U2", "expected_U", "expected_U2", "expected_R"):
        (R / f"{n}.json").write_text(json.dumps(ok))
    (R / "fixture_check.json").write_text(json.dumps({"passed": True, "checks": {}}))
    (R / "benchmark.json").write_text(json.dumps({"results": [{"seconds_per_update_median": 1.4}]}))
    for d in ("M", "U"):
        (R / "runs" / d).mkdir(parents=True)
    (R / "runs/M/summary.json").write_text(json.dumps({"timers": {"eval_seconds": 95.45}}))
    (R / "runs/M/selection.json").write_text(json.dumps({"history": [{"update": 1000, "score": 0.4}]}))
    (R / "runs/M/evals.jsonl").write_text(json.dumps({"masked_loss": 0.6}) + "\n")
    (R / "runs/U/metrics.jsonl").write_text("".join(
        json.dumps({"loss": 0.6, "step_seconds": 1.4, "train_update_seconds": 3.0 + 1.5 * k}) + "\n"
        for k in range(10)))
    (R / "runs/U/events.jsonl").write_text(json.dumps({"event": "checkpoint_saved"}) + "\n"
                                           + json.dumps({"event": "checkpoint_saved"}) + "\n")
    (R / "runs/U/summary.json").write_text(json.dumps({"timers": {"checkpoint_seconds": 120.0}}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/tacc/pilot_report.py"), "--report-dir", str(R),
                        "--job-seconds", "3600"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((R / "pilot_report.json").read_text())
    e = rep["estimates"]
    assert e["additional_updates_within_2h_training_cap"] == 4800 and e["final_update_if_cap_binds"] == 5800
    assert e["scheduled_evaluations_incl_baseline"] == 5 and e["checkpoint_saves"] == 10
    # 180 + 4800*1.5 + (5 + 1 unscheduled)*95.45 + 10*60
    assert e["estimated_job_work_seconds"] == round(180 + 7200 + 6 * 95.45 + 600)
    assert e["retained_checkpoints_max"] == 7  # 1500, 2000..5000, plus keep_last 2
    assert (R / "PILOT_PASSED").exists() and rep["measurements"]["pilot_su"] == 3.0  # 1 h on a full gpu-a100 node
    assert e["su_rate_per_node_hour"] == 3.0
    assert e["continuation_su_upper_bound"] == e["recommended_slurm_time_limit_hours"] * 3.0

    (R / "PILOT_PASSED").unlink()
    (R / "expected_R.json").write_text(json.dumps({"passed": False}))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/tacc/pilot_report.py"), "--report-dir", str(R),
                        "--job-seconds", "3600"], capture_output=True, text=True)
    assert r.returncode == 1 and not (R / "PILOT_PASSED").exists()
