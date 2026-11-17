"""Status, resume guard and best-checkpoint lookup for the warmup continuation (D-033).

Reads only JSON and hashes files; never imports torch, so it is safe on a login node.

  status:        python scripts/tacc/continuation_status.py --run-dir RUN --durable DUR --origin CKPT1000
  resume guard:  ... --check-resume PATH      (exit 0 only if PATH may be resumed; see below)
  best:          ... --best                   (prints the verified best checkpoint; continuation must be finished)

Resume guard. PATH must verify against its COMPLETE.json, and either
  * PATH is the original checkpoint 1000 (pinned model sha256) and the continuation has no
    checkpoints yet, or
  * PATH is a continuation checkpoint (timer origin 1000) and is the newest complete
    checkpoint in the durable and local directories.
Resuming anything older would roll back the update counter, the training-time cap or the
early-stopping state. A continuation that already reached a stopping rule is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ORIGIN_UPDATE = 1000
ORIGIN_MODEL_SHA256 = "f48abe457d483b19b854e1fdf6be4bb294c702f949f9d76d7c44a2274b0d5a6a"
TERMINAL = ("max_updates", "early_stop_patience", "max_train_seconds")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(ck: Path) -> dict:
    info = json.loads((ck / "COMPLETE.json").read_text())
    for name, digest in info["files"].items():
        got = sha256(ck / name)
        if got != digest:
            raise SystemExit(f"REFUSED: {ck / name} sha256 {got} != recorded {digest}")
    return info


def checkpoints(root: Path) -> dict[int, Path]:
    d = root / "checkpoints"
    if not d.is_dir():
        return {}
    return {int(p.name.split("-")[1]): p for p in d.iterdir()
            if p.name.startswith("ckpt-") and not p.name.endswith(".tmp") and (p / "COMPLETE.json").exists()}


def finished(state: dict) -> str | None:
    lim, t, s = state["config"]["limits"], state.get("timers") or {}, state.get("selection")
    if state["update"] >= lim["max_updates"]:
        return "max_updates"
    if s and s["evals_without_improvement"] >= s["patience"]:
        return "early_stop_patience"
    if lim.get("max_train_seconds") is not None and t.get("train_update_seconds", 0) >= lim["max_train_seconds"]:
        return "max_train_seconds"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--durable", required=True, help="<durable_dir>/<run_name>")
    ap.add_argument("--origin", required=True, help="original checkpoint 1000")
    ap.add_argument("--origin-update", type=int, default=ORIGIN_UPDATE)
    ap.add_argument("--origin-model-sha256", default=ORIGIN_MODEL_SHA256)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check-resume", default=None)
    g.add_argument("--best", action="store_true")
    a = ap.parse_args()
    run, dur, origin = Path(a.run_dir), Path(a.durable), Path(a.origin)

    local, durable = checkpoints(run), checkpoints(dur)
    newest = max([*local, *durable], default=None)
    newest_path = (durable.get(newest) or local.get(newest)) if newest is not None else None
    latest_state = json.loads((newest_path / "state.json").read_text()) if newest_path else None
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else None

    if a.check_resume:
        ck = Path(a.check_resume)
        info = verify(ck)
        state = json.loads((ck / "state.json").read_text())
        if os.path.realpath(ck) == os.path.realpath(origin):
            if info["files"]["model.safetensors"] != a.origin_model_sha256 or state["update"] != a.origin_update:
                raise SystemExit("REFUSED: the origin checkpoint is not the pinned checkpoint 1000")
            if newest is not None:
                raise SystemExit(f"REFUSED: the continuation already has checkpoints up to {newest} "
                                 f"({newest_path}); resume that one, not checkpoint 1000")
        else:
            if (state.get("timers") or {}).get("origin_update") != a.origin_update:
                raise SystemExit(f"REFUSED: not a continuation checkpoint (training-time origin is not update "
                                 f"{a.origin_update})")
            if state["update"] != newest:
                raise SystemExit(f"REFUSED: {ck} is update {state['update']} but update {newest} exists "
                                 f"({newest_path}); resuming an older checkpoint would roll back the cap")
            done = finished(state)
            if done:
                raise SystemExit(f"REFUSED: the continuation already stopped ({done}); nothing to resume")
        print(f"RESUME OK: {ck} (update {state['update']}, model {info['files']['model.safetensors']})")
        return 0

    sel = (latest_state or {}).get("selection")
    if a.best:
        if latest_state is None:
            raise SystemExit("no continuation checkpoints")
        done = finished(latest_state)
        if not done:
            raise SystemExit(f"continuation not finished (latest update {newest}); refusing to name a best checkpoint")
        best = sel["best_update"]
        path = origin if best == a.origin_update else (durable.get(best) or local.get(best))
        if path is None:
            raise SystemExit(f"best checkpoint {best} not found")
        verify(path)
        print(path)
        return 0

    rep = {"local_checkpoints": sorted(local), "durable_checkpoints": sorted(durable), "newest": newest,
           "newest_path": str(newest_path) if newest_path else None,
           "timers": (latest_state or {}).get("timers"), "selection": sel,
           "finished": finished(latest_state) if latest_state else None,
           "last_summary": {k: summary.get(k) for k in ("stop_reason", "update", "last_checkpoint", "timers")}
           if summary else None}
    print(json.dumps(rep, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
