#!/usr/bin/env python3
"""Status helper for milestone-5 correction runs (login node safe; no model load)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _checkpoints(root: Path) -> list[str]:
    ck = root / "checkpoints"
    if not ck.exists():
        return []
    return sorted(p.name for p in ck.glob("ckpt-*") if p.is_dir() and not p.name.endswith(".tmp"))


def status(run_dir: Path, durable: Path | None = None) -> dict:
    run_dir = Path(run_dir)
    out = {"run_dir": str(run_dir), "durable": str(durable) if durable else None}
    for name in ("summary.json", "config.resolved.json", "bank_identity.json", "repair_identity.json"):
        out[name] = _read_json(run_dir / name)
    metrics = run_dir / "metrics.jsonl"
    evals = run_dir / "evals.jsonl"
    out["n_metrics"] = sum(1 for _ in metrics.open()) if metrics.exists() else 0
    out["n_evals"] = sum(1 for _ in evals.open()) if evals.exists() else 0
    if evals.exists() and out["n_evals"]:
        last = None
        for line in evals.read_text().splitlines():
            if line:
                last = json.loads(line)
        out["last_eval"] = last
    out["local_checkpoints"] = _checkpoints(run_dir)
    latest = run_dir / "checkpoints" / "latest"
    out["latest"] = latest.read_text().strip() if latest.exists() else None
    if durable:
        durable = Path(durable)
        out["durable_checkpoints"] = _checkpoints(durable)
        out["durable_summary"] = _read_json(durable / "summary.json")
        dl = durable / "checkpoints" / "latest"
        out["durable_latest"] = dl.read_text().strip() if dl.exists() else None
    return out


def screen_overview(root: Path) -> dict:
    """Summarize all m5-*-v1 arms under a retained/correction root."""
    root = Path(root)
    arms = ["uniform", "high_conf", "gen_difficulty", "current_loss", "low_conf", "shuffled_conf"]
    rows = []
    for arm in arms:
        name = f"m5-{arm}-v1"
        durable = root / name
        st = status(durable if durable.exists() else root / name, durable if durable.exists() else None)
        summary = st.get("summary.json") or st.get("durable_summary")
        repair = (st.get("last_eval") or {}).get("repair")
        rows.append({
            "arm": arm, "run_name": name,
            "update": (summary or {}).get("update"),
            "stop_reason": (summary or {}).get("stop_reason"),
            "latest": st.get("durable_latest") or st.get("latest"),
            "checkpoints": st.get("durable_checkpoints") or st.get("local_checkpoints"),
            "last_repair": repair,
        })
    return {"root": str(root), "arms": rows}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None, help="one arm's run directory")
    p.add_argument("--durable", default=None, help="durable copy root for that arm")
    p.add_argument("--screen", default=None, help="retained/correction root: summarize all arms")
    args = p.parse_args(argv)
    if args.screen:
        print(json.dumps(screen_overview(Path(args.screen)), indent=2, default=str))
        return 0
    if not args.run_dir:
        p.error("run_dir or --screen is required")
    print(json.dumps(status(Path(args.run_dir), Path(args.durable) if args.durable else None),
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
