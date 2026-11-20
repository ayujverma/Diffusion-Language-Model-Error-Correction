"""Progress and integrity of a milestone-4 draft bank. Reads JSON and hashes files only (no torch).

  status:  python scripts/tacc/bank_status.py BANK_DIR
  verify:  python scripts/tacc/bank_status.py BANK_DIR --verify
           (finalized bank: re-hash MANIFEST.json against BANK_COMPLETE and every shard against
           MANIFEST.json; exit 0 and print "BANK VERIFIED" only if all match)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from diffusion_correction.drafts.identity import sha256_file


def status(root: Path) -> dict:
    out: dict = {"bank_dir": str(root), "exists": root.exists()}
    if not root.exists():
        return out
    plan = json.loads((root / "plan.json").read_text()) if (root / "plan.json").exists() else None
    sidecars = sorted((root / "shards").rglob("*.json")) if (root / "shards").exists() else []
    out["shards_complete"] = len(sidecars)
    if plan:
        out["shards_planned"] = len(plan["shards"])
        out["drafts_planned"] = plan["totals"]["drafts"]
        done = {str(p.relative_to(root / "shards"))[:-5] for p in sidecars}
        nxt = [s["id"] for s in plan["shards"] if s["id"] not in done]
        out["next_shard"] = nxt[0] if nxt else None
    out["drafts_complete"] = sum(json.loads(p.read_text())["n_records"] for p in sidecars)
    out["model_seconds"] = round(sum(json.loads(p.read_text())["model_seconds_with_signals"] for p in sidecars), 1)
    out["bytes"] = sum((p.with_suffix(".jsonl")).stat().st_size for p in sidecars if p.with_suffix(".jsonl").exists())
    for name in ("status.json", "validation.json"):
        if (root / name).exists():
            d = json.loads((root / name).read_text())
            out[name] = {k: d.get(k) for k in ("state", "passed", "n_problems", "missing_shards", "error",
                                               "shards_complete", "job_id", "finished_utc", "totals",
                                               "reproduction_vs_reference_records") if k in d}
    out["incomplete_dirs"] = sorted(p.name for p in (root / "incomplete").iterdir()) \
        if (root / "incomplete").exists() else []
    out["bank_complete"] = (root / "BANK_COMPLETE").exists()
    if out["bank_complete"]:
        out["BANK_COMPLETE"] = (root / "BANK_COMPLETE").read_text().strip()
    derived = root / "derived"
    if (derived / "step0" / "summary.json").exists():
        out["step0"] = json.loads((derived / "step0" / "summary.json").read_text())
    out["diagnostics"] = (derived / "diagnostics" / "diagnostics.json").exists()
    if (derived / "report" / "report_verification.json").exists():
        out["report_verification"] = json.loads((derived / "report" / "report_verification.json").read_text())["passed"]
    if (root / "events.jsonl").exists():
        out["last_events"] = [json.loads(x) for x in (root / "events.jsonl").read_text().splitlines()[-3:]]
    return out


def verify(root: Path) -> list[str]:
    bad = []
    marker = (root / "BANK_COMPLETE").read_text().split()
    msha = marker[marker.index("manifest_sha256") + 1]
    if sha256_file(root / "MANIFEST.json") != msha:
        return ["MANIFEST.json does not match BANK_COMPLETE"]
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for s in manifest["shards"]:
        if sha256_file(root / s["file"]) != s["sha256"]:
            bad.append(f"{s['file']}: sha256 differs from MANIFEST.json")
    return bad


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("bank_dir")
    p.add_argument("--verify", action="store_true")
    a = p.parse_args()
    root = Path(a.bank_dir)
    if a.verify:
        if not (root / "BANK_COMPLETE").exists():
            print(f"BANK NOT VERIFIED: {root} has no BANK_COMPLETE", file=sys.stderr)
            return 1
        bad = verify(root)
        if bad:
            print("BANK NOT VERIFIED:\n" + "\n".join(bad[:20]), file=sys.stderr)
            return 1
        n = len(json.loads((root / "MANIFEST.json").read_text())["shards"])
        print(f"BANK VERIFIED {root} ({n} shards; {(root / 'BANK_COMPLETE').read_text().split()[-1]})")
        return 0
    print(json.dumps(status(root), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
