"""Verify and record the frozen milestone-4 generator (checkpoint 8000; D-038).

Hashes files and reads JSON only (no torch), so it may run on a login node.

  record:  python scripts/tacc/generator_identity.py --checkpoint CK --model-sha256 SHA --update 8000 \
               --eval-records RECORDS --eval-records-sha256 SHA --out IDENTITY.json
  check:   ... --check        (the same, but IDENTITY.json must already exist and match exactly)

Exits non-zero, writing nothing, if any file differs from COMPLETE.json, the model hash or
update differs from the pin, or the evaluation records differ from their pinned hash. An
existing identity file is never overwritten with different content.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from diffusion_correction.drafts.identity import generator_identity, sha256_file


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-sha256", required=True)
    p.add_argument("--update", type=int, required=True)
    p.add_argument("--eval-records", required=True)
    p.add_argument("--eval-records-sha256", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    try:
        ident = generator_identity(a.checkpoint, a.model_sha256, a.update)
    except ValueError as e:
        print(f"GENERATOR NOT VERIFIED: {e}", file=sys.stderr)
        return 1
    got = sha256_file(Path(a.eval_records))
    if got != a.eval_records_sha256:
        print(f"GENERATOR NOT VERIFIED: {a.eval_records} sha256 {got} != {a.eval_records_sha256}", file=sys.stderr)
        return 1
    ident["validation_eval_records"] = {"path": str(Path(a.eval_records).resolve()), "sha256": got,
                                        "job": "3595133", "split": "val"}
    out = Path(a.out)
    if out.exists():
        old = json.loads(out.read_text())
        if {k: v for k, v in old.items() if k != "recorded"} != ident:
            print(f"GENERATOR IDENTITY CHANGED: {out} differs from the files now on disk", file=sys.stderr)
            return 1
        print(f"GENERATOR VERIFIED (matches {out}) model {ident['model_sha256']} "
              f"COMPLETE.json {ident['complete_json_sha256']}")
        return 0
    if a.check:
        print(f"GENERATOR NOT VERIFIED: {out} does not exist; record it first (without --check)", file=sys.stderr)
        return 1
    ident["recorded"] = {"time_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
                         "user": os.environ.get("USER"), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(ident, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(json.dumps({k: ident[k] for k in ("checkpoint", "update", "model_sha256", "complete_json_sha256",
                                            "config_hash", "source_revision_at_save", "tokenizer_json_sha256",
                                            "training_data_identity", "selection")}, indent=2, default=str))
    print(f"GENERATOR VERIFIED AND RECORDED {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
