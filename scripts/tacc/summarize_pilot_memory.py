"""Summarize the memory records and memory-related log lines of a retained pilot directory.

Standard library only; reads files, runs no model. Written for pilot 3592809
($DLMC_RETAINED/pilot/<jobid>, laid out by scripts/tacc/pilot.slurm).

Definitions (diffusion_correction.env.memory_snapshot, one record per update with memory_every=1):
- cuda_allocated_gb / cuda_peak_allocated_gb: bytes held by live tensors (torch.cuda.memory_allocated,
  max_memory_allocated), current and peak since process start (the trainer never resets peaks);
- cuda_reserved_gb / cuda_peak_reserved_gb: bytes the caching allocator holds from the driver,
  including cached free blocks (memory_reserved, max_memory_reserved). pilot_report's
  peak_memory_U is the maximum of cuda_peak_reserved_gb over U's records, i.e. peak reserved;
- cuda_device_used_gb: total - free from cudaMemGetInfo: every process and context on the
  device, including the CUDA context and libraries, so it exceeds reserved.
The pilot records no allocator retry or OOM counters; those are added only in
fixture_update_diagnostic.py. The application has no OOM handling or retry path, and its
cache release (release_cached_memory) is MPS-only. benchmark.py calls torch.cuda.empty_cache()
and resets peak statistics between benchmark configurations.

    python scripts/tacc/summarize_pilot_memory.py --pilot-dir $DLMC_RETAINED/pilot/3592809 \
        --log runs/slurm/dlmc-pilot-3592809.out --out <dir>/memory_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIELDS = ("cuda_allocated_gb", "cuda_peak_allocated_gb", "cuda_reserved_gb", "cuda_peak_reserved_gb",
          "cuda_device_used_gb", "cuda_total_gb", "process_peak_rss_gb")
PATTERN = re.compile(r"warn|out of memory|\boom\b|retr(y|ies)|alloc|fragment|cudamalloc|expandable|empty_cache",
                     re.IGNORECASE)
MAX_LINES_PER_FILE = 200


def jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def run_memory(metrics: Path) -> dict | None:
    recs = [r for r in jsonl(metrics) if isinstance(r.get("memory"), dict)]
    if not recs:
        return None
    out: dict = {"n_records": len(recs), "updates": [recs[0].get("update"), recs[-1].get("update")]}
    for f in FIELDS:
        vals = [r["memory"][f] for r in recs if r["memory"].get(f) is not None]
        if vals:
            out[f] = {"first": vals[0], "last": vals[-1], "min": min(vals), "max": max(vals)}
    al, rs = out.get("cuda_peak_allocated_gb"), out.get("cuda_peak_reserved_gb")
    if al and rs:
        out["peak_reserved_minus_peak_allocated_gb"] = round(rs["max"] - al["max"], 3)
    du = out.get("cuda_device_used_gb")
    if du and rs:
        out["device_used_minus_peak_reserved_gb"] = round(du["max"] - rs["max"], 3)
    return out


def scan(p: Path) -> dict:
    hits = []
    with open(p, errors="replace") as f:
        for i, line in enumerate(f, 1):
            if PATTERN.search(line):
                hits.append({"line": i, "text": line.rstrip()[:500]})
    low = " ".join(h["text"].lower() for h in hits)
    return {"n_matching_lines": len(hits), "lines": hits[:MAX_LINES_PER_FILE],
            "mentions_out_of_memory": "out of memory" in low or "outofmemoryerror" in low,
            "mentions_retry": bool(re.search(r"retr(y|ies)", low))}


def summarize(pilot: Path, extra_logs: list[Path]) -> dict:
    s: dict = {"pilot_dir": str(pilot), "definitions": __doc__.split("Definitions", 1)[1].split("\n\n")[0]}
    s["runs"] = {r: run_memory(pilot / "runs" / r / "metrics.jsonl") for r in ("M", "U", "U2", "R")}
    u = s["runs"].get("U")
    s["peak_memory_U_recomputed"] = u["cuda_peak_reserved_gb"]["max"] if u and "cuda_peak_reserved_gb" in u else None
    rep = pilot / "pilot_report.json"
    if rep.exists():
        s["peak_memory_U_in_report"] = json.loads(rep.read_text()).get("measurements", {}).get("peak_memory_U")
    bench = pilot / "benchmark.json"
    if bench.exists():
        b = json.loads(bench.read_text())
        res = b.get("results", [b])
        s["benchmark"] = [{k: r.get(k) for k in ("batch_shapes", "memory_after_load", "memory_after_first_update",
                                                  "memory_last_update", "optimizer_state_allocated_gb")} for r in res]
    files = sorted({*pilot.glob("*.log"), *pilot.glob("*.txt"), *pilot.glob("runs/*/events.jsonl"),
                    *pilot.glob("runs/*/*.log"), *extra_logs})
    s["log_scan_pattern"] = PATTERN.pattern
    s["logs"] = {str(p): scan(p) for p in files if p.is_file() and p.stat().st_size < 200_000_000}
    s["any_out_of_memory_line"] = any(v["mentions_out_of_memory"] for v in s["logs"].values())
    s["any_retry_line"] = any(v["mentions_retry"] for v in s["logs"].values())
    s["allocator_counters_recorded_by_pilot"] = False
    s["application_retry_paths"] = ("none: no OOM catch, retry or requeue in src/ or scripts/tacc/ (code inspection); "
                                    "release_cached_memory is MPS-only")
    return s


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pilot-dir", required=True)
    p.add_argument("--log", action="append", default=[], help="extra log file (e.g. the Slurm .out/.err)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    pilot = Path(a.pilot_dir)
    if not (pilot / "steps.txt").exists():
        raise SystemExit(f"{pilot} is not a pilot report directory (no steps.txt)")
    s = summarize(pilot, [Path(x) for x in a.log])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps({k: s[k] for k in ("runs", "peak_memory_U_recomputed", "any_out_of_memory_line",
                                        "any_retry_line")}, indent=2))
    for f, v in s["logs"].items():
        for h in v["lines"][:20]:
            print(f"{f}:{h['line']}: {h['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
