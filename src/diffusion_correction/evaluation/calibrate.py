"""Validation-only difficulty calibration: monotone-sampler generation scored by the checker.

Only ``train`` (for wiring checks) and ``val`` may be evaluated here; test splits are
refused, so calibration decisions never see test outcomes. Every per-example record
is written, and summaries are computed from those records.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import torch

from ..data.checker import check_response
from ..data.format import TokenizedFormat
from ..data.tasks import Example
from ..env import environment_report, synchronize
from ..sampling.monotone import monotone_unmask
from ..training.data import setting_key
from .generation import generate, public_response_length, score

CALIBRATION_SPLITS = ("train", "val")


def calibrate(adapter, examples: Sequence[Example], budgets: Sequence[int | str], tf: TokenizedFormat,
              allowed: torch.Tensor, out_dir: Path, meta: dict, batch_size: int = 32) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(ex.split not in CALIBRATION_SPLITS for ex in examples):
        raise ValueError("calibration evaluates train/val only; test splits are refused")
    by_setting: dict[tuple, list[Example]] = {}
    for ex in examples:
        by_setting.setdefault(setting_key(ex), []).append(ex)
    rows = []
    t_all = time.perf_counter()
    rec_path = out_dir / "records.jsonl"
    rec_path.write_text("")
    for key in sorted(by_setting):
        exs = by_setting[key]
        for b in budgets:
            run = generate(adapter, exs, b, tf.fmt.slot_len, allowed, batch_size=batch_size)
            records, agg = score(exs, run, tf)
            with open(rec_path, "a") as f:
                for r in records:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            L = public_response_length(exs[0], tf.fmt.slot_len)
            row = {"family": key[0], "length": key[1], "operand_digits": key[2], "response_tokens": L,
                   "budget": str(b), "calls": L if b == "L" else b, **agg}
            rows.append(row)
            print(f"{key} budget={b}: final={agg['final_answer_accuracy']} value={agg['value_accuracy']} "
                  f"malformed={agg['malformed_example_rate']} n={agg['n_examples']} "
                  f"({agg['model_seconds']}s)", flush=True)
    synchronize(adapter.device)
    summary = {"meta": meta, "budgets": [str(b) for b in budgets], "rows": rows,
               "total_wall_seconds": round(time.perf_counter() - t_all, 2),
               "environment": environment_report(adapter.device)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (out_dir / "summary.md").write_text(render_table(rows))
    return summary


def render_table(rows: list[dict]) -> str:
    head = ("| family | len | digits | budget | calls/ex | n | final | value | token | trace | malformed ex | "
            "value err/resp | model s |\n|" + "---|" * 13 + "\n")

    def f(x):
        return "n/a" if x is None else f"{x:.3f}"

    lines = [f"| {r['family']} | {r['length']} | {r['operand_digits']} | {r['budget']} | "
             f"{f(r['model_calls_per_example'])} | {r['n_examples']} | {f(r['final_answer_accuracy'])} | "
             f"{f(r['value_accuracy'])} | {f(r['token_accuracy'])} | {f(r['trace_accuracy'])} | "
             f"{f(r['malformed_example_rate'])} | {f(r['value_errors_per_response'])} | {r['model_seconds']} |"
             for r in rows]
    return head + "\n".join(lines) + "\n"


def write_traces(adapter, examples: Sequence[Example], calls: int, tf: TokenizedFormat, allowed, tokenizer,
                 path: Path) -> None:
    """Human-readable token-state traces (one row per call) for a few examples."""
    lines = []
    for ex in examples:
        L = public_response_length(ex, tf.fmt.slot_len)
        res = monotone_unmask(adapter, [ex.prompt_ids], [L], min(calls, L), allowed, record_traces=True)
        lines.append(f"## {ex.problem_id} ({ex.family}, length {ex.setting['length']}, digits "
                     f"{ex.setting['operand_digits']}), L={L}, T={min(calls, L)}, schedule {res.schedule[0]}")
        lines.append(ex.prompt_text)
        for s, state in enumerate(res.traces[0], start=1):
            toks = ["<M>" if t == adapter.tokens.mask_token_id else tokenizer.decode([t]) for t in state]
            lines.append(f"call {s}: " + "|".join(toks))
        lines.append("commit events (pos, token, call, mask_frac_before, raw_logp, policy_logp): " + "; ".join(
            f"({e.position},{tokenizer.decode([e.token_id])!r},{e.call},{e.mask_fraction_before:.2f},"
            f"{e.raw_logp:.4f},{e.policy_logp:.4f})" for e in res.events))
        chk = check_response(res.responses[0], ex, tf)
        lines.append(f"checker (after generation): parsed {chk.parsed_values} target {ex.values} "
                     f"status {chk.slot_status} "
                     f"final_correct={chk.final_correct}")
        lines.append("")
    Path(path).write_text("\n".join(lines) + "\n")
