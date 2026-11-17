"""Full generation from a fully masked response, scored by the checker afterwards.

The sampler sees only prompt IDs and the PUBLIC response length
(``len(value_names) * slot_len``: the prompt names the values and states the slot
format). Targets are read only by the checker, after generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import torch

from ..data.checker import aggregate, check_response
from ..data.format import TokenizedFormat
from ..data.tasks import Example
from ..env import release_cached_memory, synchronize
from ..models.adapter import DiffusionAdapter
from ..sampling.monotone import monotone_unmask

ONE_TOKEN_PER_CALL = "L"


def public_response_length(ex: Example, slot_len: int) -> int:
    return len(ex.value_names) * slot_len


def parse_budgets(text: str) -> list[int | str]:
    out: list[int | str] = []
    for part in text.split(","):
        part = part.strip()
        out.append(ONE_TOKEN_PER_CALL if part == ONE_TOKEN_PER_CALL else int(part))
    return out


@dataclass
class GenerationRun:
    budget: int | str
    responses: dict[int, list[int]]       # example index -> generated response IDs
    calls: dict[int, int]                 # example index -> model calls used
    infeasible: list[int]                 # example indices with T > L (not run)
    batched_forward_calls: int
    model_seconds: float                  # synchronized forward/selection time
    wall_seconds: float


def generate(adapter: DiffusionAdapter, examples: Sequence[Example], budget: int | str, slot_len: int,
             allowed: torch.Tensor, batch_size: int = 32) -> GenerationRun:
    """Group by (calls, length) so a batch shares T; the order of examples never matters."""
    t_wall = time.perf_counter()
    lengths = [public_response_length(ex, slot_len) for ex in examples]
    groups: dict[tuple[int, int], list[int]] = {}
    infeasible = []
    for i, n in enumerate(lengths):
        T = n if budget == ONE_TOKEN_PER_CALL else int(budget)
        if not 1 <= T <= n:
            infeasible.append(i)
            continue
        groups.setdefault((T, n), []).append(i)
    responses, calls = {}, {}
    n_fwd, model_s = 0, 0.0
    device = adapter.device
    for (T, _), idx in sorted(groups.items()):
        for k in range(0, len(idx), batch_size):
            chunk = idx[k:k + batch_size]
            synchronize(device)
            t0 = time.perf_counter()
            res = monotone_unmask(adapter, [examples[i].prompt_ids for i in chunk], [lengths[i] for i in chunk],
                                  T, allowed)
            synchronize(device)
            model_s += time.perf_counter() - t0
            # Outside the timed region: varying shapes otherwise grow the MPS cache until the machine swaps.
            release_cached_memory(device)
            n_fwd += res.batched_forward_calls
            for j, i in enumerate(chunk):
                responses[i] = res.responses[j]
                calls[i] = res.calls_per_example[j]
    return GenerationRun(budget=budget, responses=responses, calls=calls, infeasible=infeasible,
                         batched_forward_calls=n_fwd, model_seconds=model_s,
                         wall_seconds=time.perf_counter() - t_wall)


def by_setting(records: Sequence[dict]) -> list[dict]:
    """Per (family, length, digits) counts and rates from ``score`` records; explicit denominators."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault((r["family"], int(r["length"]), int(r["operand_digits"])), []).append(r)
    rows = []
    for (fam, length, digits), rs in sorted(groups.items()):
        n = len(rs)
        n_values = sum(len(r["value_correct"]) for r in rs)
        n_tokens = sum(len(r["response_ids"]) for r in rs)
        malformed = sum(any(s != "ok" for s in r["slot_status"]) for r in rs)
        rows.append({"family": fam, "length": length, "operand_digits": digits, "n_examples": n,
                     "final_correct": sum(r["final_correct"] for r in rs),
                     "final_answer_accuracy": sum(r["final_correct"] for r in rs) / n,
                     "trace_accuracy": sum(r["trace_correct"] for r in rs) / n,
                     "value_accuracy": sum(sum(r["value_correct"]) for r in rs) / n_values,
                     "token_accuracy": 1 - sum(r["n_token_errors"] for r in rs) / n_tokens,
                     "malformed_examples": malformed, "malformed_example_rate": malformed / n,
                     "model_calls_total": sum(r["calls"] for r in rs)})
    return rows


def selection_score(records: Sequence[dict]) -> dict:
    """D-033 monitoring score: final-answer accuracy averaged over the settings of each family
    (equal weight per setting), then over families (equal weight per family)."""
    rows = by_setting(records)
    fams: dict[str, list[float]] = {}
    for r in rows:
        fams.setdefault(r["family"], []).append(r["final_answer_accuracy"])
    per_family = {f: sum(v) / len(v) for f, v in sorted(fams.items())}
    return {"score": sum(per_family.values()) / len(per_family), "per_family": per_family,
            "settings_per_family": {f: len(v) for f, v in sorted(fams.items())},
            "n_examples": sum(r["n_examples"] for r in rows),
            "weighting": "mean over families of the mean over that family's settings of final-answer accuracy"}


def score(examples: Sequence[Example], run: GenerationRun, tf: TokenizedFormat) -> tuple[list[dict], dict]:
    """Per-example records and an aggregate over the examples that were run."""
    records, checks = [], []
    for i, ex in enumerate(examples):
        if i not in run.responses:
            continue
        chk = check_response(run.responses[i], ex, tf)
        checks.append(chk)
        records.append({
            "problem_id": ex.problem_id, "family": ex.family, "split": ex.split,
            "length": ex.setting["length"], "operand_digits": ex.setting["operand_digits"],
            "budget": run.budget, "calls": run.calls[i], "response_ids": run.responses[i],
            "parsed_values": chk.parsed_values, "slot_status": chk.slot_status,
            "value_correct": chk.value_correct, "n_token_errors": chk.n_tokens - sum(chk.token_correct),
            "n_value_errors": len(chk.value_correct) - sum(chk.value_correct),
            "final_correct": chk.final_correct, "trace_correct": chk.trace_correct,
        })
    agg = aggregate(checks)
    n = len(checks)
    agg["value_errors_per_response"] = (sum(r["n_value_errors"] for r in records) / n) if n else None
    agg["token_errors_per_response"] = (sum(r["n_token_errors"] for r in records) / n) if n else None
    agg["responses_with_any_value_error"] = sum(r["n_value_errors"] > 0 for r in records)
    agg["model_calls_total"] = sum(run.calls[i] for i in run.calls)
    agg["model_calls_per_example"] = (agg["model_calls_total"] / n) if n else None
    agg["batched_forward_calls"] = run.batched_forward_calls
    agg["infeasible_examples"] = len(run.infeasible)
    agg["model_seconds"] = round(run.model_seconds, 3)
    agg["wall_seconds"] = round(run.wall_seconds, 3)
    return records, agg
