"""Compare two training runs batch by batch and by final parameters (Gate C resume check)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

EXACT_FIELDS = ("update", "epoch", "example_indices", "problem_ids", "mask_digest", "mask_t", "n_masked", "lr")


def _metrics(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (Path(run_dir) / "metrics.jsonl").read_text().splitlines() if x]


def compare_runs(run_a: Path, run_b: Path, ckpt_name: str | None = None, loss_rtol: float = 0.0,
                 param_atol: float = 0.0, reference_model: Path | None = None,
                 divergence_rtol: float | None = None) -> dict:
    """``reference_model`` (initial weights) enables the scale-free check
    ||theta_B - theta_A|| / ||theta_A - theta_0|| <= divergence_rtol."""
    from safetensors.torch import load_file

    ma, mb = _metrics(run_a), _metrics(run_b)
    report = {"run_a": str(run_a), "run_b": str(run_b), "n_batches_a": len(ma), "n_batches_b": len(mb),
              "field_mismatches": {}, "tolerances": {"loss_rtol": loss_rtol, "param_atol": param_atol}}
    for f in EXACT_FIELDS:
        bad = [x["batch"] for x, y in zip(ma, mb) if x.get(f) != y.get(f)]
        report["field_mismatches"][f] = bad
    loss_rel = [abs(x["loss"] - y["loss"]) / max(abs(x["loss"]), 1e-12)
                for x, y in zip(ma, mb) if "loss" in x and "loss" in y]
    gn_rel = [abs(x["grad_norm"] - y["grad_norm"]) / max(abs(x["grad_norm"]), 1e-12)
              for x, y in zip(ma, mb) if "grad_norm" in x and "grad_norm" in y]
    report["loss_max_rel_diff"] = max(loss_rel) if loss_rel else None
    report["loss_bitwise_equal_batches"] = sum(x.get("loss") == y.get("loss") for x, y in zip(ma, mb))
    report["grad_norm_max_rel_diff"] = max(gn_rel) if gn_rel else None
    name = ckpt_name or (Path(run_a) / "checkpoints" / "latest").read_text().strip()
    ca, cb = Path(run_a) / "checkpoints" / name, Path(run_b) / "checkpoints" / name
    fa = next(f for f in ("model.safetensors", "adapter.safetensors") if (ca / f).exists())
    sa, sb = load_file(str(ca / fa)), load_file(str(cb / fa))
    diffs = {k: float((sa[k].double() - sb[k].double()).abs().max()) for k in sa}
    report["final_checkpoint"] = name
    report["param_tensors"] = len(diffs)
    report["param_tensors_bitwise_equal"] = sum(torch.equal(sa[k], sb[k]) for k in sa)
    report["param_max_abs_diff"] = max(diffs.values())
    report["param_worst_tensor"] = max(diffs, key=diffs.get)
    report["model_files_identical"] = (json.loads((ca / "COMPLETE.json").read_text())["files"][fa]
                                       == json.loads((cb / "COMPLETE.json").read_text())["files"][fa])
    passed = (len(ma) == len(mb) and not any(report["field_mismatches"].values())
              and (report["loss_max_rel_diff"] or 0.0) <= loss_rtol and report["param_max_abs_diff"] <= param_atol)
    if reference_model is not None:
        s0 = load_file(str(reference_model))
        s0.pop("lm_head.weight", None)
        moved = sum(float(((sa[k].double() - s0[k].double()) ** 2).sum()) for k in sa) ** 0.5
        apart = sum(float(((sa[k].double() - sb[k].double()) ** 2).sum()) for k in sa) ** 0.5
        report["update_norm_a"] = moved
        report["divergence_norm"] = apart
        report["divergence_rel_to_update"] = apart / moved if moved else None
        report["tolerances"]["divergence_rtol"] = divergence_rtol
        if divergence_rtol is not None:
            passed = passed and report["divergence_rel_to_update"] <= divergence_rtol
    report["passed"] = passed
    return report
