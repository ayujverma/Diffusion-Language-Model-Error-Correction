"""Milestone 5: shared-bank correction trainer, repair metrics, bank identity gates (tiny backbone)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from diffusion_correction.drafts.bank import write_bytes_atomic, write_json_atomic
from diffusion_correction.drafts.identity import sha256_file, sha256_json
from diffusion_correction.evaluation.repair import repair_drafts, summarize_repair
from diffusion_correction.losses.weighting import WeightRule, compute_error_weights, make_group_ids
from diffusion_correction.models.tiny import build_tiny_adapter
from diffusion_correction.training.bank_data import (BankDraft, budget_code, collate_correction_batch,
                                                      gather_error_signals, load_bank_drafts,
                                                      verify_bank_pins)
from diffusion_correction.training.correction_config import (load_correction_config,
                                                              resolve_correction_config)
from diffusion_correction.training.correction_trainer import CorrectionTrainer
from diffusion_correction.training.model_io import build_training_adapter

CPU = torch.device("cpu")
IDENT_SHA = "a" * 64
MANIFEST_SHA_PLACEHOLDER = "b" * 64


def _draft(i: int, *, budget: str = "4", n_err: int = 2, R: int = 6, split: str = "train") -> BankDraft:
    prompt = [10, 11, 12]
    ref = [(i + j) % 10 for j in range(R)]
    draft = list(ref)
    for k in range(n_err):
        draft[k] = (draft[k] + 1) % 10
    mf = [0.9 - 0.1 * j for j in range(R)]
    log_c = [-0.1 * (j + 1) for j in range(R)]
    log_q = [-1.0 - 0.2 * j for j in range(R)]
    return BankDraft(
        problem_id=f"p{i:04d}", split=split, budget=budget, budget_code=budget_code(budget),
        family="arithmetic" if i % 2 == 0 else "programs", length=2, operand_digits=1,
        setting="arithmetic:2:1", prompt_ids=prompt, draft_ids=draft, reference_ids=ref,
        log_c_gen=log_c, log_q_gen=log_q, mask_fraction_before=mf,
        token_correct=[a == b for a, b in zip(draft, ref)], final_correct=draft[-1] == ref[-1],
        well_formed=True, n_token_errors=n_err, shard=f"{split}/b{budget}/part-000",
        identity_sha256=IDENT_SHA,
    )


def _write_tiny_bank(root: Path, drafts: list[BankDraft], *, identity_sha: str = IDENT_SHA) -> tuple[str, str]:
    """Minimal finalized bank layout for load_bank_drafts (verify_shards path)."""
    root.mkdir(parents=True, exist_ok=True)
    shard_rel = "shards/train/b4/arithmetic-n2-d1/part-000.jsonl"
    # Group by (split, budget) into separate shard files matching manifest entries.
    by_key: dict[tuple, list[BankDraft]] = {}
    for d in drafts:
        by_key.setdefault((d.split, d.budget), []).append(d)
    shard_rows = []
    for (split, budget), group in sorted(by_key.items()):
        rel = f"shards/{split}/b{budget}/part-000.jsonl"
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for d in group:
            rec = {
                "schema": "dlmc.draft.v1", "split": d.split, "problem_id": d.problem_id,
                "family": d.family, "length": d.length, "operand_digits": d.operand_digits,
                "setting": d.setting, "budget": d.budget, "calls": 4, "response_length": len(d.draft_ids),
                "prompt_ids": d.prompt_ids, "bank": "tiny-m5", "shard": rel,
                "identity_sha256": identity_sha, "generator_model_sha256": "c" * 64,
                "tokenizer_json_sha256": "d" * 64, "sampler": "d029_monotone_greedy", "seed": None,
                "generation": {
                    "draft_ids": d.draft_ids, "schedule": [], "commit_call": [0] * len(d.draft_ids),
                    "commit_order": list(range(len(d.draft_ids))),
                    "mask_fraction_before": d.mask_fraction_before, "log_c_gen": d.log_c_gen,
                    "rank_score": d.log_c_gen, "policy_log_prob": d.log_c_gen,
                    "raw_max_log_prob": d.log_c_gen,
                },
                "analysis": {
                    "reference_ids": d.reference_ids, "values": [0], "log_q_gen": d.log_q_gen,
                    "token_correct": d.token_correct, "parsed_values": [0], "slot_status": ["ok"],
                    "value_correct": [True], "final_correct": d.final_correct,
                    "trace_correct": False, "well_formed": d.well_formed,
                    "n_token_errors": d.n_token_errors, "n_value_errors": 0,
                },
            }
            lines.append(json.dumps(rec, sort_keys=True))
        path.write_text("\n".join(lines) + "\n")
        shard_rows.append({"id": rel, "file": rel, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                           "n_records": len(group)})
    pinned = {"bank": "tiny-m5", "plan_sha256": "e" * 64, "identity_extra": 1}
    # Force identity_sha256 of the bank to the configured pin by writing IDENTITY with that field.
    identity = {"schema": "dlmc.bank_identity.v1", "pinned": pinned, "identity_sha256": identity_sha}
    write_json_atomic(root / "IDENTITY.json", identity)
    plan = {"schema": "dlmc.bank_plan.v1", "bank": "tiny-m5", "shards": [], "totals": {"drafts": len(drafts)},
            "infeasible": []}
    write_json_atomic(root / "plan.json", plan)
    manifest = {"schema": "dlmc.bank_manifest.v1", "bank": "tiny-m5", "identity_sha256": identity_sha,
                "plan_sha256": "e" * 64, "identity": identity, "shards": shard_rows,
                "totals": {"drafts": len(drafts)}, "plan_totals": {"drafts": len(drafts)},
                "infeasible": [], "created_utc": "t"}
    write_json_atomic(root / "MANIFEST.json", manifest)
    msha = sha256_file(root / "MANIFEST.json")
    write_bytes_atomic(root / "BANK_COMPLETE", f"bank tiny-m5\nmanifest_sha256 {msha}\n".encode())
    return identity_sha, msha


def tiny_corr_cfg(bank_dir: Path, identity_sha: str, manifest_sha: str, **over) -> dict:
    raw = {
        "run_name": "m5-tiny", "backbone": "tiny", "dtype": "float64", "seed": 7, "batch_size": 4,
        "init_checkpoint": None,
        "data": {"dir": None, "manifest_sha256": None, "verify": False},
        "bank": {"dir": str(bank_dir), "identity_sha256": identity_sha, "manifest_sha256": manifest_sha,
                 "split": "train", "budgets": ["4"], "verify_shards": True},
        "weighting": {"rule": "uniform", "beta": 0.5, "shuffle_seed": 7},
        "objective": {"lambda": 1.0, "mu": 1.0},
        "optimizer": {"lr": 3e-3}, "scheduler": {"warmup_updates": 2},
        "limits": {"max_updates": 4}, "checkpoint": {"every_updates": 2, "keep_last": 5},
        "eval": {"every_updates": 2, "at_start": False, "batch_size": 4,
                 "repair": {"split": "val", "budget": "4", "n_examples": None}},
        "log_every": 0,
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k] = {**raw[k], **v}
        else:
            raw[k] = v
    return resolve_correction_config(raw)


def make_corr_trainer(cfg, run_dir, train_drafts, bank_ident, repair_drafts=None, repair_ident=None):
    adapter, info = build_training_adapter(cfg, CPU)
    return CorrectionTrainer(cfg, adapter, info, train_drafts, bank_ident, CPU, run_dir,
                             repair_drafts_list=repair_drafts, repair_identity=repair_ident)


def test_bank_identity_mismatch_refused(tmp_path):
    drafts = [_draft(i) for i in range(8)]
    ident, msha = _write_tiny_bank(tmp_path / "bank", drafts)
    with pytest.raises(ValueError, match="identity"):
        load_bank_drafts(tmp_path / "bank", identity_sha256="f" * 64, manifest_sha256=msha,
                         split="train", budgets=["4"])
    with pytest.raises(ValueError, match="manifest"):
        verify_bank_pins(tmp_path / "bank", ident, "0" * 64)
    loaded, bid = load_bank_drafts(tmp_path / "bank", identity_sha256=ident, manifest_sha256=msha,
                                   split="train", budgets=["4"])
    assert len(loaded) == 8 and bid.n_drafts == 8


def test_beta_zero_matches_uniform_weights():
    groups = torch.tensor([0, 0, 1, 1, 1])
    log_c = torch.tensor([-0.1, -2.0, -0.5, -1.0, -3.0])
    u = compute_error_weights(WeightRule.UNIFORM, groups, beta=0.5, log_c=log_c)
    z = compute_error_weights(WeightRule.HIGH_CONF, groups, beta=0.0, log_c=log_c)
    torch.testing.assert_close(z.weights, u.weights)
    torch.testing.assert_close(z.weights, torch.ones(5))


def test_gather_error_signals_row_major_order():
    drafts = [_draft(0, n_err=2), _draft(1, n_err=1)]
    batch = collate_correction_batch(drafts, pad_token_id=17)
    err = batch.draft_layout.input_ids != batch.target_layout.input_ids
    err = err & batch.target_layout.editable_mask
    log_c, log_q, mfrac, bud = gather_error_signals(
        err, batch.target_layout.editable_mask, batch.budgets,
        batch.log_c_response, batch.log_q_response, batch.mask_fraction_response)
    assert log_c.numel() == 3
    assert bud.tolist() == [4, 4, 4]
    # First example's first two response positions were flipped.
    assert log_c[0].item() == pytest.approx(drafts[0].log_c_gen[0])
    assert log_c[1].item() == pytest.approx(drafts[0].log_c_gen[1])
    assert log_c[2].item() == pytest.approx(drafts[1].log_c_gen[0])
    groups = make_group_ids(bud, mfrac)
    assert groups.numel() == 3


def test_one_step_per_arm_finite(tmp_path):
    train = [_draft(i, split="train") for i in range(12)]
    val = [_draft(100 + i, split="val") for i in range(6)]
    ident, msha = _write_tiny_bank(tmp_path / "bank", train + val)
    bank_ident = {"bank_dir": str(tmp_path / "bank"), "identity_sha256": ident, "manifest_sha256": msha,
                  "n_drafts": len(train), "split": "train", "budgets": ("4",), "draft_ids_sha256": "x",
                  "bank_name": "tiny-m5", "plan_sha256": "e" * 64}
    repair_ident = {**bank_ident, "split": "val", "n_drafts": len(val)}
    for rule in ("uniform", "high_conf", "gen_difficulty", "current_loss", "low_conf", "shuffled_conf"):
        cfg = tiny_corr_cfg(tmp_path / "bank", ident, msha, weighting={"rule": rule, "beta": 0.5, "shuffle_seed": 7},
                            limits={"max_updates": 1}, checkpoint={"every_updates": 1},
                            eval={"every_updates": 0, "at_start": False, "repair": {"split": "val", "budget": "4"}})
        tr = make_corr_trainer(cfg, tmp_path / f"run-{rule}", train, bank_ident, val, repair_ident)
        summary = tr.train()
        assert summary["update"] == 1
        metrics = [json.loads(x) for x in (tmp_path / f"run-{rule}" / "metrics.jsonl").read_text().splitlines()]
        assert math.isfinite(metrics[-1]["loss"])
        assert math.isfinite(metrics[-1]["L_mask"])
        assert math.isfinite(metrics[-1]["L_error"])
        assert math.isfinite(metrics[-1]["L_keep"])
        assert metrics[-1]["n_error"] >= 0


def test_repair_metrics_have_denominators(tmp_path):
    drafts = [_draft(i, n_err=2 if i % 2 == 0 else 0) for i in range(8)]
    adapter = build_tiny_adapter(seed=1, dtype=torch.float64)
    records = repair_drafts(adapter, drafts, batch_size=4)
    summary = summarize_repair(records)
    assert summary["n_drafts"] == 8
    assert summary["n_wrong_before"] + summary["n_correct_before"] == summary["n_tokens"]
    assert summary["n_repaired"] <= summary["n_wrong_before"]
    assert summary["n_damaged"] <= summary["n_correct_before"]
    if summary["n_wrong_before"]:
        assert 0.0 <= summary["repair_rate"] <= 1.0
    if summary["n_correct_before"]:
        assert 0.0 <= summary["damage_rate"] <= 1.0
    # Empty-error edge: all-correct drafts give repair_rate None.
    ok = [_draft(i, n_err=0) for i in range(3)]
    s2 = summarize_repair(repair_drafts(adapter, ok, batch_size=3))
    assert s2["n_wrong_before"] == 0 and s2["repair_rate"] is None


def test_resume_refuses_rule_change(tmp_path):
    train = [_draft(i) for i in range(8)]
    val = [_draft(50 + i, split="val") for i in range(4)]
    ident, msha = _write_tiny_bank(tmp_path / "bank", train + val)
    bank_ident = {"bank_dir": str(tmp_path / "bank"), "identity_sha256": ident, "manifest_sha256": msha,
                  "n_drafts": 8, "split": "train", "budgets": ("4",), "draft_ids_sha256": "x",
                  "bank_name": "tiny-m5", "plan_sha256": "e" * 64}
    repair_ident = {**bank_ident, "split": "val", "n_drafts": 4}
    cfg = tiny_corr_cfg(tmp_path / "bank", ident, msha, limits={"max_updates": 2},
                        checkpoint={"every_updates": 2},
                        eval={"every_updates": 0, "at_start": False, "repair": {"split": "val", "budget": "4"}})
    tr = make_corr_trainer(cfg, tmp_path / "r", train, bank_ident, val, repair_ident)
    tr.train()
    ck = tmp_path / "r" / "checkpoints" / "ckpt-0000002"
    bad = tiny_corr_cfg(tmp_path / "bank", ident, msha, weighting={"rule": "high_conf", "beta": 0.5},
                        eval={"every_updates": 0, "at_start": False, "repair": {"split": "val", "budget": "4"}})
    with pytest.raises(ValueError, match="config incompatible"):
        make_corr_trainer(bad, tmp_path / "r", train, bank_ident, val, repair_ident).resume(ck)


def test_arm_yaml_files_differ_only_by_rule_and_name():
    root = Path("configs/correction")
    arms = ["uniform", "high_conf", "gen_difficulty", "current_loss", "low_conf", "shuffled_conf"]
    cfgs = [load_correction_config(root / f"m5_{a}.yaml") for a in arms]
    keys = set(cfgs[0]) - {"run_name", "weighting"}
    for c in cfgs[1:]:
        for k in keys:
            assert c[k] == cfgs[0][k], k
        assert c["weighting"]["beta"] == cfgs[0]["weighting"]["beta"]
        assert c["weighting"]["stage_edges"] == cfgs[0]["weighting"]["stage_edges"]
        assert c["objective"] == cfgs[0]["objective"]
    assert [c["weighting"]["rule"] for c in cfgs] == arms
    assert cfgs[0]["eval"]["repair"]["budget"] == "4"
    assert cfgs[0]["objective"]["lambda"] == 1.0 and cfgs[0]["objective"]["mu"] == 1.0
