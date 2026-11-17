"""Trainer, checkpoints, resume and LoRA on the tiny backbone (CPU float64; no assets needed)."""

import json
import os
from pathlib import Path

import pytest
import torch

from diffusion_correction.models.tiny import build_tiny_adapter
from diffusion_correction.training.checkpoint import copy_to_durable, verify_checkpoint
from diffusion_correction.training.config import resolve_config
from diffusion_correction.training.lora import LoRAConfig, LoRALinear, apply_lora, expected_lora_parameter_count
from diffusion_correction.training.model_io import build_training_adapter, load_trained_adapter
from diffusion_correction.training.trainer import Trainer

from tiny_tasks import make_tiny_examples

CPU = torch.device("cpu")
IDENT = {"data_dir": "synthetic", "manifest_sha256": "none", "split": "train", "n": 24}


def tiny_cfg(**over):
    raw = {"run_name": "t", "backbone": "tiny", "dtype": "float64", "seed": 3, "batch_size": 4,
           "optimizer": {"lr": 3e-3}, "scheduler": {"warmup_updates": 3},
           "limits": {"max_updates": 8}, "checkpoint": {"every_updates": 4, "keep_last": 5},
           "eval": {"split": "train_subset", "batch_size": 8}, "log_every": 0}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k] = {**raw[k], **v}
        else:
            raw[k] = v
    return resolve_config(raw)


def make_trainer(cfg, run_dir, examples=None, ident=IDENT):
    adapter, info = build_training_adapter(cfg, CPU)  # tiny base seeded by cfg["seed"]
    return Trainer(cfg, adapter, info, examples or make_tiny_examples(24, 1), ident, CPU, run_dir)


def read_metrics(run_dir):
    return [json.loads(x) for x in (Path(run_dir) / "metrics.jsonl").read_text().splitlines()]


def params(trainer):
    return {k: v.detach().clone() for k, v in trainer.adapter.model.state_dict().items()}


def test_uninterrupted_vs_resumed_run_are_identical(tmp_path):
    cfg = tiny_cfg()
    a = make_trainer(cfg, tmp_path / "a")
    a.train()
    b1 = make_trainer(tiny_cfg(limits={"max_updates": 4}), tmp_path / "b")
    b1.train()
    del b1
    b2 = make_trainer(cfg, tmp_path / "b")  # fresh process state: new model, optimizer, generators
    b2.resume(tmp_path / "b" / "checkpoints" / "ckpt-0000004")
    b2.train()
    ma, mb = read_metrics(tmp_path / "a"), read_metrics(tmp_path / "b")
    assert len(ma) == len(mb) == 8
    for x, y in zip(ma, mb):
        for k in ("batch", "update", "problem_ids", "example_indices", "mask_digest", "mask_t", "lr", "loss",
                  "grad_norm", "epoch"):
            assert x[k] == y[k], (k, x[k], y[k])
    pa, pb = params(a), params(b2)
    assert all(torch.equal(pa[k], pb[k]) for k in pa)
    assert a.optimizer.state_dict()["state"][0]["exp_avg"].equal(b2.optimizer.state_dict()["state"][0]["exp_avg"])
    assert ma[-1]["epoch"] == 2  # 8 batches of 4 over 24 examples crosses an epoch boundary
    from diffusion_correction.evaluation.compare import compare_runs

    rep = compare_runs(tmp_path / "a", tmp_path / "b", "ckpt-0000008")
    assert rep["passed"] and rep["param_tensors_bitwise_equal"] == rep["param_tensors"]
    c = make_trainer(tiny_cfg(seed=4), tmp_path / "c")  # a different seed must be detected
    c.train()
    assert not compare_runs(tmp_path / "a", tmp_path / "c", "ckpt-0000008")["passed"]


def test_evaluation_does_not_perturb_training(tmp_path):
    plain = make_trainer(tiny_cfg(), tmp_path / "p")
    plain.train()
    evald = make_trainer(tiny_cfg(eval={"every_updates": 2}), tmp_path / "e")
    evald.train()
    assert [r["loss"] for r in read_metrics(tmp_path / "p")] == [r["loss"] for r in read_metrics(tmp_path / "e")]
    assert len((tmp_path / "e" / "evals.jsonl").read_text().splitlines()) == 4
    evals = [json.loads(x) for x in (tmp_path / "e" / "evals.jsonl").read_text().splitlines()]
    assert len({e["n_masked_tokens"] for e in evals}) == 1  # same eval masks every time


def test_zero_mask_batches_skip_optimizer_and_scheduler(tmp_path):
    cfg = tiny_cfg(mask={"t_min": 0.0, "t_max": 0.0}, limits={"max_updates": 3, "max_wall_seconds": 2})
    tr = make_trainer(cfg, tmp_path / "z")
    before = params(tr)
    lr0 = tr.optimizer.param_groups[0]["lr"]
    tr.train()
    assert tr.update == 0 and tr.batches_consumed > 0 and tr.skipped_zero_mask == tr.batches_consumed
    assert tr.optimizer.param_groups[0]["lr"] == lr0 and tr.scheduler.last_epoch == 0
    assert all(torch.equal(before[k], v) for k, v in params(tr).items())
    recs = read_metrics(tmp_path / "z")
    assert all(r["event"] == "skipped_zero_mask" and r["update"] == 0 for r in recs)


def test_learning_rate_schedule_and_logged_values(tmp_path):
    tr = make_trainer(tiny_cfg(), tmp_path / "s")
    tr.train()
    lrs = [r["lr"] for r in read_metrics(tmp_path / "s")]
    assert lrs == pytest.approx([1e-3, 2e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3, 3e-3])


def test_resume_rejects_incompatible_config_or_data(tmp_path):
    tr = make_trainer(tiny_cfg(limits={"max_updates": 4}), tmp_path / "r")
    tr.train()
    ck = tmp_path / "r" / "checkpoints" / "ckpt-0000004"
    with pytest.raises(ValueError, match="config incompatible"):
        make_trainer(tiny_cfg(optimizer={"lr": 1e-3}), tmp_path / "r").resume(ck)
    with pytest.raises(ValueError, match="dataset differs"):
        make_trainer(tiny_cfg(), tmp_path / "r", ident={**IDENT, "manifest_sha256": "other"}).resume(ck)
    make_trainer(tiny_cfg(limits={"max_updates": 99}, checkpoint={"every_updates": 7}), tmp_path / "r").resume(ck)


def test_checkpoint_integrity_and_durable_copy(tmp_path):
    tr = make_trainer(tiny_cfg(limits={"max_updates": 4}, checkpoint={"durable_dir": str(tmp_path / "drive")}),
                      tmp_path / "c")
    tr.train()
    ck = tmp_path / "c" / "checkpoints" / "ckpt-0000004"
    info = verify_checkpoint(ck)
    assert set(info["files"]) == {"model.safetensors", "optimizer.pt", "rng.pt", "state.json"}
    assert (tmp_path / "c" / "checkpoints" / "latest").read_text().strip() == "ckpt-0000004"
    durable = tmp_path / "drive" / "t" / "checkpoints" / "ckpt-0000004"
    assert verify_checkpoint(durable) == info
    events = [json.loads(x) for x in (tmp_path / "c" / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "durable_copy" and e["ok"] for e in events)
    state = json.loads((ck / "state.json").read_text())
    for key in ("sampler", "compat_hash", "data_identity", "base_model", "git", "versions", "config"):
        assert key in state
    # Corruption is detected; an incomplete directory is never treated as a checkpoint.
    with open(ck / "optimizer.pt", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="sha256"):
        verify_checkpoint(ck)
    os.remove(ck / "COMPLETE.json")
    with pytest.raises(ValueError, match="incomplete"):
        verify_checkpoint(ck)
    bad = copy_to_durable(ck, tmp_path / "drive2")
    assert bad["ok"] is False and "incomplete" in bad["error"]


def test_trained_checkpoint_reloads_bit_exactly(tmp_path):
    tr = make_trainer(tiny_cfg(limits={"max_updates": 4}), tmp_path / "l")
    tr.train()
    adapter, info = load_trained_adapter(tmp_path / "l" / "checkpoints" / "ckpt-0000004", CPU)
    trained = params(tr)
    assert all(torch.equal(trained[k], v.to(trained[k].dtype)) for k, v in adapter.model.state_dict().items())
    assert adapter.model.lm_head.weight is adapter.model.model.embed_tokens.weight


def test_grad_accum_other_than_one_is_rejected():
    with pytest.raises(ValueError, match="grad_accum"):
        tiny_cfg(grad_accum=2)


# ------------------------------------------------------------------ LoRA

def test_lora_starts_equal_to_base_and_matches_merged_weights():
    base = build_tiny_adapter(seed=2)
    ids = torch.tensor([[10, 11, 1, 2, 3, 18]])
    att = torch.ones_like(ids, dtype=torch.bool)
    ref = base.model(ids, att)
    wrapped = apply_lora(base.model, LoRAConfig(rank=4, alpha=8, init_seed=1))
    assert len(wrapped) == 7 * 2
    assert torch.equal(base.model(ids, att), ref)  # B = 0 at init
    gen = torch.Generator().manual_seed(0)
    for m in base.model.modules():
        if isinstance(m, LoRALinear):
            with torch.no_grad():
                m.lora_B.copy_(torch.randn(m.lora_B.shape, generator=gen, dtype=m.lora_B.dtype))
    out = base.model(ids, att)
    merged = build_tiny_adapter(seed=2)
    with torch.no_grad():
        for (name, m) in base.model.named_modules():
            if isinstance(m, LoRALinear):
                merged.model.get_submodule(name).weight.copy_(m.merged_weight())
    assert torch.allclose(out, merged.model(ids, att), atol=1e-12, rtol=0)


def test_lora_trains_only_adapters_and_counts_parameters(tmp_path):
    cfg = tiny_cfg(method="lora", lora={"rank": 4, "alpha": 8}, limits={"max_updates": 3})
    adapter, info = build_training_adapter(cfg, CPU)
    expected = expected_lora_parameter_count(adapter.model.config, LoRAConfig(rank=4, alpha=8))
    assert info["trainable_parameters"] == expected
    names = [n for n, p in adapter.model.named_parameters() if p.requires_grad]
    assert names and all(".lora_A" in n or ".lora_B" in n for n in names)
    before = {n: p.detach().clone() for n, p in adapter.model.named_parameters()}
    tr = Trainer(cfg, adapter, info, make_tiny_examples(24, 1), IDENT, CPU, tmp_path / "lora")
    tr.train()
    after = dict(adapter.model.named_parameters())
    frozen_same = all(torch.equal(before[n], after[n]) for n in before if ".lora_" not in n)
    b_changed = all(not torch.equal(before[n], after[n]) for n in before if ".lora_B" in n)
    assert frozen_same and b_changed
    reloaded, _ = load_trained_adapter(tmp_path / "lora" / "checkpoints" / "ckpt-0000003", CPU)
    for n, p in reloaded.model.named_parameters():
        assert torch.equal(p, after[n].detach())
