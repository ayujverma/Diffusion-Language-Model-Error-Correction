"""Family B trainer: shared batch, resume refuses a different arm, checkpoints load."""

import copy
from pathlib import Path

import pytest
import torch
import yaml

from diffusion_correction.cli import main
from diffusion_correction.models.tiny import TINY_TOKENS, build_tiny_adapter
from diffusion_correction.training.family_b_config import FAMILY_B_ARMS, resolve_family_b_config
from diffusion_correction.training.family_b_run import allowed_for
from diffusion_correction.training.family_b_trainer import FamilyBTrainer
from diffusion_correction.training.masking import MaskDraw
from diffusion_correction.training.model_io import build_training_adapter, load_trained_adapter

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "configs" / "correction" / f"m7_{arm}.yaml" for arm in FAMILY_B_ARMS]


class Row:
    def __init__(self, problem_id, prompt_ids, response_ids):
        self.problem_id = problem_id
        self.prompt_ids = list(prompt_ids)
        self.response_ids = list(response_ids)


def _examples():
    return [Row(f"p{i}", [1, 2], [3, 4, 5, 6]) for i in range(4)]


def _cfg(arm, updates=2):
    return resolve_family_b_config({
        "arm": arm, "backbone": "tiny", "dtype": "float64", "seed": 20261120,
        "batch_size": 2, "run_name": f"tiny-{arm}",
        "limits": {"max_updates": updates},
        "checkpoint": {"every_updates": 1, "keep_last": 2},
        "log_every": 1, "memory_every": 0, "release_cache_every": 1,
        "data": {"split": "train", "verify": False},
    })


def _trainer(tmp_path, arm, updates=2):
    cfg = _cfg(arm, updates)
    adapter, info = build_training_adapter(cfg, torch.device("cpu"))
    ident = {"split": "train", "n_selected": 4, "source": "caller", "backbone": "tiny"}
    return FamilyBTrainer(cfg, adapter, info, _examples(), ident, torch.device("cpu"),
                          tmp_path / arm, allowed_for(adapter, None))


def test_screen_configs_share_batch_seed_init_and_refuse_test_splits():
    loaded = [yaml.safe_load(p.read_text()) for p in CONFIGS]
    assert [c["arm"] for c in loaded] == list(FAMILY_B_ARMS)
    assert len({c["batch_size"] for c in loaded}) == 1 and loaded[0]["batch_size"] == 16
    assert len({c["seed"] for c in loaded}) == 1 and loaded[0]["seed"] == 20261120
    assert len({c["limits"]["max_updates"] for c in loaded}) == 1
    assert len({c["init_checkpoint"] for c in loaded}) == 1
    assert all(c["init_checkpoint"].endswith("ckpt-0008000") for c in loaded)
    assert all(c["data"]["split"] == "train" for c in loaded)
    assert all(c["corrector"]["time_weight"] == "inv_t" for c in loaded)
    for split in ("test_id", "test_long", "test_unseen", "val"):
        raw = copy.deepcopy(loaded[0])
        raw["backbone"] = "tiny"
        raw["data"]["split"] = split
        with pytest.raises(ValueError, match="refuses"):
            resolve_family_b_config(raw)


def test_cli_refuses_a_test_split(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("arm: proseco_basic\nbackbone: tiny\ndata:\n  split: test_id\n")
    with pytest.raises(SystemExit, match="test_id"):
        main(["train-family-b", "--config", str(path), "--device", "cpu"])


@pytest.mark.parametrize("arm", FAMILY_B_ARMS)
def test_each_arm_trains_and_matches_a_second_seeded_run(tmp_path, arm):
    first = _trainer(tmp_path / "a", arm, updates=1)
    second = _trainer(tmp_path / "b", arm, updates=1)
    ra = first.train_step([0, 1])
    rb = second.train_step([0, 1])
    assert ra["arm"] == arm and ra.get("event") == rb.get("event")
    if ra.get("event") != "skipped_zero_mask":
        assert math_finite(ra["loss"]) and ra["loss"] == rb["loss"]
    summary = first.train()
    assert summary["update"] == 1 and summary["experiment_completed"] is False
    assert summary["arm"] == arm


def math_finite(value):
    return value == value and abs(value) != float("inf")


def test_zero_mask_skips_the_optimizer(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path, "proseco_basic", updates=1)
    before = [p.detach().clone() for p in trainer.params]

    def empty(layout, generator, t_min, t_max):
        del generator, t_min, t_max
        b, length = layout.editable_mask.shape
        return MaskDraw(torch.zeros(b, length, dtype=torch.bool), [0.5] * b, [0.5] * b, [0] * b)

    monkeypatch.setattr("diffusion_correction.training.family_b_trainer.sample_response_masks", empty)
    rec = trainer.train_step([0, 1])
    assert rec["event"] == "skipped_zero_mask"
    assert trainer.update == 0
    for old, new in zip(before, trainer.params):
        assert torch.equal(old, new)


def test_resume_refuses_a_different_arm_and_reloads_for_evaluate_calls(tmp_path):
    trainer = _trainer(tmp_path, "proseco_basic", updates=1)
    summary = trainer.train()
    ckpt = tmp_path / "proseco_basic" / "checkpoints" / summary["last_checkpoint"]
    again = _trainer(tmp_path / "resume", "proseco_basic", updates=2)
    again.resume(ckpt)
    assert again.update == 1
    other = _trainer(tmp_path / "other", "scope", updates=2)
    with pytest.raises(ValueError, match="arm"):
        other.resume(ckpt)
    loaded, info = load_trained_adapter(ckpt, torch.device("cpu"))
    assert info["method"] == "full"
    layout_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    attention = torch.ones_like(layout_ids, dtype=torch.bool)
    logits = loaded.aligned_logits_from_ids(layout_ids, attention) if hasattr(loaded, "aligned_logits_from_ids") else None
    del logits
    from diffusion_correction.models.alignment import build_layout
    lay = build_layout([[1, 2]], [[3, 4, 5, 6]], TINY_TOKENS.pad_token_id)
    out = loaded.aligned_logits(lay)
    assert out.shape[0] == 1 and torch.isfinite(out).all()


def test_tiny_builder_matches_the_trainer_adapter():
    cfg = _cfg("proseco_gen_conf")
    adapter = build_tiny_adapter(seed=cfg["seed"], dtype=torch.float64)
    built, _info = build_training_adapter(cfg, torch.device("cpu"))
    assert adapter.tokens.mask_token_id == built.tokens.mask_token_id
