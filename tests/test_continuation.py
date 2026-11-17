"""D-033 continuation rules on the tiny backbone: early-stopping state, training-time cap,
graceful deadline, checkpoint protection, and the monitoring score's weighting."""

import json

import pytest
import torch

from diffusion_correction.evaluation.generation import selection_score
from diffusion_correction.training.trainer import Trainer

from test_trainer import make_trainer, read_metrics, tiny_cfg

SEL = {"every_updates": 2, "generate_budgets": ["L"], "selection": {"budget": "L", "patience": 2}}


class Scripted(Trainer):
    """Replaces generation with a scripted monitoring score per update (masked loss stays real)."""

    scores: dict = {}

    def evaluate(self, scheduled=None):
        ev = {"update": self.update, "masked_loss": 1.0, "scheduled": scheduled,
              "selection_score": {"score": self.scores.get(self.update, 0.0), "per_family": {}}}
        self.last_eval = ev
        return ev


def scripted(cfg, run_dir, scores):
    tr = make_trainer(cfg, run_dir)
    tr.__class__ = Scripted
    tr.scores = scores
    return tr


def make_pre_d033(ck):
    """Rewrite a checkpoint as the pre-D-033 trainer wrote it (no timers/selection), re-hashed."""
    import hashlib

    state = json.loads((ck / "state.json").read_text())
    state.pop("timers")
    state.pop("selection")
    (ck / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    info = json.loads((ck / "COMPLETE.json").read_text())
    info["files"]["state.json"] = hashlib.sha256((ck / "state.json").read_bytes()).hexdigest()
    (ck / "COMPLETE.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")


def cfg(**over):
    base = dict(eval=SEL, checkpoint={"every_updates": 2, "keep_last": 1}, limits={"max_updates": 20})
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    return tiny_cfg(**base)


def test_selection_score_weights_families_then_settings_equally():
    def rec(fam, n, d, ok):
        return {"family": fam, "length": n, "operand_digits": d, "final_correct": ok, "trace_correct": ok,
                "value_correct": [ok], "response_ids": [1], "n_token_errors": 0, "slot_status": ["ok"], "calls": 1}
    # arithmetic: one setting at 1.0 (1 example) and one at 0.0 (9 examples) -> family 0.5
    # programs: a single setting at 0.25 -> family 0.25; overall 0.375 (not pooled 3/14)
    recs = [rec("arithmetic", 2, 1, True)] + [rec("arithmetic", 4, 2, False)] * 9 + \
           [rec("programs", 2, 1, k == 0) for k in range(4)]
    s = selection_score(recs)
    assert s["per_family"] == {"arithmetic": 0.5, "programs": 0.25}
    assert s["score"] == pytest.approx(0.375)


def test_patience_stops_after_three_non_improving_evals_and_ties_keep_earlier(tmp_path):
    scores = {0: 0.1, 2: 0.3, 4: 0.3, 6: 0.2, 8: 0.29, 10: 0.9}
    c = cfg(eval={"at_start": True, "selection": {"budget": "L", "patience": 3}})
    tr = scripted(c, tmp_path / "a", scores)
    s = tr.train()
    assert s["stop_reason"] == "early_stop_patience" and tr.update == 8
    sel = tr.selection
    assert sel["best_update"] == 2 and sel["best_score"] == 0.3  # the tie at 4 keeps update 2
    assert [h["update"] for h in sel["history"]] == [0, 2, 4, 6, 8]
    assert sel["evals_without_improvement"] == 3
    ckpts = sorted(p.name for p in (tmp_path / "a" / "checkpoints").glob("ckpt-*"))
    assert "ckpt-0000002" in ckpts and "ckpt-0000008" in ckpts  # best kept despite keep_last=1


def test_selection_state_and_timer_survive_resume(tmp_path):
    scores = {4: 0.5, 6: 0.4, 8: 0.6, 10: 0.6, 12: 0.1}
    first = make_trainer(tiny_cfg(limits={"max_updates": 4}, checkpoint={"every_updates": 2}), tmp_path / "w")
    first.train()
    ck4 = tmp_path / "w" / "checkpoints" / "ckpt-0000004"
    make_pre_d033(ck4)  # like ckpt-0001000: no timers, no selection state
    c = cfg(limits={"max_updates": 7, "max_train_seconds": 1e6, "train_seconds_origin_update": 4})
    a = scripted(c, tmp_path / "c", scores)
    a.resume(ck4)
    a.train()
    assert [h["update"] for h in a.selection["history"]] == [4, 6]  # baseline at the resumed update
    assert a.selection["best_checkpoint"] == str(ck4)
    t7 = a.timers["train_update_seconds"]
    assert a.timers["origin_update"] == 4 and t7 > 0
    b = scripted(cfg(limits={"max_updates": 20, "max_train_seconds": 1e6, "train_seconds_origin_update": 4}),
                 tmp_path / "c", scores)
    b.resume(tmp_path / "c" / "checkpoints" / "ckpt-0000007")
    assert b.selection["evals_without_improvement"] == 1 and b.timers["train_update_seconds"] == pytest.approx(t7)
    s = b.train()
    # 8 improves (0.6); 10 ties; 12 worse -> 2 non-improving evals = patience 2
    assert s["stop_reason"] == "early_stop_patience" and b.update == 12 and b.selection["best_update"] == 8
    with pytest.raises(ValueError, match="cannot be re-based"):
        bad = scripted(cfg(limits={"max_train_seconds": 1e6, "train_seconds_origin_update": 0}), tmp_path / "c",
                       scores)
        bad.resume(tmp_path / "c" / "checkpoints" / "ckpt-0000012")
        bad.train()
    with pytest.raises(ValueError, match="cannot be reset or changed"):
        bad = scripted(cfg(eval={"selection": {"budget": "L", "patience": 5}}), tmp_path / "c", scores)
        bad.resume(tmp_path / "c" / "checkpoints" / "ckpt-0000012")
        bad.train()


def test_training_time_cap_and_deadline(tmp_path):
    tr = make_trainer(tiny_cfg(limits={"max_updates": 50, "max_train_seconds": 1e-9,
                                       "train_seconds_origin_update": 0}), tmp_path / "t")
    s = tr.train()
    assert s["stop_reason"] == "max_train_seconds" and tr.update == 1  # checked before each update
    m = read_metrics(tmp_path / "t")
    assert m[-1]["train_update_seconds"] > 0
    d = make_trainer(tiny_cfg(limits={"max_updates": 50, "stop_at_unix": 1.0}), tmp_path / "d")
    s = d.train()
    assert s["stop_reason"] == "deadline" and d.update == 0
    assert not (tmp_path / "d" / "evals.jsonl").exists()  # graceful exit skips the unscheduled evaluation
    assert (tmp_path / "d" / "checkpoints" / "ckpt-0000000" / "COMPLETE.json").exists()


def test_protected_checkpoints_survive_pruning(tmp_path):
    tr = make_trainer(tiny_cfg(limits={"max_updates": 10},
                               checkpoint={"every_updates": 1, "keep_last": 1, "protect_updates": [3],
                                           "protect_every": 4, "durable_dir": str(tmp_path / "keep")}),
                      tmp_path / "p")
    tr.train()
    local = sorted(p.name for p in (tmp_path / "p" / "checkpoints").glob("ckpt-*"))
    durable = sorted(p.name for p in (tmp_path / "keep" / "t" / "checkpoints").glob("ckpt-*"))
    assert local == durable == ["ckpt-0000003", "ckpt-0000004", "ckpt-0000008", "ckpt-0000010"]


def test_evaluation_restores_mode_and_leaves_training_streams(tmp_path):
    tr = make_trainer(tiny_cfg(), tmp_path / "e")
    tr.run_dir.mkdir(parents=True)
    tr.adapter.model.train()
    before = {k: g.get_state().clone() for k, g in tr.gens.items() if k != "eval"}
    tr.evaluate()
    assert tr.adapter.model.training
    assert all(torch.equal(before[k], tr.gens[k].get_state()) for k in before)
    tr.adapter.model.eval()
    tr.evaluate()
    assert not tr.adapter.model.training


def test_selection_config_is_validated():
    with pytest.raises(ValueError, match="dividing eval.every_updates"):
        cfg(checkpoint={"every_updates": 3})
    with pytest.raises(ValueError, match="generate_budgets"):
        cfg(eval={"generate_budgets": [4]})
    with pytest.raises(ValueError, match="train_seconds_origin_update"):
        tiny_cfg(limits={"max_train_seconds": 10})
    assert json.dumps(cfg()["eval"]["selection"]) == '{"budget": "L", "patience": 2}'
