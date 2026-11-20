"""Milestone-4 draft bank: same-forward signals, Gate D, schema, atomic shards, resume, split separation,
step-0 signals, diagnostics and the verified examples report (tiny backbone, CPU)."""

from __future__ import annotations

import copy
import json
import random

import numpy as np
import pytest
import torch

from diffusion_correction.data.format import ResponseFormat, TokenizedFormat
from diffusion_correction.data.splits import sha256_bytes
from diffusion_correction.data.tasks import Example
from diffusion_correction.drafts import preflight as pf
from diffusion_correction.drafts.bank import (ShardStore, bank_identity, build_plan, collect_bank, load_bank_config,
                                              load_bank_examples, open_bank, read_complete_bank, validate_bank)
from diffusion_correction.drafts.collector import (DRAFT_SCHEMA, check_record, collect_batch, generation_view)
from diffusion_correction.drafts.diagnostics import compute, eta_squared, rankdata, spearman, write_diagnostics
from diffusion_correction.drafts.identity import generator_identity, sha256_file, sha256_json
from diffusion_correction.drafts.report import build_report, select_examples, verify_report
from diffusion_correction.drafts.step0 import read_step0, run_step0
from diffusion_correction.models.tiny import TINY_VOCAB, build_tiny_adapter
from diffusion_correction.sampling.monotone import allowed_token_mask, monotone_unmask

PLUS, MINUS = TINY_VOCAB.index("+"), TINY_VOCAB.index("-")
SPECIALS = frozenset({TINY_VOCAB.index("<pad>"), TINY_VOCAB.index("<mask>"), TINY_VOCAB.index("<eos>")})
TF = TokenizedFormat(fmt=ResponseFormat(9), sign_ids={"+": PLUS, "-": MINUS}, digit_ids=tuple(range(10)),
                     special_ids=SPECIALS, tokenizer_size=len(TINY_VOCAB))
SETTINGS = [("arithmetic", 2, 1), ("arithmetic", 3, 1), ("programs", 2, 1)]


def tiny_example(split: str, i: int, key: tuple, rng: random.Random) -> Example:
    fam, n, dig = key
    values = [rng.choice([v for v in range(-9, 10) if v != 0] + [0]) for _ in range(n)]
    return Example(problem_id=f"{split}-{fam}-{n}-{i:03d}", family=fam, split=split,
                   setting={"length": n, "operand_digits": dig}, signature=[], heldout_composition=False,
                   normalized_key=f"{split}{fam}{n}{i}", source="", inputs={},
                   value_names=[f"v{k}" for k in range(n)], values=values, final_answer=values[-1], value_bound=9,
                   width=1, prompt_text=f"prompt {split} {i}", response_text=TF.fmt.render(values),
                   prompt_ids=[rng.randrange(10, 17) for _ in range(rng.randint(3, 6))],
                   response_ids=TF.encode_values(values), slot_spans=TF.fmt.slot_spans(n))


def write_dataset(root, n_train=12, n_val=6):
    rng = random.Random(7)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"response_format": {"value_bound": 9}, "splits": {}}
    for split, n in (("train", n_train), ("val", n_val)):
        exs = [tiny_example(split, i, k, rng) for k in SETTINGS for i in range(n)]
        rng.shuffle(exs)
        data = "".join(json.dumps(e.to_dict(), sort_keys=True) + "\n" for e in exs).encode()
        (root / f"{split}.jsonl").write_bytes(data)
        manifest["splits"][split] = {"file": f"{split}.jsonl", "sha256": sha256_bytes(data), "n_examples": len(exs)}
    (root / "test.jsonl").write_text("this file must never be read\n")
    manifest["splits"]["test"] = {"file": "test.jsonl", "sha256": "0" * 64, "n_examples": 1}
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (root / "manifest.json").write_text(text)
    return sha256_bytes(text.encode()), manifest


def bank_config(data_dir, manifest_sha, manifest, budgets=("1", "2", "L"), shard_train=5):
    return {"name": "tiny-bank", "generator": {"update": 1, "model_sha256": "x" * 64},
            "data": {"dir": str(data_dir), "manifest_sha256": manifest_sha, "order": ["val", "train"],
                     "splits": {"val": {"sha256": manifest["splits"]["val"]["sha256"],
                                        "n_examples": manifest["splits"]["val"]["n_examples"], "settings": "all",
                                        "shard_problems": 6},
                                "train": {"sha256": manifest["splits"]["train"]["sha256"],
                                          "n_examples": manifest["splits"]["train"]["n_examples"], "settings": "all",
                                          "shard_problems": shard_train}}},
            "sampler": {"name": "d029_monotone_greedy", "budgets": list(budgets), "drafts_per_problem_per_budget": 1,
                        "rng": "none", "batch_size": 4, "precision": "fp32"},
            "preflight": {"batches": [{"split": "train", "setting": "arithmetic:3:1", "budget": "2", "n": 4}]}}


@pytest.fixture(scope="module")
def adapter():
    torch.manual_seed(0)
    return build_tiny_adapter(seed=3, dtype=torch.float32)


@pytest.fixture(scope="module")
def allowed():
    return allowed_token_mask(len(TINY_VOCAB), len(TINY_VOCAB), sorted(SPECIALS),
                              extra_forbidden=(TINY_VOCAB.index("<mask>"),))


@pytest.fixture
def dataset(tmp_path):
    sha, manifest = write_dataset(tmp_path / "data")
    cfg = bank_config(tmp_path / "data", sha, manifest)
    cfg_path = tmp_path / "bank.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(cfg))
    cfg = load_bank_config(cfg_path)
    examples, idents = load_bank_examples(cfg)
    plan = build_plan(cfg, examples, idents, TF.fmt.slot_len)
    return cfg, examples, plan


def fake_identity(cfg, plan, commit="abc"):
    gen = {"model_sha256": "x" * 64, "complete_json_sha256": "y" * 64, "update": 1, "tokenizer_json_sha256": None}
    env = {"git": {"commit": commit, "dirty": False}, "torch": torch.__version__, "device": "cpu",
           "precision": {"fp32": True}}
    return bank_identity(cfg, plan, gen, env, None)


def batch(examples, key=("arithmetic", 3, 1), n=4, split="train"):
    return [e for e in examples[split] if (e.family, e.setting["length"], e.setting["operand_digits"]) == key][:n]


# ---------------------------------------------------------------- signals and Gate D

def test_signals_come_from_the_commitment_forward(adapter, allowed, dataset):
    _, examples, _ = dataset
    exs = batch(examples)
    for budget in ("1", "2", "L"):
        res = collect_batch(adapter, exs, budget, allowed, TF, keep_states_for_calls=(1,))
        assert res.rank_score_mismatches == 0 and res.rng_unchanged
        for r, ex in zip(res.records, exs):
            assert r["schema"] == DRAFT_SCHEMA
            assert check_record(r, ex, TF) == []
            g = r["generation"]
            assert g["rank_score"] == g["log_c_gen"]
            # rank_score is the sampler's raw_logp (log_softmax), not float32 exp(log_c).
            for rs, lc in zip(g["rank_score"], g["log_c_gen"]):
                assert rs == lc
                c32 = float(torch.exp(torch.tensor(lc, dtype=torch.float32)))
                assert rs != c32
        # Recompute call 1 from the saved pre-commit state: same tensor, same op -> exact.
        state = res.observer.saved_states[1]
        from diffusion_correction.models.alignment import build_layout
        mask = adapter.tokens.mask_token_id
        L = len(exs[0].response_ids)
        layout = build_layout([e.prompt_ids for e in exs], [[mask] * L] * len(exs), adapter.tokens.pad_token_id)
        with torch.no_grad():
            logits = adapter.aligned_logits(layout.with_input_ids(state))
        for b, e in enumerate(exs):
            for p, c in enumerate(res.observer.commits[b]):
                if c.call == 1:
                    lp = torch.log_softmax(logits[b, len(e.prompt_ids) + p].float(), -1)
                    assert abs(float(lp[c.token_id]) - c.log_c) < 1e-6
                    assert abs(float(lp[e.response_ids[p]]) - c.log_q) < 1e-6


def test_gate_d_reference_targets_change_only_analysis(adapter, allowed, dataset):
    _, examples, _ = dataset
    exs = batch(examples, n=4)
    true = collect_batch(adapter, exs, "2", allowed, TF)
    perm = collect_batch(adapter, pf.permuted_references(exs), "2", allowed, TF)
    plain = monotone_unmask(adapter, [e.prompt_ids for e in exs], [len(exs[0].response_ids)] * 4, 2, allowed)
    assert [generation_view(r) for r in true.records] == [generation_view(r) for r in perm.records]
    assert [r["generation"]["draft_ids"] for r in true.records] == plain.responses
    assert any(a["analysis"] != b["analysis"] for a, b in zip(true.records, perm.records))


def test_preflight_gates_pass_and_catch_target_leakage(adapter, allowed, dataset, monkeypatch):
    _, examples, _ = dataset
    exs = batch(examples)
    ok = pf.run_batch_gates(adapter, TF, allowed, exs, "2", None)
    assert ok["passed"], ok["checks"]
    assert ok["checks"]["same_forward_recompute_compared"] > 0
    assert ok["checks"]["independent_single_example_max_abs_logp_diff"] <= pf.INDEPENDENT_LOGP_TOL
    # Extra forwards (same-forward recompute + B=1 checks) are counted on the preflight
    # report only; they are not mixed into collection shard accounting.
    assert ok["batched_forwards"] == 3 * ok["calls"] + len(ok["saved_calls"]) * (1 + len(exs))
    assert set(ok["seconds"]) >= {"collect_true", "collect_permuted", "generate_plain",
                                  "recompute_and_independent"}

    real = pf.collect_batch

    def leaky(adapter_, exs_, budget, allowed_, tf, **kw):
        res = real(adapter_, exs_, budget, allowed_, tf, **kw)
        for r, e in zip(res.records, exs_):     # generation that depends on the reference
            r["generation"]["draft_ids"] = list(e.response_ids)
        return res
    monkeypatch.setattr(pf, "collect_batch", leaky)
    bad = pf.run_batch_gates(adapter, TF, allowed, exs, "2", None)
    assert not bad["passed"]
    assert bad["checks"]["gate_d_generation_identical_true_vs_permuted"] is False


def test_preflight_reproduction_gate(adapter, allowed, dataset, tmp_path):
    cfg, examples, plan = dataset
    exs = batch(examples, split="val", n=4)
    res = collect_batch(adapter, exs, "2", allowed, TF)
    ref = {(e.problem_id, "2"): r["generation"]["draft_ids"] for e, r in zip(exs, res.records)}
    assert pf.run_batch_gates(adapter, TF, allowed, exs, "2", ref)["passed"]
    k = next(iter(ref))
    ref[k] = [0] * len(ref[k])
    out = pf.run_batch_gates(adapter, TF, allowed, exs, "2", ref)
    assert not out["passed"] and out["checks"]["reproduction_differing"] == [k[0]]


def test_check_record_detects_tampering(adapter, allowed, dataset):
    _, examples, _ = dataset
    exs = batch(examples)
    rec = collect_batch(adapter, exs, "L", allowed, TF).records[0]
    assert check_record(rec, exs[0], TF) == []
    for mutate in (lambda r: r["generation"]["log_c_gen"].__setitem__(0, r["generation"]["log_c_gen"][0] - 0.1),
                   lambda r: r["analysis"]["token_correct"].__setitem__(0, not r["analysis"]["token_correct"][0]),
                   lambda r: r["generation"]["mask_fraction_before"].__setitem__(1, 0.123),
                   lambda r: r["generation"]["commit_order"].__setitem__(0, 99),
                   lambda r: r["analysis"]["reference_ids"].__setitem__(0, 3 if r["analysis"]["reference_ids"][0] != 3 else 4),
                   lambda r: r["generation"]["log_c_gen"].__setitem__(0, 0.1 + 1e-9)):
        bad = copy.deepcopy(rec)
        mutate(bad)
        assert check_record(bad, exs[0], TF), mutate


# ---------------------------------------------------------------- plan, shards, resume, validation

def test_plan_is_train_val_only_and_deterministic(dataset, tmp_path):
    cfg, examples, plan = dataset
    assert {s["split"] for s in plan["shards"]} == {"train", "val"}
    assert plan["totals"]["drafts"] == 3 * (12 + 6) * 3
    assert plan["shards"][0]["split"] == "val"
    again = build_plan(cfg, examples, plan["data_identity"], TF.fmt.slot_len)
    assert sha256_json(again) == sha256_json(plan)
    bad = copy.deepcopy(cfg)
    bad["data"]["splits"]["test"] = {"sha256": "0", "n_examples": 1, "shard_problems": 1}
    import yaml
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="test splits are never loaded"):
        load_bank_config(p)
    bad["data"]["order"] = ["test"]
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError):
        load_bank_config(p)


def _collect(root, cfg, plan, examples, adapter, allowed, **kw):
    return collect_bank(root, plan, fake_identity(cfg, plan), adapter, TF, allowed, examples, log=lambda s: None, **kw)


def test_bank_collect_stop_resume_validate_finalize(adapter, allowed, dataset, tmp_path):
    cfg, examples, plan = dataset
    root = tmp_path / "bank"
    ident = fake_identity(cfg, plan)
    open_bank(root, ident, plan, resume=False)
    st = _collect(root, cfg, plan, examples, adapter, allowed, max_shards=3)
    assert st["state"] == "stopped_max_shards" and st["shards_complete"] == 3
    store = ShardStore(root)
    first = {sid: sha256_file(store.data_path(sid)) for sid in store.complete_ids()}
    with pytest.raises(SystemExit, match="not empty"):
        open_bank(root, ident, plan, resume=False)
    # Simulate an interrupted write: a .tmp file and a data file without its sidecar.
    nxt = [s["id"] for s in plan["shards"] if s["id"] not in first]
    store.data_path(nxt[0]).parent.mkdir(parents=True, exist_ok=True)
    store.data_path(nxt[0]).write_text("partial")
    (store.data_path(nxt[1]).parent / "x.jsonl.tmp").parent.mkdir(parents=True, exist_ok=True)
    (store.data_path(nxt[1]).parent / "x.jsonl.tmp").write_text("partial")
    info = open_bank(root, ident, plan, resume=True)
    assert len(info["moved_leftovers"]) == 2
    st = _collect(root, cfg, plan, examples, adapter, allowed, stop_at_unix=0)
    assert st["state"] == "stopped_deadline"
    st = _collect(root, cfg, plan, examples, adapter, allowed)
    assert st["state"] == "collection_complete" and st["shards_complete"] == len(plan["shards"])
    assert all(sha256_file(store.data_path(s)) == h for s, h in first.items())    # never rewritten
    rep = validate_bank(root, plan, ident, examples, TF, finalize=True)
    assert rep["passed"], rep["problems"]
    assert rep["totals"]["drafts"] == plan["totals"]["drafts"]
    manifest, shards = read_complete_bank(root)
    assert sum(len(r) for _, r in shards) == plan["totals"]["drafts"]
    with pytest.raises(SystemExit, match="already complete"):
        open_bank(root, ident, plan, resume=True)


def test_resume_refuses_changed_identity_and_corrupt_shards(adapter, allowed, dataset, tmp_path):
    cfg, examples, plan = dataset
    root = tmp_path / "bank"
    open_bank(root, fake_identity(cfg, plan), plan, resume=False)
    _collect(root, cfg, plan, examples, adapter, allowed, max_shards=2)
    with pytest.raises(SystemExit, match="identity changed"):
        open_bank(root, fake_identity(cfg, plan, commit="other"), plan, resume=True)
    store = ShardStore(root)
    sid = sorted(store.complete_ids())[0]
    p = store.data_path(sid)
    p.write_text(p.read_text().replace('"budget"', '"budget" ', 1))
    open_bank(root, fake_identity(cfg, plan), plan, resume=True)
    with pytest.raises(ValueError, match="sha256"):
        _collect(root, cfg, plan, examples, adapter, allowed)


def test_validation_detects_missing_duplicate_and_split_mixing(adapter, allowed, dataset, tmp_path):
    cfg, examples, plan = dataset
    root = tmp_path / "bank"
    ident = fake_identity(cfg, plan)
    open_bank(root, ident, plan, resume=False)
    _collect(root, cfg, plan, examples, adapter, allowed)
    store = ShardStore(root)
    sids = [s["id"] for s in plan["shards"]]
    # Rewrite one train shard so a record claims a val problem (and duplicates a val record).
    tr = next(s for s in sids if s.startswith("train/"))
    va = next(s for s in sids if s.startswith("val/"))
    recs, vrec = store.read(tr), store.read(va)[0]
    side = json.loads(store.sidecar_path(tr).read_text())
    recs[0] = {**vrec, "shard": tr}
    keep = {k: side[k] for k in side if k not in ("schema", "shard_id", "file", "sha256", "bytes", "n_records",
                                                   "problem_ids_sha256")}
    store.write(tr, recs, keep)
    rep = validate_bank(root, plan, ident, examples, TF, finalize=True)
    assert not rep["passed"] and not (root / "BANK_COMPLETE").exists()
    assert any("problem_ids_sha256" in p for p in rep["problems"])
    # Forge a self-consistent sidecar: the record-level checks must still catch the val record.
    side = json.loads(store.sidecar_path(tr).read_text())
    side["problem_ids_sha256"] = sha256_json(next(s for s in plan["shards"] if s["id"] == tr)["problem_ids"])
    store.sidecar_path(tr).write_text(json.dumps(side))
    rep = validate_bank(root, plan, ident, examples, TF, finalize=True)
    text = " ".join(rep["problems"])
    assert not rep["passed"] and not (root / "BANK_COMPLETE").exists()
    assert "duplicate record" in text and "planned problem IDs" in text and "differ from its shard" in text
    store.sidecar_path(sids[-1]).unlink()
    rep = validate_bank(root, plan, ident, examples, TF)
    assert rep["missing_shards"] == 1 and not rep["passed"]


# ---------------------------------------------------------------- identity

def test_generator_identity_verifies_files(tmp_path):
    ck = tmp_path / "ckpt"
    ck.mkdir()
    (ck / "model.safetensors").write_bytes(b"weights")
    (ck / "state.json").write_text(json.dumps({"update": 5, "model_file": "model.safetensors",
                                               "config": {"run_name": "r"}, "base_model": {"file_sha256": {
                                                   "tokenizer.json": "t" * 64}}}))
    files = {n: sha256_file(ck / n) for n in ("model.safetensors", "state.json")}
    (ck / "COMPLETE.json").write_text(json.dumps({"files": files, "update": 5}))
    g = generator_identity(ck, files["model.safetensors"], 5)
    assert g["model_sha256"] == files["model.safetensors"] and g["tokenizer_json_sha256"] == "t" * 64
    with pytest.raises(ValueError, match="pinned"):
        generator_identity(ck, "0" * 64)
    (ck / "model.safetensors").write_bytes(b"other")
    with pytest.raises(ValueError, match="COMPLETE.json"):
        generator_identity(ck)


# ---------------------------------------------------------------- step 0, diagnostics, report

@pytest.fixture
def complete_bank(adapter, allowed, dataset, tmp_path):
    cfg, examples, plan = dataset
    root = tmp_path / "bank"
    ident = fake_identity(cfg, plan)
    open_bank(root, ident, plan, resume=False)
    _collect(root, cfg, plan, examples, adapter, allowed)
    assert validate_bank(root, plan, ident, examples, TF, finalize=True)["passed"]
    return root, examples


def test_step0_signals(adapter, complete_bank):
    root, _ = complete_bank
    gen = {"model_sha256": "x" * 64}
    env = {"git": {"commit": "abc"}, "torch": torch.__version__, "device": "cpu"}
    s = run_step0(root, adapter, gen, env, batch_size=4)
    assert s["state"] == "complete"
    manifest, shards = read_complete_bank(root)
    s0 = read_step0(root, manifest["manifest_sha256"])
    r = shards[0][1][0]
    x = s0[(r["split"], r["problem_id"], r["budget"])]
    from diffusion_correction.models.alignment import build_layout
    lay = build_layout([r["prompt_ids"]], [r["generation"]["draft_ids"]], adapter.tokens.pad_token_id)
    with torch.no_grad():
        lp = torch.log_softmax(adapter.aligned_logits(lay)[0].float(), -1)
    P = len(r["prompt_ids"])
    for p, t in enumerate(r["analysis"]["reference_ids"]):
        assert abs(float(lp[P + p, t]) - x["step0_log_p_ref"][p]) < 1e-5
    with pytest.raises(SystemExit):
        run_step0(root, adapter, {"model_sha256": "z" * 64}, env, batch_size=4, resume=True)


def test_diagnostics_counts_match_bank(adapter, complete_bank):
    root, _ = complete_bank
    run_step0(root, adapter, {"model_sha256": "x" * 64}, {"git": {}, "device": "cpu"}, batch_size=4)
    d = write_diagnostics(root)
    _, shards = read_complete_bank(root)
    recs = [r for _, rs in shards for r in rs]
    assert d["n_drafts"] == len(recs)
    assert d["n_tokens"] == sum(r["response_length"] for r in recs)
    wrong = sum(r["analysis"]["n_token_errors"] for r in recs)
    assert sum(row["wrong"] for row in d["rates"]["by_split_family_budget"]) == wrong
    assert d["has_step0"] and "q7_step0_keep_loss" in d
    assert (root / "derived" / "diagnostics" / "diagnostics.md").exists()


def test_report_is_deterministic_and_verified(complete_bank, tmp_path):
    root, examples = complete_bank
    texts = {(e.split, e.problem_id): {"prompt_text": e.prompt_text, "response_text": e.response_text}
             for v in examples.values() for e in v}
    decode = lambda i: TINY_VOCAB[i]  # noqa: E731
    rep = build_report(root, decode, texts)
    assert rep["verification"]["passed"], rep["verification"]["problems"]
    _, shards = read_complete_bank(root)
    assert select_examples(shards) == select_examples(shards)
    cells = {(r["setting"], r["budget"]) for ex in rep["examples"] for r in [ex["display"]]
             if ex["category"].startswith("coverage")}
    train_cells = {(f"{r['family']}:{r['length']}:{r['operand_digits']}", r["budget"])
                   for _, rs in shards for r in rs if r["split"] == "train"}
    assert cells == train_cells
    out = root / "derived" / "report"
    md = (out / "examples.md").read_text()
    assert "Diagnostic sample" in md and "Full-bank counts" in md
    j = json.loads((out / "examples.json").read_text())
    j["examples"][0]["display"]["tokens"][0]["log_c_gen"] += 0.5
    (out / "examples.json").write_text(json.dumps(j))
    assert not verify_report(root, out, decode, texts)["passed"]


# ---------------------------------------------------------------- statistics helpers

def test_rank_statistics():
    assert rankdata(np.array([3.0, 1.0, 1.0, 2.0])).tolist() == [4.0, 1.5, 1.5, 3.0]
    x = np.arange(10.0)
    assert spearman(x, x ** 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert spearman(x, np.ones(10)) is None
    y = np.array([1.0, 1.0, 5.0, 5.0])
    assert eta_squared(y, np.array(["a", "a", "b", "b"])) == pytest.approx(1.0)
    assert eta_squared(y, np.array(["a", "b", "a", "b"])) == pytest.approx(0.0)


def test_diagnostics_compute_needs_complete_bank(dataset, tmp_path):
    with pytest.raises(FileNotFoundError):
        compute(tmp_path / "nothing")
