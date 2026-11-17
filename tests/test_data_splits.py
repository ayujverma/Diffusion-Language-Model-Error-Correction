"""Split disjointness, held-out compositions, deterministic manifests, and the CLI."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from diffusion_correction.cli import main
from diffusion_correction.data.format import ResponseFormat, TokenizedFormat
from diffusion_correction.data.splits import (SPLITS, DataConfig, assert_grouped_by_problem, contains_bigram,
                                              generate_splits, load_split, sha256_bytes, verify_dataset,
                                              write_splits)
from diffusion_correction.data.tasks import problem_id

pytestmark = pytest.mark.tokenizer
SMOKE = Path("configs/data/smoke.yaml")


def _generate(tmp: Path, name: str, seed: int | None = None) -> Path:
    out = tmp / name
    args = ["generate-data", "--config", str(SMOKE), "--out", str(out)]
    if seed is not None:
        args += ["--seed", str(seed)]
    assert main(args) == 0
    return out


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory, task_tokenizer):
    return _generate(tmp_path_factory.mktemp("data"), "smoke")


def test_generation_is_deterministic_and_seed_sensitive(tmp_path, smoke_dir):
    again = _generate(tmp_path, "again")
    other = _generate(tmp_path, "other", seed=1)
    m1, m2, m3 = ((d / "manifest.json").read_bytes() for d in (smoke_dir, again, other))
    assert m1 == m2
    assert m1 != m3
    for s in SPLITS:
        assert (smoke_dir / f"{s}.jsonl").read_bytes() == (again / f"{s}.jsonl").read_bytes()
    prov = json.loads((again / "provenance.json").read_text())
    assert prov["manifest_sha256"] == sha256_bytes(m2) and prov["verification"]["passed"]


def test_splits_are_disjoint_and_compositions_are_held_out(smoke_dir):
    cfg = DataConfig.from_dict(yaml.safe_load(SMOKE.read_text()))
    owner, keys = {}, {}
    for s in SPLITS:
        recs = load_split(smoke_dir, s)
        assert recs
        for r in recs:
            assert r.problem_id not in owner and r.normalized_key not in keys
            owner[r.problem_id], keys[r.normalized_key] = s, s
            fam = cfg.families[r.family]
            assert contains_bigram(r.signature, fam.heldout_bigrams) == (s == "test_unseen")
            if s == "test_long":
                assert r.setting["length"] in fam.long_lengths
            else:
                assert r.setting["length"] in fam.lengths
    # Every (family, length, digits) setting of each split has exactly the configured count.
    manifest = json.loads((smoke_dir / "manifest.json").read_text())
    for s in SPLITS:
        assert set(manifest["splits"][s]["counts"].values()) == {cfg.per_setting[s]}


def test_verify_detects_tampering_and_cross_split_duplicates(tmp_path, smoke_dir, task_tokenizer):
    bad = tmp_path / "bad"
    shutil.copytree(smoke_dir, bad)
    with open(bad / "val.jsonl", "a") as f:
        f.write((bad / "train.jsonl").read_text().splitlines()[0] + "\n")
    with pytest.raises(AssertionError, match="hash"):
        verify_dataset(bad, task_tokenizer)

    # Same duplicate but with the manifest re-hashed: the disjointness check must catch it.
    manifest = json.loads((bad / "manifest.json").read_text())
    data = (bad / "val.jsonl").read_bytes()
    manifest["splits"]["val"].update(sha256=sha256_bytes(data), n_examples=manifest["splits"]["val"]["n_examples"] + 1)
    (bad / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="split|also in"):
        verify_dataset(bad, task_tokenizer)


def test_verify_detects_a_wrong_label(tmp_path, smoke_dir, task_tokenizer):
    bad = tmp_path / "label"
    shutil.copytree(smoke_dir, bad)
    lines = (bad / "test_id.jsonl").read_text().splitlines()
    rec = json.loads(lines[0])
    rec["values"][0] += 1
    lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    data = ("\n".join(lines) + "\n").encode()
    (bad / "test_id.jsonl").write_bytes(data)
    manifest = json.loads((bad / "manifest.json").read_text())
    manifest["splits"]["test_id"]["sha256"] = sha256_bytes(data)
    (bad / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match="values differ"):
        verify_dataset(bad, task_tokenizer)


def _tampered_copy(tmp_path: Path, smoke_dir: Path, edit) -> Path:
    """Copy the dataset, apply ``edit(records_by_split)``, and re-hash the manifest so that
    verification gets past the file-hash and count checks to the semantic checks."""
    bad = tmp_path / "tampered"
    shutil.copytree(smoke_dir, bad)
    recs = {s: [json.loads(line) for line in (bad / f"{s}.jsonl").read_text().splitlines()] for s in SPLITS}
    edit(recs)
    manifest = json.loads((bad / "manifest.json").read_text())
    for s in SPLITS:
        data = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in recs[s]).encode()
        (bad / f"{s}.jsonl").write_bytes(data)
        manifest["splits"][s].update(sha256=sha256_bytes(data), n_examples=len(recs[s]))
    (bad / "manifest.json").write_text(json.dumps(manifest))
    return bad


def _first(recs, split, family):
    return next(r for r in recs[split] if r["family"] == family)


@pytest.mark.parametrize("family, candidates", [
    ("arithmetic", ("*", "+")),                   # constant signatures: no "-" then "*"
    ("programs", ("add_const", "sub_const")),     # constant signatures: no "mul_const" then "sub_var"
])
def test_verify_rejects_wrong_signature_with_correct_heldout_status(tmp_path, smoke_dir, task_tokenizer,
                                                                    family, candidates):
    cfg = DataConfig.from_dict(yaml.safe_load(SMOKE.read_text()))

    def edit(recs):
        r = _first(recs, "train", family)
        new = next(sig for c in candidates if (sig := [c] * len(r["signature"])) != r["signature"])
        assert not contains_bigram(new, cfg.families[family].heldout_bigrams)
        r["signature"] = new

    with pytest.raises(AssertionError, match="stored signature .* != source signature"):
        verify_dataset(_tampered_copy(tmp_path, smoke_dir, edit), task_tokenizer)


@pytest.mark.parametrize("family", ["arithmetic", "programs"])
def test_verify_rejects_heldout_source_labelled_as_not_heldout(tmp_path, smoke_dir, task_tokenizer, family):
    def edit(recs):
        r = _first(recs, "test_unseen", family)
        recs["test_unseen"].remove(r)
        # The source still contains the held-out pair; the metadata claims otherwise.
        r.update(split="train", heldout_composition=False,
                 signature=["+"] * len(r["signature"]) if family == "arithmetic"
                 else ["add_const"] * len(r["signature"]))
        recs["train"].insert(0, r)

    with pytest.raises(AssertionError, match="is held out, but split is train"):
        verify_dataset(_tampered_copy(tmp_path, smoke_dir, edit), task_tokenizer)


def test_verify_rejects_stale_heldout_flag(tmp_path, smoke_dir, task_tokenizer):
    def edit(recs):
        _first(recs, "test_unseen", "programs")["heldout_composition"] = False

    with pytest.raises(AssertionError, match="heldout_composition flag differs"):
        verify_dataset(_tampered_copy(tmp_path, smoke_dir, edit), task_tokenizer)


def test_verify_rejects_normalized_key_not_rebuilt_from_source(tmp_path, smoke_dir, task_tokenizer):
    def edit(recs):
        r = _first(recs, "val", "programs")
        # Self-consistent but wrong: problem_id is the hash of the stored (wrong) key.
        r["normalized_key"] = r["normalized_key"].replace("=", "= ", 1)
        r["problem_id"] = problem_id(r["normalized_key"])

    with pytest.raises(AssertionError, match="stored normalized_key differs"):
        verify_dataset(_tampered_copy(tmp_path, smoke_dir, edit), task_tokenizer)


def test_verify_rejects_problem_id_not_hash_of_rebuilt_key(tmp_path, smoke_dir, task_tokenizer):
    def edit(recs):
        _first(recs, "test_id", "arithmetic")["problem_id"] = "0" * 16

    with pytest.raises(AssertionError, match="problem_id"):
        verify_dataset(_tampered_copy(tmp_path, smoke_dir, edit), task_tokenizer)


def test_verify_rejects_duplicate_problem_across_splits(tmp_path, smoke_dir, task_tokenizer):
    """Uniqueness is checked on the identity rebuilt from source, after re-hashing."""
    def edit(recs):
        r = dict(_first(recs, "val", "programs"))
        r["split"] = "train"
        recs["train"].append(r)

    with pytest.raises(AssertionError, match="also in"):
        verify_dataset(_tampered_copy(tmp_path, smoke_dir, edit), task_tokenizer)


def test_verify_rejects_source_outside_generated_structure(tmp_path, smoke_dir, task_tokenizer):
    def edit(recs):
        r = _first(recs, "train", "arithmetic")
        r["source"] = r["source"].replace(" + ", "+", 1).replace(" - ", "-", 1).replace(" * ", "*", 1)

    with pytest.raises(AssertionError, match="outside the generated structure"):
        verify_dataset(_tampered_copy(tmp_path, smoke_dir, edit), task_tokenizer)


def test_derived_records_must_stay_in_their_problems_split(smoke_dir):
    train = load_split(smoke_dir, "train")[0]
    val = load_split(smoke_dir, "val")[0]
    ok = [{"problem_id": train.problem_id, "split": "train", "draft": 0},
          {"problem_id": train.problem_id, "split": "train", "draft": 1},
          {"problem_id": val.problem_id, "split": "val", "draft": 0}]
    assert_grouped_by_problem(ok, smoke_dir)
    with pytest.raises(AssertionError, match="is in val"):
        assert_grouped_by_problem([{"problem_id": train.problem_id, "split": "val"}], smoke_dir)
    with pytest.raises(AssertionError, match="unknown problem"):
        assert_grouped_by_problem([{"problem_id": "0" * 16, "split": "train"}], smoke_dir)


def test_too_small_problem_space_fails_loudly(tmp_path, task_tokenizer):
    raw = yaml.safe_load(SMOKE.read_text())
    raw["operand_digits"] = [1]
    raw["families"] = {"arithmetic": {"lengths": [2], "long_lengths": [3], "heldout_bigrams": [["-", "*"]]}}
    raw["per_setting"] = {"train": 6000, "val": 1, "test_id": 1, "test_long": 1, "test_unseen": 1}
    raw["max_attempts_per_problem"] = 40
    # One-digit, two-operation chains: (81 + 81 + 72) first statements x (9 + 9 + 8) second
    # operations = 6084 problems, minus the 648 held-out "- then *" chains, leaves at most 5436 for
    # train/val/test_id. Asking for 6000 training problems must stop with an explicit error.
    with pytest.raises(RuntimeError, match="train/arithmetic/length=2/digits=1.*too small"):
        generate_splits(DataConfig.from_dict(raw), task_tokenizer)


def test_config_rejects_overlapping_long_lengths():
    raw = yaml.safe_load(SMOKE.read_text())
    raw["families"]["arithmetic"]["long_lengths"] = [6, 8]
    with pytest.raises(ValueError, match="overlap"):
        DataConfig.from_dict(raw)


def test_manifest_records_format_and_tokenizer_facts(smoke_dir):
    m = json.loads((smoke_dir / "manifest.json").read_text())
    assert m["response_format"]["eos_in_response"] is False
    assert m["response_format"]["sign_ids"] == {"+": 488, "-": 481}
    assert m["tokenizer"]["revision"] == "d0d86d5b99960c05258bb1f8265dd91564dbac67"
    assert m["heldout_compositions"] == {"arithmetic": [["-", "*"]], "programs": [["mul_const", "sub_var"]]}
    assert "time" not in json.dumps(m)  # deterministic manifest: run-time facts live in provenance.json


# ---------------------------------------------------------------- inspect-example

def test_inspect_example_fixture_table(tmp_path, capsys):
    out = tmp_path / "t.json"
    assert main(["inspect-example", "--fixture", "arith_17_28", "--draft-values", "46,135,116",
                 "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "HYPOTHETICAL" in printed and "' +0045 +0135 +0116'" in printed
    t = json.loads(out.read_text())
    resp = [r for r in t["rows"] if r["seg"] == "response"]
    pads = [r for r in t["rows"] if r["seg"] == "pad"]
    assert len(resp) == 15 and len(pads) == 2
    for r in resp:
        assert int(r["aligned_from_raw_row"]) == int(r["pos"]) - 1
        assert (r["attn"], r["prompt"], r["resp"], r["edit"]) == ("1", "0", "1", "1")
        assert all(r[c] == "n/a" for c in ("c", "q", "-log_q", "bin", "p_now", "w", "CE", "wCE"))
    assert all((r["attn"], r["edit"], r["token_id"]) == ("0", "0", "151643") for r in pads)
    wrong = [r for r in resp if r["label"].startswith("E")]
    assert [(r["slot"], r["role"], r["token"], r["draft_token"]) for r in wrong] == [("a[0]", "digit3", "'5'", "'6'")]
    assert resp[0]["token_id"] == "488" and resp[0]["role"] == "sign" and resp[0]["slot_value"] == "45"
    assert t["check"]["value_correct"] == [False, True, True] and t["check"]["final_correct"] is True


def test_inspect_example_length_mismatch_draft_is_not_aligned(tmp_path, capsys):
    out = tmp_path / "t.json"
    assert main(["inspect-example", "--fixture", "arith_2_plus_3", "--draft-text", " +005", "--json", str(out)]) == 0
    t = json.loads(out.read_text())
    assert "draft_id" not in t["columns"]
    assert t["check"]["length_ok"] is False and t["check"]["slot_status"] == ["length_mismatch"]
    assert "not aligned" in capsys.readouterr().out


def test_inspect_example_from_generated_split(smoke_dir, capsys):
    rec = load_split(smoke_dir, "test_unseen")[0]
    assert main(["inspect-example", "--data-dir", str(smoke_dir), "--split", "test_unseen",
                 "--problem-id", rec.problem_id, "--prompt-rows", "-1"]) == 0
    printed = capsys.readouterr().out
    assert rec.problem_id in printed and "held-out composition: True" in printed
    assert "elided" not in printed
