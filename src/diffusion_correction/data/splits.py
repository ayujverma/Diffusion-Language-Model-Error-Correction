"""Split generation, de-duplication, manifests and verification.

Splits: ``train``, ``val``, ``test_id`` (same distribution as train),
``test_long`` (longer chains/programs), ``test_unseen`` (held-out compositions).

- Held-out compositions are declared explicitly as adjacent pairs in the
  composition signature (operator pairs for arithmetic, statement-shape pairs for
  programs). ``train``/``val``/``test_id``/``test_long`` never contain a declared pair;
  every ``test_unseen`` problem contains at least one. Seeds play no role in this.
- De-duplication is global across splits by the normalized key (canonical program
  text up to variable renaming, plus inputs). Splits are generated in the order
  ``test_unseen, test_long, test_id, val, train``; a key already used anywhere is
  rejected, so evaluation problems never reappear in training.
- ``problem_id`` is a hash of the normalized key. Drafts and augmentations must
  carry it; :func:`assert_grouped_by_problem` checks they stay in the problem's split.
- :func:`verify_dataset` rebuilds every record's problem from its source
  (``data/structure.py``) and derives the signature, held-out status, normalized key
  and problem ID from it; stored metadata must match exactly.
- ``manifest.json`` is deterministic (no timestamps): config, tokenizer and format
  facts, and per-split sha256 / counts / problem-ID hashes. Run-time provenance goes to
  ``provenance.json``. Difficulty settings are provisional until validation calibration.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .arithmetic import ArithmeticSpec, sample_chain
from .format import ResponseFormat, TokenizedFormat, encode_and_verify
from .programs import LanguageSpec, ProgramSpec, run_program, sample_program
from .structure import StructureError, derive_problem
from .tasks import FAMILIES, Example, build_example, problem_id, render_prompt

SPLITS = ("train", "val", "test_id", "test_long", "test_unseen")
GENERATION_ORDER = ("test_unseen", "test_long", "test_id", "val", "train")
FORMAT_VERSION = 1


@dataclass(frozen=True)
class FamilyConfig:
    lengths: tuple[int, ...]
    long_lengths: tuple[int, ...]
    heldout_bigrams: tuple[tuple[str, str], ...]
    n_inputs: int = 2  # programs only


@dataclass(frozen=True)
class DataConfig:
    name: str
    seed: int
    value_bound: int
    allow_negative: bool
    mul_range: tuple[int, int]
    operand_digits: tuple[int, ...]
    families: dict[str, FamilyConfig]
    per_setting: dict[str, int]
    max_attempts_per_problem: int = 200

    @classmethod
    def from_dict(cls, d: dict) -> "DataConfig":
        fams = {}
        for name, f in d["families"].items():
            if name not in FAMILIES:
                raise ValueError(f"unknown family {name!r}")
            fams[name] = FamilyConfig(lengths=tuple(f["lengths"]), long_lengths=tuple(f["long_lengths"]),
                                      heldout_bigrams=tuple(tuple(b) for b in f["heldout_bigrams"]),
                                      n_inputs=int(f.get("n_inputs", 2)))
            if set(fams[name].lengths) & set(fams[name].long_lengths):
                raise ValueError("long_lengths must not overlap the training lengths")
            if not fams[name].heldout_bigrams:
                raise ValueError("each family needs at least one held-out composition")
        per = {k: int(v) for k, v in d["per_setting"].items()}
        if set(per) != set(SPLITS):
            raise ValueError(f"per_setting needs exactly {SPLITS}")
        return cls(name=d["name"], seed=int(d["seed"]), value_bound=int(d["value_bound"]),
                   allow_negative=bool(d["allow_negative"]), mul_range=tuple(d["mul_range"]),
                   operand_digits=tuple(d["operand_digits"]), families=fams, per_setting=per,
                   max_attempts_per_problem=int(d.get("max_attempts_per_problem", 200)))

    def to_dict(self) -> dict:
        return {"name": self.name, "seed": self.seed, "value_bound": self.value_bound,
                "allow_negative": self.allow_negative, "mul_range": list(self.mul_range),
                "operand_digits": list(self.operand_digits),
                "families": {k: {"lengths": list(f.lengths), "long_lengths": list(f.long_lengths),
                                 "heldout_bigrams": [list(b) for b in f.heldout_bigrams],
                                 **({"n_inputs": f.n_inputs} if k == "programs" else {})}
                             for k, f in self.families.items()},
                "per_setting": dict(self.per_setting), "max_attempts_per_problem": self.max_attempts_per_problem}

    def response_format(self) -> ResponseFormat:
        return ResponseFormat(self.value_bound)


def contains_bigram(signature: Iterable[str], bigrams: Iterable[tuple[str, str]]) -> bool:
    sig = list(signature)
    pairs = set(zip(sig, sig[1:]))
    return any(tuple(b) in pairs for b in bigrams)


def derive_seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def _settings(cfg: DataConfig, family: str, split: str) -> list[tuple[int, int]]:
    f = cfg.families[family]
    lengths = f.long_lengths if split == "test_long" else f.lengths
    return [(n, d) for n in lengths for d in cfg.operand_digits]


def _sampler(cfg: DataConfig, family: str, length: int, digits: int):
    lo, hi = cfg.mul_range
    if family == "arithmetic":
        spec = ArithmeticSpec(n_ops=length, operand_digits=digits, mul_min=lo, mul_max=hi,
                              value_bound=cfg.value_bound, allow_negative=cfg.allow_negative)
        return lambda rng: sample_chain(rng, spec)
    spec = ProgramSpec(n_statements=length, operand_digits=digits, n_inputs=cfg.families[family].n_inputs,
                       mul_min=lo, mul_max=hi, value_bound=cfg.value_bound, allow_negative=cfg.allow_negative)
    return lambda rng: sample_program(rng, spec)


def generate_splits(cfg: DataConfig, tokenizer) -> dict[str, list[Example]]:
    tf = TokenizedFormat.from_tokenizer(tokenizer, cfg.response_format())
    seen: set[str] = set()
    out: dict[str, list[Example]] = {s: [] for s in SPLITS}
    for split in GENERATION_ORDER:
        want_heldout = split == "test_unseen"
        for family in FAMILIES:
            if family not in cfg.families:
                continue
            bigrams = cfg.families[family].heldout_bigrams
            for length, digits in _settings(cfg, family, split):
                rng = random.Random(derive_seed(cfg.seed, split, family, length, digits))
                sample = _sampler(cfg, family, length, digits)
                need = cfg.per_setting[split]
                got, attempts = 0, 0
                limit = need * cfg.max_attempts_per_problem
                while got < need:
                    attempts += 1
                    if attempts > limit:
                        raise RuntimeError(
                            f"{split}/{family}/length={length}/digits={digits}: only {got}/{need} unique "
                            f"problems after {limit} attempts; the setting's problem space is too small")
                    prob = sample(rng)
                    if contains_bigram(prob.signature, bigrams) != want_heldout:
                        continue
                    key = prob.normalized_key()
                    if key in seen:
                        continue
                    seen.add(key)
                    out[split].append(build_example(prob, split, tokenizer, tf, heldout=want_heldout))
                    got += 1
    return out


# ---------------------------------------------------------------- files, manifest, verification

def _dumps(record: dict) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_splits(out_dir: Path, cfg: DataConfig, splits: dict[str, list[Example]], tokenizer_facts: dict,
                 tf: TokenizedFormat) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"format_version": FORMAT_VERSION, "config": cfg.to_dict(), "tokenizer": tokenizer_facts,
                "response_format": {"value_bound": tf.fmt.value_bound, "width": tf.fmt.width,
                                    "slot_len": tf.fmt.slot_len, "sign_ids": tf.sign_ids,
                                    "digit_ids": list(tf.digit_ids), "eos_in_response": False,
                                    "separator": "none (each slot starts with ' +' or ' -')"},
                "dedup": "normalized_key (canonical program up to variable renaming, plus inputs), global",
                "heldout_compositions": {k: [list(b) for b in f.heldout_bigrams] for k, f in cfg.families.items()},
                "generation_order": list(GENERATION_ORDER), "splits": {}}
    for split in SPLITS:
        recs = splits[split]
        data = "".join(_dumps(r.to_dict()) + "\n" for r in recs).encode()
        (out_dir / f"{split}.jsonl").write_bytes(data)
        counts: dict[str, int] = {}
        for r in recs:
            key = f"{r.family}/length={r.setting['length']}/digits={r.setting['operand_digits']}"
            counts[key] = counts.get(key, 0) + 1
        ids = sorted(r.problem_id for r in recs)
        manifest["splits"][split] = {"file": f"{split}.jsonl", "sha256": sha256_bytes(data),
                                     "n_examples": len(recs), "counts": dict(sorted(counts.items())),
                                     "problem_ids_sha256": sha256_bytes("\n".join(ids).encode())}
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    (out_dir / "manifest.json").write_text(text)
    return {"manifest_sha256": sha256_bytes(text.encode()), **manifest}


def load_split(data_dir: Path, split: str) -> list[Example]:
    lines = (Path(data_dir) / f"{split}.jsonl").read_text().splitlines()
    return [Example.from_dict(json.loads(line)) for line in lines if line]


def verify_dataset(data_dir: Path, tokenizer) -> dict:
    """Re-check a written dataset from its files alone. Raises AssertionError on failure."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    cfg = DataConfig.from_dict(manifest["config"])
    tf = TokenizedFormat.from_tokenizer(tokenizer, cfg.response_format())
    assert manifest["response_format"]["sign_ids"] == tf.sign_ids, "tokenizer sign IDs changed"
    assert manifest["response_format"]["digit_ids"] == list(tf.digit_ids), "tokenizer digit IDs changed"
    lang = LanguageSpec(max_statements=64, literal_max=999, mul_min=cfg.mul_range[0], mul_max=cfg.mul_range[1],
                        value_bound=cfg.value_bound, allow_negative=cfg.allow_negative)
    owner: dict[str, str] = {}
    keys: dict[str, str] = {}
    n = 0
    for split in SPLITS:
        info = manifest["splits"][split]
        data = (data_dir / info["file"]).read_bytes()
        assert sha256_bytes(data) == info["sha256"], f"{split}: file hash differs from manifest"
        recs = load_split(data_dir, split)
        assert len(recs) == info["n_examples"], f"{split}: example count differs"
        for r in recs:
            n += 1
            where = f"{split}/{r.problem_id}"
            assert r.split == split, f"{where}: record says split {r.split}"
            assert r.family in cfg.families, f"{where}: family {r.family!r} not in the config"
            fam = cfg.families[r.family]
            lengths = fam.long_lengths if split == "test_long" else fam.lengths
            assert r.setting.get("length") in lengths, f"{where}: length"
            assert r.setting.get("operand_digits") in cfg.operand_digits, f"{where}: operand digits"
            # Rebuild the problem from its source; never trust stored structure metadata.
            try:
                prob = derive_problem(r.family, r.source, r.inputs, r.setting, mul_range=cfg.mul_range,
                                      value_bound=cfg.value_bound, allow_negative=cfg.allow_negative,
                                      n_inputs=fam.n_inputs if r.family == "programs" else None)
            except StructureError as e:
                raise AssertionError(f"{where}: source is outside the generated structure: {e}") from e
            signature = list(prob.signature)
            held = contains_bigram(signature, fam.heldout_bigrams)
            assert held == (split == "test_unseen"), \
                f"{where}: source composition {signature} is {'' if held else 'not '}held out, but split is {split}"
            assert r.heldout_composition == held, f"{where}: heldout_composition flag differs from the source"
            assert r.signature == signature, f"{where}: stored signature {r.signature} != source signature {signature}"
            # Identity: rebuild the normalized key and problem ID from source plus inputs.
            key = prob.normalized_key()
            pid = problem_id(key)
            assert r.normalized_key == key, f"{where}: stored normalized_key differs from the key rebuilt from source"
            assert r.problem_id == pid, f"{where}: stored problem_id differs from the hash of the rebuilt key"
            assert pid not in owner, f"{where}: problem also in {owner.get(pid)}"
            assert key not in keys, f"{where}: normalized key also in {keys.get(key)}"
            owner[pid] = split
            keys[key] = split
            # Labels: the generator-independent interpreter, and the rebuilt problem, must agree.
            assert list(run_program(r.source, r.inputs, lang)) == r.values, f"{where}: values differ"
            assert list(prob.values) == r.values and r.final_answer == r.values[-1], f"{where}: values differ"
            assert r.value_names == list(prob.names), f"{where}: value names differ from the source targets"
            p, rid, _ = encode_and_verify(tokenizer, tf, r.prompt_text, r.values)
            assert p == r.prompt_ids and rid == r.response_ids, f"{where}: token IDs differ on re-encoding"
            assert r.slot_spans == [list(s) for s in tf.fmt.slot_spans(len(r.values))], f"{where}: spans"
            assert r.prompt_text == render_prompt(r.family, r.source, r.inputs, r.value_names, tf.fmt), \
                f"{where}: prompt is not a function of the value-free fields"
    return {"n_examples": n, "n_problems": len(owner), "splits": {s: manifest["splits"][s]["n_examples"]
                                                                   for s in SPLITS}, "passed": True}


def assert_grouped_by_problem(records: Iterable[dict], data_dir: Path) -> None:
    """Every derived record (draft, augmentation) must sit in its problem's split."""
    owner = {}
    for split in SPLITS:
        for r in load_split(data_dir, split):
            owner[r.problem_id] = split
    for rec in records:
        pid, split = rec["problem_id"], rec["split"]
        if pid not in owner:
            raise AssertionError(f"record for unknown problem {pid}")
        if owner[pid] != split:
            raise AssertionError(f"record for problem {pid} is in {split}, but the problem is in {owner[pid]}")
