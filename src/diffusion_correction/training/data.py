"""Training/evaluation inputs from verified dataset files, and the resumable sampler.

Examples are always read from a written dataset (``data.splits.load_split``) whose
manifest and split-file hashes are checked first; nothing is regenerated here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch

from ..data.splits import load_split, sha256_bytes
from ..data.tasks import Example
from ..models.alignment import SequenceLayout, build_layout

TRAINING_SPLITS = ("train", "val")


@dataclass(frozen=True)
class DatasetIdentity:
    data_dir: str
    manifest_sha256: str
    split: str
    split_sha256: str
    n_split_examples: int
    subset: dict | None
    selected_problem_ids_sha256: str
    n_selected: int

    def to_dict(self) -> dict:
        return asdict(self)


def setting_key(ex: Example) -> tuple[str, int, int]:
    return ex.family, int(ex.setting["length"]), int(ex.setting["operand_digits"])


def select_subset(examples: Sequence[Example], n: int, seed: int) -> list[Example]:
    """Deterministic stratified subset: round-robin over (family, length, digits) groups,
    taking within each group the examples with the smallest sha256(seed|problem_id)."""
    groups: dict[tuple, list[Example]] = {}
    for ex in examples:
        groups.setdefault(setting_key(ex), []).append(ex)
    for k in groups:
        groups[k].sort(key=lambda e: hashlib.sha256(f"{seed}|{e.problem_id}".encode()).hexdigest())
    out: list[Example] = []
    depth = 0
    keys = sorted(groups)
    while len(out) < n:
        progressed = False
        for k in keys:
            if depth < len(groups[k]) and len(out) < n:
                out.append(groups[k][depth])
                progressed = True
        if not progressed:
            raise ValueError(f"only {len(out)} examples available, {n} requested")
        depth += 1
    return out


def select_per_setting(examples: Sequence[Example], n_per_setting: int,
                       settings: set[tuple] | None = None) -> list[Example]:
    """The first ``n_per_setting`` examples of each setting, in file order."""
    counts: dict[tuple, int] = {}
    out = []
    for ex in examples:
        k = setting_key(ex)
        if settings is not None and k not in settings:
            continue
        if counts.get(k, 0) < n_per_setting:
            out.append(ex)
            counts[k] = counts.get(k, 0) + 1
    return out


def load_verified_examples(data_dir: str | Path, split: str, expected_manifest_sha256: str | None = None,
                           subset: dict | None = None, allowed_splits: Sequence[str] = TRAINING_SPLITS
                           ) -> tuple[list[Example], DatasetIdentity]:
    """Load a split after checking the manifest hash and the split file hash against it.

    The full semantic re-verification (`dlmc verify-data`) is run separately by the
    trainer when ``data.verify`` is set; here the file hashes tie the records to a
    verified manifest.
    """
    if split not in allowed_splits:
        raise ValueError(f"split {split!r} is not allowed here (allowed: {tuple(allowed_splits)})")
    data_dir = Path(data_dir)
    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    if expected_manifest_sha256 and manifest_sha != expected_manifest_sha256:
        raise ValueError(f"{data_dir}: manifest sha256 {manifest_sha} != expected {expected_manifest_sha256}")
    manifest = json.loads(manifest_bytes)
    info = manifest["splits"][split]
    data = (data_dir / info["file"]).read_bytes()
    if sha256_bytes(data) != info["sha256"]:
        raise ValueError(f"{data_dir}/{info['file']}: sha256 differs from the manifest")
    examples = load_split(data_dir, split)
    if len(examples) != info["n_examples"]:
        raise ValueError(f"{split}: {len(examples)} examples, manifest says {info['n_examples']}")
    selected = examples
    if subset:
        selected = select_subset(examples, int(subset["n"]), int(subset["seed"]))
    ids = "\n".join(ex.problem_id for ex in selected).encode()
    ident = DatasetIdentity(data_dir=str(data_dir), manifest_sha256=manifest_sha, split=split,
                            split_sha256=info["sha256"], n_split_examples=len(examples), subset=subset,
                            selected_problem_ids_sha256=sha256_bytes(ids), n_selected=len(selected))
    return selected, ident


class EpochSampler:
    """Draws example indices from per-epoch permutations of ``range(n)``.

    Batches may straddle an epoch boundary, so no example is dropped. The state is
    (epoch, permutation, position); the generator's state is checkpointed with the
    other RNG streams.
    """

    def __init__(self, n: int, generator: torch.Generator):
        if n <= 0:
            raise ValueError("empty dataset")
        self.n = n
        self.generator = generator
        self.epoch = 0
        self.perm: list[int] = []
        self.pos = 0
        self.examples_consumed = 0

    def _refill(self) -> None:
        self.perm = torch.randperm(self.n, generator=self.generator).tolist()
        self.pos = 0
        self.epoch += 1

    def next_indices(self, k: int) -> list[int]:
        out = []
        while len(out) < k:
            if self.pos >= len(self.perm):
                self._refill()
            take = min(k - len(out), len(self.perm) - self.pos)
            out.extend(self.perm[self.pos:self.pos + take])
            self.pos += take
        self.examples_consumed += k
        return out

    def state_dict(self) -> dict:
        return {"n": self.n, "epoch": self.epoch, "perm": list(self.perm), "pos": self.pos,
                "examples_consumed": self.examples_consumed}

    def load_state_dict(self, s: dict) -> None:
        if s["n"] != self.n:
            raise ValueError(f"sampler state is for {s['n']} examples, dataset has {self.n}")
        self.epoch, self.perm, self.pos = s["epoch"], list(s["perm"]), s["pos"]
        self.examples_consumed = s["examples_consumed"]


def collate_targets(examples: Sequence[Example], pad_token_id: int) -> SequenceLayout:
    """Layout whose response tokens are the canonical targets (training/evaluation only)."""
    return build_layout([ex.prompt_ids for ex in examples], [ex.response_ids for ex in examples], pad_token_id)
