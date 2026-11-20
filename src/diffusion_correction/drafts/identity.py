"""Generator-checkpoint identity from its files (hashlib and json only; safe on a login node).

The identity re-hashes every file listed in ``COMPLETE.json`` and extracts, from
``state.json``, the training configuration and its hash, the source revision recorded
at save time, the tokenizer and base-model file hashes, the training-data identity and
the continuation's selection and timer state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

GENERATOR_IDENTITY_SCHEMA = "dlmc.generator_identity.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def generator_identity(checkpoint: str | Path, expected_model_sha256: str | None = None,
                       expected_update: int | None = None) -> dict:
    """Raise ValueError unless every file matches COMPLETE.json (and the pins, when given)."""
    ck = Path(checkpoint)
    complete_path = ck / "COMPLETE.json"
    if not complete_path.exists():
        raise ValueError(f"{ck}: no COMPLETE.json; not a complete checkpoint")
    complete = json.loads(complete_path.read_text())
    files = {}
    for name, digest in sorted(complete["files"].items()):
        got = sha256_file(ck / name)
        if got != digest:
            raise ValueError(f"{ck / name}: sha256 {got} != COMPLETE.json {digest}")
        files[name] = {"sha256": got, "bytes": (ck / name).stat().st_size}
    extra = sorted(p.name for p in ck.iterdir() if p.name not in complete["files"] and p.name != "COMPLETE.json")
    state = json.loads((ck / "state.json").read_text())
    model_file = state.get("model_file", "model.safetensors")
    model_sha = complete["files"].get(model_file)
    if state["update"] != complete["update"]:
        raise ValueError(f"{ck}: state.json update {state['update']} != COMPLETE.json {complete['update']}")
    if expected_model_sha256 is not None and model_sha != expected_model_sha256:
        raise ValueError(f"{ck}: model sha256 {model_sha} != pinned {expected_model_sha256}")
    if expected_update is not None and state["update"] != expected_update:
        raise ValueError(f"{ck}: update {state['update']} != pinned {expected_update}")
    cfg = state.get("config", {})
    base = state.get("base_model", {})
    return {
        "schema": GENERATOR_IDENTITY_SCHEMA,
        "checkpoint": str(ck.resolve()),
        "update": state["update"],
        "method": state.get("method"),
        "model_file": model_file,
        "model_sha256": model_sha,
        "complete_json_sha256": sha256_file(complete_path),
        "files": files,
        "unlisted_files": extra,
        "config_hash": state.get("config_hash"),
        "compat_hash": state.get("compat_hash"),
        "config": cfg,
        "run_name": cfg.get("run_name"),
        "source_revision_at_save": state.get("git"),
        "versions_at_save": state.get("versions"),
        "device_at_save": state.get("device"),
        "base_model": {"backbone": base.get("backbone"), "revision": base.get("revision"),
                       "file_sha256": base.get("file_sha256")},
        "tokenizer_json_sha256": (base.get("file_sha256") or {}).get("tokenizer.json"),
        "training_data_identity": state.get("data_identity"),
        "eval_identity": state.get("eval_identity"),
        "selection": state.get("selection"),
        "timers": state.get("timers"),
        "resumed_from": state.get("resumed_from"),
        "examples_consumed": state.get("examples_consumed"),
        "batches_consumed": state.get("batches_consumed"),
    }
