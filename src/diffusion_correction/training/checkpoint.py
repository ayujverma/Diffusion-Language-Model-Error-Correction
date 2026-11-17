"""Atomic checkpoints and verified copies to durable storage.

A checkpoint is a directory ``ckpt-<update:07d>/`` holding:

- ``model.safetensors`` (full fine-tuning; tied ``lm_head`` omitted, as in the
  pinned checkpoint) or ``adapter.safetensors`` (LoRA tensors only);
- ``optimizer.pt``: optimizer and LR-scheduler state dicts;
- ``rng.pt``: global RNG states and the dedicated stream generators' states;
- ``state.json``: trainer counters, sampler position, resolved config, dataset and
  base-model identity, git state;
- ``COMPLETE.json``: sha256 of every file above, written last.

The directory is written as ``<name>.tmp`` and renamed only after every file is
flushed, so a crash never leaves a directory that looks complete. ``latest`` is
replaced atomically. A durable copy is made by copying into ``<name>.tmp`` at the
destination, re-hashing every file against ``COMPLETE.json`` and renaming; a copy is
reported as successful only after that check.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

COMPLETE = "COMPLETE.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def _fsync_file(path: Path) -> None:
    with open(path, "rb") as f:
        os.fsync(f.fileno())


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    _fsync_file(tmp)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def checkpoint_name(update: int) -> str:
    return f"ckpt-{update:07d}"


def save_checkpoint(ckpt_root: Path, update: int, *, model_tensors: dict[str, torch.Tensor], model_file: str,
                    optimizer_state: dict, rng_state: dict, state: dict) -> Path:
    from safetensors.torch import save_file

    ckpt_root.mkdir(parents=True, exist_ok=True)
    name = checkpoint_name(update)
    final = ckpt_root / name
    tmp = ckpt_root / (name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    save_file({k: v.detach().to("cpu").contiguous() for k, v in model_tensors.items()}, str(tmp / model_file))
    torch.save(optimizer_state, tmp / "optimizer.pt")
    torch.save(rng_state, tmp / "rng.pt")
    (tmp / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True, default=str) + "\n")
    files = {}
    for f in (model_file, "optimizer.pt", "rng.pt", "state.json"):
        _fsync_file(tmp / f)
        files[f] = _sha256(tmp / f)
    (tmp / COMPLETE).write_text(json.dumps({"update": update, "files": files}, indent=2, sort_keys=True) + "\n")
    _fsync_file(tmp / COMPLETE)
    _fsync_dir(tmp)
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    _fsync_dir(ckpt_root)
    write_text_atomic(ckpt_root / "latest", name + "\n")
    return final


def verify_checkpoint(path: Path) -> dict:
    """Return COMPLETE.json contents after re-hashing every file; raise if incomplete or corrupt."""
    path = Path(path)
    marker = path / COMPLETE
    if not marker.exists():
        raise ValueError(f"{path}: no {COMPLETE}; the checkpoint is incomplete")
    info = json.loads(marker.read_text())
    for f, digest in info["files"].items():
        got = _sha256(path / f)
        if got != digest:
            raise ValueError(f"{path / f}: sha256 {got} != recorded {digest}")
    return info


def latest_checkpoint(ckpt_root: Path) -> Path | None:
    ptr = Path(ckpt_root) / "latest"
    if not ptr.exists():
        return None
    path = Path(ckpt_root) / ptr.read_text().strip()
    return path if (path / COMPLETE).exists() else None


def copy_to_durable(src: Path, durable_root: Path) -> dict:
    """Copy a completed checkpoint and verify it. Never raises; returns a status record."""
    src, durable_root = Path(src), Path(durable_root)
    dst = durable_root / src.name
    tmp = durable_root / (src.name + ".tmp")
    record = {"source": str(src), "destination": str(dst), "ok": False}
    try:
        verify_checkpoint(src)
        durable_root.mkdir(parents=True, exist_ok=True)
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp)
        verify_checkpoint(tmp)
        if dst.exists():
            shutil.rmtree(dst)
        os.replace(tmp, dst)
        write_text_atomic(durable_root / "latest", src.name + "\n")
        record["ok"] = True
    except Exception as e:  # reported, never claimed as a backup
        record["error"] = f"{type(e).__name__}: {e}"
    return record


def prune_checkpoints(ckpt_root: Path, keep_last: int, protect: set[str] = frozenset()) -> list[str]:
    ckpts = sorted(p for p in Path(ckpt_root).glob("ckpt-*") if p.is_dir() and not p.name.endswith(".tmp"))
    removed = []
    for p in ckpts[:-keep_last] if keep_last > 0 else []:
        if p.name in protect:
            continue
        shutil.rmtree(p)
        removed.append(p.name)
    return removed
