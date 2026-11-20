"""Step-0 correction signals on a finalized bank (diagnostic; separately runnable and resumable).

The corrector at initialization is the frozen generator checkpoint itself (every
correction arm starts from it). It sees the prompt and the COMPLETE draft (no mask
tokens) and gives, at every response position, from the aligned logits:

- ``step0_log_p_ref``: float32 full-vocabulary log probability of the reference
  token, so ``-step0_log_p_ref`` is the step-0 per-token correction cross-entropy;
- ``step0_log_p_draft``: the same at the draft's own token (the current model's
  confidence in the committed token, given the whole draft as context).

These are *current* probabilities given the draft. The bank's ``log_c_gen`` is the
*historical* generation confidence, from a partially masked context at commit time.
Output: ``<bank>/derived/step0/`` with one shard per bank shard and its own identity.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from ..env import synchronize
from ..models.alignment import build_layout
from .bank import ShardStore, append_event, now_utc, read_complete_bank, write_bytes_atomic, write_json_atomic
from .identity import sha256_json

STEP0_SCHEMA = "dlmc.step0.v1"
STEP0_SHARD_SCHEMA = "dlmc.step0_shard.v1"


def step0_batch(adapter, records: list[dict]) -> list[dict]:
    prompts = [r["prompt_ids"] for r in records]
    drafts = [r["generation"]["draft_ids"] for r in records]
    layout = build_layout(prompts, drafts, adapter.tokens.pad_token_id).to(adapter.device)
    adapter.model.eval()
    with torch.no_grad():
        logits = adapter.aligned_logits(layout)
        rows = logits[layout.editable_mask].float()
        lsm = torch.log_softmax(rows, dim=-1)
        refs = torch.tensor([t for r in records for t in r["analysis"]["reference_ids"]], device=lsm.device)
        drs = torch.tensor([t for d in drafts for t in d], device=lsm.device)
        lp_ref = lsm.gather(1, refs[:, None]).squeeze(1).cpu().tolist()
        lp_dr = lsm.gather(1, drs[:, None]).squeeze(1).cpu().tolist()
    out, k = [], 0
    for r in records:
        n = r["response_length"]
        out.append({"schema": STEP0_SCHEMA, "split": r["split"], "problem_id": r["problem_id"],
                    "budget": r["budget"], "step0_log_p_ref": lp_ref[k:k + n], "step0_log_p_draft": lp_dr[k:k + n]})
        k += n
    if k != len(lp_ref):
        raise AssertionError("step-0 rows do not match the response lengths")
    return out


def run_step0(bank_root: Path, adapter, corrector_identity: dict, environment: dict, batch_size: int = 50,
              resume: bool = False, stop_at_unix: float | None = None, should_stop=lambda: None) -> dict:
    bank_root = Path(bank_root)
    manifest, shards = read_complete_bank(bank_root)
    root = bank_root / "derived" / "step0"
    pinned = {"bank_manifest_sha256": manifest["manifest_sha256"],
              "corrector_init_model_sha256": corrector_identity["model_sha256"],
              "code_commit": (environment.get("git") or {}).get("commit"),
              "torch": environment.get("torch"), "device_type": str(environment.get("device", "")).split(":")[0],
              "batch_size": int(batch_size)}
    ident = {"schema": STEP0_SCHEMA, "pinned": pinned, "identity_sha256": sha256_json(pinned),
             "definition": __doc__, "corrector_init": corrector_identity, "environment": environment}
    idp = root / "IDENTITY.json"
    if idp.exists():
        if not resume:
            raise SystemExit(f"{root} exists; pass --resume")
        old = json.loads(idp.read_text())
        if old["identity_sha256"] != ident["identity_sha256"]:
            raise SystemExit(f"{root}: resume refused, identity changed")
    else:
        root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(idp, ident)
    if (root / "STEP0_COMPLETE").exists():
        return json.loads((root / "summary.json").read_text())
    store = ShardStore(root, STEP0_SCHEMA, STEP0_SHARD_SCHEMA)
    store.move_leftovers()
    done = store.complete_ids()
    t0 = time.perf_counter()
    state = "complete"
    for s, recs in shards:
        sid = s["id"]
        expect = {"identity_sha256": ident["identity_sha256"], "bank_shard_sha256": s["sha256"],
                  "n_records": len(recs)}
        if sid in done:
            store.verify(sid, expect)
            continue
        if should_stop() or (stop_at_unix is not None and time.time() >= stop_at_unix):
            state = "stopped"
            break
        ts = time.perf_counter()
        out = []
        for b0 in range(0, len(recs), batch_size):
            out.extend(step0_batch(adapter, recs[b0:b0 + batch_size]))
        synchronize(adapter.device)
        store.write(sid, out, {**expect, "seconds": round(time.perf_counter() - ts, 3), "time_utc": now_utc()})
        done.add(sid)
    summary = {"state": state, "shards_complete": len(done), "shards_total": len(shards),
               "identity_sha256": ident["identity_sha256"], "seconds_this_job": round(time.perf_counter() - t0, 1)}
    write_json_atomic(root / "summary.json", summary)
    if state == "complete":
        rows = [{"id": s["id"], "sha256": json.loads(store.sidecar_path(s["id"]).read_text())["sha256"]}
                for s, _ in shards]
        write_json_atomic(root / "MANIFEST.json", {"schema": STEP0_SCHEMA, "identity": ident["pinned"],
                                                    "identity_sha256": ident["identity_sha256"], "shards": rows})
        write_bytes_atomic(root / "STEP0_COMPLETE", b"complete\n")
    append_event(root, f"step0_{state}", shards_complete=len(done))
    return summary


def read_step0(bank_root: Path, manifest_sha256: str) -> dict | None:
    """(split, problem_id, budget) -> step-0 record, after hash checks; None when absent or incomplete."""
    root = Path(bank_root) / "derived" / "step0"
    if not (root / "STEP0_COMPLETE").exists():
        return None
    m = json.loads((root / "MANIFEST.json").read_text())
    if m["identity"]["bank_manifest_sha256"] != manifest_sha256:
        raise ValueError("step-0 output belongs to a different bank manifest")
    store = ShardStore(root, STEP0_SCHEMA, STEP0_SHARD_SCHEMA)
    out = {}
    for s in m["shards"]:
        store.verify(s["id"], {"sha256": s["sha256"]})
        for r in store.read(s["id"]):
            out[(r["split"], r["problem_id"], r["budget"])] = r
    return out
