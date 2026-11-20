"""Immutable draft bank: deterministic plan, pinned identity, atomic shards, resume, validation.

Layout of a bank directory::

    IDENTITY.json        pinned identities (config, plan, generator, code, tokenizer, data, environment)
    plan.json            every shard: split, budget, setting, ordered problem IDs
    shards/<split>/b<budget>/<family>-n<len>-d<digits>/part-NNN.jsonl   draft records
    shards/.../part-NNN.json                                            shard sidecar (sha256, counts, cost)
    status.json, events.jsonl                                           progress (rewritten / appended)
    incomplete/<stamp>/...                                              leftovers moved aside on resume
    validation.json, MANIFEST.json, BANK_COMPLETE                       written by validate_bank

A shard is complete only when its sidecar exists and the data file's sha256 matches
it. Both files are written to ``.tmp``, fsynced and renamed (data first). Resuming
recomputes the plan and the identity and refuses any difference; it skips verified
shards (never rewriting them), moves leftovers aside, and stops on a corrupt shard.
Nothing retries automatically.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import yaml

from ..data.format import TokenizedFormat
from ..data.tasks import Example
from ..evaluation.generation import public_response_length
from ..training.data import load_verified_examples, setting_key
from .collector import (DRAFT_SCHEMA, PROBABILITY_DEFINITION, budget_feasible, calls_for, check_record,
                        collect_batch)
from .identity import sha256_file, sha256_json

PLAN_SCHEMA = "dlmc.bank_plan.v1"
SHARD_SCHEMA = "dlmc.bank_shard.v1"
MANIFEST_SCHEMA = "dlmc.bank_manifest.v1"
IDENTITY_SCHEMA = "dlmc.bank_identity.v1"
BANK_SPLITS = ("train", "val")
SAMPLER_NAME = "d029_monotone_greedy"


# ---------------------------------------------------------------- config and plan

def load_bank_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for key in ("name", "generator", "data", "sampler"):
        if key not in cfg:
            raise ValueError(f"{path}: missing '{key}'")
    s = cfg["sampler"]
    if s.get("name") != SAMPLER_NAME:
        raise ValueError(f"sampler must be {SAMPLER_NAME} (D-029)")
    s["budgets"] = [str(b) for b in s["budgets"]]
    if int(s.get("drafts_per_problem_per_budget", 1)) != 1:
        raise ValueError("the D-029 sampler is greedy: exactly one draft per problem and budget")
    if s.get("rng") not in (None, "none"):
        raise ValueError("the D-029 sampler consumes no random numbers; rng must be 'none'")
    for split in cfg["data"]["order"]:
        if split not in BANK_SPLITS:
            raise ValueError(f"split {split!r} is not collected into a bank here (train/val only)")
        if split not in cfg["data"]["splits"]:
            raise ValueError(f"split {split!r} in order but not configured")
    if set(cfg["data"]["splits"]) - set(BANK_SPLITS):
        raise ValueError("only train and val splits may be configured; test splits are never loaded")
    return cfg


def load_bank_examples(cfg: dict, data_dir: str | None = None) -> tuple[dict[str, list[Example]], dict]:
    """Verified train/val examples (manifest and split-file hashes against the pinned config)."""
    d = cfg["data"]
    data_dir = data_dir or d["dir"]
    out, idents = {}, {}
    for split in d["order"]:
        exs, ident = load_verified_examples(data_dir, split, d["manifest_sha256"], allowed_splits=BANK_SPLITS)
        want = d["splits"][split]
        if ident.split_sha256 != want["sha256"] or ident.n_split_examples != int(want["n_examples"]):
            raise ValueError(f"{split}: split identity {ident.split_sha256}/{ident.n_split_examples} differs "
                             f"from the bank config")
        out[split] = exs
        idents[split] = ident.to_dict()
    ids = {s: {e.problem_id for e in v} for s, v in out.items()}
    if "train" in ids and "val" in ids and ids["train"] & ids["val"]:
        raise ValueError("train and val share problem IDs")
    return out, idents


def _settings_filter(spec) -> set | None:
    if spec in (None, "all"):
        return None
    out = set()
    for s in spec:
        fam, n, dig = s.split(":")
        out.add((fam, int(n), int(dig)))
    return out


def shard_id(split: str, budget: str, key: tuple, part: int) -> str:
    fam, n, dig = key
    return f"{split}/b{budget}/{fam}-n{n}-d{dig}/part-{part:03d}"


def build_plan(cfg: dict, examples: dict[str, list[Example]], data_identity: dict, slot_len: int) -> dict:
    d, s = cfg["data"], cfg["sampler"]
    shards, infeasible = [], []
    for split in d["order"]:
        sc = d["splits"][split]
        keep = _settings_filter(sc.get("settings", "all"))
        size = int(sc["shard_problems"])
        cap = sc.get("problems_per_setting")             # None = every problem (no filtering by outcome)
        groups: dict[tuple, list[Example]] = {}
        for ex in examples[split]:                       # file order within each setting
            k = setting_key(ex)
            if (keep is None or k in keep) and (cap is None or len(groups.get(k, [])) < int(cap)):
                groups.setdefault(k, []).append(ex)
        if keep is not None and set(groups) != keep:
            raise ValueError(f"{split}: configured settings missing from the data: {sorted(keep - set(groups))}")
        for budget in s["budgets"]:
            for key in sorted(groups):
                exs = groups[key]
                L = public_response_length(exs[0], slot_len)
                if any(public_response_length(e, slot_len) != L for e in exs):
                    raise ValueError(f"{split} {key}: mixed public response lengths")
                if not budget_feasible(budget, L):
                    infeasible.append({"split": split, "budget": budget, "setting": list(key), "n": len(exs)})
                    continue
                for part, k0 in enumerate(range(0, len(exs), size)):
                    ids = [e.problem_id for e in exs[k0:k0 + size]]
                    shards.append({"id": shard_id(split, budget, key, part), "split": split, "budget": budget,
                                   "setting": list(key), "response_length": L, "calls": calls_for(budget, L),
                                   "problem_ids": ids})
    totals: dict = {"shards": len(shards), "drafts": sum(len(x["problem_ids"]) for x in shards)}
    for split in d["order"]:
        sel = [x for x in shards if x["split"] == split]
        totals[split] = {"shards": len(sel), "drafts": sum(len(x["problem_ids"]) for x in sel),
                         "problems": len({p for x in sel for p in x["problem_ids"]}),
                         "example_model_calls": sum(len(x["problem_ids"]) * x["calls"] for x in sel),
                         "response_tokens": sum(len(x["problem_ids"]) * x["response_length"] for x in sel)}
    return {"schema": PLAN_SCHEMA, "bank": cfg["name"], "config_sha256": sha256_json(cfg),
            "data_identity": data_identity, "budgets": list(s["budgets"]), "batch_size": int(s["batch_size"]),
            "order": list(d["order"]), "shards": shards, "infeasible": infeasible, "totals": totals}


def bank_identity(cfg: dict, plan: dict, generator: dict, environment: dict, lock_sha256: str | None) -> dict:
    """Identity fields that must not change across resumes, and their hash."""
    git = environment.get("git") or {}
    pinned = {
        "bank": cfg["name"],
        "config_sha256": sha256_json(cfg),
        "plan_sha256": sha256_json(plan),
        "generator_model_sha256": generator["model_sha256"],
        "generator_complete_json_sha256": generator["complete_json_sha256"],
        "generator_update": generator["update"],
        "tokenizer_json_sha256": generator.get("tokenizer_json_sha256"),
        "dataset_manifest_sha256": cfg["data"]["manifest_sha256"],
        "code_commit": git.get("commit"),
        "torch": environment.get("torch"),
        "cuda_runtime": environment.get("cuda_runtime"),
        "device_type": str(environment.get("device", "")).split(":")[0],
        "precision": environment.get("precision"),
        "environment_lock_sha256": lock_sha256,
        "sampler": SAMPLER_NAME,
        "batch_size": int(cfg["sampler"]["batch_size"]),
    }
    return {"schema": IDENTITY_SCHEMA, "pinned": pinned, "identity_sha256": sha256_json(pinned),
            "probability_definition": PROBABILITY_DEFINITION,
            "rng_policy": "greedy D-029 sampler; generation consumes no random numbers (CPU and CUDA RNG states "
                          "checked unchanged for every batch); no seed is used",
            "mask_fractions": "global response mask fraction before the commit; no blocks, so no separate "
                              "active-block fraction",
            "generator": generator, "environment": environment, "config": cfg}


# ---------------------------------------------------------------- atomic files

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


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def write_json_atomic(path: Path, obj) -> None:
    write_bytes_atomic(path, (json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n").encode())


def dump_record(rec: dict) -> str:
    return json.dumps(rec, sort_keys=True, separators=(",", ":"))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class ShardStore:
    """Atomic shard files under ``<root>/shards`` (data + sidecar)."""

    def __init__(self, root: Path, data_schema: str = DRAFT_SCHEMA, sidecar_schema: str = SHARD_SCHEMA):
        self.root = Path(root)
        self.dir = self.root / "shards"
        self.data_schema, self.sidecar_schema = data_schema, sidecar_schema

    def data_path(self, sid: str) -> Path:
        return self.dir / f"{sid}.jsonl"

    def sidecar_path(self, sid: str) -> Path:
        return self.dir / f"{sid}.json"

    def write(self, sid: str, records: Sequence[dict], meta: dict) -> dict:
        data = "".join(dump_record(r) + "\n" for r in records).encode()
        dp = self.data_path(sid)
        write_bytes_atomic(dp, data)
        side = {"schema": self.sidecar_schema, "shard_id": sid, "file": dp.name,
                "sha256": sha256_file(dp), "bytes": len(data), "n_records": len(records),
                "problem_ids_sha256": sha256_json([r["problem_id"] for r in records]), **meta}
        write_json_atomic(self.sidecar_path(sid), side)
        return side

    def verify(self, sid: str, expected: dict) -> dict:
        """Sidecar must exist and match the data file and every ``expected`` field; raise otherwise."""
        side = json.loads(self.sidecar_path(sid).read_text())
        got = sha256_file(self.data_path(sid))
        if got != side["sha256"]:
            raise ValueError(f"shard {sid}: data sha256 {got} != sidecar {side['sha256']}")
        for k, v in expected.items():
            if side.get(k) != v:
                raise ValueError(f"shard {sid}: sidecar {k}={side.get(k)!r} != expected {v!r}")
        return side

    def read(self, sid: str) -> list[dict]:
        return [json.loads(line) for line in self.data_path(sid).read_text().splitlines() if line]

    def complete_ids(self) -> set[str]:
        if not self.dir.exists():
            return set()
        return {str(p.relative_to(self.dir))[:-len(".json")] for p in self.dir.rglob("*.json")
                if not p.name.endswith(".tmp")}

    def move_leftovers(self) -> list[str]:
        """Move ``.tmp`` files and data files without a sidecar to ``incomplete/<stamp>/``."""
        if not self.dir.exists():
            return []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        moved = []
        for p in sorted(self.dir.rglob("*")):
            if not p.is_file():
                continue
            leftover = p.name.endswith(".tmp") or (p.suffix == ".jsonl" and not p.with_suffix(".json").exists())
            if leftover:
                dst = self.root / "incomplete" / stamp / p.relative_to(self.dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dst))
                moved.append(str(p.relative_to(self.root)))
        return moved


def append_event(root: Path, event: str, **fields) -> None:
    with open(Path(root) / "events.jsonl", "a") as f:
        f.write(json.dumps({"time_utc": now_utc(), "event": event, **fields}, sort_keys=True, default=str) + "\n")


def open_bank(root: Path, identity: dict, plan: dict, resume: bool) -> dict:
    """Create a new bank directory, or check that a resumed one has the same identity and plan."""
    root = Path(root)
    idp, pp = root / "IDENTITY.json", root / "plan.json"
    if not resume:
        if root.exists() and any(root.iterdir()):
            raise SystemExit(f"{root} exists and is not empty; pass --resume to continue that collection")
        root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(pp, plan)
        write_json_atomic(idp, identity)
        append_event(root, "created", identity_sha256=identity["identity_sha256"],
                     plan_sha256=identity["pinned"]["plan_sha256"])
        return {"moved_leftovers": []}
    if not idp.exists() or not pp.exists():
        raise SystemExit(f"{root}: nothing to resume (IDENTITY.json or plan.json missing)")
    if (root / "BANK_COMPLETE").exists():
        raise SystemExit(f"{root}: the bank is already complete; nothing to resume")
    old = json.loads(idp.read_text())
    if old["identity_sha256"] != identity["identity_sha256"]:
        diff = {k: (old["pinned"].get(k), v) for k, v in identity["pinned"].items() if old["pinned"].get(k) != v}
        raise SystemExit(f"{root}: resume refused, identity changed: {json.dumps(diff, default=str)}")
    if sha256_json(json.loads(pp.read_text())) != identity["pinned"]["plan_sha256"]:
        raise SystemExit(f"{root}: plan.json does not match the pinned plan hash")
    moved = ShardStore(root).move_leftovers()
    append_event(root, "resumed", identity_sha256=identity["identity_sha256"], moved_leftovers=moved)
    return {"moved_leftovers": moved}


def _shard_expect(entry: dict, identity_sha: str, plan_sha: str) -> dict:
    return {"identity_sha256": identity_sha, "plan_sha256": plan_sha, "n_records": len(entry["problem_ids"]),
            "problem_ids_sha256": sha256_json(entry["problem_ids"]), "split": entry["split"],
            "budget": entry["budget"], "setting": entry["setting"]}


# ---------------------------------------------------------------- collection

def collect_bank(root: Path, plan: dict, identity: dict, adapter, tf: TokenizedFormat, allowed,
                 examples: dict[str, list[Example]], *, stop_at_unix: float | None = None,
                 max_shards: int | None = None, should_stop: Callable[[], str | None] = lambda: None,
                 log: Callable[[str], None] = print, memory: Callable[[], dict] = dict) -> dict:
    """Collect every planned shard not already complete. Returns the final status record."""
    from ..env import synchronize

    root = Path(root)
    store = ShardStore(root)
    by_id = {s: {e.problem_id: e for e in v} for s, v in examples.items()}
    ident_sha, plan_sha = identity["identity_sha256"], identity["pinned"]["plan_sha256"]
    pinned = identity["pinned"]
    done = store.complete_ids()
    planned = {s["id"] for s in plan["shards"]}
    unexpected = sorted(done - planned)
    if unexpected:
        raise SystemExit(f"{root}: shards not in the plan: {unexpected[:5]}")
    for entry in plan["shards"]:
        if entry["id"] in done:
            store.verify(entry["id"], _shard_expect(entry, ident_sha, plan_sha))  # raises on corruption
    status = {"bank": plan["bank"], "identity_sha256": ident_sha, "shards_total": len(plan["shards"]),
              "shards_complete_at_start": len(done), "state": "running", "started_utc": now_utc(),
              "job_id": os.environ.get("SLURM_JOB_ID")}
    write_json_atomic(root / "status.json", {**status, "shards_complete": len(done)})
    t_start = time.perf_counter()
    written, model_s_job = 0, 0.0
    bs = int(plan["batch_size"])
    for entry in plan["shards"]:
        sid = entry["id"]
        if sid in done:
            continue
        reason = should_stop()
        if reason is None and stop_at_unix is not None and time.time() >= stop_at_unix:
            reason = "deadline"
        if reason is None and max_shards is not None and written >= max_shards:
            reason = "max_shards"
        if reason is not None:
            status.update(state=f"stopped_{reason}")
            break
        exs = [by_id[entry["split"]][p] for p in entry["problem_ids"]]
        t0 = time.perf_counter()
        records, batches = [], []
        mismatches, rng_ok, fwd, ex_calls, model_s = 0, True, 0, 0, 0.0
        for k, b0 in enumerate(range(0, len(exs), bs)):
            chunk = exs[b0:b0 + bs]
            synchronize(adapter.device)
            tb = time.perf_counter()
            res = collect_batch(adapter, chunk, entry["budget"], allowed, tf)
            synchronize(adapter.device)
            dt = time.perf_counter() - tb
            model_s += dt
            mismatches += res.rank_score_mismatches
            rng_ok &= res.rng_unchanged
            fwd += res.batched_forward_calls
            ex_calls += res.example_model_calls
            batches.append({"index": k, "size": len(chunk), "forwards": res.batched_forward_calls,
                            "seconds_with_signals": round(dt, 4),
                            "problem_ids_sha256": sha256_json([e.problem_id for e in chunk])})
            for j, rec in enumerate(res.records):
                rec.update(bank=plan["bank"], shard=sid, batch_index=k, batch_slot=j, batch_size=len(chunk),
                           identity_sha256=ident_sha, generator_model_sha256=pinned["generator_model_sha256"],
                           tokenizer_json_sha256=pinned["tokenizer_json_sha256"], sampler=SAMPLER_NAME, seed=None)
                records.append(rec)
        if mismatches or not rng_ok:
            status.update(state="failed", error=f"shard {sid}: rank_score mismatches {mismatches}, "
                                                f"rng_unchanged {rng_ok}")
            write_json_atomic(root / "status.json", status)
            append_event(root, "failed", shard=sid, error=status["error"])
            raise RuntimeError(status["error"])
        meta = {**_shard_expect(entry, ident_sha, plan_sha), "batches": batches,
                "batched_forward_calls": fwd, "example_model_calls": ex_calls,
                "model_seconds_with_signals": round(model_s, 3), "wall_seconds": None,
                "rank_score_mismatches": mismatches, "rng_unchanged": rng_ok, "job_id": os.environ.get("SLURM_JOB_ID"),
                "time_utc": now_utc(), "memory": memory()}
        meta["wall_seconds"] = round(time.perf_counter() - t0, 3)
        store.write(sid, records, meta)
        written += 1
        model_s_job += model_s
        done.add(sid)
        status.update(shards_complete=len(done), last_shard=sid,
                      job_wall_seconds=round(time.perf_counter() - t_start, 1),
                      job_model_seconds=round(model_s_job, 1), shards_written_this_job=written)
        write_json_atomic(root / "status.json", status)
        append_event(root, "shard_complete", shard=sid, n=len(records), seconds=meta["wall_seconds"])
        log(f"{sid}: {len(records)} drafts, {fwd} forwards, {meta['wall_seconds']}s "
            f"({len(done)}/{len(plan['shards'])})")
    else:
        status.update(state="collection_complete")
    status.update(shards_complete=len(done), shards_written_this_job=written,
                  job_wall_seconds=round(time.perf_counter() - t_start, 1), finished_utc=now_utc())
    write_json_atomic(root / "status.json", status)
    append_event(root, status["state"], shards_complete=len(done))
    return status


# ---------------------------------------------------------------- validation and manifest

def _reference_drafts(path: str | Path | None) -> dict | None:
    if not path:
        return None
    out = {}
    for line in Path(path).read_text().splitlines():
        if line:
            r = json.loads(line)
            out[(r["split"], r["problem_id"], str(r["budget"]))] = r["response_ids"]
    return out


def validate_bank(root: Path, plan: dict, identity: dict, examples: dict[str, list[Example]], tf: TokenizedFormat,
                  finalize: bool = False, reference_records: str | Path | None = None) -> dict:
    """Re-check every shard and record; on success (and ``finalize``) write MANIFEST.json and BANK_COMPLETE."""
    root = Path(root)
    store = ShardStore(root)
    ident_sha, plan_sha = identity["identity_sha256"], identity["pinned"]["plan_sha256"]
    stored = json.loads((root / "IDENTITY.json").read_text())
    problems: list[str] = []
    if stored["identity_sha256"] != ident_sha:
        problems.append("IDENTITY.json differs from the identity recomputed for this validation")
    if sha256_json(json.loads((root / "plan.json").read_text())) != plan_sha:
        problems.append("plan.json differs from the pinned plan")
    by_id = {s: {e.problem_id: e for e in v} for s, v in examples.items()}
    ref = _reference_drafts(reference_records)
    seen: set[tuple] = set()
    split_ids: dict[str, set] = {s: set() for s in examples}
    shard_rows, missing = [], []
    totals = {"drafts": 0, "tokens": 0, "wrong_tokens": 0, "final_correct": 0, "trace_correct": 0,
              "malformed_drafts": 0, "batched_forward_calls": 0, "example_model_calls": 0,
              "model_seconds_with_signals": 0.0, "rank_score_mismatches": 0, "rng_changed_shards": 0}
    reproduction = {"compared": 0, "identical": 0, "differing": []} if ref is not None else None
    planned = {s["id"] for s in plan["shards"]}
    extra = sorted(store.complete_ids() - planned)
    if extra:
        problems.append(f"unexpected shards: {extra[:5]}")
    for entry in plan["shards"]:
        sid = entry["id"]
        if not store.sidecar_path(sid).exists():
            missing.append(sid)
            continue
        try:
            side = store.verify(sid, _shard_expect(entry, ident_sha, plan_sha))
        except ValueError as e:
            problems.append(str(e))
            continue
        recs = store.read(sid)
        if [r["problem_id"] for r in recs] != entry["problem_ids"]:
            problems.append(f"{sid}: records differ from the planned problem IDs/order")
        for r in recs:
            key = (r["split"], r["problem_id"], r["budget"])
            if key in seen:
                problems.append(f"duplicate record {key}")
            seen.add(key)
            if r["split"] != entry["split"] or r["budget"] != entry["budget"] or r.get("shard") != sid \
                    or [r["family"], r["length"], r["operand_digits"]] != entry["setting"]:
                problems.append(f"{sid}/{r['problem_id']}: record fields differ from its shard")
            if r.get("identity_sha256") != ident_sha or r.get("bank") != plan["bank"]:
                problems.append(f"{sid}/{r['problem_id']}: record identity differs from the bank")
            ex = by_id.get(r["split"], {}).get(r["problem_id"])
            if ex is None:
                problems.append(f"{sid}/{r['problem_id']}: not in the {r['split']} split")
                continue
            split_ids[r["split"]].add(r["problem_id"])
            problems.extend(check_record(r, ex, tf))
            a = r["analysis"]
            totals["drafts"] += 1
            totals["tokens"] += r["response_length"]
            totals["wrong_tokens"] += a["n_token_errors"]
            totals["final_correct"] += a["final_correct"]
            totals["trace_correct"] += a["trace_correct"]
            totals["malformed_drafts"] += not a["well_formed"]
            if ref is not None and key in ref:
                reproduction["compared"] += 1
                if ref[key] == r["generation"]["draft_ids"]:
                    reproduction["identical"] += 1
                elif len(reproduction["differing"]) < 50:
                    reproduction["differing"].append(list(key))
        totals["batched_forward_calls"] += side["batched_forward_calls"]
        totals["example_model_calls"] += side["example_model_calls"]
        totals["model_seconds_with_signals"] += side["model_seconds_with_signals"]
        totals["rank_score_mismatches"] += side["rank_score_mismatches"]
        totals["rng_changed_shards"] += not side["rng_unchanged"]
        shard_rows.append({"id": sid, "file": str(store.data_path(sid).relative_to(root)), "sha256": side["sha256"],
                           "bytes": side["bytes"], "n_records": side["n_records"]})
    if missing:
        problems.append(f"{len(missing)} planned shards missing (first: {missing[:3]})")
    if totals["rank_score_mismatches"] or totals["rng_changed_shards"]:
        problems.append("rank-score mismatches or RNG changes recorded in shard sidecars")
    if "train" in split_ids and "val" in split_ids and split_ids["train"] & split_ids["val"]:
        problems.append("train and val drafts share problem IDs")
    expected = plan["totals"]["drafts"]
    if not missing and totals["drafts"] != expected:
        problems.append(f"{totals['drafts']} drafts, plan expects {expected}")
    totals["model_seconds_with_signals"] = round(totals["model_seconds_with_signals"], 1)
    report = {"bank": plan["bank"], "identity_sha256": ident_sha, "plan_sha256": plan_sha,
              "time_utc": now_utc(), "passed": not problems, "n_problems": len(problems),
              "problems": problems[:200], "missing_shards": len(missing), "totals": totals,
              "per_split_problem_counts": {s: len(v) for s, v in split_ids.items()},
              "reproduction_vs_reference_records": reproduction,
              "reference_records_sha256": sha256_file(Path(reference_records)) if reference_records else None}
    write_json_atomic(root / "validation.json", report)
    if report["passed"] and finalize:
        manifest = {"schema": MANIFEST_SCHEMA, "bank": plan["bank"], "identity_sha256": ident_sha,
                    "plan_sha256": plan_sha, "identity": stored, "shards": shard_rows, "totals": totals,
                    "plan_totals": plan["totals"], "infeasible": plan["infeasible"],
                    "validation_sha256": sha256_file(root / "validation.json"), "created_utc": now_utc()}
        if (root / "BANK_COMPLETE").exists():
            raise SystemExit(f"{root}: BANK_COMPLETE exists; the bank is immutable")
        write_json_atomic(root / "MANIFEST.json", manifest)
        msha = sha256_file(root / "MANIFEST.json")
        write_bytes_atomic(root / "BANK_COMPLETE", f"bank {plan['bank']}\nmanifest_sha256 {msha}\n".encode())
        report["manifest_sha256"] = msha
        append_event(root, "bank_complete", manifest_sha256=msha)
    return report


def read_complete_bank(root: Path) -> tuple[dict, list[tuple[dict, list[dict]]]]:
    """Manifest and records of a finalized bank, after re-hashing every shard against the manifest."""
    root = Path(root)
    marker = (root / "BANK_COMPLETE").read_text().split()
    msha = marker[marker.index("manifest_sha256") + 1]
    if sha256_file(root / "MANIFEST.json") != msha:
        raise ValueError(f"{root}: MANIFEST.json does not match BANK_COMPLETE")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    out = []
    for s in manifest["shards"]:
        p = root / s["file"]
        if sha256_file(p) != s["sha256"]:
            raise ValueError(f"{p}: sha256 differs from the manifest")
        out.append((s, [json.loads(line) for line in p.read_text().splitlines() if line]))
    return {**manifest, "manifest_sha256": msha}, out
