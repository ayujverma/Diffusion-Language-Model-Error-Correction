"""Shared-bank correction trainer (milestone 5).

Every arm starts from the same init checkpoint, consumes identical bank draft IDs,
and uses the same λ/μ, optimizer, mask policy and minibatch sampler. Only the
error-weighting rule changes. Saved generation-time signals are never recomputed
from the corrector.
"""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Sequence

import torch

from ..env import git_state, memory_snapshot, release_cached_memory, synchronize
from ..evaluation.repair import repair_drafts, summarize_repair
from ..losses.correction import combined_objective, correction_losses, token_error_masks
from ..losses.masked import masked_diffusion_loss
from ..losses.weighting import WeightRule, compute_error_weights, make_group_ids
from ..models.adapter import DiffusionAdapter
from .bank_data import (BankDraft, BankIdentity, collate_correction_batch, gather_error_signals,
                        load_bank_drafts)
from .checkpoint import copy_to_durable, prune_checkpoints, save_checkpoint, verify_checkpoint
from .correction_config import correction_compat_hash, correction_config_hash
from .data import EpochSampler
from .lora import lora_state_dict
from .masking import apply_masks, sample_response_masks
from .model_io import FULL_MODEL_FILE, LORA_FILE, full_model_tensors, load_full_model_tensors
from .rng import STREAMS, capture_global_rng, make_generator, restore_global_rng
from .trainer import find_resume_checkpoint, lr_factor, mask_digest, package_versions


def _jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")


class CorrectionTrainer:
    def __init__(self, cfg: dict, adapter: DiffusionAdapter, model_info: dict,
                 train_drafts: Sequence[BankDraft], bank_identity: BankIdentity | dict,
                 device: torch.device, run_dir: Path,
                 repair_drafts_list: Sequence[BankDraft] | None = None,
                 repair_identity: dict | None = None,
                 data_identity: dict | None = None,
                 allowed: torch.Tensor | None = None):
        self.cfg = cfg
        self.adapter = adapter
        self.model_info = model_info
        self.drafts = list(train_drafts)
        self.bank_identity = bank_identity.to_dict() if isinstance(bank_identity, BankIdentity) else bank_identity
        self.data_identity = data_identity
        self.device = device
        self.run_dir = Path(run_dir)
        self.ckpt_root = self.run_dir / "checkpoints"
        self.repair_drafts = list(repair_drafts_list) if repair_drafts_list is not None else None
        self.repair_identity = repair_identity
        self.allowed = allowed
        self.params = [p for p in adapter.model.parameters() if p.requires_grad]
        oc = cfg["optimizer"]
        if oc["name"] != "adamw":
            raise ValueError("only adamw is implemented")
        self.optimizer = torch.optim.AdamW(self.params, lr=oc["lr"], betas=tuple(oc["betas"]), eps=oc["eps"],
                                           weight_decay=oc["weight_decay"], foreach=False)
        sc = cfg["scheduler"]
        if sc["name"] != "linear_warmup_constant":
            raise ValueError("only linear_warmup_constant is implemented")
        warm = max(int(sc["warmup_updates"]), 0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda completed: lr_factor(completed, warm))
        self.gens = {name: make_generator(cfg["seed"], name) for name in STREAMS}
        self.sampler = EpochSampler(len(self.drafts), self.gens["data_order"])
        self.update = 0
        self.batches_consumed = 0
        self.skipped_empty = 0
        self.train_wall_seconds = 0.0
        self.last_checkpoint: str | None = None
        self.last_eval: dict | None = None
        self.resumed_from: str | None = None
        self.autocast = cfg["autocast"]
        self.timers = {"origin_update": 0, "train_update_seconds": 0.0, "eval_seconds": 0.0,
                       "checkpoint_seconds": 0.0}
        self.stop_signal: str | None = None
        self.rule = WeightRule(cfg["weighting"]["rule"])
        self.beta = float(cfg["weighting"]["beta"])
        self.stage_edges = tuple(float(x) for x in cfg["weighting"]["stage_edges"])
        self.lam = float(cfg["objective"]["lambda"])
        self.mu = float(cfg["objective"]["mu"])
        self.shuffle_seed = int(cfg["weighting"]["shuffle_seed"])

    def _autocast(self):
        if self.autocast is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def describe(self) -> dict:
        oc = self.cfg["optimizer"]
        return {
            "kind": "shared_bank_correction",
            "weighting_rule": self.rule.value, "beta": self.beta, "stage_edges": list(self.stage_edges),
            "lambda": self.lam, "mu": self.mu, "shuffle_seed": self.shuffle_seed,
            "optimizer": f"AdamW(lr={oc['lr']}, betas={tuple(oc['betas'])}, eps={oc['eps']}, "
                         f"weight_decay={oc['weight_decay']}, foreach=False)",
            "scheduler": f"linear warmup over {self.cfg['scheduler']['warmup_updates']} updates, then constant",
            "clip_grad_norm": self.cfg["clip_grad_norm"], "physical_batch_size": self.cfg["batch_size"],
            "param_dtype": self.cfg["dtype"], "autocast": self.autocast, "method": self.cfg["method"],
            "trainable_parameters": self.model_info["trainable_parameters"],
            "total_parameters": self.model_info["total_parameters"], "device": str(self.device),
            "bank_identity": self.bank_identity, "repair_identity": self.repair_identity,
            "mask_policy": "D-028 response-only Bernoulli on the target sequence (L_mask branch); "
                           "correction pass uses the immutable complete draft",
            "signals": "saved bank log_c_gen / log_q_gen / mask_fraction_before (never recomputed)",
            "labels": "engineering defaults; objective coefficients logged at start (D-044)",
        }

    def _event(self, kind: str, **kw) -> None:
        _jsonl(self.run_dir / "events.jsonl",
               {"event": kind, "update": self.update, "batches_consumed": self.batches_consumed,
                "time": time.time(), **kw})

    def _error_weights(self, err_ce_detached: torch.Tensor, groups: torch.Tensor,
                       log_c: torch.Tensor, log_q: torch.Tensor):
        kwargs = dict(rule=self.rule, groups=groups, beta=self.beta, log_c=log_c, log_q=log_q,
                      shuffle_seed=self.shuffle_seed if self.rule is WeightRule.SHUFFLED_CONF else None)
        if self.rule is WeightRule.CURRENT_LOSS:
            kwargs["current_ce"] = err_ce_detached
        return compute_error_weights(**{k: v for k, v in kwargs.items() if v is not None or k == "rule"})

    def train_step(self, indices: Sequence[int]) -> dict:
        batch_drafts = [self.drafts[i] for i in indices]
        batch = collate_correction_batch(batch_drafts, self.adapter.tokens.pad_token_id)
        draw = sample_response_masks(batch.target_layout, self.gens["train_mask"],
                                     self.cfg["mask"]["t_min"], self.cfg["mask"]["t_max"])
        self.batches_consumed += 1
        rec = {"batch": self.batches_consumed, "epoch": self.sampler.epoch, "example_indices": list(indices),
               "problem_ids": batch.problem_ids, "budgets": batch.budget_labels,
               "mask_t": draw.t, "n_masked": draw.total_masked,
               "n_masked_per_example": draw.n_masked, "mask_digest": mask_digest(draw.masked_positions),
               "shape": list(batch.target_layout.input_ids.shape),
               "real_tokens": int(batch.target_layout.attention_mask.sum()),
               "response_tokens": int(batch.target_layout.editable_mask.sum()),
               "weighting_rule": self.rule.value, "beta": self.beta, "lambda": self.lam, "mu": self.mu}

        dev = self.device
        tgt = batch.target_layout.to(dev)
        dft = batch.draft_layout.to(dev)
        masked = draw.masked_positions.to(dev)
        err_mask, keep_mask = token_error_masks(dft.input_ids, tgt.input_ids, tgt.editable_mask)
        n_err = int(err_mask.sum())
        n_keep = int(keep_mask.sum())
        rec.update(n_error=n_err, n_keep=n_keep)

        if draw.total_masked == 0 and n_err == 0 and n_keep == 0:
            self.skipped_empty += 1
            rec.update(event="skipped_empty", update=self.update)
            self._event("skipped_empty", batch=self.batches_consumed)
            return rec

        lr = self.optimizer.param_groups[0]["lr"]
        self.adapter.model.train()
        synchronize(dev)
        t0 = time.perf_counter()
        with self._autocast():
            # Ordinary masking branch on the correct target sequence (prompt fixed).
            if draw.total_masked:
                inp_m = apply_masks(tgt.input_ids, masked, self.adapter.tokens.mask_token_id)
                logits_m = self.adapter.aligned_logits(tgt.with_input_ids(inp_m))
                l_mask = masked_diffusion_loss(logits_m, tgt.input_ids, masked, tgt.editable_mask).loss
            else:
                from ..losses.token_ce import zero_with_graph
                # Graph-connected zero; the draft forward below supplies the graph.
                l_mask = None

            # Correction pass on the immutable complete draft.
            logits_d = self.adapter.aligned_logits(dft)
            if l_mask is None:
                l_mask = zero_with_graph(logits_d)
            log_c, log_q, mfrac, bud = gather_error_signals(
                err_mask, tgt.editable_mask, batch.budgets.to(dev),
                [t.to(dev) for t in batch.log_c_response],
                [t.to(dev) for t in batch.log_q_response],
                [t.to(dev) for t in batch.mask_fraction_response])
            groups = make_group_ids(bud.cpu(), mfrac.cpu(), self.stage_edges) if n_err else torch.zeros(0, dtype=torch.long)

            def weight_fn(err_ce_det: torch.Tensor) -> torch.Tensor:
                wr = self._error_weights(err_ce_det.cpu(), groups, log_c.cpu(), log_q.cpu())
                rec["weight_diagnostics"] = wr.diagnostics
                return wr.weights.to(device=err_ce_det.device, dtype=err_ce_det.dtype)

            corr = correction_losses(logits_d, tgt.input_ids, err_mask, keep_mask,
                                     weights=weight_fn if n_err else None)
            loss, parts = combined_objective(l_mask, corr, self.lam, self.mu)

        loss.backward()
        max_norm = self.cfg["clip_grad_norm"]
        gn = torch.nn.utils.clip_grad_norm_(self.params, max_norm if max_norm else math.inf)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        loss_f, gn_f = float(loss.detach()), float(gn)
        synchronize(dev)
        dt = time.perf_counter() - t0
        self.update += 1
        rec.update(update=self.update, loss=loss_f, lr=lr, grad_norm=gn_f,
                   clipped=bool(max_norm and gn_f > max_norm), step_seconds=round(dt, 4),
                   grads_finite=math.isfinite(gn_f), **parts)
        if not math.isfinite(loss_f) or not math.isfinite(gn_f):
            raise FloatingPointError(f"non-finite loss/grad at update {self.update}: loss={loss_f} grad_norm={gn_f}")
        return rec

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Held-out complete-draft repair on the pinned validation bank drafts."""
        if self.repair_drafts is None:
            raise ValueError("no repair drafts configured")
        was = self.adapter.model.training
        self.adapter.model.eval()
        synchronize(self.device)
        t0 = time.perf_counter()
        records = repair_drafts(self.adapter, self.repair_drafts, batch_size=self.cfg["eval"]["batch_size"],
                                allowed=self.allowed)
        summary = summarize_repair(records)
        result = {"update": self.update, "device": str(self.device), "repair_identity": self.repair_identity,
                  "weighting_rule": self.rule.value, "repair": summary,
                  "eval_seconds": round(time.perf_counter() - t0, 2)}
        if self.cfg["eval"].get("write_records"):
            rdir = self.run_dir / "eval_records"
            rdir.mkdir(parents=True, exist_ok=True)
            with open(rdir / f"repair-update-{self.update:07d}.jsonl", "w") as f:
                for r in records:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
        release_cached_memory(self.device)
        self.adapter.model.train(was)
        self.last_eval = result
        _jsonl(self.run_dir / "evals.jsonl", result)
        return result

    def _timed_eval(self) -> dict:
        t0 = time.perf_counter()
        ev = self.evaluate()
        self.timers["eval_seconds"] += time.perf_counter() - t0
        return ev

    def _state(self) -> dict:
        return {
            "update": self.update, "batches_consumed": self.batches_consumed,
            "examples_consumed": self.sampler.examples_consumed, "skipped_empty": self.skipped_empty,
            "train_wall_seconds": self.train_wall_seconds, "sampler": self.sampler.state_dict(),
            "config": self.cfg, "config_hash": correction_config_hash(self.cfg),
            "compat_hash": correction_compat_hash(self.cfg),
            "bank_identity": self.bank_identity, "repair_identity": self.repair_identity,
            "data_identity": self.data_identity,
            "base_model": self.model_info["base"], "init_checkpoint": self.model_info["init_checkpoint"],
            "model_info": self.model_info, "method": self.cfg["method"],
            "model_file": FULL_MODEL_FILE if self.cfg["method"] == "full" else LORA_FILE,
            "git": git_state(), "versions": package_versions(), "device": str(self.device),
            "last_eval": self.last_eval, "resumed_from": self.resumed_from, "training": self.describe(),
            "timers": self.timers, "kind": "shared_bank_correction",
        }

    def _protected_names(self, root: Path, current: str) -> set[str]:
        ck = self.cfg["checkpoint"]
        names = {current} | {f"ckpt-{int(u):07d}" for u in ck["protect_updates"] or []}
        every = int(ck["protect_every"] or 0)
        if every:
            names |= {p.name for p in Path(root).glob("ckpt-*")
                      if not p.name.endswith(".tmp") and int(p.name[5:]) % every == 0}
        return names

    def save(self) -> Path:
        t0 = time.perf_counter()
        if self.cfg["method"] == "full":
            tensors, fname = full_model_tensors(self.adapter.model), FULL_MODEL_FILE
        else:
            tensors, fname = lora_state_dict(self.adapter.model), LORA_FILE
        rng = {"global": capture_global_rng(self.device),
               "streams": {name: g.get_state() for name, g in self.gens.items()}}
        path = save_checkpoint(self.ckpt_root, self.update, model_tensors=tensors, model_file=fname,
                               optimizer_state={"optimizer": self.optimizer.state_dict(),
                                                "scheduler": self.scheduler.state_dict()},
                               rng_state=rng, state=self._state())
        info = json.loads((path / "COMPLETE.json").read_text())
        self.last_checkpoint = path.name
        self._event("checkpoint_saved", path=str(path), model_sha256=info["files"][fname],
                    seconds=round(time.perf_counter() - t0, 2))
        durable = self.cfg["checkpoint"]["durable_dir"]
        keep = self.cfg["checkpoint"]["keep_last"]
        if durable:
            droot = Path(durable) / self.cfg["run_name"] / "checkpoints"
            rec = copy_to_durable(path, droot)
            self._event("durable_copy", **rec)
            if rec["ok"]:
                gone = prune_checkpoints(droot, keep, protect=self._protected_names(droot, path.name))
                if gone:
                    self._event("pruned_durable_checkpoints", removed=gone)
        removed = prune_checkpoints(self.ckpt_root, keep, protect=self._protected_names(self.ckpt_root, path.name))
        if removed:
            self._event("pruned_local_checkpoints", removed=removed)
        self.timers["checkpoint_seconds"] += time.perf_counter() - t0
        return path

    def resume(self, path: Path) -> None:
        path = Path(path)
        complete = verify_checkpoint(path)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        state = json.loads((path / "state.json").read_text())
        if state.get("kind") != "shared_bank_correction":
            raise ValueError(f"{path}: not a shared-bank correction checkpoint")
        if state["compat_hash"] != correction_compat_hash(self.cfg):
            old, new = state["config"], self.cfg
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f"config incompatible with checkpoint {path} (differing keys: {diff})")
        if state["bank_identity"] != self.bank_identity:
            raise ValueError(f"bank identity differs from checkpoint {path}")
        if state.get("repair_identity") != self.repair_identity:
            raise ValueError(f"repair identity differs from checkpoint {path}")
        if state["base_model"] != self.model_info["base"]:
            raise ValueError("base model identity differs from the checkpoint")
        if state["init_checkpoint"] != self.model_info["init_checkpoint"]:
            raise ValueError("init_checkpoint differs from the checkpoint")
        fname = state["model_file"]
        if self.cfg["method"] == "full":
            load_full_model_tensors(self.adapter.model, path / fname)
        else:
            from safetensors.torch import load_file
            from .lora import load_lora_state_dict
            load_lora_state_dict(self.adapter.model, load_file(str(path / fname)))
        opt = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(opt["optimizer"])
        self.scheduler.load_state_dict(opt["scheduler"])
        rng = torch.load(path / "rng.pt", map_location="cpu", weights_only=False)
        for name, g in self.gens.items():
            g.set_state(rng["streams"][name])
        accel_restored = restore_global_rng(rng["global"], self.device)
        self.update = state["update"]
        self.batches_consumed = state["batches_consumed"]
        self.skipped_empty = state.get("skipped_empty", 0)
        self.train_wall_seconds = state["train_wall_seconds"]
        self.sampler.load_state_dict(state["sampler"])
        self.last_eval = state.get("last_eval")
        self.last_checkpoint = path.name
        self.resumed_from = str(path)
        self.timers = dict(state.get("timers") or {**self.timers, "origin_update": self.update})
        cross_device = state["device"].split(":")[0] != self.device.type
        metrics = self.run_dir / "metrics.jsonl"
        if metrics.exists():
            kept, dropped = [], 0
            for line in metrics.read_text().splitlines():
                if line and json.loads(line)["batch"] <= self.batches_consumed:
                    kept.append(line)
                elif line:
                    dropped += 1
            if dropped:
                shutil.copy(metrics, self.run_dir / f"metrics.pre-resume-{self.update:07d}.jsonl")
                metrics.write_text("".join(x + "\n" for x in kept))
        self._event("resumed", path=str(path), model_sha256=complete["files"][fname],
                    cross_device=cross_device, accelerator_rng_restored=accel_restored)

    def _stop_reason(self, max_updates: int, max_wall, wall0: float, t_start: float) -> str | None:
        limits = self.cfg["limits"]
        if self.update >= max_updates:
            return "max_updates"
        if max_wall is not None and wall0 + (time.perf_counter() - t_start) >= max_wall:
            return "max_wall_seconds"
        if limits["max_train_seconds"] is not None and \
                self.timers["train_update_seconds"] >= limits["max_train_seconds"]:
            return "max_train_seconds"
        if limits["stop_at_unix"] is not None and time.time() >= limits["stop_at_unix"]:
            return "deadline"
        if self.stop_signal:
            return f"signal_{self.stop_signal}"
        return None

    def train(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        limits = self.cfg["limits"]
        max_updates, max_wall = int(limits["max_updates"]), limits["max_wall_seconds"]
        origin = limits["train_seconds_origin_update"]
        if origin is not None and int(origin) != self.timers["origin_update"]:
            raise ValueError(f"limits.train_seconds_origin_update={origin} but timer origin is "
                             f"{self.timers['origin_update']}")
        ck_every = int(self.cfg["checkpoint"]["every_updates"] or 0)
        ev_every = int(self.cfg["eval"]["every_updates"] or 0)
        mem_every = int(self.cfg["memory_every"] or 0)
        rc_every = int(self.cfg["release_cache_every"] or 0)
        wall0, t_start = self.train_wall_seconds, time.perf_counter()
        self._event("train_start", describe=self.describe(), max_updates=max_updates, max_wall_seconds=max_wall,
                    limits=limits, timers=self.timers)
        if self.cfg["eval"]["at_start"] and self.update == 0 and self.last_eval is None and self.repair_drafts is not None:
            print(f"repair eval @ 0: {json.dumps(self._timed_eval())}", flush=True)
        stop = None
        while True:
            stop = self._stop_reason(max_updates, max_wall, wall0, t_start)
            if stop:
                break
            t_iter = time.perf_counter()
            rec = self.train_step(self.sampler.next_indices(self.cfg["batch_size"]))
            self.train_wall_seconds = wall0 + (time.perf_counter() - t_start)
            rec["train_wall_seconds"] = round(self.train_wall_seconds, 2)
            if rec.get("event") != "skipped_empty" and mem_every and self.update % mem_every == 0:
                rec["memory"] = memory_snapshot(self.device)
            if rc_every and self.update % rc_every == 0:
                release_cached_memory(self.device)
            skipped = rec.get("event") == "skipped_empty"
            if not skipped and self.cfg["log_every"] and self.update % self.cfg["log_every"] == 0:
                print(f"update {self.update} loss {rec['loss']:.4f} "
                      f"L_mask {rec['L_mask']:.4f} L_error {rec['L_error']:.4f} L_keep {rec['L_keep']:.4f} "
                      f"lr {rec['lr']:.2e} gn {rec['grad_norm']:.2f} {rec['step_seconds']:.2f}s", flush=True)
            rec["train_update_seconds"] = round(self.timers["train_update_seconds"] + time.perf_counter() - t_iter, 3)
            _jsonl(self.run_dir / "metrics.jsonl", rec)
            self.timers["train_update_seconds"] += time.perf_counter() - t_iter
            if skipped:
                continue
            if ev_every and self.update % ev_every == 0 and self.repair_drafts is not None:
                ev = self._timed_eval()
                print(f"repair eval @ {self.update}: {json.dumps(ev)}", flush=True)
            if ck_every and self.update % ck_every == 0:
                self.save()
            self.train_wall_seconds = wall0 + (time.perf_counter() - t_start)
        if self.repair_drafts is not None and (self.last_eval is None or self.last_eval["update"] != self.update):
            self._timed_eval()
        self.train_wall_seconds = wall0 + (time.perf_counter() - t_start)
        if self.last_checkpoint != f"ckpt-{self.update:07d}":
            self.save()
        summary = {"stop_reason": stop, "update": self.update, "batches_consumed": self.batches_consumed,
                   "examples_consumed": self.sampler.examples_consumed, "skipped_empty": self.skipped_empty,
                   "train_wall_seconds": round(self.train_wall_seconds, 2), "last_checkpoint": self.last_checkpoint,
                   "last_eval": self.last_eval, "describe": self.describe(), "bank_identity": self.bank_identity,
                   "resumed_from": self.resumed_from, "timers": self.timers,
                   "loop_wall_seconds_this_process": round(time.perf_counter() - t_start, 2)}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        self._event("train_end", stop_reason=stop, timers=self.timers)
        return summary


def prepare_bank_drafts(cfg: dict) -> tuple[list[BankDraft], BankIdentity, list[BankDraft] | None, dict | None]:
    """Train drafts + optional repair drafts from the same bank pins."""
    b = cfg["bank"]
    train, ident = load_bank_drafts(
        b["dir"], identity_sha256=b["identity_sha256"], manifest_sha256=b["manifest_sha256"],
        split=b["split"], budgets=b["budgets"], verify_shards=bool(b["verify_shards"]))
    repair_cfg = cfg["eval"]["repair"]
    repair, repair_ident = None, None
    if repair_cfg is not None:
        all_repair, rident = load_bank_drafts(
            b["dir"], identity_sha256=b["identity_sha256"], manifest_sha256=b["manifest_sha256"],
            split=repair_cfg["split"], budgets=[repair_cfg["budget"]],
            verify_shards=False)  # already verified on the train load when verify_shards
        if repair_cfg["n_examples"] is not None:
            # Deterministic prefix by sorted order already applied in load_bank_drafts.
            n = int(repair_cfg["n_examples"])
            if n > len(all_repair):
                raise ValueError(f"repair n_examples={n} > available {len(all_repair)}")
            # Stratified-ish: take first n in stable order (same across arms).
            repair = all_repair[:n]
            from ..data.splits import sha256_bytes
            digest = sha256_bytes("\n".join(f"{d.split}\t{d.problem_id}\t{d.budget}" for d in repair).encode())
            repair_ident = {**rident.to_dict(), "n_drafts": len(repair), "draft_ids_sha256": digest,
                            "subset": {"n_examples": n, "subset_seed": repair_cfg["subset_seed"]}}
        else:
            repair, repair_ident = all_repair, rident.to_dict()
    return train, ident, repair, repair_ident
