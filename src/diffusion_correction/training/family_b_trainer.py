"""Online Family B trainer (milestone 7).

ProSeCo arms: two forwards with gradient (L_mask, then the corrector on a stop-grad
draft). SCOPE: a Bernoulli mix of one-forward MDM and a no_grad draft plus one
corrector forward. Batch size is the caller's; this module does not shrink an arm.
Checkpoints use the same atomic format as the warmup, so ``dlmc evaluate-calls``
can load them. Test splits are not read.
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

from ..env import configure_precision, git_state, memory_snapshot, release_cached_memory, synchronize
from ..losses.family_b import mix_is_mdm, proseco_training_step, sample_scope_masks, scope_training_step
from ..losses.masked import masked_diffusion_loss
from ..models.adapter import DiffusionAdapter
from ..models.alignment import build_layout
from .checkpoint import copy_to_durable, prune_checkpoints, save_checkpoint, verify_checkpoint
from .data import EpochSampler
from .family_b_config import FAMILY_B_ARMS, family_b_compat_hash, family_b_config_hash
from .masking import apply_masks, sample_response_masks
from .model_io import FULL_MODEL_FILE, full_model_tensors, load_full_model_tensors
from .rng import capture_global_rng, make_generator, restore_global_rng
from .trainer import find_resume_checkpoint, lr_factor, mask_digest, package_versions

PROSECO_STREAMS = ("data_order", "train_mask")
SCOPE_STREAMS = ("data_order", "train_mask", "scope_mix", "scope_mask", "scope_sample")


def streams_for(arm: str) -> tuple[str, ...]:
    if arm not in FAMILY_B_ARMS:
        raise ValueError(f"unknown Family B arm {arm!r}")
    return SCOPE_STREAMS if arm == "scope" else PROSECO_STREAMS


def _jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")


class FamilyBTrainer:
    def __init__(self, cfg: dict, adapter: DiffusionAdapter, model_info: dict, examples: Sequence,
                 data_identity: dict, device: torch.device, run_dir: Path, allowed: torch.Tensor):
        if cfg["arm"] not in FAMILY_B_ARMS:
            raise ValueError(f"unknown Family B arm {cfg['arm']!r}")
        self.cfg = cfg
        self.arm = cfg["arm"]
        self.adapter = adapter
        self.model_info = model_info
        self.examples = list(examples)
        self.data_identity = data_identity
        self.device = device
        self.allowed = allowed
        self.run_dir = Path(run_dir)
        self.ckpt_root = self.run_dir / "checkpoints"
        self.precision = configure_precision(device)
        self.params = [p for p in adapter.model.parameters() if p.requires_grad]
        oc = cfg["optimizer"]
        if oc["name"] != "adamw":
            raise ValueError("only adamw is implemented")
        self.optimizer = torch.optim.AdamW(self.params, lr=oc["lr"], betas=tuple(oc["betas"]), eps=oc["eps"],
                                           weight_decay=oc["weight_decay"], foreach=False)
        warm = max(int(cfg["scheduler"]["warmup_updates"]), 0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda completed: lr_factor(completed, warm))
        self.stream_names = streams_for(self.arm)
        self.gens = {name: make_generator(cfg["seed"], name) for name in self.stream_names}
        self.sampler = EpochSampler(len(self.examples), self.gens["data_order"])
        self.update = 0
        self.batches_consumed = 0
        self.skipped_zero_mask = 0
        self.train_wall_seconds = 0.0
        self.last_checkpoint: str | None = None
        self.resumed_from: str | None = None
        self.autocast = cfg["autocast"]
        self.timers = {"origin_update": 0, "train_update_seconds": 0.0, "checkpoint_seconds": 0.0}
        self.stop_signal: str | None = None

    def _autocast(self):
        if self.autocast is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def describe(self) -> dict:
        oc = self.cfg["optimizer"]
        forwards = ("two forwards with gradient (L_mask and the corrector); batch is not reduced"
                    if self.arm != "scope" else
                    "SCOPE mix: one gradient forward, or one no_grad draft plus one gradient forward. "
                    "Same batch as the ProSeCo arms")
        return {
            "kind": "family_b", "arm": self.arm, "forwards": forwards,
            "time_weight": "1/t on the ProSeCo corrector only" if self.arm != "scope" else "none",
            "beta": self.cfg["corrector"]["beta"], "stage_edges": self.cfg["corrector"]["stage_edges"],
            "groups": "mask-rate bin within the minibatch" if self.arm == "proseco_gen_conf" else None,
            "scope_mix": self.cfg["scope"]["mix_probability"] if self.arm == "scope" else None,
            "optimizer": f"AdamW(lr={oc['lr']}, betas={tuple(oc['betas'])}, eps={oc['eps']}, "
                         f"weight_decay={oc['weight_decay']}, foreach=False)",
            "clip_grad_norm": self.cfg["clip_grad_norm"], "physical_batch_size": self.cfg["batch_size"],
            "param_dtype": self.cfg["dtype"], "precision": self.precision,
            "init_checkpoint": self.model_info.get("init_checkpoint"),
            "trainable_parameters": self.model_info["trainable_parameters"],
            "seed": self.cfg["seed"], "max_updates": self.cfg["limits"]["max_updates"],
            "labels": "Family B online drafts. Not a paper reproduction. Not Family A.",
        }

    def _event(self, kind: str, **kw) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        _jsonl(self.run_dir / "events.jsonl",
               {"event": kind, "update": self.update, "batches_consumed": self.batches_consumed,
                "arm": self.arm, "time": time.time(), **kw})

    def _layout(self, indices: Sequence[int]):
        rows = [self.examples[i] for i in indices]
        layout = build_layout([r.prompt_ids for r in rows], [r.response_ids for r in rows],
                              self.adapter.tokens.pad_token_id)
        return rows, layout

    def _skip(self, indices, rows, layout, reason_bits: dict) -> dict:
        self.skipped_zero_mask += 1
        rec = {"event": "skipped_zero_mask", "batch": self.batches_consumed, "update": self.update,
               "arm": self.arm, "example_indices": list(indices),
               "problem_ids": [r.problem_id for r in rows],
               "shape": list(layout.input_ids.shape), **reason_bits}
        self._event("skipped_zero_mask", batch=self.batches_consumed)
        return rec

    def train_step(self, indices: Sequence[int]) -> dict:
        rows, layout = self._layout(indices)
        self.batches_consumed += 1
        dev = self.device
        mask_cfg = self.cfg["mask"]
        if self.arm == "scope":
            return self._scope_step(indices, rows, layout, dev, mask_cfg)
        draw = sample_response_masks(layout, self.gens["train_mask"], mask_cfg["t_min"], mask_cfg["t_max"])
        if draw.total_masked == 0:
            return self._skip(indices, rows, layout, {"mask_t": draw.t, "n_masked": 0, "branch": self.arm})
        layout_d = layout.to(dev)
        masked = draw.masked_positions.to(dev)
        self.adapter.model.train()
        synchronize(dev)
        t0 = time.perf_counter()
        with self._autocast():
            step = proseco_training_step(
                self.adapter, layout_d, masked, draw.t, self.allowed.to(dev), arm=self.arm,
                beta=float(self.cfg["corrector"]["beta"]),
                stage_edges=tuple(self.cfg["corrector"]["stage_edges"]))
        return self._optimize(step.loss, step.parts, indices, rows, layout, draw.t, draw.masked_positions, t0)

    def _scope_step(self, indices, rows, layout, dev, mask_cfg) -> dict:
        mdm = mix_is_mdm(self.gens["scope_mix"], float(self.cfg["scope"]["mix_probability"]))
        if mdm:
            draw = sample_response_masks(layout, self.gens["train_mask"], mask_cfg["t_min"], mask_cfg["t_max"])
            if draw.total_masked == 0:
                return self._skip(indices, rows, layout, {"mask_t": draw.t, "n_masked": 0, "branch": "mdm"})
            layout_d = layout.to(dev)
            masked = draw.masked_positions.to(dev)
            self.adapter.model.train()
            synchronize(dev)
            t0 = time.perf_counter()
            with self._autocast():
                z = apply_masks(layout_d.input_ids, masked, self.adapter.tokens.mask_token_id)
                logits = self.adapter.aligned_logits(layout_d.with_input_ids(z))
                l_mask = masked_diffusion_loss(logits, layout_d.input_ids, masked, layout_d.editable_mask).loss
            parts = {"arm": "scope", "branch": "mdm", "L_mask": float(l_mask.detach()), "L_corr": 0.0,
                     "n_masked": draw.total_masked, "draft_pass": None}
            return self._optimize(l_mask, parts, indices, rows, layout, draw.t, draw.masked_positions, t0)
        drawn = sample_scope_masks(layout, self.gens["scope_mask"])
        if drawn.total_masked == 0:
            return self._skip(indices, rows, layout, {"mask_t": drawn.rates, "n_masked": 0, "branch": "scope"})
        layout_d = layout.to(dev)
        masked = drawn.masked_positions.to(dev)
        self.adapter.model.train()
        synchronize(dev)
        t0 = time.perf_counter()
        with self._autocast():
            step = scope_training_step(
                self.adapter, layout_d, masked, self.allowed.to(dev), self.gens["scope_sample"],
                temperature=float(self.cfg["scope"]["sample_temperature"]),
                rho=float(self.cfg["scope"]["commit_rho"]))
        parts = dict(step.parts)
        parts["branch"] = "scope"
        parts["L_mask"] = 0.0
        parts["L_corr"] = float(step.loss.detach())
        parts["mask_rate"] = drawn.rates
        return self._optimize(step.loss, parts, indices, rows, layout, drawn.rates, drawn.masked_positions, t0)

    def _optimize(self, loss, parts, indices, rows, layout, rates, masked_cpu, t0) -> dict:
        dev = self.device
        lr = self.optimizer.param_groups[0]["lr"]
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
        rec = {"batch": self.batches_consumed, "update": self.update, "arm": self.arm,
               "example_indices": list(indices), "problem_ids": [r.problem_id for r in rows],
               "mask_t": rates, "mask_digest": mask_digest(masked_cpu),
               "shape": list(layout.input_ids.shape), "loss": loss_f, "lr": lr, "grad_norm": gn_f,
               "clipped": bool(max_norm and gn_f > max_norm), "step_seconds": round(dt, 4),
               "grads_finite": math.isfinite(gn_f), **parts}
        if not math.isfinite(loss_f) or not math.isfinite(gn_f):
            raise FloatingPointError(f"non-finite loss/grad at update {self.update}: loss={loss_f} grad_norm={gn_f}")
        return rec

    def _state(self) -> dict:
        return {
            "update": self.update, "batches_consumed": self.batches_consumed,
            "examples_consumed": self.sampler.examples_consumed,
            "skipped_zero_mask": self.skipped_zero_mask,
            "train_wall_seconds": self.train_wall_seconds, "sampler": self.sampler.state_dict(),
            "config": self.cfg, "config_hash": family_b_config_hash(self.cfg),
            "compat_hash": family_b_compat_hash(self.cfg),
            "data_identity": self.data_identity, "base_model": self.model_info["base"],
            "init_checkpoint": self.model_info.get("init_checkpoint"),
            "model_info": self.model_info, "method": self.cfg["method"], "model_file": FULL_MODEL_FILE,
            "git": git_state(), "versions": package_versions(), "device": str(self.device),
            "resumed_from": self.resumed_from, "training": self.describe(), "timers": self.timers,
            "kind": "family_b", "arm": self.arm, "loss_name": self.arm,
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
        rng = {"global": capture_global_rng(self.device),
               "streams": {name: g.get_state() for name, g in self.gens.items()}}
        path = save_checkpoint(self.ckpt_root, self.update, model_tensors=full_model_tensors(self.adapter.model),
                               model_file=FULL_MODEL_FILE,
                               optimizer_state={"optimizer": self.optimizer.state_dict(),
                                                "scheduler": self.scheduler.state_dict()},
                               rng_state=rng, state=self._state())
        info = json.loads((path / "COMPLETE.json").read_text())
        self.last_checkpoint = path.name
        self._event("checkpoint_saved", path=str(path), model_sha256=info["files"][FULL_MODEL_FILE],
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
        if state.get("kind") != "family_b":
            raise ValueError(f"{path}: not a Family B checkpoint")
        if state.get("arm") != self.arm or state.get("loss_name") != self.arm:
            raise ValueError(f"checkpoint arm {state.get('arm')!r} != config arm {self.arm!r}")
        if state["compat_hash"] != family_b_compat_hash(self.cfg):
            old, new = state["config"], self.cfg
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f"config incompatible with checkpoint {path} (differing keys: {diff})")
        if state["data_identity"] != self.data_identity:
            raise ValueError(f"training data identity differs from checkpoint {path}")
        if state["base_model"] != self.model_info["base"]:
            raise ValueError("base model identity differs from the checkpoint")
        if state.get("init_checkpoint") != self.model_info.get("init_checkpoint"):
            raise ValueError("init_checkpoint differs from the checkpoint")
        load_full_model_tensors(self.adapter.model, path / state["model_file"])
        opt = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
        self.optimizer.load_state_dict(opt["optimizer"])
        self.scheduler.load_state_dict(opt["scheduler"])
        rng = torch.load(path / "rng.pt", map_location="cpu", weights_only=False)
        if set(rng["streams"]) != set(self.gens):
            raise ValueError(f"checkpoint RNG streams {sorted(rng['streams'])} != {list(self.stream_names)}")
        for name, g in self.gens.items():
            g.set_state(rng["streams"][name])
        restore_global_rng(rng["global"], self.device)
        self.update = state["update"]
        self.batches_consumed = state["batches_consumed"]
        self.skipped_zero_mask = state.get("skipped_zero_mask", 0)
        self.train_wall_seconds = state["train_wall_seconds"]
        self.sampler.load_state_dict(state["sampler"])
        self.last_checkpoint = path.name
        self.resumed_from = str(path)
        self.timers = dict(state.get("timers") or {**self.timers, "origin_update": self.update})
        self._event("resumed", path=str(path), model_sha256=complete["files"][state["model_file"]],
                    arm=self.arm)

    def _stop_reason(self, max_updates, max_wall, wall0, t_start) -> str | None:
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
        mem_every = int(self.cfg["memory_every"] or 0)
        rc_every = int(self.cfg["release_cache_every"] or 0)
        wall0, t_start = self.train_wall_seconds, time.perf_counter()
        self._event("train_start", describe=self.describe(), max_updates=max_updates)
        stop = None
        while True:
            stop = self._stop_reason(max_updates, max_wall, wall0, t_start)
            if stop:
                break
            t_iter = time.perf_counter()
            rec = self.train_step(self.sampler.next_indices(self.cfg["batch_size"]))
            self.train_wall_seconds = wall0 + (time.perf_counter() - t_start)
            rec["train_wall_seconds"] = round(self.train_wall_seconds, 2)
            if rec.get("event") != "skipped_zero_mask" and mem_every and self.update % mem_every == 0:
                rec["memory"] = memory_snapshot(self.device)
            if rc_every and self.update and self.update % rc_every == 0:
                release_cached_memory(self.device)
            skipped = rec.get("event") == "skipped_zero_mask"
            if not skipped and self.cfg["log_every"] and self.update % self.cfg["log_every"] == 0:
                print(f"update {self.update} arm {self.arm} loss {rec['loss']:.4f} "
                      f"lr {rec['lr']:.2e} gn {rec['grad_norm']:.2f} {rec['step_seconds']:.2f}s", flush=True)
            rec["train_update_seconds"] = round(self.timers["train_update_seconds"] + time.perf_counter() - t_iter, 3)
            _jsonl(self.run_dir / "metrics.jsonl", rec)
            self.timers["train_update_seconds"] += time.perf_counter() - t_iter
            if skipped:
                continue
            if ck_every and self.update % ck_every == 0:
                self.save()
            self.train_wall_seconds = wall0 + (time.perf_counter() - t_start)
        self.train_wall_seconds = wall0 + (time.perf_counter() - t_start)
        if self.last_checkpoint != f"ckpt-{self.update:07d}":
            self.save()
        summary = {"stop_reason": stop, "update": self.update, "batches_consumed": self.batches_consumed,
                   "examples_consumed": self.sampler.examples_consumed,
                   "skipped_zero_mask": self.skipped_zero_mask,
                   "train_wall_seconds": round(self.train_wall_seconds, 2),
                   "last_checkpoint": self.last_checkpoint, "arm": self.arm, "describe": self.describe(),
                   "data_identity": self.data_identity, "resumed_from": self.resumed_from, "timers": self.timers,
                   "experiment_completed": False,
                   "note": "software run; Family B has not been scored on TACC by this trainer"}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        self._event("train_end", stop_reason=stop)
        return summary


# Re-exported so a resume helper and tests share one lookup.
find_family_b_resume = find_resume_checkpoint
