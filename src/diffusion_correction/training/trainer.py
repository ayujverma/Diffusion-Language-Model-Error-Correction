"""Resumable masked-diffusion trainer (common warmup; D-017, D-026, D-028, D-030).

One physical minibatch = one loss normalisation unit = at most one optimizer update.
A batch whose masks are all empty is consumed but skipped (no optimizer or scheduler
step) and logged. ``update`` counts completed optimizer updates;
``batches_consumed`` counts data batches drawn.

Run directory::

    config.resolved.json   metrics.jsonl (one record per consumed batch)
    events.jsonl           evals.jsonl    summary.json
    checkpoints/ckpt-XXXXXXX/ + checkpoints/latest
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Sequence

import torch

from ..data.tasks import Example
from ..env import git_state, memory_snapshot, release_cached_memory, synchronize
from ..losses.masked import masked_diffusion_loss
from ..losses.token_ce import selected_token_ce
from ..models.adapter import DiffusionAdapter
from .checkpoint import (copy_to_durable, latest_checkpoint, prune_checkpoints, save_checkpoint,
                         verify_checkpoint)
from .config import compat_hash, config_hash
from .data import DatasetIdentity, EpochSampler, collate_targets
from .lora import lora_state_dict
from .masking import apply_masks, sample_response_masks
from .model_io import FULL_MODEL_FILE, LORA_FILE, full_model_tensors, load_full_model_tensors
from .rng import STREAMS, capture_global_rng, make_generator, restore_global_rng


def _jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def mask_digest(masked: torch.Tensor) -> str:
    m = masked.to("cpu", torch.uint8).contiguous()
    return hashlib.sha256(str(tuple(m.shape)).encode() + m.numpy().tobytes()).hexdigest()[:16]


def package_versions() -> dict:
    import numpy
    import safetensors
    import tokenizers

    return {"torch": torch.__version__, "numpy": numpy.__version__, "tokenizers": tokenizers.__version__,
            "safetensors": safetensors.__version__}


class Trainer:
    def __init__(self, cfg: dict, adapter: DiffusionAdapter, model_info: dict, examples: Sequence[Example],
                 data_identity: DatasetIdentity | dict, device: torch.device, run_dir: Path,
                 eval_examples: Sequence[Example] | None = None, eval_identity: dict | None = None,
                 tf=None, allowed: torch.Tensor | None = None):
        self.cfg = cfg
        self.adapter = adapter
        self.model_info = model_info
        self.examples = list(examples)
        self.data_identity = data_identity.to_dict() if isinstance(data_identity, DatasetIdentity) else data_identity
        self.device = device
        self.run_dir = Path(run_dir)
        self.ckpt_root = self.run_dir / "checkpoints"
        self.eval_examples = list(eval_examples) if eval_examples is not None else None
        self.eval_identity = eval_identity
        self.tf, self.allowed = tf, allowed
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
        # Factor for the update about to be taken, given `completed` updates so far.
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda completed: min(1.0, (completed + 1) / warm) if warm > 0 else 1.0)
        self.gens = {name: make_generator(cfg["seed"], name) for name in STREAMS}
        self.sampler = EpochSampler(len(self.examples), self.gens["data_order"])
        self.update = 0
        self.batches_consumed = 0
        self.skipped_zero_mask = 0
        self.train_wall_seconds = 0.0
        self.last_checkpoint: str | None = None
        self.last_eval: dict | None = None
        self.resumed_from: str | None = None
        self.autocast = cfg["autocast"]

    # ------------------------------------------------------------------ bookkeeping
    @property
    def tied(self) -> bool:
        return bool(getattr(self.adapter.model.config, "tie_word_embeddings", False))

    def _autocast(self):
        if self.autocast is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)

    def describe(self) -> dict:
        oc = self.cfg["optimizer"]
        return {
            "optimizer": f"AdamW(lr={oc['lr']}, betas={tuple(oc['betas'])}, eps={oc['eps']}, "
                         f"weight_decay={oc['weight_decay']}, foreach=False)",
            "scheduler": f"linear warmup over {self.cfg['scheduler']['warmup_updates']} updates, then constant",
            "clip_grad_norm": self.cfg["clip_grad_norm"], "physical_batch_size": self.cfg["batch_size"],
            "grad_accum": self.cfg["grad_accum"], "param_dtype": self.cfg["dtype"], "autocast": self.autocast,
            "method": self.cfg["method"], "trainable_parameters": self.model_info["trainable_parameters"],
            "total_parameters": self.model_info["total_parameters"], "device": str(self.device),
            "mask_policy": "D-028: per-example t=clamp(u,1/500,1-1/500), Bernoulli(t) per response token, "
                           "zero-mask examples kept, all-zero batches skipped, pooled CE (D-017), no 1/t",
            "labels": "engineering defaults, not paper-mandated (D-030)",
        }

    def _event(self, kind: str, **kw) -> None:
        rec = {"event": kind, "update": self.update, "batches_consumed": self.batches_consumed,
               "time": time.time(), **kw}
        _jsonl(self.run_dir / "events.jsonl", rec)

    # ------------------------------------------------------------------ one batch
    def train_step(self, indices: Sequence[int]) -> dict:
        exs = [self.examples[i] for i in indices]
        lay = collate_targets(exs, self.adapter.tokens.pad_token_id)
        draw = sample_response_masks(lay, self.gens["train_mask"], self.cfg["mask"]["t_min"],
                                     self.cfg["mask"]["t_max"])
        self.batches_consumed += 1
        rec = {"batch": self.batches_consumed, "epoch": self.sampler.epoch, "example_indices": list(indices),
               "problem_ids": [e.problem_id for e in exs], "mask_t": draw.t, "n_masked": draw.total_masked,
               "n_masked_per_example": draw.n_masked, "mask_digest": mask_digest(draw.masked_positions),
               "shape": list(lay.input_ids.shape), "real_tokens": int(lay.attention_mask.sum()),
               "response_tokens": int(lay.editable_mask.sum())}
        if draw.total_masked == 0:
            self.skipped_zero_mask += 1
            rec.update(event="skipped_zero_mask", update=self.update)
            self._event("skipped_zero_mask", batch=self.batches_consumed)
            return rec
        dev = self.device
        lay_d = lay.to(dev)
        targets = lay_d.input_ids
        masked = draw.masked_positions.to(dev)
        inp = apply_masks(targets, masked, self.adapter.tokens.mask_token_id)
        lr = self.optimizer.param_groups[0]["lr"]
        self.adapter.model.train()
        synchronize(dev)
        t0 = time.perf_counter()
        with self._autocast():
            logits = self.adapter.aligned_logits(lay_d.with_input_ids(inp))
        out = masked_diffusion_loss(logits, targets, masked, lay_d.editable_mask)
        out.loss.backward()
        max_norm = self.cfg["clip_grad_norm"]
        gn = torch.nn.utils.clip_grad_norm_(self.params, max_norm if max_norm else math.inf)
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        loss = float(out.loss.detach())
        gn = float(gn)
        synchronize(dev)
        dt = time.perf_counter() - t0
        self.update += 1
        rec.update(update=self.update, loss=loss, lr=lr, grad_norm=gn,
                   clipped=bool(max_norm and gn > max_norm), step_seconds=round(dt, 4),
                   grads_finite=math.isfinite(gn))
        if not math.isfinite(loss) or not math.isfinite(gn):
            raise FloatingPointError(f"non-finite loss/grad at update {self.update}: loss={loss} grad_norm={gn}")
        return rec

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self) -> dict:
        """Masked loss with a freshly seeded eval stream (identical masks every time; training
        streams untouched), plus optional monotone-sampler accuracy."""
        exs = self.eval_examples if self.eval_examples is not None else self.examples
        g = make_generator(self.cfg["seed"], "eval")
        bs = self.cfg["eval"]["batch_size"]
        was = self.adapter.model.training
        self.adapter.model.eval()
        synchronize(self.device)
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for k in range(0, len(exs), bs):
            chunk = exs[k:k + bs]
            lay = collate_targets(chunk, self.adapter.tokens.pad_token_id)
            draw = sample_response_masks(lay, g, self.cfg["mask"]["t_min"], self.cfg["mask"]["t_max"])
            if draw.total_masked == 0:
                continue
            lay_d = lay.to(self.device)
            masked = draw.masked_positions.to(self.device)
            inp = apply_masks(lay_d.input_ids, masked, self.adapter.tokens.mask_token_id)
            with self._autocast():
                logits = self.adapter.aligned_logits(lay_d.with_input_ids(inp))
            ce = selected_token_ce(logits, lay_d.input_ids, masked)
            total += float(ce.sum())
            n += int(ce.numel())
        result = {"update": self.update, "n_examples": len(exs), "n_masked_tokens": n,
                  "masked_loss": total / n if n else None,
                  "eval_set": "train_subset" if self.eval_examples is None else self.cfg["eval"]["split"]}
        budgets = self.cfg["eval"]["generate_budgets"]
        if budgets:
            from ..evaluation.generation import generate, score

            if self.tf is None or self.allowed is None:
                raise ValueError("generation eval needs the tokenized format and allowed-token mask")
            result["generation"] = {}
            for b in budgets:
                run = generate(self.adapter, exs, b, self.tf.fmt.slot_len, self.allowed, batch_size=bs)
                _, agg = score(exs, run, self.tf)
                result["generation"][str(b)] = {k: agg[k] for k in (
                    "final_answer_accuracy", "value_accuracy", "token_accuracy", "trace_accuracy",
                    "malformed_example_rate", "malformed_slot_reasons", "model_calls_per_example",
                    "value_errors_per_response", "model_seconds")}
        synchronize(self.device)
        result["eval_seconds"] = round(time.perf_counter() - t0, 2)
        # Full-vocabulary eval logits leave large cached blocks that training would
        # otherwise have to fit around (MPS OOM at batch 32 after a batch-48 eval).
        release_cached_memory(self.device)
        self.adapter.model.train(was)
        self.last_eval = result
        _jsonl(self.run_dir / "evals.jsonl", result)
        return result

    def _early_stop(self, ev: dict) -> bool:
        es = self.cfg["eval"]["early_stop"]
        if not es:
            return False
        gen = ev.get("generation", {}).get(str(es["budget"]))
        if gen is None or ev["masked_loss"] is None:
            return False
        return gen["trace_accuracy"] >= es["trace_accuracy"] and ev["masked_loss"] <= es["max_masked_loss"]

    # ------------------------------------------------------------------ checkpoints
    def _state(self) -> dict:
        return {
            "update": self.update, "batches_consumed": self.batches_consumed,
            "examples_consumed": self.sampler.examples_consumed, "skipped_zero_mask": self.skipped_zero_mask,
            "train_wall_seconds": self.train_wall_seconds, "sampler": self.sampler.state_dict(),
            "config": self.cfg, "config_hash": config_hash(self.cfg), "compat_hash": compat_hash(self.cfg),
            "data_identity": self.data_identity, "eval_identity": self.eval_identity,
            "base_model": self.model_info["base"], "init_checkpoint": self.model_info["init_checkpoint"],
            "model_info": self.model_info, "method": self.cfg["method"],
            "model_file": FULL_MODEL_FILE if self.cfg["method"] == "full" else LORA_FILE,
            "git": git_state(), "versions": package_versions(), "device": str(self.device),
            "last_eval": self.last_eval, "resumed_from": self.resumed_from, "training": self.describe(),
        }

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
        if durable:
            rec = copy_to_durable(path, Path(durable) / self.cfg["run_name"] / "checkpoints")
            self._event("durable_copy", **rec)
        removed = prune_checkpoints(self.ckpt_root, self.cfg["checkpoint"]["keep_last"], protect={path.name})
        if removed:
            self._event("pruned_local_checkpoints", removed=removed)
        return path

    def resume(self, path: Path) -> None:
        path = Path(path)
        complete = verify_checkpoint(path)
        state = json.loads((path / "state.json").read_text())
        if state["compat_hash"] != compat_hash(self.cfg):
            old, new = state["config"], self.cfg
            diff = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
            raise ValueError(f"config incompatible with checkpoint {path} (differing keys: {diff})")
        if state["data_identity"] != self.data_identity:
            raise ValueError(f"dataset differs from checkpoint {path}: {state['data_identity']} vs {self.data_identity}")
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
        self.skipped_zero_mask = state["skipped_zero_mask"]
        self.train_wall_seconds = state["train_wall_seconds"]
        self.sampler.load_state_dict(state["sampler"])
        self.last_eval = state.get("last_eval")
        self.last_checkpoint = path.name
        self.resumed_from = str(path)
        cross_device = state["device"].split(":")[0] != self.device.type
        # Records past the checkpoint describe work that is being redone; keep them aside for audit.
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

    # ------------------------------------------------------------------ loop
    def train(self) -> dict:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        limits = self.cfg["limits"]
        max_updates, max_wall = int(limits["max_updates"]), limits["max_wall_seconds"]
        ck_every = int(self.cfg["checkpoint"]["every_updates"] or 0)
        ev_every = int(self.cfg["eval"]["every_updates"] or 0)
        mem_every = int(self.cfg["memory_every"] or 0)
        rc_every = int(self.cfg["release_cache_every"] or 0)
        wall0, t_start = self.train_wall_seconds, time.perf_counter()
        self._event("train_start", describe=self.describe(), max_updates=max_updates, max_wall_seconds=max_wall)
        if self.cfg["eval"]["at_start"] and self.update == 0 and self.last_eval is None:
            print(f"eval @ 0: {json.dumps(self.evaluate())}", flush=True)
        stop = None
        while True:
            if self.update >= max_updates:
                stop = "max_updates"
                break
            if max_wall is not None and wall0 + (time.perf_counter() - t_start) >= max_wall:
                stop = "max_wall_seconds"
                break
            rec = self.train_step(self.sampler.next_indices(self.cfg["batch_size"]))
            self.train_wall_seconds = wall0 + (time.perf_counter() - t_start)
            rec["train_wall_seconds"] = round(self.train_wall_seconds, 2)
            if rec.get("event") != "skipped_zero_mask" and mem_every and self.update % mem_every == 0:
                rec["memory"] = memory_snapshot(self.device)
            if rc_every and self.update % rc_every == 0:
                # Varying padded lengths make the MPS cache grow far past live memory
                # (78 GB driver vs 6 GB live after 50 updates) and the machine swaps.
                release_cached_memory(self.device)
            _jsonl(self.run_dir / "metrics.jsonl", rec)
            if rec.get("event") == "skipped_zero_mask":
                continue
            if self.cfg["log_every"] and self.update % self.cfg["log_every"] == 0:
                print(f"update {self.update} loss {rec['loss']:.4f} lr {rec['lr']:.2e} gn {rec['grad_norm']:.2f} "
                      f"{rec['step_seconds']:.2f}s", flush=True)
            if ev_every and self.update % ev_every == 0:
                ev = self.evaluate()
                print(f"eval @ {self.update}: {json.dumps(ev)}", flush=True)
                if self._early_stop(ev):
                    stop = "early_stop"
                    self.train_wall_seconds = wall0 + (time.perf_counter() - t_start)
                    break
            if ck_every and self.update % ck_every == 0:
                self.save()
            self.train_wall_seconds = wall0 + (time.perf_counter() - t_start)
        if self.last_eval is None or self.last_eval["update"] != self.update:
            self.evaluate()
        self.train_wall_seconds = wall0 + (time.perf_counter() - t_start)
        if self.last_checkpoint != f"ckpt-{self.update:07d}":
            self.save()
        summary = {"stop_reason": stop, "update": self.update, "batches_consumed": self.batches_consumed,
                   "examples_consumed": self.sampler.examples_consumed, "skipped_zero_mask": self.skipped_zero_mask,
                   "train_wall_seconds": round(self.train_wall_seconds, 2), "last_checkpoint": self.last_checkpoint,
                   "last_eval": self.last_eval, "describe": self.describe(), "data_identity": self.data_identity,
                   "resumed_from": self.resumed_from}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        self._event("train_end", stop_reason=stop)
        return summary


def find_resume_checkpoint(run_dir: Path, durable_dir: str | None, run_name: str) -> Path | None:
    """Local ``latest`` first; otherwise copy the durable ``latest`` back (Colab restart)."""
    local = latest_checkpoint(Path(run_dir) / "checkpoints")
    if local is not None:
        return local
    if durable_dir:
        droot = Path(durable_dir) / run_name / "checkpoints"
        remote = latest_checkpoint(droot)
        if remote is not None:
            dst = Path(run_dir) / "checkpoints" / remote.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copytree(remote, dst)
            verify_checkpoint(dst)
            print(f"restored {remote} from durable storage to {dst} (hashes verified)")
            return dst
    return None
