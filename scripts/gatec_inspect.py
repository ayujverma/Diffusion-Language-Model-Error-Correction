"""Gate C inspection: predictions, loss components, and which parameters an update changes.

Usage (outside the sandbox for MPS):
  .venv/bin/python scripts/gatec_inspect.py --checkpoint runs/train/gatec-overfit32/checkpoints/ckpt-XXXXXXX \
      --device mps --out runs/gatec/inspect

1. For a few of the overfit examples: per-position target, argmax, p(target) and CE
   with the whole response masked, the masked loss's pooled value, and the monotone
   sampler's output at T = L and T = 2 scored by the checker.
2. Parameter-update check on real weights: (a) the overfit checkpoint vs. the pinned
   pretrained weights (full fine-tuning: every tensor should move); (b) one real LoRA
   update: adapter tensors change, every base tensor stays bit-identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from diffusion_correction.data.checker import check_response
from diffusion_correction.env import resolve_device
from diffusion_correction.losses.masked import masked_diffusion_loss
from diffusion_correction.losses.token_ce import selected_token_ce
from diffusion_correction.models.alignment import build_layout
from diffusion_correction.sampling.monotone import monotone_unmask
from diffusion_correction.training.config import resolve_config
from diffusion_correction.training.data import load_verified_examples
from diffusion_correction.training.model_io import build_training_adapter, load_trained_adapter
from diffusion_correction.training.run import sampler_allowed, task_format
from diffusion_correction.training.trainer import Trainer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n-show", type=int, default=4)
    ap.add_argument("--out", default="runs/gatec/inspect")
    args = ap.parse_args()
    device = resolve_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state = json.loads((Path(args.checkpoint) / "state.json").read_text())
    cfg = state["config"]
    d = cfg["data"]
    tok, tf = task_format(cfg["model_dir"], d["dir"])
    examples, ident = load_verified_examples(d["dir"], d["split"], d["manifest_sha256"], d["subset"])
    assert ident.to_dict() == state["data_identity"], "subset differs from the one trained on"
    adapter, mid = load_trained_adapter(args.checkpoint, device, cfg["model_dir"])
    allowed = sampler_allowed(adapter, tf)
    mask_id = adapter.tokens.mask_token_id
    lines, report = [], {"checkpoint": mid, "examples": []}

    def dec(t):
        return "<M>" if t == mask_id else tok.decode([t])

    with torch.no_grad():
        for ex in examples[:args.n_show]:
            L = len(ex.value_names) * tf.fmt.slot_len
            lay = build_layout([ex.prompt_ids], [ex.response_ids], adapter.tokens.pad_token_id).to(device)
            masked = lay.editable_mask.clone()
            inp = lay.input_ids.masked_fill(masked, mask_id)
            logits = adapter.aligned_logits(lay.with_input_ids(inp))
            ce = selected_token_ce(logits, lay.input_ids, masked).cpu()
            loss = masked_diffusion_loss(logits, lay.input_ids, masked, lay.editable_mask)
            rows = logits[masked].float()
            pt = torch.softmax(rows, -1).gather(-1, lay.input_ids[masked][:, None]).squeeze(-1).cpu()
            am = rows.argmax(-1).cpu().tolist()
            gens = {}
            for T in (L, 2):
                r = monotone_unmask(adapter, [ex.prompt_ids], [L], T, allowed)
                chk = check_response(r.responses[0], ex, tf)
                gens[str(T)] = {"response": "".join(dec(t) for t in r.responses[0]), "parsed": chk.parsed_values,
                                "final_correct": chk.final_correct, "trace_correct": chk.trace_correct}
            lines.append(f"## {ex.problem_id} {ex.family} {ex.setting}  targets {ex.values}")
            lines.append(ex.prompt_text)
            lines.append(f"all-response-masked pooled L_mask = {float(loss.loss):.5f} over {loss.n_masked} tokens")
            lines.append("pos target argmax p(target) CE")
            for k in range(L):
                lines.append(f"{k:3d} {dec(ex.response_ids[k])!r:7} {dec(am[k])!r:7} {float(pt[k]):.4f} "
                             f"{float(ce[k]):.5f}")
            for T, g in gens.items():
                lines.append(f"sampler T={T}: {g['response']!r} parsed={g['parsed']} final={g['final_correct']} "
                             f"trace={g['trace_correct']}")
            lines.append("")
            report["examples"].append({"problem_id": ex.problem_id, "targets": ex.values,
                                       "L_mask_all_masked": float(loss.loss), "ce": ce.tolist(),
                                       "p_target": pt.tolist(), "generation": gens})

    # (a) full fine-tuning: compare with pinned pretrained weights.
    from safetensors.torch import load_file

    pre = load_file(str(Path(cfg["model_dir"]) / "model.safetensors"))
    pre.pop("lm_head.weight", None)
    post = load_file(str(Path(args.checkpoint) / "model.safetensors"))
    changed = {k: not torch.equal(pre[k].float(), post[k]) for k in post}
    report["full_ft_tensors_changed"] = sum(changed.values())
    report["full_ft_tensors_total"] = len(changed)
    report["full_ft_unchanged"] = [k for k, c in changed.items() if not c]
    lines.append(f"full fine-tuning: {sum(changed.values())}/{len(changed)} tensors differ from pretrained; "
                 f"unchanged: {report['full_ft_unchanged']}")
    del adapter

    # (b) one real LoRA update on the same data: adapters move, base tensors stay identical.
    lcfg = resolve_config({**{k: v for k, v in cfg.items()}, "method": "lora",
                           "optimizer": {**cfg["optimizer"], "lr": 2e-4}})
    ladapter, info = build_training_adapter(lcfg, device)
    before = {n: p.detach().clone() for n, p in ladapter.model.named_parameters()}
    tr = Trainer(lcfg, ladapter, info, examples, ident, device, out / "lora_step")
    rec = tr.train_step(list(range(min(16, len(examples)))))
    after = dict(ladapter.model.named_parameters())
    base_same = [n for n in before if ".lora_" not in n and torch.equal(before[n], after[n].detach())]
    base_total = [n for n in before if ".lora_" not in n]
    b_moved = [n for n in before if ".lora_B" in n and not torch.equal(before[n], after[n].detach())]
    report["lora_step"] = {"loss": rec.get("loss"), "trainable_parameters": info["trainable_parameters"],
                           "base_tensors_unchanged": len(base_same), "base_tensors_total": len(base_total),
                           "lora_B_tensors_changed": len(b_moved),
                           "lora_B_tensors_total": sum(".lora_B" in n for n in before)}
    lines.append(f"LoRA step: {report['lora_step']}")
    (out / "inspect.txt").write_text("\n".join(lines) + "\n")
    (out / "inspect.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
