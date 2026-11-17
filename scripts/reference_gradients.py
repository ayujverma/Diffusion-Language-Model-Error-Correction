"""Compare parameter gradients of our backbone with the upstream Open-dLLM implementation.

Both models load the same pinned Open-dCoder-0.5B weights. Both run the verified
upstream configuration (SDPA, bidirectional, no padding: upstream gets
``attention_mask=None, is_causal=False``; ours gets an all-True key mask). The SAME
plain-PyTorch aligned loss, defined in this file and independent of our package's
loss code, is applied to each model's RAW logits; upstream's own training loss
(liger fused CE + auxiliary path term) is deliberately not used.

Two inputs: (A) a prompt plus a partially masked response, CE on the masked
positions; (B) a prompt plus a complete flawed draft (two wrong digits), CE on every
response position with fixed non-uniform weights. Gradients are compared per
parameter and summarised by group (attention, MLP, normalisation, tied embedding /
output). Two controls show the comparison can detect plausible bugs: dropping the
logit shift in our loss, and running upstream causally.

Run in the reference environment, with both packages importable:

    PYTHONPATH=runs/sources/Open-dLLM:src .venv-ref/bin/python scripts/reference_gradients.py

Writes runs/reference/gradient_equivalence.json and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch
import transformers

from diffusion_correction.models.open_dcoder import DEFAULT_LOCAL_DIR, load_open_dcoder, load_tokenizer

OUT = Path("runs/reference/gradient_equivalence.json")
PROMPT = "# Compute a=17+28; b=a*3; c=b-19. Return a,b,c.\nanswer = "
TARGET = "045,135,116"
DRAFT = "046,135,126"  # two wrong digits
# Predeclared acceptance thresholds on the per-tensor relative L2 error
# ||g_ours - g_ref|| / ||g_ref||. float64 should agree to near machine precision
# (eps 2.2e-16, amplified by depth-24 accumulation). float32 differences can only come
# from operation-order rounding (eps 1.2e-7); 1e-4 leaves ~1000x headroom for that but
# is still far below the controls' errors (order 1).
REL_TOL = {torch.float64: 1e-9, torch.float32: 1e-4}


def external_aligned_loss(raw_logits: torch.Tensor, targets: torch.Tensor, positions: torch.Tensor,
                          weights: torch.Tensor, shift: bool = True) -> torch.Tensor:
    """sum_i w_i * -log softmax(raw[:, i-1])[target_i] / n over selected positions i.

    ``shift=False`` is the control that reads ``raw[:, i]`` instead (the classic bug).
    """
    b, i = positions.nonzero(as_tuple=True)
    rows = raw_logits[b, i - 1] if shift else raw_logits[b, i]
    logp = torch.log_softmax(rows.to(torch.promote_types(rows.dtype, torch.float32)), dim=-1)
    ce = -logp.gather(-1, targets[b, i].unsqueeze(-1)).squeeze(-1)
    return (weights.to(ce.dtype) * ce).sum() / ce.numel()


def group_of(name: str) -> str:
    if "embed_tokens" in name or "lm_head" in name:
        return "embedding_tied_with_lm_head"
    if "self_attn" in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if "norm" in name:
        return "normalisation"
    raise ValueError(name)


def grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Distinct parameters only (named_parameters de-duplicates the tied lm_head)."""
    return {n: p.grad.detach().clone() for n, p in model.named_parameters()}


def compare(g_ours: dict, g_ref: dict) -> dict:
    if g_ours.keys() != g_ref.keys():
        raise AssertionError(f"parameter names differ: {sorted(g_ours.keys() ^ g_ref.keys())[:5]}")
    per_group: dict[str, dict] = {}
    worst = {"rel_l2": 0.0, "name": None}
    for name in g_ref:
        a, r = g_ours[name].double(), g_ref[name].double()
        diff_norm, ref_norm = float((a - r).norm()), float(r.norm())
        rel = diff_norm / ref_norm if ref_norm > 0 else (0.0 if diff_norm == 0 else math.inf)
        g = per_group.setdefault(group_of(name), {"n_tensors": 0, "n_elements": 0, "max_abs_diff": 0.0,
                                                    "max_abs_ref": 0.0, "max_rel_l2": 0.0,
                                                    "min_cosine": 1.0, "diff_sq": 0.0, "ref_sq": 0.0})
        g["n_tensors"] += 1
        g["n_elements"] += r.numel()
        g["max_abs_diff"] = max(g["max_abs_diff"], float((a - r).abs().max()))
        g["max_abs_ref"] = max(g["max_abs_ref"], float(r.abs().max()))
        g["max_rel_l2"] = max(g["max_rel_l2"], rel)
        cos = float(torch.nn.functional.cosine_similarity(a.flatten(), r.flatten(), dim=0)) if ref_norm > 0 else 1.0
        g["min_cosine"] = min(g["min_cosine"], cos)
        g["diff_sq"] += diff_norm ** 2
        g["ref_sq"] += ref_norm ** 2
        if rel > worst["rel_l2"]:
            worst = {"rel_l2": rel, "name": name}
    for g in per_group.values():
        g["group_rel_l2"] = math.sqrt(g.pop("diff_sq")) / math.sqrt(g.pop("ref_sq"))
    return {"groups": per_group, "worst_tensor": worst}


def build_inputs(tok, mask_id: int) -> dict:
    prompt = tok.encode(PROMPT).ids
    target = tok.encode(TARGET).ids
    draft = tok.encode(DRAFT).ids
    assert len(target) == len(draft) == 11, (target, draft)
    n_p, n_r = len(prompt), len(target)
    L = n_p + n_r
    targets = torch.tensor([prompt + target])

    # (A) masked: positions 0, 2, 3, 6, 9, 10 of the response are masked.
    masked_slots = [0, 2, 3, 6, 9, 10]
    resp_a = list(target)
    for s in masked_slots:
        resp_a[s] = mask_id
    pos_a = torch.zeros(1, L, dtype=torch.bool)
    pos_a[0, [n_p + s for s in masked_slots]] = True
    w_a = torch.ones(int(pos_a.sum()))

    # (B) complete flawed draft: CE at every response position, errors weighted 1.5, keeps 0.5.
    pos_b = torch.zeros(1, L, dtype=torch.bool)
    pos_b[0, n_p:] = True
    wrong = [s for s in range(n_r) if draft[s] != target[s]]
    assert len(wrong) == 2, wrong
    w_b = torch.tensor([1.5 if s in wrong else 0.5 for s in range(n_r)])
    return {
        "masked": {"input_ids": torch.tensor([prompt + resp_a]), "targets": targets, "positions": pos_a,
                   "weights": w_a},
        "flawed_draft": {"input_ids": torch.tensor([prompt + draft]), "targets": targets, "positions": pos_b,
                         "weights": w_b, "wrong_slots": wrong},
        "prompt_len": n_p, "response_len": n_r,
    }


def float32_rope_forward(self, x, position_ids):
    """Upstream's RoPE: angles, cos and sin always in float32, then cast to x.dtype."""
    freqs = position_ids.float()[..., None] * self.inv_freq.float().to(x.device)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def float32_rmsnorm_forward(self, x):
    """Upstream's RMSNorm: normalise in float32 whatever the input dtype, then cast back."""
    h = x.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    return self.weight * h.to(x.dtype)


def run_dtype(dtype: torch.dtype, cases: dict, match_upstream_casts: bool) -> dict:
    from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM as VeomniQwen2
    from diffusion_correction.models.qwen2_bidir import Qwen2RMSNorm

    ref = VeomniQwen2.from_pretrained(str(DEFAULT_LOCAL_DIR), torch_dtype=torch.float32,
                                      attn_implementation="sdpa").to(dtype)
    ours = load_open_dcoder(DEFAULT_LOCAL_DIR, device="cpu", dtype=dtype, verify_weights_hash=False).model
    if match_upstream_casts:
        # Upstream computes RoPE angles and RMSNorm statistics in float32 even for float64
        # activations; ours keeps float64. Only float64 runs differ: for float32 inputs
        # both are identity casts, and for bf16 both upcast to float32.
        rope = ours.model.rotary_emb
        rope.forward = float32_rope_forward.__get__(rope)
        for m in ours.modules():
            if isinstance(m, Qwen2RMSNorm):
                m.forward = float32_rmsnorm_forward.__get__(m)
    ref.train()
    ours.train()
    tying = {
        "ours_lm_head_is_embedding": ours.lm_head.weight is ours.model.embed_tokens.weight,
        "ref_lm_head_is_embedding": ref.lm_head.weight is ref.model.embed_tokens.weight,
        "ours_distinct_parameters": sum(1 for _ in ours.named_parameters()),
        "ref_distinct_parameters": sum(1 for _ in ref.named_parameters()),
    }
    weights_equal = all(torch.equal(p, dict(ref.named_parameters())[n]) for n, p in ours.named_parameters())

    out = {"dtype": str(dtype), "rel_tol": REL_TOL[dtype],
           "match_upstream_float32_rope_and_rmsnorm": match_upstream_casts,
           "tying": tying,
           "initial_weights_identical": weights_equal, "cases": {}}
    for case_name in ("masked", "flawed_draft"):
        c = cases[case_name]
        ids = c["input_ids"]
        args = (c["targets"], c["positions"], c["weights"])

        def ref_grad(is_causal: bool):
            ref.zero_grad(set_to_none=True)
            logits = ref(input_ids=ids, attention_mask=None, is_causal=is_causal).logits
            loss = external_aligned_loss(logits, *args)
            loss.backward()
            return float(loss.detach()), logits.detach(), grads(ref)

        def our_grad(shift: bool):
            ours.zero_grad(set_to_none=True)
            logits = ours(ids, torch.ones_like(ids, dtype=torch.bool))
            loss = external_aligned_loss(logits, *args, shift=shift)
            loss.backward()
            return float(loss.detach()), logits.detach(), grads(ours)

        t0 = time.perf_counter()
        loss_r, logits_r, g_r = ref_grad(is_causal=False)
        loss_o, logits_o, g_o = our_grad(shift=True)
        main = compare(g_o, g_r)
        _, _, g_r_causal = ref_grad(is_causal=True)
        ctl_causal = compare(g_o, g_r_causal)
        del g_r_causal
        _, _, g_o_noshift = our_grad(shift=False)
        ctl_shift = compare(g_o_noshift, g_r)
        del g_o_noshift
        tol = REL_TOL[dtype]
        passed = (main["worst_tensor"]["rel_l2"] <= tol
                  and ctl_causal["worst_tensor"]["rel_l2"] > 100 * tol
                  and ctl_shift["worst_tensor"]["rel_l2"] > 100 * tol)
        out["cases"][case_name] = {
            "n_supervised_positions": int(c["positions"].sum()),
            "loss_ref": loss_r, "loss_ours": loss_o, "loss_abs_diff": abs(loss_r - loss_o),
            "raw_logits_max_abs_diff": float((logits_o - logits_r).abs().max()),
            "comparison": main,
            "control_upstream_causal": {"worst_tensor": ctl_causal["worst_tensor"],
                                        "group_rel_l2": {k: v["group_rel_l2"] for k, v in ctl_causal["groups"].items()}},
            "control_no_shift_in_our_loss": {"worst_tensor": ctl_shift["worst_tensor"],
                                             "group_rel_l2": {k: v["group_rel_l2"] for k, v in ctl_shift["groups"].items()}},
            "passed": passed, "seconds": round(time.perf_counter() - t0, 1),
        }
        if "wrong_slots" in c:
            out["cases"][case_name]["wrong_response_slots"] = c["wrong_slots"]
    out["passed"] = (all(c["passed"] for c in out["cases"].values()) and weights_equal
                     and tying["ours_lm_head_is_embedding"] and tying["ref_lm_head_is_embedding"]
                     and tying["ours_distinct_parameters"] == tying["ref_distinct_parameters"])
    del ref, ours
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="float64+upstream_casts,float32,float64",
                   help="comma list of DTYPE or DTYPE+upstream_casts; plain float64 is informational")
    args = p.parse_args()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    tok = load_tokenizer(DEFAULT_LOCAL_DIR)
    mask_id = tok.token_to_id("<M>")
    cases = build_inputs(tok, mask_id)
    commit = subprocess.run(["git", "-C", "runs/sources/Open-dLLM", "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    report = {"reference_class": "veomni.models.transformers.qwen2.modeling_qwen2.Qwen2ForCausalLM",
              "reference_call": "attention_mask=None, is_causal=False, attn_implementation=sdpa, no labels",
              "ours_call": "BidirectionalQwen2(input_ids, all-True key mask)",
              "loss": "external_aligned_loss in scripts/reference_gradients.py (plain PyTorch, shift i-1 -> i)",
              "open_dllm_commit": commit, "transformers": transformers.__version__, "torch": torch.__version__,
              "deterministic_algorithms": True, "device": "cpu", "prompt": PROMPT, "target": TARGET,
              "draft": DRAFT, "prompt_len": cases["prompt_len"], "response_len": cases["response_len"],
              "runs": []}
    for variant in args.variants.split(","):
        name, _, flag = variant.partition("+")
        dtype = getattr(torch, name)
        print(f"== {variant}", flush=True)
        r = run_dtype(dtype, cases, match_upstream_casts=flag == "upstream_casts")
        r["variant"] = variant
        # Unmatched float64 has known, documented float32 casts upstream; it is reported
        # but does not gate.
        r["gating"] = flag == "upstream_casts" or dtype != torch.float64
        for cname, c in r["cases"].items():
            print(f"  {cname}: loss ref={c['loss_ref']:.10g} ours={c['loss_ours']:.10g} "
                  f"logit max|diff|={c['raw_logits_max_abs_diff']:.3e} "
                  f"worst rel L2={c['comparison']['worst_tensor']['rel_l2']:.3e} "
                  f"({c['comparison']['worst_tensor']['name']}); controls causal="
                  f"{c['control_upstream_causal']['worst_tensor']['rel_l2']:.3e} no-shift="
                  f"{c['control_no_shift_in_our_loss']['worst_tensor']['rel_l2']:.3e} passed={c['passed']}",
                  flush=True)
            for g, s in c["comparison"]["groups"].items():
                print(f"    {g:28s} tensors={s['n_tensors']:3d} max|diff|={s['max_abs_diff']:.3e} "
                      f"max|g|={s['max_abs_ref']:.3e} max rel L2={s['max_rel_l2']:.3e} "
                      f"group rel L2={s['group_rel_l2']:.3e} min cos={s['min_cosine']:.12f}")
        report["runs"].append(r)
    report["passed"] = all(r["passed"] for r in report["runs"] if r["gating"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"passed={report['passed']}; wrote {OUT}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
