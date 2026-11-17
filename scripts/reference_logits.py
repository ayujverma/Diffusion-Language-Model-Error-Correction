"""Compute reference logits with the upstream Open-dLLM implementation of Open-dCoder.

Reference = ``veomni.models.transformers.qwen2.modeling_qwen2.Qwen2ForCausalLM`` from
Open-dLLM commit 5b9ba4d (the class CDLM's evaluation code loads for open-dcoder).
The model-card path (Hub ``modeling_qwen2.py``) is also probed and its failures are
recorded; it is not runnable as published (see docs/source_audit.md).

Run in the reference environment (transformers==4.54.1, CDLM's verified version):

    python3 -m venv .venv-ref
    .venv-ref/bin/pip install torch numpy safetensors psutil "transformers==4.54.1"
    git clone https://github.com/pengzhangzhi/Open-dLLM runs/sources/Open-dLLM   # commit 5b9ba4d
    PYTHONPATH=runs/sources/Open-dLLM .venv-ref/bin/python scripts/reference_logits.py

Writes runs/reference/open_dcoder_reference.pt (consumed by tests/test_open_dcoder.py).
Does not import our package, so it is an independent implementation.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL = Path("runs/hf/open-dcoder-0.5B")
OPEN_DLLM = Path("runs/sources/Open-dLLM")
OUT = Path("runs/reference/open_dcoder_reference.pt")
PROMPT = "# Compute a=17+28; b=a*3; c=b-19. Return a,b,c.\nanswer = "


def probe_hub_remote_code() -> dict:
    """Try the Hub modeling file as published, then with the docstring decorator neutralised."""
    out = {}

    def load(patch_doc: bool):
        if patch_doc:
            import transformers.utils as tu
            tu.replace_return_docstrings = lambda *a, **k: (lambda fn: fn)
        name = f"hub_modeling_{int(patch_doc)}"
        spec = importlib.util.spec_from_file_location(name, LOCAL / "modeling_qwen2.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod.Qwen2ForCausalLM

    for patch_doc in (False, True):
        key = "import_as_published" if not patch_doc else "forward_with_docstring_shim"
        try:
            cls = load(patch_doc)
            if patch_doc:
                m = cls.from_pretrained(str(LOCAL), torch_dtype=torch.float32, attn_implementation="sdpa").eval()
                with torch.no_grad():
                    m(input_ids=torch.tensor([[1, 2, 3]]), is_causal=False)
                del m
            out[key] = "ok"
        except Exception as e:  # noqa: BLE001 - recording the failure is the point
            out[key] = f"{type(e).__name__}: {e}".splitlines()[0]
            out[key + "_where"] = traceback.extract_tb(e.__traceback__)[-1].name
    return out


def main() -> None:
    torch.manual_seed(0)
    from veomni.models.transformers.qwen2.modeling_qwen2 import Qwen2ForCausalLM as VeomniQwen2

    tok = AutoTokenizer.from_pretrained(str(LOCAL))
    mask_id = tok.mask_token_id
    prompt_ids = tok(PROMPT)["input_ids"]
    # Row 0: fully masked 11-token response. Row 1: partially revealed draft "045,1<M><M>,<M>16".
    resp0 = [mask_id] * 11
    resp1 = tok("045,1")["input_ids"] + [mask_id, mask_id] + tok(",")["input_ids"] + [mask_id] + tok("16")["input_ids"]
    assert len(resp1) == 11, resp1
    input_ids = torch.tensor([prompt_ids + resp0, prompt_ids + resp1])
    probe = input_ids.clone()
    probe[:, -1] = tok("7")["input_ids"][0]
    row = input_ids.shape[1] - 6  # an output row left of the probed token

    def right_ctx(model, **kw) -> float:
        with torch.no_grad():
            a = model(input_ids=input_ids, **kw).logits
            b = model(input_ids=probe, **kw).logits
        return float((a[:, row] - b[:, row]).abs().max())

    meta: dict = {}
    sdpa = VeomniQwen2.from_pretrained(str(LOCAL), torch_dtype=torch.float32, attn_implementation="sdpa").eval()
    with torch.no_grad():
        ref = sdpa(input_ids=input_ids, attention_mask=None, is_causal=False).logits.float()
        causal = sdpa(input_ids=input_ids, attention_mask=None, is_causal=True).logits.float()
    meta["sdpa_mask_none_is_causal_false_right_context_effect"] = right_ctx(sdpa, attention_mask=None, is_causal=False)
    meta["sdpa_mask_none_is_causal_true_right_context_effect"] = right_ctx(sdpa, attention_mask=None, is_causal=True)
    meta["sdpa_causal_vs_bidir_max_abs_diff"] = float((ref - causal).abs().max())
    ones = torch.ones_like(input_ids)
    meta["sdpa_all_ones_2d_mask_right_context_effect"] = right_ctx(sdpa, attention_mask=ones, is_causal=False)
    # One padded key in a 2-D mask: does the model stay bidirectional?
    padded = ones.clone()
    padded[:, -1] = 0
    with torch.no_grad():
        a = sdpa(input_ids=input_ids, attention_mask=padded, is_causal=False).logits
        p2 = input_ids.clone()
        p2[:, -2] = tok("7")["input_ids"][0]
        b = sdpa(input_ids=p2, attention_mask=padded, is_causal=False).logits
    meta["sdpa_2d_mask_with_pad_right_context_effect"] = float((a[:, row] - b[:, row]).abs().max())
    del sdpa

    eager = VeomniQwen2.from_pretrained(str(LOCAL), torch_dtype=torch.float32, attn_implementation="eager").eval()
    meta["eager_mask_none_is_causal_false_right_context_effect"] = right_ctx(eager, attention_mask=None, is_causal=False)
    del eager

    auto = AutoModelForCausalLM.from_pretrained(str(LOCAL), torch_dtype=torch.float32, trust_remote_code=True).eval()
    with torch.no_grad():
        auto_logits = auto(input_ids=input_ids).logits.float()
    meta["auto_model_for_causal_lm"] = {
        "class": f"{type(auto).__module__}.{type(auto).__name__}",
        "has_diffusion_generate": hasattr(auto, "diffusion_generate"),
        "max_abs_diff_vs_reference": float((auto_logits - ref).abs().max()),
        "right_context_effect": right_ctx(auto),
    }
    del auto
    meta["hub_remote_code"] = probe_hub_remote_code()

    commit = subprocess.run(["git", "-C", str(OPEN_DLLM), "rev-parse", "HEAD"], capture_output=True, text=True)
    meta.update({"reference_class": "veomni.models.transformers.qwen2.modeling_qwen2.Qwen2ForCausalLM",
                 "open_dllm_commit": commit.stdout.strip(), "transformers": str(transformers.__version__),
                 "torch": str(torch.__version__), "prompt": PROMPT, "mask_token_id": mask_id,
                 "pad_token_id": tok.pad_token_id, "eos_token_id": tok.eos_token_id,
                 "bos_token_id": tok.bos_token_id, "probe_row": row})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_ids": input_ids, "raw_logits": ref, "meta": meta}, OUT)
    OUT.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
