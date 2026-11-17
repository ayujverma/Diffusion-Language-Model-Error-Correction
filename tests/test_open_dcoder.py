"""Real-backbone checks for the Open-dCoder adapter (skipped if weights are absent).

Reference logits come from scripts/reference_logits.py (upstream Open-dLLM class,
transformers 4.54.1, separate environment).
"""

import json
from pathlib import Path

import pytest
import torch

from diffusion_correction.losses import correction_losses, token_error_masks
from diffusion_correction.models.alignment import build_layout
from diffusion_correction.models.open_dcoder import DEFAULT_LOCAL_DIR, load_open_dcoder, load_tokenizer, resolve_token_spec

REFERENCE = Path("runs/reference/open_dcoder_reference.pt")
pytestmark = [
    pytest.mark.real_backbone,
    pytest.mark.skipif(not (DEFAULT_LOCAL_DIR / "model.safetensors").exists(),
                       reason="pinned Open-dCoder weights not downloaded"),
]


@pytest.fixture(scope="module")
def adapter():
    return load_open_dcoder(DEFAULT_LOCAL_DIR, device="cpu", dtype=torch.float32)


def test_token_ids_from_pinned_files():
    spec = resolve_token_spec(DEFAULT_LOCAL_DIR)
    assert spec.mask_token_id == 151665  # "<M>"
    assert spec.pad_token_id == spec.eos_token_id == 151643  # "<|endoftext|>"
    assert spec.vocab_size == 151936
    gen_cfg = json.loads((DEFAULT_LOCAL_DIR / "generation_config.json").read_text())
    assert gen_cfg["mask_token_id"] == 151643  # the stale value we deliberately do not use (D-003)


def test_tokenizer_digits_are_single_tokens():
    tok = load_tokenizer()
    enc = tok.encode("a=17+28;b=045", add_special_tokens=True)
    # Digits split one per token, but punctuation can merge with a neighbour (";b"),
    # so fixed character width does not imply fixed token width.
    assert enc.tokens == ["a", "=", "1", "7", "+", "2", "8", ";b", "=", "0", "4", "5"]
    assert tok.encode("<M><M>").ids == [151665, 151665]


@pytest.mark.skipif(not REFERENCE.exists(), reason="run scripts/reference_logits.py in .venv-ref first")
def test_matches_upstream_reference_logits(adapter):
    ref = torch.load(REFERENCE)
    ids = ref["input_ids"]
    with torch.no_grad():
        ours = adapter.raw_logits(ids, torch.ones_like(ids, dtype=torch.bool))
    diff = (ours - ref["raw_logits"]).abs()
    scale = ref["raw_logits"].abs().max()
    print(f"max|diff|={float(diff.max()):.3e} mean|diff|={float(diff.mean()):.3e} max|ref|={float(scale):.2f}")
    assert float(diff.max()) < 2e-3
    assert torch.equal(ours.argmax(-1), ref["raw_logits"].argmax(-1))


def test_real_model_is_bidirectional_and_pad_isolated(adapter):
    tok = load_tokenizer()
    m = adapter.tokens.mask_token_id
    prompt = tok.encode("x = 3 + 4\nprint(x)\n# output: ").ids
    resp = [m, m, m, m]
    lay = build_layout([prompt], [resp], pad_token_id=adapter.tokens.pad_token_id)
    with torch.no_grad():
        base = adapter.aligned_logits(lay)
        ids = lay.input_ids.clone()
        ids[0, -1] = tok.encode("9").ids[0]
        changed = adapter.aligned_logits(lay.with_input_ids(ids))
        assert float((base[0, len(prompt)] - changed[0, len(prompt)]).abs().max()) > 1e-3
        # Same example inside a padded batch: real positions unchanged up to float32 kernel noise.
        other = tok.encode("y = 10 * 2 - 7 + 1\n# output: ").ids
        batch = build_layout([prompt, other], [resp, [m] * 6], pad_token_id=adapter.tokens.pad_token_id)
        assert batch.pad_mask[0].any()
        padded = adapter.aligned_logits(batch)[0, : lay.input_ids.shape[1]]
    assert float((padded - base[0]).abs().max()) < 1e-3


def test_real_forward_loss_backward_step(adapter):
    """Forward, correction loss, backward, SGD step: parameters change and stay finite."""
    tok = load_tokenizer()
    prompt = tok.encode("# a=17+28; b=a*3\nanswer = ").ids
    target = tok.encode("045,135").ids
    draft = list(target)
    draft[2] = tok.encode("6").ids[0]  # "046,135": one wrong digit
    lay = build_layout([prompt], [draft], pad_token_id=adapter.tokens.pad_token_id)
    targets = build_layout([prompt], [target], pad_token_id=adapter.tokens.pad_token_id).input_ids
    E, K = token_error_masks(lay.input_ids, targets, lay.editable_mask)
    assert int(E.sum()) == 1 and int(K.sum()) == len(target) - 1

    model = adapter.model
    probe = model.model.layers[0].self_attn.q_proj.weight
    before = probe.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    opt.zero_grad()
    corr = correction_losses(adapter.aligned_logits(lay), targets, E, K)
    (corr.l_error + corr.l_keep).backward()
    grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step()
    changed = float((probe.detach() - before).abs().max())
    with torch.no_grad():  # restore so other tests see pretrained weights
        for p in model.parameters():
            if p.grad is not None:
                p.add_(p.grad, alpha=1e-4)
    opt.zero_grad(set_to_none=True)
    assert torch.isfinite(corr.l_error) and grads_finite and changed > 0
