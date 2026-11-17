"""D-029 sampler on the pinned Open-dCoder (CPU fp32): allowed set, quotas, prompt immutability."""

import pytest
import torch

from diffusion_correction.data.fixtures import get_fixture
from diffusion_correction.data.format import ResponseFormat, TokenizedFormat
from diffusion_correction.data.tasks import build_example
from diffusion_correction.models.open_dcoder import DEFAULT_LOCAL_DIR, load_open_dcoder, load_tokenizer
from diffusion_correction.sampling.monotone import monotone_unmask, reveal_schedule
from diffusion_correction.training.run import sampler_allowed

pytestmark = [
    pytest.mark.real_backbone,
    pytest.mark.skipif(not (DEFAULT_LOCAL_DIR / "model.safetensors").exists(),
                       reason="pinned Open-dCoder weights not downloaded"),
]


def test_real_sampler_invariants_and_allowed_set():
    adapter = load_open_dcoder(DEFAULT_LOCAL_DIR, device="cpu", dtype=torch.float32)
    tok = load_tokenizer()
    prob = get_fixture("arith_17_28")
    tf = TokenizedFormat.from_tokenizer(tok, ResponseFormat(prob.spec.value_bound))
    ex = build_example(prob, "fixture", tok, tf, heldout=False)
    allowed = sampler_allowed(adapter, tf)
    # Excluded: <M>, every special added token (incl. pad = EOS), and the 270 unused output rows.
    assert not allowed[151665] and not allowed[151643]
    assert not allowed[tf.tokenizer_size:].any() and int((~allowed).sum()) >= 151936 - 151666 + 2
    assert all(bool(allowed[i]) for i in list(tf.digit_ids) + list(tf.sign_ids.values()))
    L = len(ex.value_names) * tf.fmt.slot_len
    res = monotone_unmask(adapter, [ex.prompt_ids], [L], 4, allowed, record_traces=True)
    final = res.final_input_ids[0]
    assert final[:len(ex.prompt_ids)].tolist() == ex.prompt_ids
    assert 151665 not in res.responses[0] and all(bool(allowed[t]) for t in res.responses[0])
    revealed = [sum(t != 151665 for t in state) for state in res.traces[0]]
    assert revealed == [sum(reveal_schedule(L, 4)[:s]) for s in range(1, 5)]
    assert res.batched_forward_calls == 4
