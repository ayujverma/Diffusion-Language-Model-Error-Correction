# Diffusion LM error correction

Research code for *Does Generation-Time Confidence Improve Diffusion Self-Correction?*,
a controlled study of how to weight model-generated errors when fine-tuning a
masked diffusion LM (Open-dCoder-0.5B) to correct its own drafts.

**Status: milestone 1 of 8 complete.** That milestone covers the source audit, the
backbone adapter, and the loss and weighting code with its gradient checks. There
are no tasks, draft bank, training loop, or experiments yet. See
`docs/implementation_plan.md`.

| Doc | Contents |
| --- | --- |
| `docs/implementation_plan.md` | ordered checklist, milestone status |
| `docs/source_audit.md` | sources, pinned revisions, backbone findings, prior-method rules, open questions |
| `docs/decisions.md` | decisions and why |
| `docs/verification.md` | commands run, results, checks not run |
| `docs/inputs/` | project proposal PDF |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"            # exact versions: requirements-lock.txt
.venv/bin/dlmc download-backbone             # pinned Open-dCoder-0.5B (1.26 GB) -> runs/hf/, sha256-verified
```

## Checks

```bash
.venv/bin/python -m pytest -q                                   # all tests (real-backbone tests skip without weights)
.venv/bin/python -m pytest -q -m "not real_backbone"            # tiny-backbone / fixture tests only
.venv/bin/dlmc doctor --backbone tiny --device cpu --dtype float64
.venv/bin/dlmc doctor --backbone open-dcoder --device cpu       # real forward/loss/backward/AdamW step
.venv/bin/dlmc doctor --backbone open-dcoder --device mps       # same on Apple GPU
```

`doctor` writes a JSON record to `runs/doctor/`.

### Regenerating the upstream reference logits (optional)

`tests/test_open_dcoder.py::test_matches_upstream_reference_logits` compares our
forward pass with Open-dLLM's implementation. The reference runs in a separate
environment, because the upstream code needs transformers 4.54.1:

```bash
python3 -m venv .venv-ref
.venv-ref/bin/pip install -r requirements-ref-lock.txt
git clone https://github.com/pengzhangzhi/Open-dLLM runs/sources/Open-dLLM
git -C runs/sources/Open-dLLM checkout 5b9ba4d12c92be0010b99c61fc1129a89deeec04
PYTHONPATH=runs/sources/Open-dLLM .venv-ref/bin/python scripts/reference_logits.py
```

## Layout

```
src/diffusion_correction/
  models/alignment.py     logits->target alignment (the single shift rule) and layout masks
  models/qwen2_bidir.py   bidirectional Qwen2 forward (no causal mask, no KV cache)
  models/open_dcoder.py   pinned-file loader, sha256 checks, special-token resolution
  models/tiny.py          random tiny backbone for software checks only
  losses/                 plain-PyTorch CE, masked loss, correction/keep loss, rank weights
  cli.py, env.py          `dlmc` commands, device and provenance capture
scripts/reference_logits.py   upstream reference (runs in .venv-ref)
tests/                         alignment, loss/gradient (Gate B), weighting, real backbone
runs/                          git-ignored: weights, upstream clones, reference and doctor outputs
```

## Licence and attribution

Apache-2.0. `models/qwen2_bidir.py` is derived from the Open-dCoder
`modeling_qwen2.py` (Qwen team / HuggingFace / Bytedance, Apache-2.0); see its header.
