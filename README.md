# Diffusion LM error correction

Research code for *Does Generation-Time Confidence Improve Diffusion Self-Correction?*,
a controlled study of how to weight model-generated errors when fine-tuning a
masked diffusion LM (Open-dCoder-0.5B) to correct its own drafts.

**Status: milestones 1 and 2 of 8 complete (software only).**
- Milestone 1: source audit, backbone adapter, loss and weighting code, gradient
  checks against upstream.
- Milestone 2: arithmetic and program tasks, the fixed-slot response format, the
  checker, splits with hashed manifests, and `inspect-example`.

There is no training loop, draft bank or experiment yet, and no model accuracy has
been measured. See `docs/implementation_plan.md`.

| Doc | Contents |
| --- | --- |
| `docs/handoff.md` | where things stand; start here |
| `docs/implementation_plan.md` | ordered checklist, milestone status |
| `docs/source_audit.md` | sources, pinned revisions, backbone/tokenizer findings, prior-method rules, open questions |
| `docs/decisions.md` | decisions and why (append-only; later entries supersede) |
| `docs/verification.md` | commands run, results, checks not run |
| `docs/data_format.md` | task grammar, response format, record schema, checker, splits |
| `docs/ai_usage.md` | log of AI assistance |
| `docs/inputs/` | project proposal PDF and the approved build prompt (implementation and experiment protocol) |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"            # exact versions: requirements-lock.txt
.venv/bin/dlmc download-backbone             # pinned Open-dCoder-0.5B (1.26 GB) -> runs/hf/, sha256-verified
```

## Checks

```bash
.venv/bin/python -m pytest -q                                   # all 180 tests (weight/tokenizer tests skip without assets)
.venv/bin/python -m pytest -q -m "not real_backbone and not tokenizer"   # no downloaded assets needed
.venv/bin/dlmc doctor --backbone tiny --device cpu --dtype float64
.venv/bin/dlmc doctor --backbone open-dcoder --device cpu       # real forward/loss/backward/AdamW step
.venv/bin/dlmc doctor --backbone open-dcoder --device mps       # same on Apple GPU
```

`doctor` writes a JSON record to `runs/doctor/`.

## Data

The data commands need only the pinned tokenizer files (from `download-backbone`).

```bash
.venv/bin/dlmc generate-data --config configs/data/smoke.yaml           # tiny; -> runs/data/smoke-seed0
.venv/bin/dlmc generate-data --config configs/data/pilot.yaml           # provisional pilot grid; -> runs/data/pilot-seed20261116
.venv/bin/dlmc verify-data --data-dir runs/data/pilot-seed20261116      # re-check hashes, labels, splits, tokens
.venv/bin/dlmc inspect-example --fixture arith_17_28 --draft-values 46,135,126 --pad 2
.venv/bin/dlmc inspect-example --data-dir runs/data/pilot-seed20261116 --split val --index 0
```

`generate-data` writes `<split>.jsonl`, `manifest.json` and `provenance.json`, and
verifies the result before returning. It refuses to overwrite unless `--overwrite`
is given. The same config and seed give a byte-identical manifest. The
`inspect-example` columns that need a draft bank or a trained model (c, q, bin,
current p, weight, CE) print `n/a`, and a draft passed with `--draft-*` is labelled
HYPOTHETICAL. See `docs/data_format.md`.

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
PYTHONPATH=runs/sources/Open-dLLM:src .venv-ref/bin/python scripts/reference_gradients.py      # gradient equivalence
PYTHONPATH=src .venv-ref/bin/python scripts/reference_tokenizer.py --data-dir runs/data/pilot-seed20261116
```

`reference_gradients.py` writes `runs/reference/gradient_equivalence.json` (CPU, about
a minute). `reference_tokenizer.py` compares our tokenizer with transformers' fast
and slow Qwen2 tokenizers on every record.

## Layout

```
src/diffusion_correction/
  models/alignment.py     logits->target alignment (the single shift rule) and layout masks
  models/qwen2_bidir.py   bidirectional Qwen2 forward (no causal mask, no KV cache)
  models/open_dcoder.py   pinned-file loader, sha256 checks, special-token resolution
  models/tiny.py          random tiny backbone for software checks only
  losses/                 plain-PyTorch CE, masked loss, correction/keep loss, rank weights
  data/arithmetic.py      arithmetic chains (generator, exact evaluation)
  data/programs.py        restricted program language: parser, interpreter, generator
  data/format.py          fixed-slot response format, tokenizer checks, slot parsing
  data/tasks.py           prompt rendering, record schema
  data/checker.py         token/value/final/trace scoring, malformed-output policy
  data/splits.py          splits, de-duplication, held-out compositions, manifests, verification
  data/structure.py       rebuild a stored problem (signature, identity) from its source for verification
  data/fixtures.py, data/inspect.py   hand-checked fixtures; inspect-example table
  cli.py, env.py          `dlmc` commands, device and provenance capture
configs/data/            smoke and pilot data configs
scripts/reference_*.py   upstream logits / gradients / tokenizer references (run in .venv-ref)
tests/                   alignment, losses (Gate B), weighting, real backbone, tasks, format, structure, splits
runs/                    git-ignored: weights, upstream clones, datasets, reference, doctor, inspect outputs
```

## Licence and attribution

Apache-2.0. `models/qwen2_bidir.py` is derived from the Open-dCoder
`modeling_qwen2.py` (Qwen team / HuggingFace / Bytedance, Apache-2.0); see its header.
