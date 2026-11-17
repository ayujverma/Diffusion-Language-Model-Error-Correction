# Handoff for a fresh session

As of 2026-11-17. **Milestones 1 and 2 are complete and closed as software.**
The milestone-1 advisor follow-ups and the milestone-2 source-review gap (D-024) are
closed. Milestone 3 has **not** been started.
No model has been trained or evaluated on the tasks, and no accuracy has been
measured. This file should be enough to continue. The detailed records are in
`docs/decisions.md`, `docs/verification.md`, `docs/source_audit.md`,
`docs/implementation_plan.md` and `docs/data_format.md`; if they disagree with this
file, they win.

## 1. Inputs (full spec, stored in the repo)

- Build prompt: the approved implementation and experiment protocol, revised
  16 Nov 2026. It is in `docs/inputs/Cursor_Diffusion_LM_Build_Prompt.txt`, tracked
  unchanged since D-025. No other tracked document contains the complete protocol.
- Proposal (scientific scope, 2 pages):
  `docs/inputs/Diffusion_LM_Error_Correction_Project_Proposal.pdf`. Cursor's PDF
  reader fails on it, so extract the text with `pypdf` in a throwaway environment.

Both are byte-identical copies of the originals (sha256 values in D-013). Read the
build prompt before writing code. Sections 4, 6, 7, 11 (Gate C) and 12 govern
milestone 3.

## 2. Project summary

Working title: *Does Generation-Time Confidence Improve Diffusion Self-Correction?*
The author is Ayuj Verma (CS 395T).

**Question.** When training drafts are held fixed, does weighting wrong tokens by
their saved generation-time confidence improve correction beyond uniform
weighting, generation-time correct-target difficulty, and current correction
difficulty?

**Signals at a wrong draft token i:**
- c_i: the frozen generator's probability of the wrong token it committed.
- q_i: the same distribution's probability of the correct token at that step;
  d_gen = −log q_i.
- d_now_i: the current corrector's cross-entropy on the correct token.

These are four different things (uniform, c, d_gen, d_now) and must stay distinct.

**Design.**
1. Masked fine-tuning warmup of the pretrained Open-dCoder-0.5B.
2. A frozen copy of the warmed model builds an immutable draft bank.
3. Every correction arm starts from the same checkpoint with identical drafts and
   hyperparameters; only the error-weighting rule changes.

**Required arms:**
1. compute-matched ordinary masked fine-tuning;
2. uniform;
3. high c;
4. current loss;
5. ProSeCo error-prioritized, adapted;
6. SCOPE with D3IM, adapted;
7. low c;
8. shuffled c;
9. d_gen.

Arms 5 and 6 form a separate comparison family. The primary outcome is
final-answer accuracy at a validation-selected fixed model-call budget. Extensions
stay optional.

**Rules.**
- Do not change the question or drop arms; record conflicts in `decisions.md`.
- Do not launch warmup on real data, draft collection, the grid, or paid hardware
  without the user's go-ahead. Do not invent results. Do not claim any accuracy
  range (such as the proposal's 40–85% target) until it has been measured.
- Label every result IMPLEMENTED / TESTED LOCALLY / TESTED ON REAL BACKBONE /
  EXPERIMENT COMPLETED. Keep software verification separate from scientific results.
- No credentials in the repo.
- Do not record human review that has not happened. **None has happened so far.**
- Keep `decisions.md`, `verification.md` and `ai_usage.md` current. Supersede
  decisions with new entries rather than editing them.
- Ask the user before git commits.

## 3. Milestone status

| Milestone | Status |
| --- | --- |
| 1: source audit, adapter, losses, gradient checks | DONE |
| Milestone-1 advisor follow-ups (gradient equivalence, loading/tying, D-006 clarification, two wording fixes) | DONE (D-016, D-017; `verification.md`, "Milestone-1 follow-ups") |
| 2: tasks, format, checker, splits, inspect-example | DONE and closed as software (D-018 to D-021; review gap closed by D-024). Pilot task grid is **provisional** (D-019) |
| 3: warmup and difficulty calibration | **next**; not started |
| 4 to 8 | not started |

## 4. Next milestone (3): what to do, in order

Full list in `docs/implementation_plan.md`.

1. **Write a decision entry for the mask policy before any warmup** (D-006, D-017,
   D-026; open question Q-8).
   - Masking is response-only with the prompt fixed. This is protocol (D-026), not a
     choice.
   - The entry must fix the precise mask-rate sampling. Upstream draws, per packed
     sequence, u ~ U(0, 1) and sets t = clamp(u, 1/500, 1 − 1/500). That is a
     clamped uniform with probability 1/500 at each endpoint, not a uniform draw on
     [1/500, 1 − 1/500]. The entry must also choose Bernoulli vs. exact-count masking.
   - It must fix zero-mask handling. Upstream returns a loss of 0; the options are to
     resample or to force one mask.
   - Keep the pooled masked-token CE, with no 1/t weight and no path term.
2. **Memory/runtime pilot on MPS:**
   - Compare full fine-tuning with LoRA, and bf16 with fp32 (Q-3).
   - Record measured seconds per step and per example. This is the early runtime
     estimate required by D-022.
   - A100 use needs the user's approval.
   - Fix the trainable-parameter policy for all arms.
3. **Trainer:**
   - Resumable checkpoints covering model, optimizer, scheduler, RNG and data order.
   - JSONL metrics and provenance.
   - Reads the verified splits through `data.splits.load_split`.
4. **Monotone unmasking sampler:**
   - Resolve the reveal schedule, tie-breaking and commit-confidence rule in a
     decision entry **before** writing the code (D-022).
   - Constraints (D-027): training responses have 10, 20 or 30 editable tokens.
     Use only feasible integer reveal schedules (at least one reveal per call, no
     mask left at the end). Report the actual calls, and never add no-op calls to
     match a budget.
   - Count model calls and record traces.
5. **Gate C:** overfit 16–32 examples; checkpoint/resume reproduces the losses.
6. **Difficulty calibration:**
   - Run on validation only, 200–500 problems per setting, scored with
     `data.checker.aggregate`.
   - Record the chosen settings. If the grid changes, regenerate the data (new
     manifest hash).
   - The proposal's target accuracy range is a goal, not a result.
7. Start writing the methods section from the verified code.

Minimal complete-draft repair stays in **milestone 5** (D-022).

## 5. Files

| File | Responsibility |
| --- | --- |
| `src/diffusion_correction/models/alignment.py` | `align_logits` (the only shift rule: `aligned[:, i] = raw[:, i−1]`), `SequenceLayout`, `build_layout` (masks from lengths, never token IDs) |
| `models/qwen2_bidir.py` | Own bidirectional Qwen2 forward pass (SDPA `is_causal=False`, key-padding mask, no KV cache). Apache-2.0 attribution in the header |
| `models/adapter.py` | `DiffusionAdapter.raw_logits` / `.aligned_logits(layout)`; takes no labels |
| `models/open_dcoder.py` | Pinned revision and sha256 values, `download`, `verify_files`, `resolve_token_spec`, `load_open_dcoder` (strict keys, tied embeddings), `load_tokenizer` |
| `models/tiny.py` | Random tiny backbone for software checks |
| `losses/` | Plain-PyTorch CE; L_mask (pooled, D-017); L_error, L_keep; `combined_objective`; rank weights and diagnostics |
| `data/arithmetic.py` | Arithmetic chains: `ArithmeticSpec`, `make_problem`, `sample_chain`, exact evaluation |
| `data/programs.py` | Restricted language: lexer, parser, validator, interpreter (`run_program`), `normalized_key`, `sample_program` |
| `data/format.py` | `ResponseFormat` (sign + 4 digits per value, no EOS), `TokenizedFormat` (IDs from the tokenizer, `parse_slot`), `encode_and_verify` |
| `data/tasks.py` | `render_prompt` (value-free fields only), `Example` record, `build_example` |
| `data/checker.py` | `check_response`, `aggregate`: token, value, final and trace levels; malformed-output policy |
| `data/splits.py` | `DataConfig`, `generate_splits`, `write_splits` (manifest), `load_split`, `verify_dataset`, `assert_grouped_by_problem` |
| `data/structure.py` | `derive_problem`: rebuilds a record's problem from its source, enforcing the generated structure; verification derives signature, held-out status and identity from it (D-024) |
| `data/fixtures.py`, `data/inspect.py` | Hand-checked fixtures; the `inspect-example` table |
| `cli.py`, `env.py` | `dlmc doctor`, `download-backbone`, `generate-data`, `verify-data`, `inspect-example`; provenance |
| `configs/data/smoke.yaml`, `pilot.yaml` | Data configs (the pilot is provisional) |
| `scripts/reference_logits.py`, `reference_gradients.py`, `reference_tokenizer.py` | Upstream / transformers references; run in `.venv-ref` |

Tests (180 in total with MPS available):

| File | Tests | Notes |
| --- | --- | --- |
| `test_alignment.py` | 8 | |
| `test_losses.py` | 13 | |
| `test_weighting.py` | 11 | |
| `test_open_dcoder.py` | 7 | marker `real_backbone`; 6 without MPS |
| `test_data_tasks.py` | 39 | |
| `test_data_format.py` | 48 | |
| `test_data_structure.py` | 34 | hand-specified signatures, structure rejections, identity |
| `test_data_splits.py` | 20 | includes 9 tamper tests with re-hashed manifests |

68 tests carry the `tokenizer` marker.

## 6. Environment and commands

The machine is a MacBook Pro M2 Max (64 GB), macOS 26.6.2, Python 3.14.7 and torch
2.15.0. A Colab A100 is available later, with the user's approval. Cursor's sandbox
hides MPS and blocks the Hub, so use full permissions for those commands.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # pinned: requirements-lock.txt; no transformers (D-008)
.venv/bin/dlmc download-backbone                              # -> runs/hf/open-dcoder-0.5B, sha256-verified
.venv/bin/python -m pytest -q                                 # expect 180 passed (179 inside the sandbox: no MPS case)
.venv/bin/python -m pytest -q -m "not real_backbone and not tokenizer"   # 105, no assets needed
.venv/bin/dlmc doctor --backbone open-dcoder --device mps     # ~10 GB of memory
.venv/bin/dlmc generate-data --config configs/data/pilot.yaml  # -> runs/data/pilot-seed20261116 (verifies itself)
.venv/bin/dlmc verify-data --data-dir runs/data/pilot-seed20261116
.venv/bin/dlmc inspect-example --fixture arith_17_28 --draft-values 46,135,116 --pad 2
.venv/bin/dlmc inspect-example --data-dir runs/data/pilot-seed20261116 --split val --index 0
```

Reference environment (optional; transformers 4.54.1):

```bash
python3 -m venv .venv-ref && .venv-ref/bin/pip install -r requirements-ref-lock.txt
git clone https://github.com/pengzhangzhi/Open-dLLM runs/sources/Open-dLLM
git -C runs/sources/Open-dLLM checkout 5b9ba4d12c92be0010b99c61fc1129a89deeec04
PYTHONPATH=runs/sources/Open-dLLM .venv-ref/bin/python scripts/reference_logits.py
PYTHONPATH=runs/sources/Open-dLLM:src .venv-ref/bin/python scripts/reference_gradients.py
PYTHONPATH=src .venv-ref/bin/python scripts/reference_tokenizer.py --data-dir runs/data/pilot-seed20261116
```

## 7. Evidence (software verification only; no scientific results exist)

Details and tables are in `docs/verification.md`.

**Tests.** `pytest`: 180 passed outside the sandbox. Inside it, 179 pass, because
MPS is hidden and the MPS case is not collected.

**Forward pass.** Our raw logits equal upstream `veomni` exactly (max |diff| 0.0,
fp32, CPU).

**Gradients (D-016).** The same external aligned loss was applied to both models,
on a masked input and on a flawed-draft input:
- fp32: worst per-parameter relative error 3.1e-5 (tolerance 1e-4).
- fp64 with upstream's float32 RoPE/RMSNorm casts matched: 3.4e-15 (tolerance 1e-9).
- fp64 without the matched casts: about 5e-6. This is information only, and the
  cause is identified.
- Controls are detected at relative error 2.9 to 24.7: no logit shift in our loss,
  and upstream run causally.
- Artifact: `runs/reference/gradient_equivalence.json` (sha256 `cac20388…cb54`).

**Loading and tying.**
- All 291 checkpoint tensors load bit-exactly: 290 parameters plus the `lm_head`
  alias.
- `lm_head` and `embed_tokens` share storage after the move to CPU and to MPS, and
  still after an AdamW step.

**Data (pilot):**
- 27,900 records.
- Labels re-derived by the interpreter for every record.
- Composition signature, held-out status, normalized key and problem ID are rebuilt
  from each record's source and match the stored metadata exactly (D-024).
- Splits disjoint by the rebuilt problem ID and normalized key.
- Held-out compositions appear only in `test_unseen`, judged on derived signatures.
- Every record re-tokenizes to its stored IDs, with no merge across the
  prompt/response boundary.
- Manifest byte-identical on regeneration.
- `verify-data` passes.

**Tokenizer.** Our wrapper, transformers' fast tokenizer and the slow pure-Python
Qwen2 tokenizer agree on 83,713 strings with 0 mismatches.

**Mutation check.** 7 of 8 injected data/checker bugs were caught. The survivor is
an equivalent mutant.

**Real inspect-example output.** The fixture `arith_17_28` with a HYPOTHETICAL draft
46,135,116 gives tokens 14/15 correct, values 2/3, final answer correct, trace wrong.
It is stored in `runs/inspect/arith_17_28_hypothetical_draft.txt` and excerpted in
`verification.md`.

**Artifacts.** These are git-ignored under `runs/`. Recreate them with the commands
above.

| Path | Contents |
| --- | --- |
| `runs/hf/open-dcoder-0.5B/` | weights and tokenizer |
| `runs/sources/{Open-dLLM,CDLM,proseco}` | upstream clones |
| `runs/reference/open_dcoder_reference.{pt,json}`, `reference_run.log` | upstream logits |
| `runs/reference/gradient_equivalence.json`, `gradient_run.log` | gradient comparison |
| `runs/data/smoke-seed0/` | smoke dataset; manifest sha256 `527404808937a52446627e576965a0bd37a8cdb3cdb862e577fa278c51b1b2a0` |
| `runs/data/pilot-seed20261116/` | pilot dataset; manifest sha256 `990ecd1dd2341b81bc742fa91189b129877cd9b48b196fbdbc208e8586c81536`; includes `tokenizer_reference_check.json` |
| `runs/inspect/*.txt`, `*.json` | inspect-example outputs |
| `runs/doctor/*.json` | doctor records |

**Pinned revisions.**

| Item | Pin |
| --- | --- |
| Backbone | HF `fredzzp/open-dcoder-0.5B` @ `d0d86d5b99960c05258bb1f8265dd91564dbac67` (file hashes in `open_dcoder.py::PINNED_SHA256`) |
| Open-dLLM (`veomni` reference) | `pengzhangzhi/Open-dLLM` @ `5b9ba4d12c92be0010b99c61fc1129a89deeec04` |
| CDLM | `zhangshuibai/CDLM` @ `86d108e2bb6dbc3f491d217aae04de4fc7c26ce3` |
| ProSeCo | `kuleshov-group/proseco` @ `43ce881779f06f475b56d90a152acd65ddc74fe4` |
| Token IDs | mask `<M>` = 151665 (the generation config's 151643 is stale, D-003); pad = EOS = 151643; no BOS; vocabulary 151,936 (tokenizer 151,666) |
| Response format | `" +"` = 488, `" -"` = 481, digits `"0"`–`"9"` = 15–24 |

## 8. Risks, unresolved choices, and checks not run

**Risks:**
- **The pilot task grid is a guess** (D-019). The proposal's accuracy target
  is untested. Calibration may change lengths or digits, which means regenerating
  the data under a new manifest hash.
- **The value-only format has no variable names in the response.** A model must
  track slot order. If calibration shows this is the main failure mode, revisit it
  through a superseding decision, not an edit.
- **Response-only masking differs from upstream.** Upstream's collator masks whole
  packed sequences, while the protocol masks only response tokens and keeps the
  prompt fixed (D-026). Response tokens make up 13–30% of each pilot sequence
  (sequences are 59–230 tokens). This is a settled, documented difference, not an
  open choice.
- **Runtime is unknown.** No training step has been timed on the pilot sizes.
- **Normalized keys miss commuted or semantically equal programs** (a documented
  limitation).

**Unresolved choices:**
- D-006/D-017/D-026 (Q-8): the precise mask-rate sampling and zero-mask handling.
  Settle before warmup.
- D-009 / Q-3: precision (bf16 untested), and LoRA vs. full fine-tuning.
- D-011: β = 0.5 and the bin edges.
- D-012: λ and μ.
- The monotone sampler's reveal schedule and tie-breaking.
- Q-1: which liger version trained the base checkpoint.
- Q-4: the ProSeCo corrector input.
- Q-5: the SCOPE commit confidence.
- Q-6: the ReMDM adaptation.
- Q-7: whether near-zero L_keep is typical.

**Checks not run:**
- Gate C (overfit, resume).
- Gate D (samplers, observer, log c / log q same-pass).
- ProSeCo/SCOPE stop-gradient fixtures.
- bf16 on any device; any A100 run.
- The memory/runtime pilot.
- Any warmup, draft collection or experiment.
- No human review of anything.

## 9. Git state

- Parent of the milestone-2 work: `8e00216` ("Add research docs, proposal, and
  milestone-1 plan.").
- Milestone 2 is split into reviewable commits on `main` after that parent, in
  order:
  1. milestone-1 follow-ups (gradients, loading/tying, wording);
  2. arithmetic and program task generators;
  3. fixed-slot format, checker and inspect-example;
  4. splits, verification-from-source, data CLI and configs;
  5. documentation, tracked protocol inputs, and milestone-2 closure notes.
- `docs/handoff.md`, `docs/ai_usage.md` and the build prompt are tracked
  (D-023, D-025). Everything under `runs/` stays git-ignored: weights, datasets,
  reference and doctor outputs.
- Run `git log --oneline 8e00216..HEAD` for the current hashes.
