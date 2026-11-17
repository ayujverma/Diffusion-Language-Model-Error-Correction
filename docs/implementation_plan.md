# Implementation plan and status

Working title: *Does Generation-Time Confidence Improve Diffusion Self-Correction?*
The scientific scope is fixed by the proposal and the build prompt (16 Nov 2026
revision). This file is the ordered checklist that later sessions should continue
from. Status labels: IMPLEMENTED, TESTED LOCALLY (tiny backbone / fixtures),
TESTED ON REAL BACKBONE (Open-dCoder-0.5B), EXPERIMENT COMPLETED.

## Milestone 1: source audit, minimal adapter, loss/gradient checks (DONE 2026-11-16)

- [x] Source and requirement audit: `docs/source_audit.md`
- [x] Pinned backbone download and sha256 verification: `dlmc download-backbone`, `models/open_dcoder.py`
- [x] Own bidirectional Qwen2 forward, exact match to upstream `veomni` logits: TESTED ON REAL BACKBONE
- [x] Single alignment function plus explicit layout masks: TESTED LOCALLY and on the real backbone
- [x] Tiny random backbone with the same adapter contract: TESTED LOCALLY
- [x] Plain-PyTorch per-token CE, masked loss, correction/keep loss: TESTED LOCALLY
- [x] Rank-weight rules (uniform, high/low/shuffled confidence, generation-time difficulty, current loss): TESTED LOCALLY
- [x] Gate B tests (see `docs/verification.md` for what was and was not covered)
- [x] `dlmc doctor` real forward/loss/backward/AdamW step: TESTED ON REAL BACKBONE (CPU fp32, MPS fp32)

### Advisor follow-ups on milestone 1 (DONE 2026-11-16, second session)

- [x] Reference gradient comparison against upstream with the same external loss, on a masked and a flawed-draft input: TESTED ON REAL BACKBONE (D-016)
- [x] Recorded confirmation of complete weight loading and embedding tying (CPU and MPS): TESTED ON REAL BACKBONE
- [x] D-006 clarified by D-017 (pooled masked-token CE; mask-rate policy and zero-mask handling still to be fixed before warmup)
- [x] "log c" hand example and CPU/MPS wording corrected in `docs/verification.md`

## Milestone 2: tasks, data, inspect-example (DONE; closed 2026-11-17)

- [x] `data/arithmetic.py`: dependent chains, bounds, exact values; fixtures 2+3, 17+28 → (45, 135, 116), negative/zero: TESTED LOCALLY
- [x] `data/programs.py`: restricted language, hand-written parser/interpreter, generator; checked against an independent `ast` evaluator: TESTED LOCALLY
- [x] Canonical value-only fixed-slot format (D-018), checked on the real tokenizer; EOS question (Q-2) resolved; token→value spans stored: TESTED LOCALLY (real tokenizer)
- [x] Checker with separate token/value/final/trace levels and an explicit malformed-output policy (D-020): TESTED LOCALLY
- [x] Five splits, global de-duplication, held-out adjacent compositions, deterministic hashed manifest, `verify-data`, grouping helper for later drafts: TESTED LOCALLY
- [x] `dlmc inspect-example` (milestone 4–5 fields print `n/a`): TESTED LOCALLY
- [x] Gate A data items (see `docs/verification.md`, "Milestone 2")
- [x] Smoke and pilot datasets generated and verified (pilot task grid is **provisional**, D-019)
- [x] Advisor review gap closed: `verify-data` rebuilds signature, held-out status, normalized key and problem ID from source (`data/structure.py`, D-024); both existing datasets pass unchanged: TESTED LOCALLY

Format spec: `docs/data_format.md`. Protocol: `docs/inputs/Cursor_Diffusion_LM_Build_Prompt.txt` (tracked, D-025).

## Milestone 3: warmup and difficulty calibration (DONE 2026-11-17, local MPS)

Each item lists its status; the evidence is in `docs/verification.md`, "Milestone 3".

1. **Mask policy** (D-028, recorded before trainer code): per-example clamped-uniform
   t, Bernoulli response-only masking, zero-mask examples kept, all-zero minibatches
   skip the optimizer and scheduler steps, pooled CE, gradient accumulation 1,
   separate RNG streams. DONE, TESTED LOCALLY.
2. **Memory/runtime pilot and precision** (D-030): fp32 full fine-tuning, chosen over
   bf16 and LoRA from measurements on MPS. DONE. The A100 is unmeasured.
3. **Trainer** with atomic, resumable checkpoints and a verified durable copy.
   DONE, TESTED ON REAL BACKBONE.
4. **Monotone sampler** (D-029, recorded before the code). DONE, TESTED ON REAL
   BACKBONE.
5. **Gate C**: the overfit reached 100% trace accuracy on 32 examples at update 250,
   and the resume trajectory falls within the device's own noise floor. PASSED.
6. **Common warmup**: `warmup-pilot-v1`, 1,000 updates (the cap), 5,790 s. DONE
   within its limits, but still improving at the cap, so the model is not converged.
7. **Validation calibration** (D-031): an 18-setting screen, then 200 problems × 10
   selected settings × 5 budgets. DONE. The pilot dataset is kept (no new
   manifest). The final primary budget is not chosen here.
8. **Methods draft**: `paper/methods.md`. Started from the verified code; it
   contains no results.

Open after milestone 3 (not blocking milestone 4 software): whether to extend the
warmup beyond 1,000 updates before freezing the generator for the draft bank (the
loss was still falling), and any A100 measurement. Both are addressed by the
continuation below.

## Warmup continuation on TACC Lonestar6 (D-033): PREPARED 2026-11-17, NOT RUN

Runbook: `docs/warmup_continuation.md`. Done locally:
1. Trainer support. DONE, TESTED LOCALLY:
   - a training-update timer with a fixed origin and a cumulative cap;
   - a deadline and a graceful signal stop;
   - persisted early-stopping state on a pinned validation subset;
   - protected checkpoints;
   - per-setting monitoring records.
2. CUDA precision and environment reporting (TF32 off). IMPLEMENTED; CUDA untested.
3. `configs/train/warmup_continuation.yaml`, the monitoring subset, and the Linux lock
   file. DONE.
4. `scripts/tacc/`: transfer staging and verification, environment build, Ranch
   backup, environment check, CUDA pilot, continuation, final evaluation, status and
   resume guard. Helpers TESTED LOCALLY; shell and Slurm scripts syntax- and
   lint-checked only.
5. Preflight of checkpoint 1000, the CUDA reference fixture, and a real-CLI CPU smoke
   run. DONE (Mac).

Pending, in order, on Lonestar6 (the user submits):
1. transfer;
2. Ranch backup;
3. environment;
4. `env_check`;
5. pilot (gates in the runbook §8);
6. continuation from the original checkpoint 1000;
7. final stage-2 comparison of 1000 vs the selected checkpoint.

Then the user chooses the milestone-4 generator checkpoint, recorded in a new entry.

## Milestone 4: immutable draft bank and signal diagnostics

1. Frozen generator copy; monotone collector that saves, per committed token:
   commit step, mask fractions, float32 log c and log q from the SAME forward pass
   (via a detached observer), plus sampling log prob / selection score if they
   differ.
2. Gate D observer checks: changing reference targets changes only analysis fields.
3. Bank validation, hashes, separate validation/test banks.
4. Signal diagnostic: c, −log q, step-0 correction loss by budget/stage/position/difficulty.

## Milestone 5: shared-bank correction training and the four-way screen

Uniform, high c, −log q, and current CE, plus low and shuffled controls. All start
from one warmup checkpoint hash, consume identical bank/example IDs, and use the
same λ/μ, optimizer, and minibatch sampler. Held-out repair evaluation. This needs
minimal complete-draft repair support (one repair call on a saved complete draft),
which is built here; the full correction samplers stay in milestone 6 (D-022).

## Milestone 6: correction samplers and full-generation evaluation

ProSeCo-style correction sampler, repair-of-complete-draft mode, D3IM, call
accounting. Compute-matched ordinary MFT and extra-unmasking / ReMDM inference
controls. Timing harness.

## Milestone 7: prior-method adapters

ProSeCo (basic and error-prioritized, distinct names) and SCOPE + D3IM, each with
fixtures that show supervised positions and gradient paths (resolve Q-4, Q-5).

## Milestone 8 onward: cost estimate, main grid, analysis, paper

Measured per-step and per-example timing, then a total-cost extrapolation before
launching any grid. Three seeds for the main comparisons, transfer tests,
sampler-dependence runs, bootstrap-by-problem statistics, and a report generated
from raw records.

## How to resume in a new session

1. Read this file, then `docs/decisions.md`, `docs/verification.md`, and
   `docs/source_audit.md` §5 (open questions).
2. Recreate the environments and assets with the README "Setup" commands, then run
   `.venv/bin/python -m pytest -q`. All 225 tests should pass. The 8
   `real_backbone` tests need the weights and the 68 `tokenizer` tests need the
   tokenizer files. Inside a sandbox that hides MPS, the MPS case is not collected
   (224).
3. Start at the first pending item: currently the TACC warmup continuation
   (`docs/warmup_continuation.md` §4), then milestone 4.
