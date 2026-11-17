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

## Milestone 3: warmup and difficulty calibration (next)

Do these in order. Items 1–2 are decisions that must be recorded before warmup code
is run on real data.

1. **D-006/D-017/D-026 mask policy.** Masking is over editable response tokens only,
   with the prompt fixed (protocol, D-026; not an open choice). Before any warmup,
   fix in a new decision entry: the precise mask-rate sampling (for example,
   upstream-style t = clamp(u, 1/500, 1 − 1/500) with u ~ U(0,1), drawn per example;
   or a different stated distribution; Bernoulli vs. exact-count masking), and
   zero-mask handling (for example, resample or force at least one masked token).
2. **Memory/runtime pilot** on MPS (and A100 only if the user approves paid
   hardware): full fine-tuning vs. LoRA, bf16 vs. fp32 on MPS (Q-3), batch and
   sequence size for the pilot data (prompt plus response is 59–230 tokens). Record the measured seconds per step and per example; this is the
   early runtime estimate (D-022). Fix the trainable-parameter policy for all main
   arms.
3. Trainer with resumable checkpoints (model, optimizer, scheduler, step, RNG,
   data order), JSONL metrics, config/provenance capture. Inputs come from the
   verified data files (`load_split`), never regenerated on the fly.
4. Ordinary monotone unmasking sampler (integer reveal schedule) with call counting
   and token-state traces. Resolve its ambiguities (reveal schedule, ties, which
   confidence decides the commit order) in a decision entry **before** writing it
   (D-022). Constraints (D-027): training responses have 10, 20 or 30 editable
   tokens; use only feasible schedules (at least one reveal per call, no mask left at
   the end); report actual calls; never add no-op calls to match a budget.
5. Gate C wiring checks: overfit 16–32 examples; checkpoint/resume reproduces the
   loss trajectory.
6. Difficulty-calibration command and pilot (200–500 validation problems per
   setting, several budgets), using the checker's metrics. Record the chosen
   settings before any test evaluation, then regenerate the dataset if the grid
   changes (new manifest hash).
7. Start the methods section (tasks, format, checker, objective) from the verified
   code.

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
   `.venv/bin/python -m pytest -q`. All 180 tests should pass. The 7
   `real_backbone` tests need the weights and the 68 `tokenizer` tests need the
   tokenizer files. Inside a sandbox that hides MPS, the MPS case is not collected
   (179).
3. Start at the first unchecked item of the next milestone.
