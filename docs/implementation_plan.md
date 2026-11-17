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

### Pending advisor follow-ups on milestone 1 (open)

- [ ] Reference gradient comparison against upstream (logits already match; gradients were not compared)
- [ ] Recorded confirmation of complete weight loading and the required embedding tying
- [ ] Clarify D-006 (masked-loss contract) with a superseding decision entry
- [ ] Correct the positive "log c" hand example and the CPU/MPS relative-error wording in `docs/verification.md`

Details: `docs/verification.md`, "Pending advisor follow-ups"; `docs/decisions.md` D-014.

## Milestone 2: tasks, data, inspect-example (next; not started)

1. `data/arithmetic.py`: dependent integer chains (2/4/6 operations; 1–3-digit
   operands; +, −, × by bounded small integers), explicit magnitude/sign bounds,
   exact intermediate values. Hand-checked fixtures, including 2+3 and the 17+28
   example (45, 135, 116).
2. `data/programs.py`: restricted assignment language plus deterministic
   interpreter (2–6 assignments; bounded conditions later), with independent
   interpreter tests. Never execute model-generated Python.
3. Canonical fixed-slot response format. Verify the token layout against the real
   tokenizer: digits are single tokens, but punctuation merges (e.g. `";b"`), so
   choose separators whose tokenization is stable. Store token→value-span maps.
   Decide Q-2 (EOS in responses).
4. Checker: value-level and token-level correctness kept separate; malformed
   outputs count as failures and are reported.
5. Splits: train/val/test-ID/test-long/test-unseen-composition, dedup by
   normalised problem, hashed manifest. Same-problem drafts stay in one split.
6. `dlmc inspect-example`: one table showing prompt/response tokens, aligned
   target, masks, and (later) c, q, −log q, bin, current p, weight, CE, weighted CE.
7. Remaining Gate A data checks.

## Milestone 3: warmup and difficulty calibration

1. Decide the masked-warmup mask-rate distribution (D-006) and the trainable-
   parameter policy after a short real-model memory/runtime pilot on MPS and
   (if available) A100. Full fine-tuning vs. LoRA is then fixed for all main arms.
2. Trainer with resumable checkpoints (model, optimizer, scheduler, step, RNG,
   data order), JSONL metrics, config/provenance capture.
3. Ordinary monotone unmasking sampler (integer reveal schedule) with call counting
   and token-state traces.
4. Gate C wiring checks: overfit 16–32 examples; checkpoint/resume check.
5. Difficulty-calibration command and pilot (200–500 validation problems per
   setting, several budgets). Record decisions before any test evaluation.

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
same λ/μ, optimizer, and minibatch sampler. Held-out repair evaluation.

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
   `.venv/bin/python -m pytest -q`. All 36 tests should pass; 5 need the weights.
3. Settle or schedule the pending advisor follow-ups, then start at the first
   unchecked item of the next milestone.
