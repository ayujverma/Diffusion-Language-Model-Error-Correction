# AI-use log

Record of meaningful AI assistance, reused code, and human checks. Do not add a
human-review claim until the human has actually done the review.

## 2026-11-16: Milestone 1 (Cursor agent, Claude model)

What the AI did:
- Read the proposal and build prompt. Fetched and read upstream sources: the HF
  model files, the Open-dLLM, CDLM, and ProSeCo repositories, and the arXiv text
  of the ProSeCo, SCOPE, and PAPL papers. Wrote `docs/source_audit.md`.
- Wrote all code in `src/diffusion_correction/`, `scripts/reference_logits.py`, and
  `tests/`, plus the docs in `docs/` and the README.
- Ran the tests, the reference script, the doctor gates, and a mutation check, and
  recorded their outputs in `docs/verification.md`.

Reused or derived code:
- `src/diffusion_correction/models/qwen2_bidir.py` re-implements the forward
  computation of `modeling_qwen2.py` from `fredzzp/open-dcoder-0.5B` (Qwen team /
  HuggingFace / Bytedance, Apache-2.0). Attribution and a list of changes are in the
  file header.
- No code was copied from ProSeCo, SCOPE, CDLM, or Open-dLLM. Open-dLLM's `veomni`
  class is only *executed*, unmodified, in a separate environment to produce
  reference logits.

Human checks:
- None recorded yet. The author has not yet reviewed the code, tests, or audit.

## 2026-11-16: Handoff preparation (Cursor agent, Claude model)

What the AI did:
- Wrote `docs/handoff.md`. Copied the build prompt and proposal verbatim into
  `docs/inputs/` (hashes in D-013).
- Recorded four advisor follow-ups as pending (D-014; `docs/verification.md`),
  re-ran the test suite, and made the first local git commit at the user's request.
  No code changed.

Human checks:
- None. The user relayed advisor follow-up items. These are recorded as open work,
  not as a completed review.

## 2026-11-16: Milestone-1 follow-ups and milestone 2 (Cursor agent, Claude model)

What the AI did:
- Wrote `scripts/reference_gradients.py` and `scripts/reference_tokenizer.py`, plus
  the new loading/tying test and hand-value test. Ran them and recorded the results
  (D-016; `docs/verification.md`).
- Wrote D-016 to D-022 (D-017 clarifies D-006 without editing it) and corrected two
  wording errors in `docs/verification.md`.
- Wrote all of `src/diffusion_correction/data/`, the data CLI commands,
  `configs/data/`, and `tests/test_data_*.py`. Generated and verified the smoke and
  pilot datasets. Ran a mutation check on the checker, splits and format code.
- Updated the docs, wrote `docs/data_format.md`, and rewrote `docs/handoff.md`.

Reused or derived code:
- None. The task generators, the program parser and interpreter, the format and the
  checker were written from scratch. The DeepMind Mathematics generator, GSM8K and
  TinyGSM were not used. Python's `ast` module appears only in tests, as an
  independent reference evaluator. Open-dLLM's `veomni` model and transformers'
  tokenizers are only *executed*, unmodified, in `.venv-ref` as references.

Human checks:
- None. The task definitions, the provisional pilot grid (D-019) and the checker
  policy (D-020) have not been reviewed by the author or advisor.

## 2026-11-16: Milestone-2 review packet and tracking (Cursor agent, Claude model)

What the AI did:
- Assembled the milestone-2 review packet for the user (docs, pilot counts, two
  generated examples, source-map pointers). Did not start milestone 3.
- Removed the `.gitignore` exclusions for `docs/handoff.md` and `docs/ai_usage.md`
  (D-023). Left the Cursor build prompt git-ignored. Updated the handoff git-state
  section and committed the milestone-1 follow-ups plus milestone-2 work at the
  user's request.

Human checks:
- None. Preparing the review packet is not a human review of the code or tests.

## 2026-11-17: Milestone-2 closure after advisor source review (Cursor agent, Claude model)

What the AI did:
- Implemented the correction the advisor's source review asked for:
  `data/structure.py` (rebuild problems from source), and a strengthened
  `verify_dataset` that derives the signature, held-out status and identity from
  source (D-024).
- Wrote `tests/test_data_structure.py` and 9 tamper tests in `tests/test_data_splits.py`.
- Ran the targeted tests, the full suite, a mutation check of the new assertions, and
  `verify-data` on the existing smoke and pilot datasets, which pass unchanged.
- Tracked the build prompt unchanged (D-025). Recorded response-only masking as
  protocol (D-026) and the advisor's sampler constraints (D-027). Corrected the
  "30×" figure and the handoff's mask-rate wording.
- Committed the patch and documentation at the user's request.

Human checks:
- The advisor reviewed source code and reported the verification gap; the user relayed
  that finding. This is recorded as the trigger for D-024. It is not a full human
  review of the code or tests, and no human has reviewed the fix.

## 2026-11-17: Milestone 3, masked warmup and calibration (Cursor agent, Claude model)

What the AI did:
- Recorded the mask policy (D-028), the sampler policy (D-029), the precision and
  trainable-parameter policy (D-030) and the calibration protocol (D-031). Each was
  recorded before the code or the run it governs.
- Wrote `training/` (masking, RNG streams, verified data loading, LoRA, atomic
  checkpoints, config, trainer), `sampling/monotone.py`, `evaluation/` (generation
  scoring, calibration, timing pilot, precision check, run comparison), the new CLI
  commands, configs, the Colab launcher, `scripts/gatec_inspect.py`,
  `scripts/mutation_check_m3.py`, the new tests and `paper/methods.md`.
- Ran the precision check, the timing pilot, the feasibility runs, the Gate C overfit
  and resume checks, the common warmup and the validation calibration on the local
  M2 Max. It diagnosed two failed warmup attempts (an MPS out-of-memory error after
  evaluation, then allocator-cache growth that caused swapping) and fixed them by
  releasing cached memory. Both attempts are archived under `runs/train/`.
- Wrote the documentation updates and committed at the user's request.

Human checks:
- None. No human has reviewed the milestone-3 code, the runs or the documents.

## 2026-11-17: Warmup-continuation preparation for TACC Lonestar6 (Cursor agent, Claude model)

What the AI did:
- Asked the user four questions: the system (Lonestar6), the account details (not
  known yet), the backup target (Ranch) and the path layout (the default). Recorded
  the continuation policy as D-033 before any CUDA run.
- Extended the existing trainer and CLI; no second trainer. Added:
  - the training-update timer and cumulative cap;
  - the deadline and SIGUSR1/SIGTERM graceful stop;
  - persisted early stopping on a pinned validation subset;
  - checkpoint protection;
  - per-setting evaluation records;
  - CUDA precision settings and reporting.

  Also wrote the continuation config, the monitoring subset, the Linux lock file,
  `scripts/tacc/` (Python helpers, shell and Slurm scripts), two test files and a
  D-033 mutation check, and `docs/warmup_continuation.md`.
- Ran, on the Mac only:
  - the tests and both mutation checks;
  - the preflight of checkpoint 1000 and its validation on `ckpt-0000900`;
  - fixture creation and self-check;
  - a CPU smoke run of the real CLI from a clone of checkpoint 1000;
  - the helper scripts on real files;
  - `bash -n` and `shellcheck`.

  It checked Lonestar6 and Ranch rules against TACC's user guides. Following the
  Lonestar6 guide, it moved the virtual environment from the user-chosen `$WORK` to
  `$SCRATCH`.
- Submitted no job and ran nothing on TACC. Did not start milestone 4. Wrote the
  documentation updates and committed at the user's request.

Human checks:
- None. No human has reviewed the D-033 code, scripts or documents.

## 2026-11-17: TACC compatibility fallback D-034, prepared only (Cursor agent, Claude model)

- The user reported that Lonestar6 job 3591308 stopped at the driver guard (driver
  570.195.03; torch 2.15.0+cu132 needs R580) and asked for the documented fallback to be
  prepared without installing it or submitting jobs.
- The agent verified the official torch 2.14.1+cu126 wheel for CPython 3.14 on Linux
  x86-64 and its dependencies, regenerated the TACC lock, updated the driver and build
  checks and bound the environment and pass markers to the lock, added `stage_update.sh`
  for a code-only update and `tests/test_tacc_env.py`, and wrote D-034 (proposed) and the
  runbook section 11.
- Ran on the Mac only: lock resolution and a hash-checked download of every locked wheel
  for the Linux target, a dependency-closure check, the TACC tests, a mutation check of
  the guards, `bash -n` and `shellcheck`. Submitted nothing and ran nothing on TACC.

Human checks:
- The user approved D-034 on 2026-11-17 for rebuilding and validating the TACC
  environment, and supplied the job-log paths and the `$WORK` path used in runbook §11.3.
  No human has reviewed the code.

## 2026-11-18: Full-node GPU launcher, D-035 (Cursor agent, Claude model)

- The user reported that `gpu-a100-small` job 3592559 failed importing NumPy (missing
  `lahf_lm`) and that the installed environment passed a probe on a full `gpu-a100` node,
  and asked for launcher changes only.
- The agent moved the GPU jobs to `gpu-a100`, added single-GPU selection before any CUDA
  use with a separate physical-inventory record, a partition guard and a per-job
  PyTorch one-A100 check, set the pilot report's rate to 3 SU per node-hour, and wrote
  D-035 and runbook §11.4. The lock and the environment were not changed.
- Ran on the Mac only: the launcher tests, a mutation check of the guards, `bash -n` and
  `shellcheck`. Submitted nothing.

Human checks:
- None. The user ran the remote probe and reported its results.

## 2026-11-18: Pilot 3592809 fixture update-norm diagnostic (Cursor agent, Claude model)

- The user reported that pilot 3592809 failed only the fixture's embedding update-norm
  check, and asked for a separate, bounded diagnostic without changing the gate.
- The agent wrote:
  - `fixture_update_diagnostic.py`, which repeats the fixture's training step and records
    native fp32 and float64 norms, raw embedding values, alignment and optimizer state,
    and allocator counters;
  - `summarize_pilot_memory.py`;
  - `fixture_diagnostic.slurm`;
  - tests and runbook §11.5.
- Ran on the Mac CPU only: two diagnostic runs, which reproduced the reference fixture's
  norms bitwise, one comparison, and the targeted tests and `shellcheck`. Submitted
  nothing.

Human checks:
- None. The remote facts are as reported by the user. Any gate change awaits the user's
  review.

## 2026-11-18: D-036 fixture float64 per-tensor norms (Cursor agent, Claude model)

- The user confirmed diagnostic 3593175 (`norm_accumulation_explains_failure: true`) and
  asked to fix the fixture norm measurement while keeping the 1% tolerance and the
  historical fixture.
- The agent added a shared chunked CPU float64 L2 in `cuda_fixture.py`, pointed the pilot
  at `cuda_pilot_fixture_f64norms`, wrote D-036, generated and self-checked the new Mac
  fixture, and added tests. No Slurm jobs and no training.

## 2026-11-20: M5 bookkeeping + M6 handoff prompt (Cursor agent)

- Restored missing `docs/decisions.md` entries D-037–D-045 from prior session
  transcripts; recorded D-047 (M5 EXPERIMENT COMPLETED from artifacts) and D-048
  (array `squeue` EndTime multiline fix). D-046 (`proseco_gen_conf`) kept.
- Updated `docs/handoff.md`, `docs/verification.md` (M5 results table), and wrote
  `docs/m6_agent_prompt.md` for a fresh Milestone-6 agent.
- Patched `correction_screen.slurm`, `bank_collect.slurm`, `continuation.slurm` for
  D-048. No new TACC jobs; no invented remote results.

## 2026-11-20: Milestone 6 samplers and call-budget harness (Cursor agent, Grok 4.7)

What the AI did:
- Read the handoff, implementation plan, D-022 through D-048, the milestone-5
  verification section, source-audit sampler notes (Q-4, Q-5, Q-6), the bank schema,
  the build-prompt sampler section, and the pinned ProSeCo `llada/generate.py`.
- Wrote D-049 before the sampler behavior was locked in code. Implemented the NFE
  meter, the ProSeCo-style correction sampler, D3IM, the extra-unmask control
  (explicitly not ReMDM), budgeted repair, `dlmc evaluate-calls`, the eval config,
  and a user-submitted Slurm stub.
- Ran `.venv/bin/python -m pytest -q`: 300 passed. No TACC job. No experiment claim.

Reused or derived code:
- The correction sampler follows the structure of ProSeCo `llada/generate.py` at
  `43ce881` (predictor, temporary complete draft, fixed-point argmax, NFE split).
  It is a new implementation against `DiffusionAdapter.aligned_logits`, not a copy
  of that file. D-049 lists the differences. No ProSeCo training code was ported.

Human checks:
- None. The author has not reviewed the milestone-6 sampler or the D-049 contract.

## 2026-11-21: Backbone and sampler clarification (Cursor agent, Grok 4.7)

What the AI did:
- Recorded D-050. Open-dCoder stays the backbone. ProSeCo's released sampler is
  LLaDA-8B. SCOPE's audited paper has algorithms and no pinned code repo; D3IM is
  its sampler, not a third model. Milestone 6 is a local software gate, not a
  paper-faithfulness certificate. Milestone 7 starts with Q-4 and Q-5.
- Updated `docs/decisions.md`, `docs/implementation_plan.md`, `docs/handoff.md`,
  `docs/source_audit.md`, and `docs/verification.md`. No training and no TACC run.

Human checks:
- None. The user asked why LLaDA was not used and whether milestone 6 is ready
  for milestone 7. That question is not a review of the sampler code.
