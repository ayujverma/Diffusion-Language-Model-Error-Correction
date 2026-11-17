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
