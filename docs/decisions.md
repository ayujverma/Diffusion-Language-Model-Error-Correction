# Decisions log

Each entry says what was decided, why, and whether it is provisional. Change a
decision by adding a new entry that supersedes it; do not edit history silently.
Evidence references point to `docs/source_audit.md` (SA §n) and
`docs/verification.md` (V).

## Scope and conflicts

**D-000 (2026-11-16) Research question and arms are unchanged.** The proposal's
question and every required comparison in build prompt §1 (arms 1–9, two
comparison families, original and warmup reference checkpoints, three seeds) stay
in scope. Milestone 1 implemented none of the experiment arms. It built the
adapter, loss and weighting code they share.

**D-001 (2026-11-16) Schedule conflict recorded.** The proposal PDF plans five
weeks. The build prompt sets a three-week working target to results and a draft
paper. We plan to the three-week pacing in `docs/implementation_plan.md` without
removing any experiment. If the measured cost estimate (required before the grid)
shows the grid does not fit, the conflict goes back to the author rather than
being resolved by silently dropping arms.

## Backbone and alignment

**D-002 Backbone implementation: own bidirectional Qwen2, not the Hub remote code.**
The Hub `modeling_qwen2.py` fails to import and fails in `forward` (SA §2.1). The
working upstream class (`veomni`) has two silent-causal paths (eager attention; any
padded 2-D mask) and needs a large dependency tree. We re-implemented the forward
pass in `models/qwen2_bidir.py` (Apache-2.0 attribution in the file header), with
bidirectional attention as the only mode and an explicit key-padding mask. It is
exactly equal to the upstream SDPA logits on the reference inputs (V).

**D-003 Mask token is `<M>` = 151665.** The tokenizer, the Open-dLLM trainer and
sampler, and CDLM all use it. `generation_config.json`'s `mask_token_id: 151643`
is `<|endoftext|>` and is treated as stale. `resolve_token_spec` reads IDs from
the pinned tokenizer files and refuses a mask ID equal to pad/EOS.

**D-004 Attention masks come from lengths; padding is on the right.** Pad = EOS =
151643, so ID-based masks are wrong. Position IDs are `arange(L)` for every row.
Right padding leaves real positions' RoPE phases unchanged. Pad queries attend to
real keys, so no attention row is fully masked. Batch-vs-alone invariance is tested
on the tiny backbone (exact in float64) and on the real backbone (< 1e-3 in float32).

**D-005 Alignment convention.** `align_logits` sets `aligned[:, i] = raw[:, i−1]`.
Row 0 is a constant zero with no gradient path; upstream uses `raw[:, 0]`, which is
unused either way. Every loss and future sampler consumes aligned logits. Position
0 can never be a target, and prompts must be non-empty.

**D-006 (provisional for the mask-rate part) Masked-loss contract.** L_mask is the
mean CE over masked editable positions, pooled over the physical minibatch. It has
no 1/t time weighting and no Open-dLLM "path" term (SA §2.4), because the build
prompt forbids inheriting undocumented auxiliary losses and a plain token mean is
the simplest well-defined estimator. The mask-rate distribution is decided in the
warmup milestone. The default candidate is U(0,1) clamped to [1/500, 1−1/500] over
response tokens only, matching the backbone's pretraining. It must be identical
across all main arms.

**D-007 Cross-entropy implementation.** Plain PyTorch `log_softmax` + `gather` on
logits upcast to at least float32, computed only at selected positions. No fused
or chunked kernel. Any future replacement must pass the value and gradient tests in
`tests/test_losses.py` (CDLM liger bug, SA §2.4).

**D-008 Main environment has no `transformers` dependency.** The tokenizer is read
with `tokenizers` from the pinned `tokenizer.json`, and weights with `safetensors`.
This avoids remote-code execution and transformers-version drift between the Mac
and Colab. The separate `.venv-ref` (transformers 4.54.1 + Open-dLLM) exists only
to regenerate reference logits.

**D-009 Precision (provisional).** Tiny-backbone math checks run in float64. Real
backbone checks run in float32 on CPU and MPS (both pass the doctor gate). bf16
(MPS or A100) is untested and needs its own value and gradient comparison before
use. Loss and probability math always runs in at least float32.

## Loss and weighting contract

**D-010 Correction loss normalisation.** L_error = Σ_{i∈E} w_i ℓ_i / |E| and
L_keep = mean_{i∈K} ℓ_i, pooled over the physical minibatch. Empty sets return a
graph-connected zero, and emptiness is reported in the objective's parts dict. E/K
come from token comparison on editable positions; prompt and pad are excluded by
construction.

**D-011 (provisional: beta and bin edges) Rank-weighting details.**
- Groups are (decoding budget, stage bin). Stage bins use the mask fraction
  immediately before commitment, with edges (0.2, 0.4, 0.6, 0.8): [0,.2), [.2,.4),
  [.4,.6), [.6,.8), [.8,1]. Identical for every rule.
- Midranks use exact equality of the supplied scores (saved float32 log c / −log q,
  or detached current CE). There is no rounding before ranking.
- r = (rank − 0.5)/n; w_raw = 1 + β(2r − 1); w = w_raw / group mean. Default
  β = 0.5 is provisional, to be selected on validation; β ∈ [0, 1). Weights are
  computed in float64 on CPU without gradient, then cast to float32.
- `low_conf` reverses ranks, which gives exactly w_low = 2 − w_high.
  `shuffled_conf` permutes saved log c within each group using an explicit seed,
  then ranks upward. It reports how many scores moved.
- `uniform`, and any minibatch with zero errors, gives all-one weights.
- Diagnostics per call: group sizes, singleton groups, weight mean/var/min/max,
  ESS = (Σw)²/Σw², and whether any weight varies.

**D-012 Open: initial λ and μ.** Not yet chosen. They will be logged at the start
of the first correction-training milestone and selected later by a small validation
sweep whose budget is reported. Neither paper mandates a value.

## Project records

**D-013 (2026-11-16) Input documents are stored in the repository.** Byte-identical
copies of the build prompt and the proposal are in `docs/inputs/`, so that a new
session does not depend on files in `~/Downloads`. sha256:
`Cursor_Diffusion_LM_Build_Prompt.txt` `88b19a9f…c2b1`;
`Diffusion_LM_Error_Correction_Project_Proposal.pdf` `043a0648…066f`. These copies
are the authoritative spec; any later revision gets a new file and a new entry.

**D-014 (2026-11-16) Advisor follow-ups are pending, not done.** Four items were
raised for milestone 1 and are open. None of them changes an earlier decision yet.
1. Compare our gradients with the upstream reference (milestone 1 compared logits
   only).
2. Confirm that every checkpoint weight loads and that the required embedding
   tying holds, with a recorded check.
3. Clarify D-006, the masked-loss contract. D-006 is unchanged until then; the
   clarification will be a new entry that supersedes it.
4. Correct two statements in `docs/verification.md`: the hand example uses positive
   "log c" values, which are impossible for a log probability, and the sentence
   describing CPU/MPS differences as "about 1e-4 relative" understates some
   differences.
The details are in `docs/verification.md`, "Pending advisor follow-ups".

**D-015 (2026-11-16) AI session docs stay local; supersedes D-013 for the build
prompt.** The public research tree tracks the proposal PDF under `docs/inputs/`
and the scientific docs under `docs/`. The Cursor build prompt, `docs/handoff.md`,
and `docs/ai_usage.md` remain on disk for local sessions only and are listed in
`.gitignore`. D-013's claim that the build prompt is a tracked repository input
no longer holds.

## Milestone-1 follow-ups and milestone 2 (2026-11-16, second session)

**D-016 (2026-11-16) Advisor follow-ups 1, 2 and 4 of D-014 are closed; item 3 is
closed by D-017.** Evidence is in `docs/verification.md`, "Milestone-1 follow-ups".
1. Gradient equivalence. Our backbone and the upstream `veomni` class load the same
   pinned weights and run the verified upstream configuration (SDPA, no padding,
   `is_causal=False`). One external plain-PyTorch aligned loss, defined in
   `scripts/reference_gradients.py` and independent of both models and of our loss
   package, is backpropagated through each model. The comparison covers two inputs
   (masked; complete flawed draft) and all 290 distinct parameters, reported by group.
   Tolerances were fixed before the run, as the per-tensor relative L2 error
   ‖g_ours − g_ref‖ / ‖g_ref‖: 1e-4 in float32 and 1e-9 in float64.
   Upstream computes RoPE angles and RMSNorm statistics in float32 even for float64
   activations, while ours keeps float64. The gating float64 run therefore matches those
   two casts. The unmatched float64 run is reported as information only. float32 needs
   no matching, because both casts are then identity operations.
2. Loading and tying. A real-backbone test (CPU and MPS) confirms that all 291
   checkpoint tensors load bit-exactly. The 291 are 290 parameters plus the stored
   `lm_head.weight` alias, which equals the embedding. `lm_head.weight` is the embedding
   tensor itself, before and after the dtype/device move and after an AdamW step, and
   the optimizer holds 290 tensors.
4. The two `verification.md` statements are corrected in place, with a visible
   correction note.

**D-017 (2026-11-16) Clarifies D-006 (masked-loss contract); supersedes D-006's
wording, not its choice.** The objective stays the pooled masked-token CE:
L_mask = Σ_{i∈M} ℓ_i / |M|, where M is the set of masked editable response positions
across the whole physical minibatch.
- This is the same estimator as upstream Open-dLLM's plain masked-token term
  (`mdm_loss`): a sum over masked positions in the flattened micro-batch, divided by
  their count. We **retain that term** and **omit the auxiliary path term**
  (sg(exp(−ℓ)) · ℓ / t). No 1/t (or other time) weighting is applied.
- Under pooled normalization every masked token has coefficient 1/|M|. An example
  with m_b masked tokens therefore receives total weight m_b/|M|, so examples with
  more masked tokens (high mask rate, or a longer response) weigh more in total. This
  is a property of the chosen estimator, not an accident. A per-example mean would be
  a different objective.
- **Scheduled for resolution before warmup** (milestone 3, before the first warmup
  step; open now):
  (a) the precise mask-rate sampling policy. Upstream draws one t per sequence from
  U(0,1) clamped to [1/500, 1 − 1/500], then masks each token independently with
  probability t. Undecided: per-example vs. per-batch t, independent Bernoulli vs.
  exact-count masking, and response-only masking (our default) vs. upstream's
  whole-sequence masking;
  (b) zero-mask handling. With independent masking and responses of 10–60 tokens, an
  example (or, rarely, a whole minibatch) can draw no masked token. Options are to
  resample, force at least one mask, or keep it and count it. The current code returns
  a graph-connected zero for an empty minibatch, and that behaviour is unchanged.
- Whatever is chosen must be identical across all main arms. This entry introduces no
  objective change.

**D-018 (2026-11-16) Canonical response representation; resolves Q-2 (EOS).**
- A response is one slot per value, with no names or separators: a sign token
  `" +"` (488) or `" -"` (481) followed by `width` digit tokens (`"0"`..`"9"` = 15..24),
  for example `" +0045 +0135 +0116"`. `width = len(str(value_bound))` comes from the
  declared bound in the data config, never from answers. Zero is `" +0000"`, and
  `" -0000"` is non-canonical (malformed).
- The response length is public: (number of statements) × (1 + width). The prompt
  states the names in order and the slot format.
- **No EOS** is generated or supervised. The sequence ends after the last slot, and any
  padding is excluded by the length-based attention mask (D-004). An EOS or pad ID in a
  response position is a malformed `special_token`. This needs no adapter change:
  `build_layout` already uses explicit lengths.
- Why: on the pinned tokenizer, punctuation merges with neighbours (`"=-"`, `",+"`,
  `",-"`, `"(+"` are single tokens; `"=+"` is two), so `a=+0045,b=-0135` has no fixed
  token width. A sign with a leading space is a single token after a digit and after
  the prompt's final `":"`. No printable prompt-ending character merges with it. No
  tokens are added and the embeddings are not resized.
- The token IDs are derived from the tokenizer at run time and asserted to be single,
  non-special tokens. Our `tokenizers` wrapper, transformers' fast tokenizer, and the
  independent slow (pure-Python BPE) tokenizer agree on every pilot prompt, response
  and joint text.

**D-019 (2026-11-16, PROVISIONAL until validation calibration) Task grid, bounds,
held-out compositions and split sizes.** These are in `configs/data/pilot.yaml`;
details are in `docs/data_format.md`.
- Arithmetic: a chain of n operations. The first value combines two constants; each
  later value applies `+c`, `−c` (c has exactly d digits, 1..9 when d = 1) or `×k`
  (k ∈ [2, 9]) to the previous value. Lengths are n ∈ {2, 4, 6} and digits
  d ∈ {1, 2, 3}. Every intermediate satisfies |v| ≤ 9999, and negatives are allowed
  (an explicit sign slot).
- Programs: 2 inputs (x, y), and n ∈ {2, 4, 6} statements in the D-021 language,
  generated as a dependency chain. The statement shapes are add/sub/mul by a constant
  and add/sub with another variable. Same bound.
- Longer-problem test: n ∈ {8, 10, 12} for both families, with the same digits and bound.
- Held-out compositions are declared explicitly, not via seeds. For arithmetic the pair
  is `-` immediately followed by `*`; for programs it is `mul_const` immediately followed
  by `sub_var`. These pairs never occur in train/val/test_id/test_long, and every
  test_unseen problem contains one.
- Pilot sizes per (family, length, digits) setting: train 1000, val 200, test_id 200,
  test_long 50, test_unseen 100. In total: 18,000 / 3,600 / 3,600 / 900 / 1,800.
- Settings are chosen later from train/validation results only (milestone 3). Test
  performance is never used to choose difficulty. Changing the held-out compositions
  means regenerating every split.
- Known property: bound rejection and the held-out exclusion make `*` rarer than `+` in
  training arithmetic (9,198 vs 15,280 operator occurrences in the pilot train split).
  This is recorded, not corrected.

**D-020 (2026-11-16) Checker metrics and malformed-output policy.** Token-level,
value-level, final-answer and whole-trace correctness are reported separately.
Parsing is by token ID at the fixed slot positions; text is never decoded and
re-parsed.
- A malformed slot (bad sign or digit token, a special token, an ID outside the
  tokenizer, or `-0000`) is an incorrect value. Its tokens are still scored position
  by position.
- A prediction whose length differs from the public response length is never trimmed,
  padded or zipped. All its tokens and slots count as incorrect, and it is flagged
  `length_mismatch`.
- Every example stays in every denominator. Malformed rates are reported with their
  reasons.

**D-021 (2026-11-16) Restricted assignment language.** This is the grammar and
semantics in `data/programs.py` and `docs/data_format.md`: single-letter names;
non-negative integer literals; `atom` or `atom op atom` with op in {+, −, ×}; `×`
requires a literal in [mul_min, mul_max]; read-only inputs; define before use;
reassignment allowed; and bound checks after every statement. A hand-written lexer,
parser and interpreter are used, and Python `eval`/`exec` is never called. Model output
is never executed. Labels are re-checked by the interpreter, and tests compare the
interpreter with an independent `ast`-based evaluator. **Bounded conditions remain a
later extension**; they are not part of the milestone-2 language.

**D-022 (2026-11-16) Plan clarifications (no scope change).**
- Minimal complete-draft repair support stays scheduled for **milestone 5**: a repair
  call on an already-complete saved draft, which the held-out repair evaluation of the
  four-way screen needs. The full correction samplers (ProSeCo-style, D3IM) stay in
  milestone 6.
- Sampler ambiguities (reveal schedule and tie-breaking for the monotone collector;
  Q-4; Q-5; Q-6) are resolved in a decision entry before the code that depends on them.
- An early runtime estimate (measured seconds per step and per example) is produced in
  milestone 3. The methods section is written incrementally from verified code,
  starting in milestone 3.

**D-023 (2026-11-16) Track handoff and AI-use log; narrows D-015.** D-015 kept
`docs/handoff.md` and `docs/ai_usage.md` git-ignored as local session records.
There is no concrete reason for that separation: they contain no credentials, and
they are the continuation and provenance records the next session needs. Both are
tracked again. The Cursor build prompt
(`docs/inputs/Cursor_Diffusion_LM_Build_Prompt.txt`) stays local and git-ignored;
it is an external instruction document, not a research artifact, and D-013's claim
that it is a tracked repository input remains superseded.

## Milestone-2 closure (2026-11-17)

**D-024 (2026-11-17) Dataset verification rebuilds structure and identity from
source; closes milestone 2.** An advisor source review found that `verify_dataset`
applied the held-out predicate to the stored `signature` field and checked only that
`problem_id` was the hash of the stored `normalized_key`. Neither was rebuilt from
the record's source. Changes:
- New production module `data/structure.py`. `derive_problem` parses a record's
  `source` with the program-language parser and checks that it is an instance of the
  generated structure claimed by its family and public setting. It then rebuilds the
  `ArithmeticProblem` or `ProgramProblem`.
  - Arithmetic: targets `a, b, …` in order; statement 1 is `K0 op K1`; each later
    statement is `previous op K`; operand digit counts and the multiplier range are
    checked.
  - Programs: inputs exactly `x, y[, z]` with d-digit values; targets in order with no
    reassignment. Each statement reads its predecessor (statement 1: an input). The
    shape rules are `P ± K` (const), `P * K` (mul_const), and `P ± V` / `V ± P` with V
    another defined variable (var).
  - Anything else, including non-canonical spacing, is rejected.
- The signature and normalized key come from the same properties the generators use
  (`.signature`, `.normalized_key()`); there is no second definition. The composition
  definitions and the identity definition (`programs.normalized_key`,
  `problem_id = sha256(key)[:16]`) are unchanged.
- `verify_dataset` now does the following:
  - applies the held-out predicate to the **derived** signature;
  - requires that `heldout_composition` and split membership match it;
  - requires exact equality between the stored and derived signatures;
  - requires the stored `normalized_key` and `problem_id` to equal the ones rebuilt
    from source plus inputs;
  - checks cross-split uniqueness on the rebuilt identities;
  - checks values and value names against the rebuilt problem, in addition to the
    independent interpreter.
- Semantic-equivalence and commutative de-duplication are deliberately not added
  (D-019 limitation unchanged).
- Both existing datasets pass unchanged, so they are kept with their existing
  manifest hashes. Evidence: `docs/verification.md`, "Milestone-2 closure".
- Milestone 2 is closed.

**D-025 (2026-11-17) Track the approved build prompt; supersedes the build-prompt
part of D-023 (and D-015).** `docs/inputs/Cursor_Diffusion_LM_Build_Prompt.txt` is
the approved implementation and experiment protocol, and no tracked document
otherwise contains the complete protocol. It is now tracked, unchanged: sha256
`88b19a9fe8afd1221ae2294690340ec88a1bd9544eba8dde4f1ac51e7591c2b1`, the D-013 value.
Its `.gitignore` rule is removed. D-013's statement that the input documents are
stored in the repository holds again. Any revision gets a new file and a new
decision.

**D-026 (2026-11-17) Masking is response-only with the prompt fixed; supersedes the
whole-sequence option in D-017(a) and corrects D-006's wording.** The build prompt
(§7, "ordinary masked fine-tuning … with correct visible context and the prompt
fixed") fixes this: masking applies only to editable response positions, and prompt
tokens are always visible. Whole-sequence masking is therefore **not** an open
protocol choice.
- This differs from the upstream Open-dLLM collator, which masks whole packed
  sequences. D-006's phrase "matching the backbone's pretraining" was inaccurate for
  this reason.
- Still open for milestone 3, before the first warmup step:
  - the precise mask-rate sampling: per-example vs. per-batch rate, and independent
    Bernoulli vs. exact-count masking;
  - zero-mask handling.
- Upstream's rate is a **clamped** uniform: t = clamp(u, 1/500, 1 − 1/500) with
  u ~ U(0, 1). That puts probability 1/500 on each endpoint, so it is not a uniform
  draw on the interval [1/500, 1 − 1/500]. Any later description must say which of
  the two is meant.

**D-027 (2026-11-17) Constraints for the milestone-3 monotone sampler (advisor
note).**
- Pilot training responses have 10, 20 or 30 editable tokens (2, 4 or 6 slots × 5
  tokens). `test_long` responses have 40, 50 or 60.
- The ordinary monotone sampler must use only feasible integer reveal schedules
  (each call reveals at least one token, and the schedule ends with no mask left).
- It must report the actual number of model calls per example.
- It must not add no-op calls to reach a nominal budget. A budget larger than the
  response length is infeasible, and is reported as such.
- The reveal-schedule and tie-breaking decisions (D-022) must respect this.
