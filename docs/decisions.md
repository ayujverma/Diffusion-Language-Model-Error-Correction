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

## Milestone 3: warmup and calibration (2026-11-17)

**D-028 (2026-11-17) Masked-warmup mask policy; resolves Q-8 and the open parts of
D-017(a)/(b) and D-026. Recorded before any trainer code was written.** This policy
is the ordinary masking branch for the common warmup, and it stays common across all
main experimental arms later. With D-017 (pooled CE, no 1/t, no path term) and D-026
(response-only), it fully specifies L_mask.
- **Rate.** Draw u_b ~ U(0, 1) independently for each example b. Set
  t_b = clamp(u_b, 1/500, 1 − 1/500). This is a clamped uniform, with mass 1/500 at
  each endpoint, not a uniform draw on [1/500, 1 − 1/500] (D-026 wording).
- **Masking.** Mask each editable response token of b independently, with
  probability t_b (independent Bernoulli, not exact-count). Prompt and padding tokens
  never change.
- **Zero-mask examples are kept.** There is no resampling and no forced mask. Such an
  example contributes nothing to L_mask but still counts as consumed data. For a
  response of L tokens, P(no mask) = E[(1 − t)^L] ≈ 1/(L + 1) (about 9% at L = 10).
- **Estimator.** L_mask = Σ_{i∈M} ℓ_i / |M|, where M is the set of masked response
  tokens in the whole physical minibatch (D-017). There is no 1/t weighting and no
  auxiliary path term.
- **All-zero minibatch.** If |M| = 0 for the whole physical minibatch, the optimizer
  and the LR scheduler are **not** stepped, and a `skipped_zero_mask` event is logged.
  The trainer counts *data batches consumed* separately from *optimizer updates
  completed*. Run limits (`max_updates`) refer to completed updates.
- **Gradient accumulation = 1.** The trainer rejects any other value. A later change
  must first define the accumulation normalization (pooled over all micro-batches or
  per micro-batch) and the empty-micro-batch rule, and test them; it must not silently
  change pooled-token normalization.
- **RNG streams.** Separate `torch.Generator`s on the CPU, each seeded from
  sha256(run seed, stream name):
  - `data_order`: the per-epoch permutation of the training examples;
  - `train_mask`: the u_b and Bernoulli draws;
  - `eval`: re-seeded from the same seed at the start of every evaluation, so every
    evaluation sees the same masks and never advances a training stream.
  Masks are drawn on the CPU, so they are identical on CPU, MPS and CUDA. Draw order
  within a batch: for each example in batch order, one float64 u, then one float64
  draw per response position; position j is masked iff its draw < t_b.
- **Checkpoints** store the state of every stream (D-030).

**D-029 (2026-11-17) Ordinary monotone unmasking sampler; satisfies D-022 and D-027.
Recorded before the sampler code was written.** This is a calibration and collection
sampler, not the frozen final experimental budget.
- **Inputs.** Prompt token IDs and the public response length only. Response length
  is (number of named values in the prompt) × slot length, from the task layout. The
  sampler never receives targets, error labels, reference-derived lengths or
  correctness feedback.
- **Start.** The prompt is fixed and every response position is `<M>`.
- **Schedule.** For L editable tokens and T model calls, 1 ≤ T ≤ L is required;
  T > L is reported as infeasible and not run (no clamping, no no-op calls). After
  call s the total revealed count is K_s = ⌈s·L/T⌉ (K_0 = 0), so call s reveals
  K_s − K_{s−1} ≥ 1 new positions. After call T, K_T = L: no mask remains, and there
  is no extra filling call. Previously revealed tokens never change.
- **Candidates.** At every still-masked position, the candidate is the greedy argmax
  of the aligned float32 logits over the **allowed** set. Excluded: `<M>`, every
  special added token of the pinned tokenizer (this includes pad = EOS), and IDs ≥
  the tokenizer size (151,666; output rows up to 151,935 have no token). There are no
  per-slot numeric or grammar constraints: any ordinary vocabulary token may be
  chosen, and the checker counts non-numeric tokens as malformed.
- **Commit order.** Masked positions are ranked by the **raw** temperature-1
  probability of their selected candidate: softmax over the full, unrestricted output
  vocabulary in float32. The restricted policy probability (softmax over the allowed
  set) is stored separately and is not used for ranking. The top K_s − K_{s−1} are
  revealed.
- **Ties.** Equal maximal candidate logits pick the smallest token ID. Equal position
  scores (exact float32 equality) pick the smallest position index first.
- **Observation.** Each forward pass can be shown to an optional observer
  (call index, current token state, the float32 aligned logits, the commit events).
  The observer's return value is ignored, so milestone 4 can read log q from the
  commitment forward pass without rerunning generation. The full draft bank is not
  built here.
- **Accounting.** Every batched forward is counted, and per example the number of
  forwards in which it took part (= T). A `T = L` reference (one token per call) is
  supported by grouping examples by length.
- **Determinism.** Greedy selection uses no random draws.
- **Traces** record, per call, the revealed positions, tokens, raw and policy log
  probabilities, the mask fraction before the call, and the full response state.

**D-030 (2026-11-17) Precision, trainable parameters and optimizer for the warmup and
the main arms; resolves Q-3 and closes D-009 for training. Chosen before the common
warmup, from measurements on the M2 Max (MPS). No A100 measurement exists, so the
A100 parts of this entry are device-dependent and unmeasured.**
- **Full fine-tuning, float32 parameters, no autocast.** All 494,032,768 parameters
  are trainable, with the embedding tied to `lm_head`. This is fixed for the warmup and
  for every main weighting arm.
- Evidence (`docs/verification.md`, "Milestone 3"):
  1. bf16 autocast is supported on MPS and passes the pre-declared screening
     tolerances. Loss relative difference 7e-4; argmax agreement 98.8%; global
     gradient relative L2 0.024; cosine 0.9997.
  2. bf16 autocast is **slower** than fp32 on this device: 3.16 vs 2.92 s/update at
     16 × ~135 tokens.
  3. Pure bf16 parameters fail the argmax-agreement tolerance (97.7% < 98%) and are
     not faster.
  4. LoRA (r = 16, α = 32, all seven projection types, 8,798,208 trainable) is only
     about 13% faster than full fine-tuning (2.55 vs 2.92 s/update) and saves memory
     that is not binding on a 64 GB machine (Metal driver maximum 15.1 vs 21.8 GB).
     Full fine-tuning avoids an adapter-capacity confound and needs no extra
     hyperparameters.
- **Optimizer (ENGINEERING DEFAULTS, not paper-mandated).** AdamW (non-foreach),
  lr 3e-5, betas (0.9, 0.999), eps 1e-8, weight decay 0.01 on all trainable tensors.
  Linear warmup over 20 updates, then constant: a constant phase lets a bounded run be
  extended without redefining a decay horizon. Gradient-norm clipping at 1.0 (as in the
  upstream Open-dLLM configs). Gradient accumulation 1 (D-028).
- **Physical batch 32 for the common warmup.** Throughput per example is flat between
  16 and 32 (5.5 vs 5.4 examples/s), so 32 sees about twice the data within the
  1,000-update cap. The estimate is 1,000 × about 5.9 s ≈ 98 min, inside the two-hour
  cap. The Gate C overfit uses batch 16 on its 32 examples (a wiring check).
- **A100 (unmeasured).** The same fp32 full-fine-tuning policy is expected to fit:
  analytic weights + gradients + AdamW ≈ 7.9 GB plus activations. Any bf16 or TF32 use
  there needs its own comparison and a new entry, and must be identical across arms.
  Throughput on the A100 is an extrapolation until measured.

**D-031 (2026-11-17) Validation calibration protocol and setting-selection rule.
Recorded while the common warmup was running, before any post-warmup validation
result had been read.**
- **Model.** The final checkpoint of the common warmup (`warmup-pilot-v1`, whichever
  update it reaches within D-030's limits). It is identified by its model sha256 and its
  training data identity: pilot manifest `990ecd1d…1536`, train split. The pretrained
  model's validation behaviour is taken from the warmup's update-0 evaluation, not
  from a separate screen.
- **Data.** The pilot `val` split only, verified against the same manifest. `test_id`,
  `test_long` and `test_unseen` are not loaded (`dlmc calibrate` refuses them). Each
  setting's examples are its first n in file order, so the screen's examples are a
  subset of the stage-2 examples.
- **Sampler and budgets.** D-029, T ∈ {1, 2, 4, 8} plus T = L. These are calibration
  settings, not the frozen experimental budget.
- **Stage 1 (screen).** All 18 pilot settings (2 families × lengths {2, 4, 6} × digits
  {1, 2, 3}), 32 problems each, all five budgets.
- **Stage 2 (200 problems per selected setting), at most 5 settings per family.**
  Selection uses only stage-1 final-answer accuracy:
  1. the easiest setting (length 2, digits 1), always kept so that
     already-correct/preservation cases exist;
  2. the hardest setting (length 6, digits 3);
  3. the remaining slots go to settings where some budget in {1, 2, 4, 8} has
     final-answer accuracy in [0.40, 0.85]. They are ranked by the smallest distance
     |acc − 0.625| over those budgets, with ties broken by (length, digits) ascending.
     If fewer qualify, the middle setting (length 4, digits 2) fills one slot.
  The 40–85% band is a heuristic for where a correction task is likely to be
  informative. It is not a target, and weights are never altered to reach it.
- **Reported metrics.** Final-answer, value, token and trace accuracy; malformed rate
  and reasons; value errors per response; calls per example; and model seconds. These
  are broken down by family, setting and budget.
- **Dataset change rule.** The pilot dataset is kept unless the screen shows a family
  that is useless for correction at every budget, i.e. either ≥ 98% final-answer
  accuracy in every setting (no errors to correct) or < 5% at T = L in every setting
  (no near-misses). A change needs a new data config, new manifests and a new entry.
  Every checkpoint keeps recording the manifest it was trained on and the one it was
  evaluated on. The final primary budget is chosen later (milestone 4 or 5), not here.

**D-032 (2026-11-17) Outcome of the milestone-3 warmup and calibration; engineering
memory rule.** Recorded after the runs, from `docs/verification.md`, "Common warmup"
and "Validation calibration".
- **Dataset kept.** Under D-031's rule neither family is uniformly ≥ 98% or < 5% at
  T = L. The provisional pilot dataset (manifest `990ecd1d…1536`) therefore stays,
  with no new manifest. The D-019 "provisional" label stays until the primary budget
  and the milestone-4 diagnostics confirm the grid.
- **Provisional common-warmup checkpoint:** `runs/train/warmup-pilot-v1/checkpoints/
  ckpt-0001000`, model sha256
  `f48abe457d483b19b854e1fdf6be4bb294c702f949f9d76d7c44a2274b0d5a6a`, trained on the
  pilot train split. It reached the 1,000-update cap while validation loss and
  accuracy were still improving, and the gradient was clipped on every update. Before
  milestone 4 freezes a generator from it, the user should decide whether to (a) use
  it as is, or (b) continue it with `--resume` under a larger update cap. Option (b)
  changes only `limits`, which is resume-mutable, so the optimizer state, data order
  and masks continue exactly. Either choice must be recorded in a new entry together
  with the checkpoint hash.
- **Primary budget not chosen.** The calibration cells inside the 40–85% heuristic
  band are listed in `verification.md`. They inform the choice but do not make it.
- **Memory rule (engineering only, no effect on the mathematics).** On MPS the
  allocator cache is released after every training update, after every evaluation and
  after every generation batch (`release_cache_every`, default 1; resume-mutable).
  Without this, varying padded shapes made the cache grow until the machine swapped.
  The rule changes timing and memory, not losses: identical losses at matching
  updates were observed with and without it.

**D-033 (2026-11-17) Warmup continuation on Lonestar6: limits, stopping rule, CUDA
precision, checkpoint retention. Recorded before any CUDA run and before any
continuation result.** Runbook: `docs/warmup_continuation.md`.
- **User decision on D-032: option (b).** Continue `warmup-pilot-v1` from
  `ckpt-0001000` (model sha256 `f48abe45…5a6a`, optimizer `ae55b394…99af`) on TACC
  Lonestar6, `gpu-a100-small` (one A100 40 GB). The backup goes to Ranch. The
  continuation reloads the original checkpoint, never pilot-updated weights.
  Everything in the resume-compatibility hash is unchanged
  (`configs/train/warmup_continuation.yaml` vs `warmup_pilot.yaml`): data, D-028
  masking, D-030 optimizer, schedule, batch 32, fp32 full fine-tuning, clipping and
  seed. The constant learning rate stays 3e-5.
- **Limits.** At most 10,000 total updates (9,000 more). At most 7,200 s of
  *training-update time*, counted from update 1000 and cumulative across resumes. It
  covers drawing indices through the update and its bookkeeping, but not evaluation,
  checkpointing or start-up. The timer's origin is stored in every checkpoint and
  cannot be re-based. A Slurm limit is not the cap: each job's limit is larger, and
  the trainer stops itself before it (`--stop-at-unix`, SIGUSR1 backstop), saving a
  checkpoint. One attempt per submission: no automatic retries or requeueing.
  `max_wall_seconds` is null for the continuation.
- **Early stopping (an engineering rule, not an established optimum).** The 500-ID
  monitoring subset is fixed (`configs/eval/monitor_val_stage2_first50.json`, sha256
  `324d5ca8…3dc94`): the first 50 val problems of each stage-2 calibration setting.
  - *Schedule.* Scheduled evaluations run at update 1000 (the CUDA baseline) and every
    1,000 total updates.
  - *Score.* Final-answer accuracy at T = L, averaged over settings within each
    family, then over families.
  - *Patience.* Training stops after 3 consecutive scheduled evaluations without a
    strictly higher score; ties keep the earlier checkpoint.
  - *Persistence.* The best score and update, the counter, the history and the subset
    identity persist in checkpoints. Changing any of them on resume is refused.
  - *Eligibility.* Only scheduled evaluations are eligible. A stop between scheduled
    points gets one unscheduled evaluation for reporting.
- **Retention.** Checkpoints every 500 updates, with a verified copy in `$WORK`.
  Pruning (`keep_last 2`) never removes the current checkpoint, 1500, any multiple
  of 1000, or the best. Checkpoint 1000 stays read-only in `$WORK/dlmc/transfer`
  and on Ranch.
- **CUDA precision.** fp32 with TF32 disabled for matmul and cuDNN, and
  `float32_matmul_precision("highest")`. This keeps the arithmetic fp32 as in D-030.
  The SDPA backend is torch's default and is recorded in the environment report. The
  MPS cache release (D-032) is a no-op on CUDA and CPU. Exact MPS-to-CUDA numerical
  agreement is not expected. Data order, masks and learning rates come from CPU
  streams and must match exactly.
- **Environment.** CPython 3.14.7 and torch 2.15.0 (Linux build for CUDA 13.2,
  driver ≥ R580). Every package is pinned with hashes in
  `scripts/tacc/requirements-tacc-lock.txt`; shared packages are at the Mac versions.
  Any other torch build (e.g. 2.14.1+cu126 for an older driver) needs a new entry.
- **Pilot gates.** These are declared before any CUDA result (tolerances in
  `scripts/tacc/cuda_fixture.py` and the runbook §8):
  - the CPU-reference fixture check (forward, gradients, one AdamW step, greedy
    generation);
  - CUDA save/reload/resume against an uninterrupted CUDA run, at the Gate C
    tolerances;
  - reproduction of the Mac-predicted batches 1001–1010;
  - a baseline monitoring evaluation;
  - checkpoint 1000 unchanged.

  A failed gate is recorded; it is not retried or loosened without a new entry.
- **Final comparison.** The full D-031 stage-2 protocol on CUDA, for checkpoint 1000
  and the selected checkpoint (validation only).
- **Still open.** Which checkpoint becomes the milestone-4 generator: the user
  decides after the comparison, and a new entry records it. The bank-collection
  budget and the final primary budget stay unselected. Test splits stay untouched.

**D-034 (2026-11-17) APPROVED: torch 2.14.1+cu126 on Lonestar6 for an R570 driver.**
- **Approval.** The user approved this fallback on 2026-11-17 (23:38 UTC-6) for rebuilding
  and validating the TACC environment. The approval does not waive any gate: a fresh
  environment check and the CUDA pilot must pass before the continuation. Remote
  execution is the user's; none has happened at the time of this entry.
- **Observed.** Environment-check job 3591308 (account ASC26098, `gpu-a100-small`, node
  `v330-007.ls6.tacc.utexas.edu`, one NVIDIA A100-PCIE-40GB) at commit `944905e` with the
  D-033 environment (CPython 3.14.7, torch 2.15.0, CUDA 13.2). The driver is 570.195.03;
  the R580 guard stopped the job before any later check (as reported by the user). No
  CUDA pilot or continuation has run. Checkpoint 1000 has a hash-verified Ranch backup.
- **Proposal.** Replace only the TACC torch build: torch 2.14.1+cu126, the official
  PyTorch wheel `cp314-cp314-manylinux_2_28_x86_64` from `download.pytorch.org/whl/cu126`
  (sha256 `eb7564fd…834b`), taken by direct URL; every other package from PyPI. CUDA 12.6
  Update 3 needs Linux driver ≥ 560.35.05 (NVIDIA release notes), which R570 meets
  without minor-version compatibility. The CUDA 13 packages are gone; the CUDA 12
  libraries are those torch 2.14.1+cu126 pins, with nvJitLink held at the CUDA 12.6.3
  component version (12.6.85). Every package shared with the Mac keeps its Mac version.
  The Mac environment (torch 2.15.0) and its records are unchanged.
- **Unchanged.** Model, data, optimizer, scheduler, masking, batch size, fp32 precision,
  TF32 off, clipping, evaluation and selection, stopping limits, and every pilot and
  preflight tolerance (D-033).
- **Consequence.** The CUDA runs use a different torch version from the Mac. The checkpoint
  and the preflight compare state, data order, masks and learning rates, not versions; the
  preflight comparison also tests that torch 2.14.1 loads the 2.15.0 checkpoint and
  reproduces the Mac-predicted batches. If it does not, the fallback fails and nothing runs.
- **Gates.** A fresh environment check and the existing CUDA pilot with this environment.
  The venv's build record, `ENV_CHECK_PASSED` and the pilot's copy of it carry the lock's
  sha256, and every job refuses a mismatch, so the version change cannot count as a
  passed check. The failed job's logs, the old venv and its records are kept.

**D-035 (2026-11-18) GPU jobs on full `gpu-a100` nodes, one GPU per job.**
- **Observed (reported by the user).** At `906dcaa`, `gpu-a100-small` job 3592559 failed
  importing NumPy: the virtual node's CPU flags lack `lahf_lm`, which NumPy's X86_V2
  baseline requires. The same installed environment (the D-034 build, unchanged) passed a
  diagnostic probe on full `gpu-a100` node c301-001: NumPy 2.5.5, torch 2.14.1+cu126,
  CUDA 12.6, driver 570.195.03; three physical A100-PCIE-40GB GPUs;
  `CUDA_VISIBLE_DEVICES=0` exposed exactly one; NumPy → PyTorch → CUDA → back arithmetic
  passed. Log: `/work/10692/ayuj/ls6/dlmc/retained/full-a100-probe.CkLNzB.log`.
- **Decision.** Every GPU job (`env_check`, `pilot`, `continuation`, `final_eval`) runs on
  the `gpu-a100` partition and refuses any other. Each job selects one GPU in the job shell
  before any process initializes CUDA: a scheduler-provided `CUDA_VISIBLE_DEVICES` is
  respected (its first device), otherwise device 0, with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
  The node's physical inventory (from `nvidia-smi`) is recorded separately from the
  selected device, and PyTorch must see exactly one A100. No multi-GPU or distributed
  training; the other two GPUs stay idle.
- **Cost.** `gpu-a100` charges 3 SU per node-hour (Lonestar6 user guide), twice the
  1.5 SU of `gpu-a100-small`, with the same 0.25 h minimum per job and 48 h limit. The
  pilot report's default rate is now 3.0.
- **Unchanged.** The dependency lock, the installed environment and its build record;
  every scientific setting, tolerance, checkpoint, validation and selection rule, and
  stopping limit (D-033, D-034).
- **Gates.** The probe is a diagnostic, not a gate: it does not replace
  `ENV_CHECK_PASSED` or `PILOT_PASSED`. The full environment check (including the
  checkpoint preflight), the pilot and the continuation are pending.

**D-036 (2026-11-18) CUDA pilot fixture: per-tensor norms measured with chunked CPU float64.**
- **Evidence.** Pilot 3592809 failed only `tensor_update_norm_rel` on
  `model.embed_tokens.weight` (1.0615e-2 against 1e-2). Diagnostic job 3593175 reported
  `norm_accumulation_explains_failure: true`: CUDA and Mac embedding updates agreed
  elementwise (~1.8e-4 relative), while the Mac native fp32 L2 of that update was ~1.06%
  below a chunked float64 L2 of the same saved fp32 values. Tolerance 1e-2 is unchanged.
- **Change.** In `scripts/tacc/cuda_fixture.py`, `make` and `check` both record per-tensor
  gradient and update norms with a shared chunked CPU float64 L2
  (`norm_measurement: chunked_cpu_float64_l2`). Training, clipping (`clip_grad_norm_`) and
  AdamW stay fp32. The historical `runs/reference/cuda_pilot_fixture.*` (native fp32 norms)
  is preserved; the pilot uses `runs/reference/cuda_pilot_fixture_f64norms.*`.
- **Gates.** This replaces the measurement for those two per-tensor checks only. A fresh
  CUDA pilot against the new fixture is required for `PILOT_PASSED`. No pilot is marked
  passed by this entry.

## Milestone 4: frozen generator and draft bank (2026-11-19)

**D-037 (2026-11-19) Outcome of the CUDA pilot, the warmup continuation and the final
validation comparison.** Recorded from the downloaded job records
(`runs/tacc_results/job-3595133/`, not in Git) and the user's reports.
- **Pilot.** Job 3593382 passed every D-033/D-036 gate (reported by the user; recorded in
  the previous session).
- **Continuation.** Job 3593457 resumed checkpoint 1000 and stopped at update 8712 on the
  7,200 s training-time cap (reported by the user; the continuation logs are on TACC and
  were not downloaded). In job 3595133, `continuation_status.py --best` applied the D-033
  rule and selected `/work/10692/ayuj/ls6/dlmc/retained/warmup-pilot-v1-cont/checkpoints/
  ckpt-0008000` (verified in `final-eval-3595133.out`).
- **Final comparison.** Job 3595133 (exit 0:0, 00:21:13 as reported; commit
  `9cf1a6eb8c67d7368eb844988c2d914bf2bddff0`, clean; torch 2.14.1+cu126; one A100 on node
  c317-015) ran the full D-031 stage-2 protocol on the validation split (10 settings ×
  200 problems, budgets 1, 2, 4, 8, L, batch 50). At T = L, checkpoint 1000 scored 857/2000
  final answers; checkpoint 8000 scored 1812/2000 final answers, 1769/2000 whole traces and
  0 malformed. Equal-weight final accuracy, checkpoint 1000 → 8000: T=1 0.206 → 0.520,
  T=2 0.246 → 0.583, T=4 0.337 → 0.732, T=8 0.398 → 0.859, T=L 0.429 → 0.906 (recomputed
  from `records.jsonl`). These are validation results, not test results. Two CUDA
  caching-allocator retry warnings appear in the job's stderr; the job completed.

**D-038 (2026-11-19) APPROVED: checkpoint 8000 is the frozen milestone-4 generator.**
- **Approval.** The user approved checkpoint 8000 as the generator and asked that no
  earlier checkpoint be substituted to obtain more errors.
- **Identity (verified from the job-3595133 records).** Path
  `/work/10692/ayuj/ls6/dlmc/retained/warmup-pilot-v1-cont/checkpoints/ckpt-0008000`; run
  `warmup-pilot-v1-cont`, update 8000, full fine-tuning, fp32; model file sha256
  `a5ceb4ffbf70ecc49e9fb48d7a0cdd01ccafa4ce81a6b2f91c73e0bd1b755341`. Trained on the pilot
  train split: manifest `990ecd1d…1536`, split sha256 `215fdb4a…e18a`, selected problem IDs
  sha256 `b9a59da9…9c64`, 18,000 problems. Base `fredzzp/open-dcoder-0.5B` at revision
  `d0d86d5b99960c05258bb1f8265dd91564dbac67`; `tokenizer.json` sha256
  `4f3ead5ffbb2abfb9e51757b104af5174f59c994ff3b2ca715d344dc66ee1f4e`. Its validation
  records: `retained/final_eval/selected-ckpt-0008000-cuda/records.jsonl`, sha256
  `9f332bbcaedf1075c4dbf19be304806453e5a38d98664193ca864330f3c6d8e7` (local copy).
- **Pending remote verification.** The `COMPLETE.json` hash, the optimizer, RNG and
  `state.json` hashes, the configuration hash, the source revision recorded at save time
  and the selection/timer state are on TACC only. `scripts/tacc/generator_identity.py`
  re-hashes every file against `COMPLETE.json`, checks the pinned model hash, update and
  evaluation-record hash, and writes `retained/generator/ckpt-0008000.identity.json`. The
  collection job refuses to start unless the files still match that record.
- **Preservation.** Checkpoint 1000 (transfer copy and Ranch backup) and every
  continuation output stay as they are. Checkpoint 8000 is read-only input. Its Ranch
  backup (`backup_to_ranch.sh ckpt8000`) has **not** been made; the checkpoint-1000
  backup does not cover it.

**D-039 (2026-11-19) Milestone-4 bank collection protocol. The protocol is fixed; the
scope is PENDING USER APPROVAL.** Code: `src/diffusion_correction/drafts/`; runbook:
`docs/milestone4_bank.md`.
- **No approved scope existed.** Neither the build prompt nor D-000–D-038 fixes which
  problems, settings or budgets the bank holds. Two configs are prepared; the user picks
  one (or another) and a later entry records it. Nothing has been collected.
  - *Recommended (`configs/bank/m4_ckpt8000_v1.yaml`):* every train problem (18,000; all 18
    settings) and every validation problem (3,600), budgets 1, 2, 4, 8, L: 108,000 drafts,
    756,000 example-calls, 15,120 batched forwards, 2.16 M token records.
  - *Alternative (`configs/bank/m4_ckpt8000_stage2_v1.yaml`):* the 10 stage-2 settings
    only: 60,000 drafts, 384,000 example-calls, 7,680 batched forwards, 1.02 M tokens.
- **Fixed regardless of scope.**
  - *Splits.* Train drafts form the training bank; validation drafts are stored in the
    same bank directory under their own `split` and shards and are never mixed into
    training. Test splits are never loaded (`load_bank_config` and `load_verified_examples`
    refuse them). The dataset is not regenerated; within a configured setting every
    problem is collected and nothing is filtered by outcome.
  - *Generator and sampler.* Checkpoint 8000 (D-038), the unchanged D-029 sampler, one
    greedy draft per problem and budget. No random numbers are consumed (CPU and CUDA RNG
    states are checked unchanged per batch), so there is no seed. Budgets T ∈ {1, 2, 4, 8, L};
    T > L would be infeasible and is listed, not run (none is, since L ≥ 10).
  - *Batching and precision.* Batch 50 in file order within a setting (identical to job
    3595133's batches on validation); fp32, TF32 off, matmul precision "highest"; one A100
    on a full `gpu-a100` node (D-035), torch 2.14.1+cu126 (D-034), the existing environment.
  - *Order and shards.* Validation first, then train; within a split by budget, then
    setting, then part (200 problems per validation shard, 250 per train shard). The plan
    (every shard's ordered problem IDs) is hashed; the bank identity pins the config, plan,
    generator model and `COMPLETE.json` hashes, tokenizer, data manifest, code commit (clean
    tree required on CUDA), torch, CUDA runtime, device type, precision, lock and batch size.
- **Primary correction-comparison budget: still unselected.** Collecting several budgets
  does not choose it.

**D-040 (2026-11-19) Generation-time signals, record schema and the collection checks.**
- **Signals, from the commitment forward pass.** The sampler passes a read-only observer
  the cloned pre-commit state and the detached aligned logits after it has fixed the
  commits of that call. The observer recomputes the sampler's exact row tensor
  (`logits[editable & masked].float()`) and `log_softmax`, and stores per committed token:
  `log_c_gen` (committed token) and `log_q_gen` (reference target), both float32 over the
  **full output vocabulary** at temperature 1; `raw_max_log_prob`; the commit call,
  per-example commit order and the global mask fraction before the call (no blocks exist,
  so there is no separate active-block fraction). The sampler's own ranking score is
  stored as `rank_score` and must equal `log_c_gen` bitwise; the restricted-policy log
  probability (renormalized over the allowed set) is stored separately as
  `policy_log_prob`. Floats are float32 values written exactly as JSON doubles.
- **Target isolation.** References reach only the observer and the checker; the
  observer's return value is ignored. Records split target-independent fields
  (`generation`) from target-dependent ones (`analysis`: reference, `log_q_gen`, checker
  labels).
- **Startup gates in the full job (no separate trial job).** On four 50-problem batches
  (val arithmetic:4:2 at L, val programs:6:3 at 4, val arithmetic:2:1 at 1, train
  programs:4:2 at 8): Gate D (true vs permuted references vs no observer: identical
  `generation`, calls and sampler scores; target-dependent fields must change, and
  `log_q` must not change where the target did not); exact same-forward recompute of saved
  pre-commit states; an independent single-example raw forward with a manual shift within
  2e-3 in log probability (D-004's < 1e-3 batch-vs-alone logit bound, doubled for the
  log-normalizer); exact reproduction of job 3595133's drafts for the validation batches;
  record invariants; unchanged RNG. A failure stops the job before any shard is written;
  tolerances are not loosened and nothing is retried without a new entry.
- **Storage and validation.** JSONL shards written atomically with sidecar hashes;
  resume only with an identical identity, never rewriting a verified shard, moving partial
  files aside and stopping on corruption. Finalization re-checks every record against the
  dataset (schema, schedule, signal identities `log_c == log_q` on correct tokens and
  `c + q ≤ 1` on wrong ones, checker labels), detects missing, duplicate and unexpected
  records, checks split separation and counts, and writes `MANIFEST.json` and
  `BANK_COMPLETE`. Whole-bank agreement with the 3595133 records is reported (not gated).
- **Derived diagnostics (separately runnable; never alter the bank).** Step-0 correction
  signals with the corrector at initialization (= checkpoint 8000) given the full draft,
  signal diagnostics, and a verified examples report.

**D-041 (2026-11-19) Milestone-4 bank scope: the recommended full scope is approved.**
Supersedes the "pending" scope status in D-039; the protocol in D-039 and D-040 is unchanged.
- The user chose `configs/bank/m4_ckpt8000_v1.yaml` (bank `m4-ckpt8000-v1`): all 18,000 train
  and 3,600 validation problems (all 18 settings), budgets 1, 2, 4, 8, L, one greedy draft per
  problem and budget: 108,000 drafts, 756,000 example-calls, 15,120 batched forwards.
- The stage-2 config is not approved and is not to be submitted.
- Only the header comments of the two config files changed; the bank identity hashes the
  parsed config, so `config_sha256` and `plan_sha256` are unaffected.
- The primary correction-comparison budget remains unselected. Collection has not run.

**D-042 (2026-11-19) Clarification: the D-029 sampler ranks by float32 raw LOG probability.**
Does not change generation, ranking, candidate choice or any prior result (including job 3595133).
- **Ambiguity.** D-029

 \

 \

 \

 \

 \

 \

'
 \

 \

 \

 \

 \

 \

 \

 \

 \

 \

 \

 \

 \

**D-043 (2026-11-20) Milestone-4 draft bank completed on Lonestar6 (job 3597054).**
Supersedes the "collection has not run" lines of D-039/D-041 for this bank name.
- Bank `m4-ckpt8000-v1` from checkpoint 8000 (D-038), full scope (D-041), protocol D-039/D-040/D-042.
- Job `3597054` (`bank_collect.slurm`): collection_complete at 2026-11-20T01:25:24Z;
  `BANK_COMPLETE` with `manifest_sha256` `c5901617bbc6467483d1a91d8679c881dc055d62bbe993aa03fe181092e71ac4`;
  `identity_sha256` `e3e810bef987753ebdb8c9cf002de0f113eb9c4e5c61843094893ca5b22203ea`;
  `plan_sha256` `ed18b8a9a91148d7f390c1476c2c033a76ffd50709c518c6c63c424619bae3f4`.
- Validation passed (`n_problems` 0): 108,000 drafts, 15,120 batched forwards, 756,000
  example-model calls, 2,160,000 tokens; `rank_score_mismatches` 0; `rng_changed_shards` 0;
  `model_seconds_with_signals` 5341; final_correct 73,048; trace_correct 70,047;
  wrong_tokens 126,265; malformed_drafts 118.
- Reproduction vs job-3595133 generator validation records: 10,000 compared, 10,000 identical.
- Step-0 signals, diagnostics, and the verified examples report completed in the same job.
- Local Mac checks on downloaded metadata and a shard subset (val b4, val bL, train b8
  programs:4:2; 8,200 records): sidecar hashes matched; record invariants passed; report
  line_sha256 matched overlapping examples. Full local `--verify` of every shard optional.
- The primary correction-comparison budget remains unselected. Milestone 5 not started.
- Ranch backup of the finished bank: confirm with the user if `BACKUP VERIFIED` was printed;
  do not claim it here without that evidence.

## Milestone 5: shared-bank correction screen (2026-11-20)

**D-044 (2026-11-20) Milestone-5 screen protocol: primary budget, λ/μ/β, scale.
Recorded from bank diagnostics before any correction arm was trained.**
Supersedes the "primary budget unselected" lines of D-031/D-039/D-041/D-043 for the
first shared-bank screen. Does not change the bank, the generator, or D-011's bin edges.
- **Primary comparison budget: T = 4.** Chosen from `m4-ckpt8000-v1` diagnostics (job
  3597054): validation final-answer accuracy ≈ 0.653 at T=4 (inside the 40–85%
  heuristic band), with 3,555 wrong tokens / 1,248 wrong finals on val and 13,453 wrong
  tokens on train — enough errors for ranking without T=1's single-stage collapse or
  T=8/L's near-ceiling scarcity. Held-out repair evaluation uses **all validation drafts
  at budget 4** (3,600), identical across arms.
- **Training bank filter:** all **train** drafts at **all budgets** {1,2,4,8,L}
  (90,000 drafts). Ranking groups remain (budget code, stage bin) per D-011. Arms share
  identical example IDs and minibatch sampler seed.
- **Objective coefficients (logged initials):** λ = 1.0, μ = 1.0 (closes the "not yet
  chosen" status of D-012 for this screen; a later validation sweep may supersede).
- **Rank weights:** β = 0.5 and stage edges (0.2, 0.4, 0.6, 0.8) as in D-011 (no longer
  provisional for this screen). β = 0 must match uniform (tested).
- **Arms:** uniform, high_conf, gen_difficulty (−log q), current_loss, plus low_conf and
  shuffled_conf. Configs `configs/correction/m5_*.yaml` share every non-weight setting;
  only `run_name` and `weighting.rule` differ. All start from ckpt-0008000 (D-038).
- **Screen scale:** 1,000 updates, one seed (`20261120`), batch 32, same AdamW/schedule/
  fp32 full fine-tuning as D-030. Three seeds deferred to the main grid (milestone 8).
  No separate pilot/trial job. Slurm array of six arms on `gpu-a100` (D-035); resume with
  `DLMC_CORR_RESUME=1` only; no identity-changing auto-retries.
- **Repair:** one forward on the complete saved draft; greedy re-prediction of every
  editable position (allowed set). Report repair_rate = n_repaired / n_wrong_before and
  damage_rate = n_damaged / n_correct_before (rates None if denom = 0). Full correction
  samplers remain milestone 6 (D-022).
- **Bank pins:** identity `e3e810be…03ea`, manifest `c5901617…1ac4`. Mismatched identity
  or incomplete bank is refused.
- **Status at entry time:** software IMPLEMENTED / TESTED LOCALLY only. No TACC correction
  job has run; do not mark EXPERIMENT COMPLETED until artifacts are verified.
  (Superseded for completion status by D-047; batch size superseded by D-045.)

**D-045 (2026-11-20) Correction-screen physical batch 16 on A100 40GB (engineering).**
Does not change D-044 science (arms, λ/μ/β, budget, 1000 updates, seed).
- **Observed.** Array job 3600168 (arm uniform) hit `torch.OutOfMemoryError` during the
  second (draft) forward of `train_step` at batch 32: ~33.6 GiB allocated of 39.5 GiB,
  failed +2.52 GiB. Cause: two fp32 forwards with grad per update (L_mask + correction)
  plus full AdamW; warmup/collection used one forward or `no_grad`.
- **Change.** Train and eval `batch_size` for the M5 screen configs and
  `correction_screen.slurm` default to **16** (override `DLMC_CORR_BATCH`). Same for every
  arm. If 16 still OOMs, drop to 8 with the same override; do not change only one arm.

## Milestone 7 proposal: ProSeCo + generation-confidence weighting (2026-11-20)

**D-046 (2026-11-20) Proposed Family-B arm `proseco_gen_conf` (train-time); eval
call budgets; does not change Family A.**

- **Motivation.** The user wants to improve ProSeCo by using generation-confidence
  probabilities at **train time**, not only to study fixed-bank weighting (Family A,
  milestone 5). Family A remains the shared-bank control. This entry defines a
  **separate** ProSeCo-adapted variant for milestone 7.
- **Where.** Implement only in **milestone 7**, after `proseco_basic` and
  `proseco_error_prioritized` adapters work and their stop-gradient fixtures pass.
  Milestone 6 builds the **samplers and NFE/call accounting** used to evaluate every
  family under matched budgets; it does not train this arm.
- **Inference call budget (all families).** ProSeCo's published sampler uses a fixed
  correction schedule (corrector every ω unmask steps, up to S fixed-point argmax
  iterations; each iteration is one NFE). For the project's primary metric, **every**
  method—including Family A checkpoints and all ProSeCo/SCOPE variants—is compared at
  a **validation-selected fixed model-call (NFE) budget** (handoff). Family A training
  itself stays shared-bank (`L_mask + λ L_error + μ L_keep`); the fixed correction-pass
  budget is an **evaluation/sampling** contract from milestone 6 onward, not a change
  to the M5 trainer. Milestone 5's one-forward greedy repair remains a screen metric
  only.
- **Draft construction (unchanged from ProSeCo).** On each step, sample `z_t` from the
  true response; first forward `x_θ(z_t)`; discrete draft
  `ŷ = sg(argmax x_θ(z_t))` (Q-4 resolves visible vs all positions for Open-dCoder).
  Generation confidence for weighting uses the **same** draft-construction forward:
  \[
  \log \hat{c}_i = \mathrm{sg}\big(\log \mathrm{softmax}(x_θ(z_t))_{i,\,\hat{y}_i}\big)
  \]
  i.e. stop-grad log-probability of the **committed draft token** (not of the correct
  target, not max-over-vocab unless they coincide). Never recompute \(\hat{c}\) from
  the corrector forward or from the frozen bank.
- **Proposed corrector loss.** Let \(E\) / \(K\) be editable positions where
  \(\hat{y}_i \neq x^\star_i\) / \(\hat{y}_i = x^\star_i\). Let \(\ell_i = -\log p_θ(x^\star_i\mid\hat{y})\)
  from the **second** (gradient) forward on the draft. Let \(w_i\) be D-011 rank
  weights on \(\{\log\hat{c}_j:j\in E\}\) within the same (budget, stage-bin) groups
  as Family A, with provisional \(\beta=0.5\), **no gradient through \(w\)**. Time
  weight \(a(t)\) matches the ProSeCo-adapted MDM/corrector schedule chosen in the
  adapter (published: \(\dot\alpha_t/(1-\alpha_t)\); code often \(1/t\)). Then
  \[
  \mathcal{L}_{\mathrm{corr}}^{\mathrm{gen\_conf}}
  = a(t)\left(
      \frac{1}{|E|}\sum_{i\in E} w_i\,\ell_i
      + \frac{1}{|K|}\sum_{i\in K} \ell_i
    \right)
  \]
  with the usual graph-connected zero if \(E\) or \(K\) is empty. The MDM term on
  `z_t` stays as in the ProSeCo basic adapter. **Default proposed rule ranks
  \(\log\hat{c}\) upward on errors** (high generation confidence on a wrong commit
  gets larger \(w\)); low/shuffled ablations are optional diagnostics, not required
  grid arms.
- **Relation to ProSeCo Eq. 11 (error-prioritized).** Published error-prioritized uses
  \(-\log p_θ(x^\star)\) on wrong positions and \(-\,p_θ(x^\star)\) (probability, not
  CE) on already-correct positions. `proseco_gen_conf` does **not** replace Eq. 11; it
  keeps CE on both \(E\) and \(K\) and only reweights \(E\) by \(\hat{c}\). The three
  names stay distinct: `proseco_basic`, `proseco_error_prioritized`, `proseco_gen_conf`.
- **Gradient / learning check (honest).**
  - Because \(w=\mathrm{sg}(w(\log\hat{c}))\) and \(\hat{y}=\mathrm{sg}(\mathrm{argmax}\,\cdot)\),
    the first forward's discrete choices and the ranks do not receive gradient.
  - On an error token, \(\partial\mathcal{L}/\partial\theta\) scales the usual CE
    gradient by \(w_i\): same direction (toward \(x^\star_i\)), larger magnitude when
    \(w_i>1\). This **rebalances which mistakes dominate the batch**, it does not
    invent a new local optimum condition per token.
  - On correct tokens, CE gradients are unweighted (mean over \(K\)); this differs
    from Eq. 11's \(-\,p\) term, whose gradient on an already-correct position is not
    the same as CE.
  - **Does this necessarily help?** No. Family A (fixed bank, analogous rank weights
    on saved \(\log c\)) did **not** show high-\(c\) weighting beating uniform on the
    M5 screen. Online ProSeCo may still differ because \(\hat{c}\) tracks the current
    model and drafts are on-policy. The arm is justified as a **direct test** of the
    “improve ProSeCo with generation confidence” idea, not as a guaranteed gain.
  - **Failure modes to watch:** \(\hat{c}\) saturated near 1 on many errors (weights
    collapse); stop-grad/argmax mismatch with Q-4; accidentally weighting by
    \(p(x^\star)\) or corrector-forward probs (forbidden by this entry).
- **Eval.** Same milestone-6 call-budget protocol as other Family B arms; report
  final-answer accuracy and NFE. Do not treat M5 greedy repair as the ProSeCo
  comparison metric.
- **Unchanged.** Family A six-arm shared-bank training and its M5 screen conclusions;
  SCOPE + D3IM still a separate prior-method arm; no identity-changing retry of M5.

**D-047 (2026-11-20) Milestone-5 shared-bank screen EXPERIMENT COMPLETED (artifacts).**
Does not change D-044/D-045 science settings. Recorded from downloaded
`runs/tacc_results/correction/` (not in Git) and TACC job records reported by the user.
- **Training.** Job `3600435` array tasks 0–4 COMPLETED 0:0 (~24–28 min) for arms
  uniform, high_conf, gen_difficulty, current_loss, low_conf. Task 5 (`shuffled_conf`)
  FAILED at startup (multiline `squeue` EndTime; see D-048). Resubmit job `3600756`
  COMPLETED 0:0 for shuffled_conf. All six arms reached update 1000 / `max_updates`
  from init `ckpt-0008000` on bank `m4-ckpt8000-v1` (D-043), seed 20261120, batch 16
  (D-045), λ=μ=1, β=0.5, primary repair budget T=4 (D-044).
- **Dedicated repair.** Job `3600803` array 0–5 COMPLETED 0:0 (~1.5–3.5 min/arm).
  Each wrote `retained/correction/repair/m5-<arm>-v1/{summary.json,records.jsonl}`
  on `ckpt-0001000`, val@T=4, 3600 drafts. Aggregates **match** each arm's in-screen
  `last_eval.repair` at update 1000 bit-for-bit on repair_rate, damage_rate, and
  n_wrong_after.
- **Shared eval identity.** All arms: n_wrong_before=3555, n_correct_before=68445,
  draft_ids_sha256 `b8420264…c61b`. @0 repair identical across arms (repair_rate
  ≈0.1381, damage_rate ≈0.0515, wrong_after=6589).
- **Final ranking by n_wrong_after (lower better):** low_conf 3175 (net −380) ≈
  uniform 3180 (−375) > high_conf 3314 > shuffled_conf 3367 > gen_difficulty 3483 >
  current_loss 3571 (net +16). High gen-time confidence weighting did **not** beat
  uniform on this screen; correction training itself improved sharply vs @0.
- **Label.** EXPERIMENT COMPLETED for the M5 screen and dedicated repair on these
  jobs/artifacts only. Not a test-split result; not a ProSeCo comparison; not a
  multi-seed main-grid claim.

**D-048 (2026-11-20) Slurm array EndTime: take one line before `date -d`.**
- **Observed.** Job `3600435_5` failed: `squeue -h -j "$SLURM_JOB_ID" -o %e` returned
  six EndTime lines (one per array task) when `SLURM_JOB_ID` names the array parent;
  `date -d` rejected the multiline string.
- **Change.** In `correction_screen.slurm`, `bank_collect.slurm`, and
  `continuation.slurm`, set
  `END=$(squeue -h -j "$SLURM_JOB_ID" -o %e | sort -u | head -n 1)` and refuse an
  empty END. Prefer identical EndTimes; if they differ, the earliest unique line after
  sort is used (conservative stop margin). Does not change science settings.

## Milestone 6: samplers and call-budget accounting (2026-11-20)

**D-049 (2026-11-20) Inference samplers, NFE, and call budgets. This is the
milestone-6 contract. Does not train Family B and does not retrain Family A.**

- **Scope.** This entry freezes the milestone-6 inference contract. It does not
  implement `proseco_basic`, `proseco_error_prioritized`, `proseco_gen_conf` (D-046),
  or SCOPE training. Those stay milestone 7. Family A's six checkpoints stay as they
  are and are only consumers of this harness. Q-4 (corrector input at train time),
  Q-5 (SCOPE training confidence), and Q-6 (ReMDM's schedule and σ_t) stay open.
- **One NFE.** One NFE is one successful call to `DiffusionAdapter.aligned_logits`.
  `sampling/nfe.py::ForwardMeter.forward` checks the cap and then calls the adapter,
  so a sampler cannot spend a forward without recording it. A call that would exceed
  `max_calls` does not run the model and returns None. Kinds are `predictor`,
  `corrector`, `d3im`, and `revision`. A batched forward charges every example in the
  batch, as in D-029. Prompt and padding stay immutable. The allowed-token mask is
  the D-029 mask. Targets, error masks, and the checker never enter a sampler.
- **Budgets.** The harness interface is a per-example cap `max_calls`. Each cap is
  its own horizon. A prefix of a longer run is not the result at a shorter cap,
  because reveal quotas and D3IM keep-counts depend on the planned horizon. No
  primary budget is selected here. `dlmc evaluate-calls` and `dlmc evaluate-repair`
  accept `train` (wiring) and `val` only and refuse `test_id`, `test_long`, and
  `test_unseen`.
- **Monotone (D-029, unchanged schedule).** `monotone_unmask` still runs exactly T
  forwards and still finishes with no mask. An optional meter records those forwards
  as `predictor`. If the meter refuses one, that is an error, not an early stop.
  Under a cap B the harness sets T = min(B, L) and reports B − T unused. It does not
  add no-op calls when B > L (D-027).
- **ProSeCo-style correction sampler** (`sampling/proseco.py`). An adaptation of
  `llada/generate.py` at ProSeCo `43ce881`, not a numerical reproduction and not
  Family A. Provisional defaults, from that file, until a validation sweep: ω = 2
  (`apply_corrector_every_n_steps`) and S = 4 (`max_corrector_steps_per_loop`).
  Differences from the upstream sampler, all deliberate:
  1. Logits are aligned (D-005). Argmax is over the allowed set, smallest id on ties.
     Upstream argmax is unrestricted.
  2. Greedy only. No Gumbel noise and no random remasking.
  3. One block, the whole response. No EOS early stop (D-018: an EOS in the response
     is malformed). Semi-autoregressive blocks are not implemented for this sampler.
  4. Unmask quotas are D-029 `reveal_schedule` (ceil), so S = 0 matches monotone
     unmasking at T = min(B, L). Upstream `get_num_transfer_tokens` puts the
     remainder on the first steps instead. Ranking is the D-029 float32 full-vocabulary
     log-probability of the allowed argmax, not upstream's float64 softmax probability.
  5. On step s (1-based), if s is a multiple of ω and S > 0, the corrector runs.
     Its input is a temporary complete draft: masked editable positions are filled
     with the predictor's allowed argmax, and already visible tokens stay. The
     corrector then rewrites every editable position. A match stops the loop; that
     forward still counts. Already-visible main-state tokens are then replaced by
     the corrected tokens. Masks in the main state stay masks until the step's
     unmask commit, which uses the last completed logits and is not an extra NFE.
- **Call plan for the correction sampler.** Let W(t) = t + S·⌊t/ω⌋. The number of
  unmask steps is the largest t in 1..L with W(t) ≤ B, or one step that reveals
  every token when no such t exists (the corrector loop stops when the next forward
  would exceed B). A finished run with B ≥ 1 has no mask left. Early fixed points
  are not backfilled: `calls_used` may be less than B, and the difference is
  reported. Stopping inside a corrector loop skips iterations that do not fit; it
  does not roll back forwards that already ran.
- **Complete-draft repair.** `repair_drafts` stays the milestone-5 one-forward
  greedy screen (D-022, D-044). `repair_drafts_proseco` repeats that same greedy
  rewrite (`_greedy_response`) for up to `max_calls` iterations (default S = 4).
  `max_calls = 1` matches `repair_drafts` token for token. The batch stops when
  every draft is unchanged and does not backfill. Every draft in a batch is charged
  that batch's forward count. `dlmc evaluate-repair --method greedy` is the default
  and does not change the milestone-5 records. `--method proseco` is the new path.
- **D3IM** (`sampling/d3im.py`). Adapted from source_audit §3.2, not from a local
  SCOPE code drop (none is pinned). Each step is one `d3im` NFE. The candidate is
  the allowed argmax; its confidence is that candidate's float32 full-vocabulary
  softmax probability (ties: smallest position). This can tie distinct log-probs
  under float32 exp. Keep K positions in the active block and remask the rest,
  including tokens that were already visible. K = ⌊L_block · t / T⌋ before the
  block's last step, and the whole block on the last step. K = 0 remasks the block
  and still counts. Default is one block (the response). Optional `block_length`
  freezes earlier blocks; the horizon must be divisible by the number of blocks.
  EOS/pad/mask are excluded by the allowed mask, which stands in for the paper's
  "EOS confidence = −∞" on a format that has no EOS. A finished horizon has no mask.
- **Extra-unmask control** (`sampling/extra_unmask.py`). Not ReMDM. Q-6 stays open.
  The CDLM threshold decoder is not implemented. This control spends exactly B
  forwards: monotone unmasking for min(B, L) calls, then one revision per leftover
  call. A revision replaces the editable position whose current token has the lowest
  float32 full-vocabulary log-probability (ties: smallest index) with the greedy
  allowed candidate from that same forward. It does not remask. A call whose
  replacement equals the current token still counts. D3IM scores the new candidate
  and can remask; this control scores the token that is already written.
- **Timing.** The harness records synchronized model seconds and wall seconds around
  each batch (D-032 cache release after the batch). These are the milestone-8 cost
  hooks. No TACC call-budget job has been run. Do not mark EXPERIMENT COMPLETED
  from this entry.

**D-050 (2026-11-21) The backbone stays Open-dCoder. Milestone 6 is a software gate
for starting milestone 7, not a certificate that ProSeCo or SCOPE match their papers.**
Does not change D-000, D-046, or D-049.
- **Why not LLaDA.** The approved protocol names Open-dCoder-0.5B as the backbone
  and builds the question on drafts from that warmed model. ProSeCo's released
  sampler loads LLaDA-8B (`llada/generate.py` at the pinned commit: `GSAI-ML/LLaDA-8B-Instruct`).
  Moving the project onto LLaDA would replace the tokenizer, the logit shift, the
  warmup, the bank, and Family A. A second backbone is an optional extension in the
  build prompt, not part of the main grid.
- **What the Family B comparison is.** ProSeCo, SCOPE, and `proseco_gen_conf` are
  trained and decoded on this same Open-dCoder checkpoint. That compares procedures
  on one model. It does not reproduce the papers' published scores. LLaDA has
  copy-over, so a visible token's prediction is that token. Open-dCoder does not, so
  the same visible token can be predicted as something else. That is why Q-4 is still
  open for ProSeCo training.
- **SCOPE's model.** D3IM is the sampler in the SCOPE paper's appendix, not a third
  backbone. The milestone-1 audit read SCOPE's algorithms and found no code
  repository. It does not record SCOPE as a LLaDA method. Our D3IM path is the
  D-049 adaptation on Open-dCoder.
- **Milestone 6 status.** The D-049 samplers and `dlmc evaluate-calls` are
  IMPLEMENTED and TESTED LOCALLY (tiny backbone; suite 300 passed at that check).
  No Open-dCoder call-budget job has been run. The tests check call counts and
  token-state rules. They do not check numerical agreement with LLaDA or SCOPE.
- **Starting milestone 7.** The harness is ready, so milestone 7 can start. The first
  acts are written decisions for Q-4 and Q-5, then `proseco_basic` stop-gradient
  fixtures, then the other Family B trainers including `proseco_gen_conf` (D-046).
  Q-6 (ReMDM) stays an inference follow-up. Do not treat milestone 6 as proof that
  those trainers are faithful.

## Milestone 7: Family B training adapters (2026-11-22)

**D-051 (2026-11-22) ProSeCo training draft, corrector time weight, and online
rank groups. Closes Q-4. Recorded before any Family B trainer code.** Does not
change D-046's loss formula, D-049's sampler, or D-050's backbone. Supersedes, for
the online `proseco_gen_conf` arm only, D-046's phrase "the same (budget, stage-bin)
groups as Family A".

- **Training draft.** On each step the mask is a D-028 response-only Bernoulli draw
  on the gold response (labels are allowed for the mask and for the loss). `z_t`
  is that gold sequence with masked editable positions replaced by the mask id.
  Prompt and pad are unchanged. The first forward's aligned logits (D-005) produce
  `ŷ = sg(argmax)` over the D-029 allowed set, smallest id on ties, at **every
  editable response position**. The corrector input writes that token at every
  editable position, including positions that were visible in `z_t`. Prompt and pad
  stay as they are. Gold is not written into any position that is masked in `z_t`.
  This is ProSeCo Algorithm 1 (argmax at all positions) and the same allowed-set
  argmax the milestone-6 temporary complete draft uses. It is not a sampler: eval
  samplers still receive no gold, no error mask, and no checker output.
- **Alternative that is not the training rule.** Pinned `diffusion.py` at `43ce881`
  defaults `use_model_outputs_as_corrector_input: False` with `use_argmax_for_corrector:
  True`: argmax only where `z_t` is the mask, and the visible token (the gold `x_0`)
  is kept elsewhere. LLaDA's copy-over makes that equal to argmax-everywhere.
  Open-dCoder has no copy-over (D-050), so a visible token's prediction can differ
  from the token sitting there. Keeping the visible gold token would train a
  different corrector input. That code default is not used.
- **Time weight `a(t) = 1/t`.** `t` is that example's D-028 mask rate (clamped to
  `[1/500, 1 − 1/500]`, so `1/t` is finite). For the log-linear schedule the paper's
  `|α̇/(1−α)|` equals `1/t` when `α = 1 − t` and `t` is the mask rate. The pinned
  code's `dsigma / expm1(sigma)` is that scalar, and `use_weighted_corrector_loss`
  is true. This project uses `1/t` directly and does not introduce a second
  continuous-time noise schedule. `a(t_b)` multiplies each of example `b`'s corrector
  terms **before** the pool. It is not applied to `L_mask`. The same `a(t)` is used
  for `proseco_basic`, `proseco_error_prioritized`, and `proseco_gen_conf`.
- **`L_mask`.** D-017 / D-028 on the first forward: pooled CE over masked response
  positions, no `1/t`, no Open-dLLM path term. The first forward keeps its gradient
  for this term. The argmax and `log ĉ` are detached from it. An all-zero mask
  minibatch skips the optimizer and the scheduler and logs `skipped_zero_mask`. It
  does not run the corrector.
- **Corrector softmax.** Before the corrector CE or probability, the mask id is
  removed from the normalizer (its probability is 0). That is the pinned line
  `corrector_logits[..., mask_index] += neg_infinity` before `log_softmax`. `L_mask`
  does not do this. The target at a loss position must not be the mask id.
- **Pools, coefficient 1.** The pinned `configs/config.yaml` leaves
  `mdlm_loss_weight` and `corrector_loss_weight` at 0 (placeholders). The training
  objective is `L_mask + L_corr`.
  - `proseco_basic`: `L_corr = Σ_i a(t_b(i)) ℓ_i / N` over editable positions
    (one pool; `ℓ_i` is CE of the gold token on the second forward).
  - `proseco_error_prioritized`: the same single pool and the same `a(t)`, but the
    summand is `ℓ_i` where `ŷ_i ≠ gold` and `−p(gold)` where `ŷ_i = gold`. One
    denominator, the editable count, matching the pinned reduction
    `(term * attention).sum() / count`. This is not CE on the keep set.
  - `proseco_gen_conf`: D-046's two means, with `a(t_b)` inside each sum,
    `(Σ_{i∈E} a(t_b) w_i ℓ_i) / |E| + (Σ_{i∈K} a(t_b) ℓ_i) / |K|`. CE on both
    sets. Not Eq. 11. Not Family A's `high_conf`.
  Empty `E` or empty `K` contributes a graph-connected zero and is logged (D-010).
  The second forward is the one that receives the corrector gradient. Both ProSeCo
  forwards carry gradient, so the physical batch stays 16 for every Family B arm
  (D-045). SCOPE's step count is stated in the Q-5 entry; it does not get a smaller
  batch.
- **`proseco_gen_conf` groups.** An online step has no bank budget and no saved
  stage. The group is the D-011 stage bin of that example's mask rate `t_b`, edges
  `(0.2, 0.4, 0.6, 0.8)`, ranked inside the physical minibatch. The score is
  stop-grad `log ĉ` of the committed draft token from the **first** forward, on
  error positions only, ranked upward, `β = 0.5`, float64 on CPU, then cast.
  Weights have no gradient. Saved bank `log_c` is not read. The score is not `q`
  and not a probability from the second forward.
- **What this is not.** Not a ProSeCo paper reproduction, not LLaDA, not a change
  to the six Family A arms, not a selected call budget, and not Q-6 / ReMDM. Init
  remains checkpoint 8000 (D-038). Screen seed `20261120`, 1000 updates. Q-5 stays
  open until its own entry.
