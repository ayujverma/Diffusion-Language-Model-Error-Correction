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

**D-034 (2026-11-17) PROPOSED, NOT APPROVED: torch 2.14.1+cu126 on Lonestar6 for an R570
driver.** Requires the user's explicit approval before the update is applied on TACC.
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
