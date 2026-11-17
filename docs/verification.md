# Verification report

Status labels: IMPLEMENTED · TESTED LOCALLY (tiny backbone / fixtures) · TESTED ON
REAL BACKBONE (Open-dCoder-0.5B @ d0d86d5b) · EXPERIMENT COMPLETED. No experiment
has been run yet; nothing here is evidence about the research hypothesis.

Machine: MacBook Pro M2 Max, 64 GB, macOS 26.6.2 (arm64). Python 3.14.7, torch
2.15.0. Main env: `requirements-lock.txt`. Reference env: `requirements-ref-lock.txt`.

## Milestone 1 (2026-11-16)

### Commands actually run and their results

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` | **36 passed** (31 tiny/fixture tests + 5 real-backbone tests), 0 warnings |
| `PYTHONPATH=runs/sources/Open-dLLM .venv-ref/bin/python scripts/reference_logits.py` | wrote `runs/reference/open_dcoder_reference.{pt,json}`; findings below |
| `.venv/bin/dlmc doctor --backbone tiny --device cpu --dtype float64` | PASS |
| `.venv/bin/dlmc doctor --backbone tiny --device mps --dtype float32` | PASS |
| `.venv/bin/dlmc doctor --backbone open-dcoder --device cpu --dtype float32` | PASS (`runs/doctor/doctor-open-dcoder-cpu-float32-20261116-210837.json`) |
| `.venv/bin/dlmc doctor --backbone open-dcoder --device mps --dtype float32` | PASS (`runs/doctor/doctor-open-dcoder-mps-float32-20261116-210843.json`) |
| `.venv/bin/dlmc download-backbone` | all 8 pinned sha256 values verified |
| fresh venv + `pip install -r requirements-ref-lock.txt`, then import `veomni` Qwen2 | imports; transformers 4.54.1 (the reference env is reproducible from the lockfile) |
| Mutation check: four bugs injected into a copy of `src/` and the tiny suite rerun | every mutant caught: no logit shift (2 failures), −log q sign flipped (1), weights ignored in L_error (5), causal attention without pad mask (1) |

### Real-backbone adapter (TESTED ON REAL BACKBONE)

- **Equivalence to upstream.** Our `BidirectionalQwen2` raw logits vs. Open-dLLM
  `veomni` Qwen2 (SDPA, mask None, `is_causal=False`, transformers 4.54.1), on 2
  sequences × 46 tokens (one fully masked response, one partially revealed draft),
  float32 CPU: max |diff| = **0.0**, argmax identical. Sensitivity control:
  scaling one final-norm weight by (1 + 1e-6) gives max |diff| 4.8e-5, and a RoPE θ
  of 999,000 gives 6.4e-3.
- **Bidirectionality.** Changing the last response token changes the aligned
  prediction at the first response position (threshold 1e-3). Upstream reference:
  right-context effect 5.35 on the SDPA/no-mask path.
- **Padding.** The same example alone vs. inside a padded batch: max |diff| < 1e-3
  (float32).
- **Token IDs.** Mask 151665 (`<M>`), pad = EOS = 151643, vocabulary 151,936. The
  stale `generation_config.json` mask ID 151643 is asserted and ignored.
- **Training gate (doctor).** A forward pass with no labels, masked loss plus
  correction losses, backward, and an AdamW step (lr 1e-5) on all 494,032,768
  parameters. Gradients are finite and the watched q_proj weight changed by 1.0e-5.

  | | CPU fp32 | MPS fp32 |
  | --- | --- | --- |
  | L_mask (5 masked) | 2.80061 | 2.80081 |
  | L_error (1 wrong digit) | 15.8192 | 15.8193 |
  | L_keep (10 tokens) | 6.35e-5 | 6.39e-5 |
  | grad norm | 802.70 | 802.70 |
  | 2 forwards + backward + step (single, unwarmed run) | 1.64 s | 1.10 s |
  | memory | peak RSS 10.2 GB | MPS driver 9.5 GB |

  The timings are single cold runs with sequence length 40, **not** benchmarks.
  Bitwise CPU/MPS identity is not expected.

  *Corrected 2026-11-16 (D-016).* The earlier sentence said that CPU/MPS differences
  are "about 1e-4 relative". That understated some quantities. The measured
  differences, recomputed from the two doctor JSON records, are:

  | Quantity | abs. difference | relative difference |
  | --- | --- | --- |
  | L_mask | 2.0e-4 | 7.2e-5 |
  | L_error | 9.3e-5 | 5.9e-6 |
  | L_keep | 4.1e-7 | **6.4e-3** (0.64%) |
  | grad norm | 5.5e-3 | 6.9e-6 |

  L_keep's large relative difference comes from its tiny magnitude, not a larger
  absolute error. ℓ ≈ 1 − p ≈ 6e-5 when p ≈ 1, so a float32 logit difference of
  similar absolute size to the other quantities' differences is a large fraction of
  it. This is one example on one run of each device. It is evidence that both
  devices compute the same function to float32 kernel precision, not a tolerance
  study.

### Upstream behaviour measured by `scripts/reference_logits.py`

The right-context effect is the maximum change of an earlier output row when the
last token changes; 0 means causal.

| Path | Right-context effect | Meaning |
| --- | --- | --- |
| `veomni`, SDPA, mask None, `is_causal=False` | 5.35 | bidirectional (the reference) |
| `veomni`, SDPA, all-ones 2-D mask | 5.35 | bidirectional |
| `veomni`, SDPA, 2-D mask with one pad | **0.0** | silently causal |
| `veomni`, eager, `is_causal=False` | **0.0** | silently causal |
| `AutoModelForCausalLM(trust_remote_code=True)` (model card) | **0.0** | stock causal Qwen2, no `diffusion_generate`; logits differ by up to 30.67 |
| Hub `modeling_qwen2.py` | n/a | import fails (docstring decorator). With a shim, `forward` fails (`attention_interface` called without `module`) |

### Gate A (alignment parts only)

| Check | Test | Status |
| --- | --- | --- |
| Shift: aligned row i = raw row i−1; row 0 unusable | `test_align_logits_shifts_by_one`, `test_position_zero_is_never_a_target` | TESTED LOCALLY |
| Prompt boundary: first response target uses the last prompt output | `test_first_response_target_uses_last_prompt_output` | TESTED LOCALLY |
| Right context affects earlier positions, and the probe detects a causal mask | `test_right_context_changes_earlier_predictions`, `test_right_context_check_detects_a_causal_mask` | TESTED LOCALLY + REAL |
| Padding isolation (alone vs. batch; pad contents irrelevant) | `test_padding_is_isolated`, real-backbone test | TESTED LOCALLY + REAL |
| Prompt/response/pad masks partition the sequence | `test_layout_masks_partition_sequence` | TESTED LOCALLY |
| Data, split, checker, fixed layout, log c / log q same-pass checks | n/a | **NOT RUN** (milestones 2 and 4) |

### Gate B (loss mathematics)

| Gate B item | Test | Status |
| --- | --- | --- |
| Per-token CE and totals vs. independent reference | `test_token_ce_matches_numpy_reference`, `test_total_loss_matches_reference` (NumPy float64, tol 1e-12) | TESTED LOCALLY |
| Autograd = w_i(softmax − onehot)/denominator with fixed weights | `test_autograd_matches_analytic_gradient` (tol 1e-14; prompt/pad rows exactly 0) | TESTED LOCALLY |
| Weight 0 zeros the row's direct gradient | `test_zero_weight_zeros_that_rows_direct_gradient` | TESTED LOCALLY |
| Weight scales the row's gradient and the optimizer update | `test_weight_scales_rows_direct_gradient`, `test_weight_scales_optimizer_update_on_tiny_model` (SGD update exactly doubles; weight 0 leaves parameters unchanged) | TESTED LOCALLY |
| β = 0 equals uniform; all-one weights equal the unweighted loss | `test_beta_zero_recovers_uniform`, `test_all_one_weights_equal_unweighted_mean` | TESTED LOCALLY |
| Ties, singleton bins, groups independent | `test_midranks_average_ties`, `test_singleton_and_fully_tied_groups_get_weight_one`, `test_groups_are_ranked_independently` | TESTED LOCALLY |
| All-correct, all-wrong, no masks, prompt/pad exclusion, finite gradients | `test_empty_sets_give_finite_zero_and_backprop`, `test_masked_loss_rejects_prompt_or_pad_positions`, `test_error_keep_masks_exclude_prompt_and_pad` | TESTED LOCALLY |
| Detached saved-confidence / saved-difficulty / current-loss weights | `test_weights_with_grad_are_rejected`, `test_current_loss_weights_act_as_constants`, `test_current_loss_weights_are_detached` | TESTED LOCALLY |
| Wrong-sign catch (ranking q upward) | `test_rule_directions` (and the mutation check) | TESTED LOCALLY |
| Bounds, mean one per bin, ESS, shuffle determinism | `test_weights_bounded_mean_one_and_diagnostics`, `test_shuffled_confidence_is_deterministic_and_preserves_weight_distribution` | TESTED LOCALLY |
| ProSeCo / SCOPE stop-gradient paths | n/a | **NOT RUN**: adapters not implemented (milestone 7) |
| Fused/chunked loss equivalence | n/a | not applicable: no fused loss is used |

### Tiny numerical example (hand-checkable)

Four errors in one (budget, stage) bin with saved log c = (−2.3, −1.2, −0.51, −0.105)
(c ≈ 0.10, 0.30, 0.60, 0.90), β = 0.5. The ranks are 1..4, so
r = (.125, .375, .625, .875) and w = 1 + .5(2r − 1) = **(0.625, 0.875, 1.125, 1.375)**,
mean 1 (`test_high_conf_hand_values_on_valid_log_probabilities`). With fixed weights (0.25, 1.75, 1.0) on three error rows and λ = 0.7,
autograd's gradient on error row i equals 0.7·w_i·(softmax(z_i) − onehot(y_i))/3 to
1e-14 (`test_autograd_matches_analytic_gradient`).

*Corrected 2026-11-16 (D-016).* This example previously said "saved log c = (10, 20,
30, 40)". A log probability is at most 0, so those values could not be saved log
probabilities. The test `test_rank_weights_hand_values` passes arbitrary positive
**ranking scores** (10, 20, 30, 40). They are not log probabilities; only their order
matters, so the ranks and weights above are unchanged. The example now uses valid
negative log c values with the same order.

### Not run in milestone 1

- Gate A data/checker items, Gate C (tiny overfit run, checkpoint/resume), Gate D (samplers, observer).
- bf16 on MPS or CUDA; any A100/Colab run.
- Memory/runtime pilot for full fine-tuning vs. LoRA (needed in milestone 3).
- Any experiment, so no EXPERIMENT COMPLETED items.

## Re-check at handoff (2026-11-16)

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` | 36 passed in 4.0 s |
| `shasum -a 256` on the tracked proposal PDF vs. `~/Downloads` | identical (`043a0648…066f`) |

No code changed after the milestone-1 runs above. Only documentation was edited.

## Advisor follow-ups (recorded open at handoff; closed 2026-11-16, see below and D-016/D-017)

1. **Reference gradient comparison.** Milestone 1 matched our forward logits to the
   upstream `veomni` class. It did not compare gradients (for example, parameter
   gradients of the same loss on the same inputs). Our gradient evidence so far is
   the tiny-model analytic checks and the real-backbone doctor step.
2. **Weight loading and tying.** `load_open_dcoder` raises if any expected key is
   missing or unexpected, and if the checkpoint's `lm_head.weight` differs from
   `embed_tokens`. It then ties `lm_head` to the embedding. No test or recorded
   check yet confirms this on the real checkpoint (key counts, shared storage after
   the dtype and device move, and an identical result after an optimizer step).
3. **Clarify D-006.** The masked-loss contract needs clarification. It is unchanged
   until a superseding decision is recorded.
4. **Two statements in this report need correcting.**
   - The tiny numerical example uses saved "log c" = (10, 20, 30, 40). A log
     probability is at most 0, so the example should use valid negative values. The
     ranks and weights are unaffected, because only the ordering is used.
   - "CPU/MPS differences are about 1e-4 relative" understates some entries in the
     doctor table. For example, L_keep is 6.35e-5 vs. 6.39e-5, which is about 0.6%
     relative. The sentence should report each quantity's actual difference.

## Milestone-1 follow-ups (2026-11-16, second session)

### 1. Gradient equivalence with upstream (TESTED ON REAL BACKBONE)

Command (reference env; ran in about 60 s on CPU):

    PYTHONPATH=runs/sources/Open-dLLM:src .venv-ref/bin/python scripts/reference_gradients.py

Output: `runs/reference/gradient_equivalence.json`; console log in
`runs/reference/gradient_run.log`. Exit 0, `passed=true`.

Setup:
- Reference model: upstream `veomni` Qwen2 (Open-dLLM `5b9ba4d`, transformers 4.54.1),
  SDPA, `attention_mask=None`, `is_causal=False`, no labels.
- Our model: `BidirectionalQwen2` with an all-True key mask.
- Both load the same pinned weights, run in train mode on CPU with
  `torch.use_deterministic_algorithms(True)`, and have gradients enabled.
- Loss: `external_aligned_loss` in the script, Σ w_i·(−log softmax(raw[i−1])[y_i]) / n.
  This is plain PyTorch and shares no code with either model or with
  `diffusion_correction.losses`. Upstream's own loss (liger fused CE plus path term) is
  not used.
- Inputs: prompt "# Compute a=17+28; b=a*3; c=b-19. Return a,b,c.\nanswer = " (46 tokens
  in total with the response).
  - (A) **masked**: 6 of the 11 response tokens are `<M>`; CE on those 6 positions with
    weight 1.
  - (B) **complete flawed draft** `046,135,126` (target `045,135,116`; 2 wrong digits):
    CE on all 11 response positions, with weight 1.5 on the errors and 0.5 on the rest.
- Metric: per-tensor relative L2 error ‖g_ours − g_ref‖ / ‖g_ref‖ over all 290 distinct
  parameters. Thresholds were declared before the run: 1e-4 (float32) and 1e-9
  (float64). Both controls must exceed 100× the threshold.

Worst per-tensor relative L2 error, by parameter group:

| Run | Input | embedding = lm_head (1) | attention (168) | MLP (72) | norms (49) | loss / logits |
| --- | --- | --- | --- | --- | --- | --- |
| float64, upstream float32 casts matched (gating) | masked | 0 | 2.0e-15 | 0 | 2.4e-15 | identical; max \|Δlogit\| 0 |
| | flawed draft | 0 | 1.4e-15 | 0 | 3.4e-15 | identical; 0 |
| float32 (gating) | masked | 6.2e-6 | 9.8e-6 | 6.1e-6 | 3.1e-5 | identical; 0 |
| | flawed draft | 5.1e-6 | 1.7e-5 | 6.8e-6 | 3.1e-5 | identical; 0 |
| float64, unmatched (information only) | masked | 8.5e-7 | 2.7e-6 | 9.8e-7 | 2.3e-6 | Δloss 8.2e-7; max \|Δlogit\| 7.5e-6 |
| | flawed draft | 1.0e-6 | 3.2e-6 | 2.1e-6 | 5.4e-6 | Δloss 2.4e-7; 1.6e-5 |

- Minimum cosine similarity per tensor: ≥ 0.9999999995 in float32, and 1.0 to 12
  digits in matched float64.
- Controls (worst tensor; identical in every run): upstream run with `is_causal=True`
  gives 24.7 (masked) and 9.9 (flawed draft). Our loss without the logit shift gives
  2.9 and 3.0. Both are about 10⁵ times the float32 threshold.
- Tying: both models have 290 distinct parameters, and in both `lm_head.weight` is
  `embed_tokens.weight`. The tied tensor's gradient therefore sums the input-embedding
  and output-projection paths in both models, and those sums agree, as shown in the
  "embedding = lm_head" column. Initial weights are identical (`torch.equal` on all 290).
- Why float32 differs while logits are bit-identical: the forward kernels happen to
  produce identical results, but the backward passes accumulate in a different order.
  Upstream uses `repeat_kv` (expand plus reshape) while ours uses `repeat_interleave`,
  and the two have different backward reductions. The resulting float32 errors (at
  most 3.1e-5 relative, largest on RMSNorm weights, whose gradients sum over all
  positions) are about 3.2× below the threshold (1e-4 / 3.1e-5). *Corrected
  2026-11-17: this previously said "about 30×".* In matched float64 the same paths
  agree to 3.4e-15, so the float32 differences are rounding, not a different
  computation.
- Why float64 needs matching: upstream's RMSNorm casts to float32 inside the norm, and
  its RoPE computes cos/sin in float32 (`modeling_qwen2.py` lines 278–283 and 386–405),
  even when activations are float64. Ours keeps float64. In float32 both casts are
  identity operations, and for bf16 both upcast to float32, so production dtypes run
  the same arithmetic. With only RoPE matched, the error stayed around 5e-6. With
  RMSNorm also matched it dropped to about 3e-15. That localises the unmatched
  float64 difference to these two casts.

### 2. Complete loading and tying (TESTED ON REAL BACKBONE)

`tests/test_open_dcoder.py::test_checkpoint_loads_completely_and_embeddings_stay_tied[cpu|mps]`:
both pass (the MPS case needs the run outside the Cursor sandbox, which hides MPS).
- `model.safetensors` holds 291 tensors: 290 parameters plus `lm_head.weight`, an alias
  equal to `model.embed_tokens.weight` (the config sets `tie_word_embeddings: true`).
  The checkpoint has no rotary keys; `inv_freq` is a non-persistent buffer recomputed
  from the config.
- Every checkpoint tensor equals the loaded fp32 parameter bit for bit (bf16 → fp32 is
  exact). The alias is checked against `lm_head.weight`.
- `lm_head.weight is embed_tokens.weight`, they share `data_ptr()`, and the parameter
  count is 494,032,768. After an AdamW step the tensors are still the same object, the
  embedding has changed, and the optimizer holds 290 tensors, so the tied weight is
  updated once.

### 3 and 4. D-006 clarification and documentation corrections

D-017 clarifies the masked-loss contract. The two statements above are corrected in
place, with "Corrected 2026-11-16" notes. A new test,
`test_high_conf_hand_values_on_valid_log_probabilities`, runs the `high_conf` rule on
valid negative log c values.

## Milestone 2: tasks, format, checker, splits, inspect-example (2026-11-16)

Labels: everything below is IMPLEMENTED and TESTED LOCALLY against the **real pinned
tokenizer**. No model was run on these tasks. **No accuracy was measured**; the
40–85% target range is only checked after warmup (milestone 3).

### Commands actually run

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` (outside the sandbox, so MPS is visible; final run of the 2026-11-16 session; see "Milestone-2 closure" for the current count) | **137 passed** in 11.5 s |
| `.venv/bin/python -m pytest -q` (inside the sandbox) | 136 passed (the MPS case is not collected) |
| `.venv/bin/python -m pytest -q -m "not real_backbone"` | 129 passed, 6 deselected |
| `.venv/bin/python -m pytest -q -m "not real_backbone and not tokenizer"` | 70 passed, 65 deselected (no runs/hf files needed) |
| `.venv/bin/dlmc generate-data --config configs/data/smoke.yaml --overwrite` | 216 problems, verification passed, manifest sha256 `52740480…b2a0` |
| `.venv/bin/dlmc generate-data --config configs/data/pilot.yaml --overwrite` | 27,900 problems (18,000 / 3,600 / 3,600 / 900 / 1,800) in 16 s, verification passed, manifest sha256 `990ecd1d…1536` |
| same pilot command with `--out /tmp/pilot-regen` | identical manifest sha256 (deterministic regeneration) |
| `.venv/bin/dlmc verify-data --data-dir runs/data/pilot-seed20261116` | passed (27,900 examples, 27,900 distinct problems) |
| `PYTHONPATH=src .venv-ref/bin/python scripts/reference_tokenizer.py --data-dir runs/data/smoke-seed0` | 661 strings, 0 mismatches |
| `PYTHONPATH=src .venv-ref/bin/python scripts/reference_tokenizer.py --data-dir runs/data/pilot-seed20261116` | **83,713 strings (27,900 records × prompt/response/joint + edge strings), 0 mismatches** between our `tokenizers` wrapper, transformers `Qwen2TokenizerFast`, and the slow pure-Python `Qwen2Tokenizer` (vocab.json + merges.txt), and the stored record IDs |
| `.venv/bin/dlmc inspect-example --fixture arith_17_28 --draft-values 46,135,116 --json …` | table below; artifacts in `runs/inspect/` |
| Mutation check (8 mutants of `src/diffusion_correction/data/` in a temp copy; non-real-backbone suite) | 7 caught. The survivor (removing `strict=True` from the checker's zip) is an **equivalent mutant**: the explicit length guard before it already returns for unequal lengths, and the guard's own mutant is caught |

Mutants and what caught them: interpreter `x − y` → `y − x` (hand-checked programs and
the independent `ast` evaluator); checker length guard disabled
(`test_checker_never_zips_unequal_lengths`); `-0000` accepted
(`test_slot_parse_statuses`); special tokens not flagged (`test_slot_parse_statuses`);
global de-duplication disabled (split disjointness test); held-out predicate leaking
into test_long (generate-data's built-in verification fails in the shared fixture);
multiplier bound off by one (`test_make_problem_rejects_out_of_spec_chains`).

### Gate A data items

| Check | Test / evidence | Status |
| --- | --- | --- |
| Hand-checked arithmetic: 2+3 → 5; 17+28, ×3, −19 → 45, 135, 116; 12−45, ×3, +99 → −33, −99, 0 | `test_hand_checked_arithmetic_fixtures` | TESTED LOCALLY |
| Interpreter hand cases (incl. reassignment, const − var, negatives) and 22 predictable rejections (syntax, undefined, multiply, literal, read-only input, bound, length) | `test_hand_checked_programs`, `test_invalid_programs_are_rejected_predictably` | TESTED LOCALLY |
| Independent reference: an `ast`-based evaluator in the test file (no shared code, no eval/exec) agrees with generated chains/programs (2,400 problems, lengths up to 12) | `test_sampled_chains_…`, `test_generated_programs_match_independent_evaluation` | TESTED LOCALLY |
| Interpreter never calls eval/exec (builtins patched to raise) | `test_interpreter_never_calls_eval_or_exec` | TESTED LOCALLY |
| Labels re-derived for every stored record by the interpreter (arithmetic as a sub-language), separately from the generators | `verify_dataset`, run by generate-data and verify-data | TESTED LOCALLY (27,900 records) |
| Format token IDs equal hand-read IDs (`" +"` 488, `" -"` 481, digits 15–24); naive formats merge (`"=-"`) | `test_format_token_ids_match_hand_inspection`, `test_naive_formats_do_not_have_fixed_token_width` | TESTED LOCALLY |
| Encode/parse round trip at signs, zero and magnitude bounds (±9999, ±1000, ±999, …, 0), plus 2,000 random multi-slot sequences | `test_single_value_tokenization_and_round_trip`, `test_random_multi_slot_sequences_…` | TESTED LOCALLY |
| Prompt/response boundary: joint encoding = concatenation for every fixture and every stored record; the check detects a real merge (`"a="` + `"-0045"`); no printable prompt ending merges with `" +"`/`" -"` | `test_fixture_prompts_have_clean_boundaries`, `test_boundary_check_detects_a_real_merge`, `test_leading_space_sign_never_merges_…`, `verify_dataset` | TESTED LOCALLY |
| Wrapper vs. reference tokenizers | `scripts/reference_tokenizer.py` (above) | TESTED LOCALLY |
| Prompt contains no computed value (rebuilt from value-free fields for every record) | `verify_dataset` | TESTED LOCALLY |
| Fixed layout: response length = slots × 5, spans `[5k, 5k+5)`, no EOS | `test_fixture_17_28_exact_ids`, manifest test | TESTED LOCALLY |
| Checker: separate token/value/final/trace levels; malformed slots counted wrong; unequal lengths never zipped; denominators include every example (hand-counted aggregate) | `test_checker_*`, `test_aggregate_denominators_include_every_example` | TESTED LOCALLY |
| Value correct ⇔ all slot tokens correct (3,000 random corruptions) | `test_value_correct_iff_all_slot_tokens_correct` | TESTED LOCALLY |
| Checker token labels = loss code's E/K masks on the real layout | `test_checker_labels_agree_with_loss_error_masks` | TESTED LOCALLY |
| Split disjointness by problem ID and normalized key; held-out compositions only in test_unseen; long lengths only in test_long; exact per-setting counts | `test_splits_are_disjoint_and_compositions_are_held_out`, `verify_dataset` | TESTED LOCALLY |
| Tampering detected (file hash), cross-split duplicate detected with a re-hashed manifest, wrong label detected | `test_verify_detects_*` | TESTED LOCALLY |
| Deterministic manifests; seed changes them | `test_generation_is_deterministic_and_seed_sensitive`, pilot regeneration | TESTED LOCALLY |
| Drafts/augmentations stay in their problem's split | `test_derived_records_must_stay_in_their_problems_split` (`assert_grouped_by_problem`) | TESTED LOCALLY (utility; no drafts exist yet) |
| Too-small problem space fails loudly | `test_too_small_problem_space_fails_loudly` | TESTED LOCALLY |
| inspect-example: aligned row = pos − 1, masks, slot roles, E/K labels, signals `n/a`, length-mismatch draft not aligned | `test_inspect_example_*` | TESTED LOCALLY |
| log c / log q same-pass checks; prompt immutability under sampling | n/a | NOT RUN (milestone 4 / samplers) |

### Real inspect-example output (fixture; draft is HYPOTHETICAL)

`.venv/bin/dlmc inspect-example --fixture arith_17_28 --draft-values 46,135,116 --pad 2 --json runs/inspect/arith_17_28_hypothetical_draft.json`
(full output in `runs/inspect/arith_17_28_hypothetical_draft.txt`; excerpt of selected
rows, columns trimmed):

    pos  seg       attn prompt resp edit token_id token  aligned_from_raw_row slot role   slot_value draft_id draft_token label
    60   prompt    1    1      0    0    25       ':'    -                    -    -      -          -        -           -
    61   response  1    0      1    1    488      ' +'   60                   a[0] sign   45         488      ' +'        K (keep)
    64   response  1    0      1    1    19       '4'    63                   a[0] digit2 45         19       '4'         K (keep)
    65   response  1    0      1    1    20       '5'    64                   a[0] digit3 45         21       '6'         E (wrong)
    66   response  1    0      1    1    488      ' +'   65                   b[1] sign   135        488      ' +'        K (keep)
    75   response  1    0      1    1    21       '6'    74                   c[2] digit3 116        21       '6'         K (keep)
    76   pad       0    0      0    0    151643   '<|endoftext|>' -           -    -      -          -        -           -
    checker: tokens correct 14/15; values correct 2/3; final answer correct True; whole trace correct False

(c, q, −log q, bin, p_now, w, CE, wCE print `n/a` on every response row.)

### Limitations and checks not run

- No model was trained or evaluated on these tasks, so nothing is known about task
  accuracy. The pilot grid is provisional (D-019).
- Normalized keys identify programs up to consistent variable renaming. They do not
  identify commuted operands (`17 + 28` and `28 + 17` are different problems) or
  semantically equal programs. This is a documented limitation, not a bug.
- Generators sample with step-wise rejection near the bounds, so operator and value
  distributions are not uniform (D-019). The generator streams are seeded with
  `random.Random`, so a dataset is reproducible from its config on this Python
  version. Regenerating on another Python version should be checked against the
  manifest hash rather than assumed.
- `vocab.json` and `merges.txt` (used only by the slow reference tokenizer) are not in
  `PINNED_SHA256`. Their sha256 values are recorded in each
  `tokenizer_reference_check.json`.
- The MPS loading test runs only outside the Cursor sandbox.
- *Gap found by advisor review and closed 2026-11-17 (next section):* the held-out
  check above used the stored `signature`, and identity was not rebuilt from source.

## Milestone-2 closure: structure and identity rebuilt from source (2026-11-17)

Labels: IMPLEMENTED and TESTED LOCALLY (real pinned tokenizer for the dataset tests).
Decision D-024.

### What `verify_dataset` now checks for every record

1. Setting: `length` is in the split's lengths, and `operand_digits` is in the config.
2. `data/structure.py::derive_problem` parses `source` and rebuilds the problem, or
   fails with "source is outside the generated structure". The rules are in D-024 and
   `docs/data_format.md` §6.
3. The held-out predicate is applied to the **derived** signature and must agree with
   the split (only `test_unseen` is held out) and with `heldout_composition`.
4. The stored `signature` must equal the derived one exactly.
5. The stored `normalized_key` must equal the key rebuilt from source plus inputs.
   The stored `problem_id` must equal sha256(rebuilt key)[:16].
6. Uniqueness across all splits uses the rebuilt problem ID and key.
7. Values must equal both the independent interpreter's output and the rebuilt
   problem's values, and `value_names` must equal the source targets. The existing
   token, span and prompt-rebuild checks are unchanged.

### New tests

`tests/test_data_structure.py` has 34 tests, no tokenizer needed. Expected values are
written by hand.
- Arithmetic signatures, keys and values from source (4 cases): `('+','*','-')`
  key `arithmetic|a=17+28;b=a*3;c=b-19`; zero and negative `('-','-')`; `(-33,-99,0)`;
  `*` first.
- Program signatures from source (4 cases):
  - first statement with an input and a constant;
  - first statement combining both inputs;
  - reuse of earlier targets and inputs;
  - swapped operand order (`b = x - a` is `sub_var`).
- The held-out predicate on derived signatures, including order sensitivity
  (`*` then `-` is not held out).
- Rejections outside the generated structure:
  - arithmetic, 10 cases: constant on the left, wrong target, reassignment, length,
    digit count, multiplier, non-canonical spacing, broken chain;
  - programs, 13 cases: constant on the left, not reading the predecessor, same
    variable twice, `NAME * NAME`, no input read, reassignment, renamed inputs, input
    digits, multiplier, constant digits, spacing, undefined name.
- Setting/config mismatches.
- Identity: renaming invariance including a renamed reassigned target; reassignment
  differs from a fresh variable; input values, operand order and statement order all
  change the key. The exact key `program|i0=5;t0=i0+1;t0=t0+2;t1=t0*3` is checked.

`tests/test_data_splits.py` gains 9 tamper tests on a copy of the smoke dataset. Each
copy has its manifest file hashes and counts re-written, so verification reaches the
semantic checks.

| Test | Tampering | Rejected with |
| --- | --- | --- |
| `test_verify_rejects_wrong_signature_with_correct_heldout_status[arithmetic, programs]` | stored signature replaced by a different signature that is still not held out (status correct for train) | "stored signature … != source signature" |
| `test_verify_rejects_heldout_source_labelled_as_not_heldout[arithmetic, programs]` | a test_unseen problem moved to train, labelled `heldout_composition=False` with a non-held-out signature | "source composition … is held out, but split is train" |
| `test_verify_rejects_stale_heldout_flag` | `heldout_composition=False` on a test_unseen program | "heldout_composition flag differs" |
| `test_verify_rejects_normalized_key_not_rebuilt_from_source` | wrong key, with `problem_id` re-hashed from it (passes the old check) | "stored normalized_key differs" |
| `test_verify_rejects_problem_id_not_hash_of_rebuilt_key` | wrong `problem_id` | "problem_id" |
| `test_verify_rejects_duplicate_problem_across_splits` | a val program copied into train | "also in" |
| `test_verify_rejects_source_outside_generated_structure` | non-canonical spacing in an arithmetic source | "outside the generated structure" |

Mutation check (each new assertion disabled in turn, then the file restored; the
diff was verified empty):
- dropping the exact signature comparison makes the wrong-signature test fail;
- applying the predicate to the stored signature makes the held-out-source test fail;
- dropping the key rebuild makes the normalized-key test fail;
- dropping the flag check makes the stale-flag test fail.

### Commands and outcomes

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q tests/test_data_structure.py tests/test_data_splits.py` | 54 passed |
| `.venv/bin/python -m pytest -q` (outside the sandbox) | **180 passed** in 13.4 s |
| `.venv/bin/python -m pytest -q -m "not real_backbone and not tokenizer"` | 105 passed, 75 deselected |
| `.venv/bin/dlmc verify-data --data-dir runs/data/smoke-seed0` | passed; manifest `527404808937a52446627e576965a0bd37a8cdb3cdb862e577fa278c51b1b2a0` (unchanged) |
| `.venv/bin/dlmc verify-data --data-dir runs/data/pilot-seed20261116` | passed, 27,900 records in 12.3 s; manifest `990ecd1dd2341b81bc742fa91189b129877cd9b48b196fbdbc208e8586c81536` (unchanged) |

No dataset was regenerated. The split files' sha256 values are unchanged, matching
their manifests.
