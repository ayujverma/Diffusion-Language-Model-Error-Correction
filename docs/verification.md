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

## Milestone 3: masked warmup and difficulty calibration (2026-11-17)

Device for every run below: Apple M2 Max (MPS), fp32. No CUDA device is available
locally and no paid compute was used; nothing in this section was measured on an A100.
Large artifacts live under `runs/` (git-ignored). Dataset for every run: the pilot,
manifest `990ecd1dd2341b81bc742fa91189b129877cd9b48b196fbdbc208e8586c81536`,
train split sha256 `215fdb4a…e18a` (18,000 examples); evaluation uses the `val` split of
the same manifest. No `test_*` split was loaded.

### Software status

| Component | Status |
| --- | --- |
| D-028 masking, RNG streams, verified data loading, epoch sampler | TESTED LOCALLY, used on real backbone |
| Trainer (AdamW + linear-warmup/constant LR, clipping, zero-mask skip, JSONL metrics) | TESTED LOCALLY + TESTED ON REAL BACKBONE |
| Atomic checkpoints, resume validation, durable copy + copy-back | TESTED LOCALLY + TESTED ON REAL BACKBONE |
| LoRA (`LoRALinear`, r 16, α 32, 7 projection types) | TESTED LOCALLY + TESTED ON REAL BACKBONE (20-update run, one-step freeze check) |
| D-029 monotone sampler with observer hook and traces | TESTED LOCALLY + TESTED ON REAL BACKBONE |
| Generation scoring, `dlmc calibrate`, `benchmark`, `precision-check`, `compare-runs` | TESTED ON REAL BACKBONE |
| Colab launcher `notebooks/colab_runner.ipynb` | IMPLEMENTED only (no CUDA runtime was used) |

### Sequence lengths

Pilot training sequences (prompt + response) are 59–141 tokens, mean ≈ 95. The
longest setting is programs, length 6, digits 3. Responses have 10, 20 or 30 tokens.

### Precision (`dlmc precision-check --device mps`, `runs/benchmark/precision_mps.json`)

Fixed fixture: 8 training examples padded to 8 × 119, 86 masked tokens (D-028 masks
from a fixed stream), fp32 reference loss 4.14837. Tolerances were declared in
`evaluation/precision.py` before the run: loss relative difference ≤ 2e-2, masked-row
argmax agreement ≥ 0.98, global gradient relative L2 ≤ 0.1, gradient cosine ≥ 0.995.

| Candidate | loss rel. diff | argmax agreement | grad rel. L2 | grad cosine | fwd+bwd s (last 2) | Passes |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 repeat (noise floor) | 0 | 1.0 | 5.5e-9 | 1.0 | — | — |
| bf16 autocast, fp32 master | 7.1e-4 | 0.988 | 0.024 | 0.9997 | 1.35, 1.31 (fp32: 1.11, 1.18) | yes |
| pure bf16 parameters | 5.9e-6 | 0.977 | 0.030 | 0.9996 | 1.29, 1.25 | **no** (argmax) |

The fp32 forward is bitwise repeatable on MPS; the backward is not quite (5.5e-9).
Decision D-030: fp32 without autocast.

### Timing/memory pilot (`dlmc benchmark`, `runs/benchmark/pilot_mps.json`)

Fresh model per configuration; 3 counted warmup updates then 10 timed updates with
`torch.mps.synchronize()` around each; timing covers forward + loss + backward + clip
+ AdamW + scheduler. Memory definitions: *live* = `torch.mps.current_allocated_memory`
(tensors); *driver* = `torch.mps.driver_allocated_memory` (includes the allocator
cache). Analytic fp32 full fine-tuning: parameters 1.976 GB, gradients 1.976 GB,
AdamW state 3.952 GB.

| Configuration | s/update | examples/s | live GB | driver max GB |
| --- | --- | --- | --- | --- |
| full, batch 16, mixed lengths | 2.85 | 5.59 | 6.05 | 30.1 |
| full, batch 16, longest setting | 2.92 | 5.47 | 6.05 | 21.8 |
| full, batch 32, longest (32 × 133–136, 4,299 real tokens) | 5.88 | 5.44 | 6.05 | 36.0 |
| LoRA, batch 16, longest | 2.55 | 6.26 | 2.12 | 15.1 |
| LoRA, batch 32, longest | 4.61 | 6.93 | 2.12 | 29.6 |
| full, batch 16, longest, bf16 autocast | 3.16 | 5.06 | — | 21.6 |

LoRA trainable parameters: 8,798,208 (matches `expected_lora_parameter_count`).

### Feasibility runs (20 updates, identical batches and masks)

`dlmc train` with `configs/train/warmup_pilot.yaml` (then batch 16) plus `--set`
overrides for `run_name`, `limits.max_updates=20`, `eval.generate_budgets=[]`,
`checkpoint.every_updates=0`, and for LoRA `method=lora optimizer.lr=2e-4`: full fine-tuning at lr 3e-5 (`runs/train/feas-full-20`) went from
loss 3.746 to 1.517, val masked loss 1.445, 74.9 s of training, 5.9 GB checkpoint
saved in 16 s. LoRA at lr 2e-4 (`runs/train/feas-lora-20`) ended at 1.652, val masked
loss 1.686, 53.6 s. These 20-update numbers show that the wiring trains; they are not
evidence about generalization or about which method is better.

### Gate C

**Overfit** (`dlmc train --config configs/train/overfit32.yaml --device mps`,
`runs/train/gatec-overfit32`): 32 fixed training examples (stratified, subset seed 0,
selected-ID sha256 `46e8ed66…1abb`), batch 16, fp32 full fine-tuning, lr 3e-5,
evaluation every 50 updates on the same 32 examples with the D-029 sampler. Early
stop rule declared in the config: trace accuracy 1.0 at T = L and masked loss ≤ 0.05.

| update | masked loss | T = L final / trace / malformed | T = 2 final / trace / malformed |
| --- | --- | --- | --- |
| 0 (pretrained) | 4.109 | 0 / 0 / 100% | 0 / 0 / 100% |
| 50 | 0.512 | 44% / 22% / 3% | 41% / 19% / 0% |
| 100 | 0.048 | 97% / 69% / 0% | 88% / 69% / 3% |
| 150 | 0.0098 | 94% / 88% / 3% | 100% / 100% / 0% |
| 200 | 0.0046 | 100% / 88% / 0% | 100% / 100% / 0% |
| 250 | 0.0007 | **100% / 100% / 0%** | **100% / 100% / 0%** |

It stopped early at update 250 of the 500 allowed (758 s), with 0 zero-mask batches
skipped. Checkpoint `ckpt-0000250`, model sha256 `e278dcfd…13b7`. The pretrained
model's malformed outputs were mostly `bad_sign_token`: it does not emit the format
before training.

**Prediction and loss inspection** (`scripts/gatec_inspect.py`,
`runs/gatec/inspect/inspect.txt`). For 4 overfit examples with the whole response
masked, every target token is the argmax, with p(target) ≥ 0.9977 and the pooled loss
≤ 2.4e-4. Per-position CE is listed in the file. The sampler output at T = L and T = 2
equals the target string, for example `' +0028 +0112 +0672 +6048'` for `a = 7 * 4, b = a * 4,
c = b * 6, d = c * 9`.

**Parameter updates.** Full fine-tuning: all 290 stored tensors of the overfit
checkpoint differ from the pinned pretrained tensors (the tied `lm_head` is not stored
separately). LoRA, one real update on MPS with the same data: all 168 `lora_B` tensors
changed and all 290 base tensors are bitwise unchanged, with 8,798,208 trainable
parameters.

**Checkpoint/resume** (`configs/train/resume_check.yaml`: batch 32, warmup
optimizer/schedule, 10 updates, checkpoints at 5 and 10):
- A: uninterrupted (`runs/train/resume-A`).
- A2: an identical uninterrupted repeat, which gives the device noise floor.
- B: `--max-updates 5 --durable-dir runs/durable/resume-B`. The process exited. Then
  `runs/train/resume-B/checkpoints` was **deleted**, and a new process ran with
  `--resume auto --durable-dir runs/durable/resume-B`. It copied `ckpt-0000005` back
  from the durable copy, re-verified the hashes and resumed (`accelerator_rng_restored:
  true`, `cross_device: false`). The durable copies logged `ok: true` only after
  re-hashing.
- `dlmc compare-runs runs/train/resume-A runs/train/resume-{B,A2} --checkpoint
  ckpt-0000010 --reference-model runs/hf/open-dcoder-0.5B/model.safetensors
  --loss-rtol 1e-4 --param-atol 1e-5 --divergence-rtol 1e-3` (`runs/gatec/compare-*.txt`).

| | A vs B (resumed) | A vs A2 (repeat) |
| --- | --- | --- |
| batch / epoch / example IDs / problem IDs / mask digests / t / n_masked / lr | identical, all 10 | identical, all 10 |
| max loss relative difference | 4.1e-7 | 3.9e-6 |
| max grad-norm relative difference | 3.6e-5 | 1.3e-4 |
| max parameter abs difference at update 10 | 9.5e-7 | 3.8e-6 |
| ‖θ_A − θ_X‖ / ‖θ_A − θ_pretrained‖ | 2.1e-5 | 4.2e-5 |
| passed (declared tolerances) | yes | yes |

The resumed trajectory is within the device's own run-to-run noise: MPS backward is
not bitwise deterministic, so bitwise equality is not expected. Data order and masks
are exact. Cross-device limits: the accelerator RNG state is restored only on the same
device type. Masks and data order are generated on the CPU and stay identical across
devices, but MPS-to-CUDA resumes are not expected to match numerically beyond the
precision tolerances.

### Common warmup (`runs/train/warmup-pilot-v1`)

Command: `dlmc train --config configs/train/warmup_pilot.yaml --device mps --run-dir
runs/train/warmup-pilot-v1`. The run started from the pinned pretrained weights, with
the settings fixed by D-028/D-030: fp32 full fine-tuning, AdamW lr 3e-5, a 20-update
linear warmup then constant, clipping at 1.0, physical batch 32, seed 20261117.
Validation monitoring used 144 `val` problems (8 per setting), every 250 updates plus
update 0, at T = L and T = 4.

**Two earlier attempts failed and are archived; neither wrote a checkpoint.**
1. `runs/train/warmup-pilot-v1.oom-attempt`: an MPS out-of-memory error after 17
   updates. The update-0 evaluation (batch 48, full-vocabulary fp32 logits) left
   large cached blocks.
2. `runs/train/warmup-pilot-v1.swap-attempt`: 59 updates in 55 minutes. Step time
   rose from 5 s to 150 s. At update 50 the Metal driver held 78.2 GB against 6.0 GB
   of live tensors (the recommended maximum is 55.7 GB), and macOS swap was full. The
   cause: padded lengths vary per batch, so the MPS allocator keeps caching new block
   sizes, and it only reclaims at 1.4× the recommended maximum.
Fixes: `env.release_cached_memory` runs after every update (`release_cache_every:
1`), after each evaluation, and after each generation batch; the evaluation batch is
24. These change memory behaviour only. All three attempts produced identical losses
at matching updates (for example update 10: 1.8015, gradient norm 8.73), because the
data order and masks come from the seeded CPU streams.

**Completed run.** Stopped by `max_updates`. 1,000 updates = 1,000 batches = 32,000
examples (1.78 epochs of the 18,000 training examples). 0 zero-mask batches;
3,043,874 real tokens; 320,654 masked tokens. Training wall time **5,790 s** (cap
7,200 s), including the evaluations (about 92 s each) and checkpoint saves (about
10 s each). Median step 5.01 s. Metal driver memory stayed between 28.7 and 30.5 GB;
live tensors 6.05 GB. Final checkpoint `ckpt-0001000`: model sha256
`f48abe457d483b19b854e1fdf6be4bb294c702f949f9d76d7c44a2274b0d5a6a`, optimizer sha256
`ae55b394…99af`; `ckpt-0000900` is also kept. Trained on the pilot `train` split;
evaluated on the pilot `val` split (same manifest).

The gradient norm before clipping was above 1.0 on **all 1,000 updates**: median
5.15, last-100 median 4.71, range 3.03–50.6. So clipping set the update magnitude
throughout. This is reported, not changed: D-030 fixed the setting before the run.

| update | mean train loss (window) | val masked loss | T = L final / value / trace / malformed | T = 4 final / value / trace / malformed |
| --- | --- | --- | --- | --- |
| 0 (pretrained) | — | 3.878 | 0 / 0 / 0 / 100% | 0 / 0 / 0 / 100% |
| 250 | | 0.909 | 13.9% / 22.4% / 11.1% / 0.7% | 9.7% / 20.7% / 6.3% / 0.7% |
| 500 | 0.781 (451–500) | 0.755 | 19.4% / 36.8% / 18.1% / 2.8% | 10.4% / 29.2% / 9.0% / 2.1% |
| 750 | | 0.671 | 23.6% / 42.7% / 22.2% / 1.4% | 16.7% / 37.0% / 16.0% / 1.4% |
| 1000 | 0.613 (951–1000) | 0.591 | 38.9% / 54.3% / 35.4% / 0.0% | 24.3% / 45.1% / 23.6% / 0.7% |

The mean training loss over updates 1–50 was 1.616. Validation loss and accuracy were
still improving at update 1,000, so the warmup reached its update cap without
converging. Whether to warm up longer before freezing the generator is left open
(handoff).

### Validation calibration (D-031)

Model: `warmup-pilot-v1/ckpt-0001000` (sha256 `f48abe45…5a6a`). Data: pilot `val`
only, verified against manifest `990ecd1d…1536`. Sampler: D-029. No test split was
loaded.

**Stage 1 screen** (`dlmc calibrate … --settings all --n-per-setting 32 --budgets
1,2,4,8,L --batch-size 32 --trace-examples 1 --trace-calls 4 --out
runs/calibration/screen-warmup1000`). It covered 18 settings × 32 problems × 5 budgets
and took 651 s. The last settings ran under the same cache-growth memory pressure;
their accuracies are unaffected (greedy decoding is deterministic), but their model
seconds are inflated. The full table is in `runs/calibration/screen-warmup1000/
summary.md`. Final-answer accuracy ranged from 59–91% (length 2, digits 1) down to
0–3% (length 6, digits 3) in both families. **Dataset rule (D-031):** neither family
is ≥ 98% everywhere or < 5% at T = L everywhere. The pilot dataset is kept: no new
manifest.

**Stage 2 selection** (`scripts/select_calibration_settings.py`, output in
`runs/calibration/screen-warmup1000/stage2_selection.txt`). The first version of the
script added the middle setting only when *no* setting was in band. D-031 says it is
added when fewer qualify than there are free slots. The script was corrected to match
the recorded rule **before** stage 2 ran. Selected per family: easiest (2, 1), hardest
(6, 3), in-band (arithmetic: (4, 1), (2, 2); programs: (2, 2), (2, 3)), and the middle
(4, 2).

**Stage 2** (`dlmc calibrate … --settings <10 settings> --n-per-setting 200 --budgets
1,2,4,8,L --batch-size 50 --trace-examples 2 --trace-calls 4 --out
runs/calibration/stage2-warmup1000`). It covered 2,000 problems × 5 budgets and took
1,404 s. One earlier attempt hit the same cache growth (63 GB process, full swap) and
was stopped after 42 of 50 cells. It is archived as
`runs/calibration/stage2-warmup1000.swap-attempt`; its accuracies for the shared cells
are identical. After the fix (cache release per generation batch) the process stayed
at about 9 GB.

Final-answer accuracy (n = 200 per cell). Every cell also records value, token and
trace accuracy, malformed rate and reasons, value errors per response, calls per
example and model seconds in `summary.md`/`summary.json`, with per-example records in
`records.jsonl`:

| family | length | digits | L | T=1 | T=2 | T=4 | T=8 | T=L | malformed (range) | model s at T=L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| arithmetic | 2 | 1 | 10 | 0.660 | 0.770 | 0.880 | 0.910 | 0.910 | 1.0–1.5% | 25.7 |
| arithmetic | 2 | 2 | 10 | 0.180 | 0.215 | 0.360 | 0.470 | 0.470 | 0.5% | 27.1 |
| arithmetic | 4 | 1 | 20 | 0.060 | 0.070 | 0.270 | 0.535 | 0.670 | 0.5–3.5% | 75.9 |
| arithmetic | 4 | 2 | 20 | 0.005 | 0.010 | 0.055 | 0.100 | 0.225 | 0–1.0% | 80.1 |
| arithmetic | 6 | 3 | 30 | 0.000 | 0.000 | 0.010 | 0.015 | 0.025 | 0–2.0% | 175.7 |
| programs | 2 | 1 | 10 | 0.695 | 0.790 | 0.860 | 0.880 | 0.880 | 1.5–4.0% | 30.0 |
| programs | 2 | 2 | 10 | 0.250 | 0.340 | 0.475 | 0.535 | 0.530 | 0.5% | 31.6 |
| programs | 2 | 3 | 10 | 0.185 | 0.215 | 0.350 | 0.405 | 0.400 | 0% | 33.2 |
| programs | 4 | 2 | 20 | 0.015 | 0.040 | 0.090 | 0.115 | 0.155 | 0–2.5% | 90.7 |
| programs | 6 | 3 | 30 | 0.010 | 0.010 | 0.015 | 0.015 | 0.020 | 0–9.0% | 193.7 |

Observations (calibration facts, not experimental results):
- More calls help in every setting, most for longer responses. T = 8 and T = L
  coincide for L = 10, where T = 8 reveals one token per call except for two calls.
- The only malformed reason observed is `negative_zero` (`" -0000"`). Malformed rates
  are 0–9%, highest at T = 1 in programs (6, 3).
- Wrong responses usually contain several wrong values: for example, 4.8 value errors
  per response in the (6, 3) settings, against 0.1–0.3 in (2, 1).
- Cells inside the 40–85% heuristic band: arithmetic (2, 1) at T = 1–2; arithmetic
  (2, 2) at T = 8/L; arithmetic (4, 1) at T = 8/L; programs (2, 1) at T = 1–2; programs
  (2, 2) at T = 4/8/L; programs (2, 3) at T = 8/L (0.405/0.400). The band is a heuristic only.
  No weights were changed to reach it, and no final primary budget is chosen here.
- Token-state traces for 2 examples per family at T = 4 are in `traces.txt` in each
  output directory. Quotas follow ⌈sL/T⌉ (for example 3, 2, 3, 2 for L = 10), no
  token is revised, and no mask remains after the last call.

### Tests and mutation check (final, after all code changes)

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` (outside the sandbox) | **216 passed** in 18.6 s (180 before milestone 3) |
| `pytest --co -m real_backbone` / `-m tokenizer` / `-m "not real_backbone and not tokenizer"` | 8 / 68 / 140 tests |
| `.venv/bin/python scripts/mutation_check_m3.py` | **12/12 mutants caught**. Each mutant is applied, the targeted tests run, then the source is restored; the targeted tests pass again afterwards (35 passed) |

Mutants: resume without mask-stream restore; sampler may revise tokens; candidate ties
→ largest ID; ranking by restricted probability; position ties → rightmost;
floor-based schedule; one mask rate per batch; maskable prompt; zero-mask batch steps
the scheduler; evaluation uses the training mask stream; LoRA leaves the base
trainable; sampler sees the reference length.

### Not run in milestone 3

- Any CUDA/A100 run: the Colab launcher is untested, and no A100 timing exists.
- Any `test_id`, `test_long` or `test_unseen` evaluation.
- A warmup longer than 1,000 updates, and any warmup hyperparameter search.
- Draft-bank collection, correction training, and any experimental arm (milestone 4
  onward).
- No human review.

## Warmup-continuation preparation (2026-11-17, Mac only)

Preparation for continuing the common warmup on Lonestar6 (D-033,
`docs/warmup_continuation.md`). Every check below ran on the local M2 Max. **Nothing
has run on TACC**: transfer, environment build, environment check, CUDA pilot, Ranch
backup, continuation and final evaluation are all PENDING, and no CUDA number exists.

### Software status

| Component | Status |
| --- | --- |
| D-033 trainer rules: training-update timer and cap, deadline/signal stop, persisted early-stopping state, protected checkpoints, monitoring subset by ID list, per-setting evaluation records | TESTED LOCALLY (tiny backbone); CLI path smoke-tested on the real backbone on CPU |
| CUDA precision settings and CUDA memory/environment reporting (`env.py`) | IMPLEMENTED; CUDA untested |
| `scripts/tacc/` Python helpers (preflight, fixture, expected batches, preflight comparison, status/resume guard, pilot report, calibration comparison) | TESTED LOCALLY on real files where noted below |
| `scripts/tacc/` shell and Slurm scripts | syntax and `shellcheck` only; not run (they refuse to run off Lonestar6) |

### Tests and mutation checks

| Command | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q` (outside the sandbox) | **225 passed** (216 before); 224 inside the sandbox (no MPS case) |
| `pytest --co -m "not real_backbone and not tokenizer"` | 149 tests |
| `tests/test_continuation.py` | 7 tests: monitoring-score weighting; patience 3 with a tie keeping the earlier checkpoint; selection state and timer surviving resume (re-basing and changed settings refused); time cap and deadline; protected checkpoints in local and durable directories; evaluation leaves training RNG streams and mode unchanged; config validation |
| `tests/test_tacc_scripts.py` | 2 tests: the resume guard on real tiny-backbone D-033 checkpoints (origin reuse, rollback to an older checkpoint, resuming a finished run, `--best` before finishing, and a corrupted file are all refused; `--best` returns the strictly best scheduled checkpoint), and the pilot report's gates and estimate arithmetic |
| `scripts/mutation_check_m3.py` | 12/12 mutants still caught |
| `scripts/mutation_check_d033.py` (new) | **7/7 caught**: ties counted as improvement; resume resets the timer; resume resets the early-stopping state; evaluation time charged to the training timer; monitoring score pooled instead of equal-weighted; deadline ignored; best checkpoint not protected |

One defect was found by the new tests and fixed: resuming into a run directory that did
not exist yet failed, because `Trainer.resume` wrote its event before the directory was
created. This is exactly the TACC case of a fresh `$SCRATCH` run directory resuming the
transferred checkpoint.

### Preflight of checkpoint 1000 (`scripts/tacc/preflight_checkpoint.py`, CPU, read-only)

Output: `runs/preflight/ckpt-0001000-mac.json` (sha256 `a09068e3…af5c`), **14/14 checks
passed**, 7 s.
- **File hashes.** Match `COMPLETE.json`.
- **Compatibility.** The resume-compatibility hash of `warmup_continuation.yaml`
  equals the checkpoint's (`d192b416…a6f6`); immutable-key differences: none; changed
  mutable sections: `checkpoint`, `eval`, `limits`, `run_name`.
- **Identities.** Data identity recomputed from the files; base-model files match.
- **Counters.** Update 1000, batches 1000, examples 32,000, zero-mask skips 0.
  `train_wall_seconds` is 5,688.05 in the checkpoint and 5,789.88 in `summary.json`
  (the summary adds the final evaluation and save).
- **Sampler.** Epoch 2, position 14,000 of 18,000.
- **Model.** 290 fp32 tensors, 494,032,768 parameters (`lm_head` tied, not stored);
  L2 distance from pretrained 17.3775, norm 1178.458.
- **Optimizer.** AdamW state for all 290 tensors at step 1000; fp32 state
  3,952,263,304 bytes.
- **Scheduler and learning rate.** `last_epoch` 1000; LR 3e-5 for every update to
  10,000 (no decay horizon).
- **RNG streams.** `data_order` and `train_mask` advanced; `eval` still at its seed.
  The global RNG state comes from MPS; on another device it is not restored (unused
  by training).
- **Predicted batches.** Batches 1001–1010 predicted from the CPU streams (mask
  digests `4696908e…`, …, `8b1442d0…`; masked tokens 339, 266, 297, 327, 299, 292,
  330, 374, 280, 256).
- **Gradient diagnostics** from `metrics.jsonl`:
  - pre-clip norm median 5.15 (p95 7.53, range 3.03–50.6); last 100 updates median
    4.71;
  - clip factor median 0.194;
  - clipped on 100% of updates.

  Not measured: post-clip per-tensor norms, per-step update norms, AdamW
  second-moment statistics, per-layer gradient norms.
- **MPS memory** from metrics: live 6.05 GB; driver 28.7–30.6 GB.

**Validation of the batch prediction.** The same preflight run on `ckpt-0000900`
(`--next-batches 100`) predicted every recorded batch 901–1000 of the warmup exactly:
indices, problem IDs, mask digests, mask rates, shapes and LR
(`check_expected_batches.py`: PASS).

### CUDA reference fixture (`scripts/tacc/cuda_fixture.py make`, CPU fp32)

Output: `runs/reference/cuda_pilot_fixture.{pt,json}` (sha256 `b5dc7081…3e74`,
`0f3c4855…fd24`), made from checkpoint 1000. Tolerances were fixed in the script
before any CUDA result, and are stored with the fixture.
- **Batch.** Shape [18, 134], 180 masked tokens, mask digest `419e0322…`.
- **Values.** Loss 0.6131474, pre-clip gradient norm 6.29932, LR 3e-5.
- **Generation margins.** Smallest committed top-1/top-2 margins in the greedy T = L
  generations range from 0.007 (programs 2/2) to 8.67 (arithmetic 2/1).
- **Self-check.** `check --device cpu` against itself passed with zero differences.

### Real-CLI smoke run (CPU, disposable APFS clone of checkpoint 1000)

`dlmc train --config configs/train/warmup_continuation.yaml --device cpu --resume
runs/cli_smoke/ckpt-0001000-clone --max-updates 1004`, with monitoring disabled by
`--set`, then SIGUSR1 sent after the first update log line.
- **Resume.** `cross_device: true`, `accelerator_rng_restored: false`.
- **Updates.** 1001 (loss 0.534) and 1002 (0.625). The signal arrived during update
  1002, which completed.
- **Stop.** `stop_reason signal_SIGUSR1`. `ckpt-0001002` verified, with timers
  `origin_update 1000` and `train_update_seconds 20.4`. No unscheduled evaluation.
- **Batches.** 1001–1002 matched the Mac preflight's predictions exactly.
- **Original.** The checkpoint's hashes were unchanged afterwards (re-hashed for the
  transfer table).

The bash wait/trap pattern of `continuation.slurm` was checked separately with a
stand-in process: the batch shell forwards SIGUSR1, and the trainer's exit status is
propagated with and without the signal.

### Helper scripts on real files

| Check | Result |
| --- | --- |
| `compare_preflight.py` Mac preflight vs itself / vs the `ckpt-0000900` preflight | PASS / FAIL with 23 differences (as intended) |
| `compare_calibrations.py` stage-2 run vs itself, with the MPS cross-device option | reproduces the stage-2 aggregates (T = L: 0.429) and all cell counts |
| `continuation_status.py --check-resume` on the real `ckpt-0001000` / on `ckpt-0000900` | accepted / refused (not a continuation checkpoint) |
| `bash -n` on every `scripts/tacc/*.sh`, `*.slurm` | no syntax errors |
| `shellcheck` | only unused-variable notices for variables used by the sourcing scripts, and bare `cd` covered by `set -e` |

### Environment pinning

`scripts/tacc/compile_lock.sh` (uv 0.13.9 on the Mac, `--python-platform
x86_64-manylinux_2_28 --python-version 3.14 --generate-hashes`) produced
`requirements-tacc-lock.txt`:
- torch 2.15.0, triton 3.9.0, CUDA 13.2 runtime packages and cuDNN 9.26.0.51;
- shared packages at the Mac versions (numpy 2.5.5, safetensors 0.8.0, tokenizers
  0.23.3, PyYAML 6.0.3, huggingface-hub 2.4.0, pytest 9.2.0).

PyPI publishes torch 2.15.0 for Linux only as the CUDA 13.2 build (driver ≥ R580).
The uv 0.13.9 Linux tarball sha256 (`a7fbaff2…71d5`) was taken from its GitHub
release. Lonestar6 rules were checked against the TACC user guides (Lonestar6,
Ranch):
- queue table and charge rates, with the 0.25 h minimum;
- no `--gres`;
- virtual environments on `$SCRATCH` and its 10-day purge;
- `$WORK` neither purged nor backed up;
- `$ARCHIVER`/`$ARCHIVE` and `scp` for Ranch.

### Not run

- Anything on TACC: no transfer, environment, CUDA run, pilot, backup, continuation or
  final evaluation. All CUDA numbers in the runbook are provisional estimates.
- The transfer staging itself (`stage_transfer.sh`) runs after the commit, because it
  needs a clean tree; its outcome is reported outside this file.
- No test-split evaluation, draft-bank collection or correction training. No human review.

## TACC compatibility fallback D-034 (2026-11-17, Mac only; approved by the user afterwards, remote validation PENDING)

**Remote evidence (reported by the user, not re-checked here).** Lonestar6 environment-check
job 3591308 at `944905e` (ASC26098, `gpu-a100-small`, `v330-007.ls6.tacc.utexas.edu`, one
A100-PCIE-40GB) with CPython 3.14.7, torch 2.15.0, CUDA 13.2: driver 570.195.03; the R580
guard stopped the job before its later checks. No pilot or continuation has run.
Checkpoint 1000 has a hash-verified Ranch backup.

**Remote validation of the fallback is PENDING**: nothing below ran on TACC, the fallback is
not installed, and no CUDA result exists.

### Checks performed on the Mac

| Check | Result |
| --- | --- |
| `download.pytorch.org/whl/cu126/torch/` index | lists `torch-2.14.1+cu126-cp314-cp314-manylinux_2_28_x86_64.whl` with sha256 `eb7564fd…834b` and core-metadata sha256 `4496899d…c1ad`; the downloaded metadata matches that hash |
| Torch metadata | `Requires-Python >=3.10`; Linux deps: `cuda-toolkit[…]==12.6.3`, `cuda-bindings>=12.9.4,<13`, `nvidia-cudnn-cu12==9.10.2.21`, `nvidia-cusparselt-cu12==0.7.1`, `nvidia-nccl-cu12==2.29.3`, `nvidia-nvshmem-cu12==3.4.5`, `nvidia-nvjitlink-cu12>=12.6.85,<13`, `triton~=3.8.0` |
| NVIDIA CUDA 12.6 Update 3 release notes | toolkit driver minimum on Linux x86_64 ≥ 560.35.05; nvJitLink component 12.6.85 |
| `scripts/tacc/compile_lock.sh` (uv 0.13.9, `x86_64-manylinux_2_28`, Python 3.14, `--generate-hashes`, PyPI index) | resolved 48 packages; without the nvJitLink pin it chose 12.9.86, so the input holds it at 12.6.85 |
| Install of the lock for the target platform (`uv pip install --target … --python-platform x86_64-manylinux_2_28 --python-version 3.14 --require-hashes --no-deps -r requirements-tacc-lock.txt`, the `setup_env.sh` flags) | all 48 wheels downloaded and hash-verified; torch reports `__version__ 2.14.1+cu126`, `cuda 12.6`, wheel tag `cp314-cp314-manylinux_2_28_x86_64` |
| Dependency closure of those 48 distributions for Linux / CPython 3.14.7 (markers evaluated, extras followed) | no missing or out-of-range requirement |
| `tests/test_tacc_env.py` (new, 8 tests) + `tests/test_tacc_scripts.py` | 10 passed: driver guard (570.195.03, 560.35.05, 580.65.06 accepted; 560.35.04, 555.42.06, unparsable refused); venv build record and pass markers bound to the lock sha256 (missing, foreign and unlabelled markers refused); lock consistency (torch URL and hash, no CUDA 13 packages, all NVIDIA libraries `-cu12`, Mac versions kept, constraints file = Mac lock without torch) |
| Mutation check of the guards (temporary edits to `tacc_env.sh`, restored) | 6/6 caught: major-only driver check, build-record check disabled, marker check ignoring the sha, torch version, driver minimum, cuDNN version |
| `bash -n` on every `scripts/tacc/*.sh`, `*.slurm`; `shellcheck` 0.11.0 | no syntax errors; only the notices recorded before (variables used by sourcing scripts, `local t0=$(…)` in `pilot.slurm`) plus `DLMC_CUDNN_VERSION`, which `env_check.slurm` uses |

The model code, configuration, data, checkpoint, tolerances and the Mac environment were
not changed, so the full suite and the D-033 mutation checks were not re-run.

### Not run

- Anything on TACC: update transfer, environment rebuild, environment check, CUDA pilot,
  continuation. Whether torch 2.14.1 loads the torch 2.15.0 checkpoint and reproduces the
  Mac preflight is tested only by the TACC environment check.
- Staging of the code-only update runs after the commit (it needs a clean tree); its
  outcome is reported outside this file.
