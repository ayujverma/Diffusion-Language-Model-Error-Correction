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
  CPU/MPS differences are about 1e-4 relative, which is expected kernel-level float32
  variation. Bitwise CPU/MPS identity is not expected. *(This wording is flagged for
  correction; see "Pending advisor follow-ups".)*

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

Four errors in one (budget, stage) bin with saved log c = (10, 20, 30, 40)
(arbitrary increasing scores), β = 0.5. The ranks are 1..4, so
r = (.125, .375, .625, .875) and w = 1 + .5(2r − 1) = **(0.625, 0.875, 1.125, 1.375)**,
mean 1 (`test_rank_weights_hand_values`). With fixed weights
(0.25, 1.75, 1.0) on three error rows and λ = 0.7, autograd's gradient on error row
i equals 0.7·w_i·(softmax(z_i) − onehot(y_i))/3 to 1e-14
(`test_autograd_matches_analytic_gradient`). *(The positive "log c" values in this
example are flagged for correction; see "Pending advisor follow-ups".)*

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

## Pending advisor follow-ups (open; none completed)

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
