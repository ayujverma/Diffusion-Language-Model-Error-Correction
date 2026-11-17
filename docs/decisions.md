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
