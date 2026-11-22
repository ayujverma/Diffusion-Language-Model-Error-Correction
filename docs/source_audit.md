# Source and implementation audit

Milestone 1 deliverable. Written 2026-11-16; sections marked "added in milestone 2"
were added in the second session the same day. Every claim below cites what was read
or what was executed. "Verified by execution" means a command in
`docs/verification.md` produced the stated output. Paper details that have **not**
yet been checked against code are labelled as such.

Local copies of upstream code live under `runs/sources/` and `runs/hf/`, which are
git-ignored. Re-fetch them with the commands in the README.

## 1. Pinned sources

| Source | Identifier / revision | How it was checked |
| --- | --- | --- |
| Project proposal | `Diffusion_LM_Error_Correction_Project_Proposal.pdf` (2 pages) | Text extracted with pypdf and read in full |
| Build prompt | `Cursor_Diffusion_LM_Build_Prompt.txt` (rev. 16 Nov 2026; local only, not tracked) | Read in full |
| Open-dCoder-0.5B weights + remote code | HF `fredzzp/open-dcoder-0.5B` @ `d0d86d5b99960c05258bb1f8265dd91564dbac67` (last modified 2025-08-27) | Downloaded. sha256 of every file used is pinned in `src/diffusion_correction/models/open_dcoder.py::PINNED_SHA256`. Weights sha256 `59c1a005…7b6e` matches the CDLM training README's pinned-revision table |
| Open-dLLM (Open-dCoder trainer, `veomni`) | github.com/pengzhangzhi/Open-dLLM @ `5b9ba4d12c92be0010b99c61fc1129a89deeec04` | Cloned. Training loss, collator, and `sample.py` read. Its Qwen2 class was executed as the numerical reference |
| CDLM code | github.com/zhangshuibai/CDLM @ `86d108e2bb6dbc3f491d217aae04de4fc7c26ce3` (2026-10-02) | Cloned. `training/README.md`, `training/corrected_gradient/README.md`, `utils.py`, `training/requirements.txt` read |
| ProSeCo code | github.com/kuleshov-group/proseco @ `43ce881779f06f475b56d90a152acd65ddc74fe4` (2026-06-21) | Cloned. `diffusion.py` corrector loss (lines 550–596), `configs/config.yaml`, `llada/generate.py` sampler skimmed |
| MDLM | arXiv 2406.07524v2, "Simple and Effective Masked Diffusion Language Models" | Title/version confirmed via arXiv API. Not yet re-read in detail |
| ProSeCo paper | arXiv 2602.11590v4, "Learn from Your Mistakes: Self-Correcting Masked Diffusion Models" | Algorithm 1, §3.2, Appendix C.3 (Eq. 11) read |
| SCOPE paper | arXiv 2606.01026v1, "Revise, Don't Freeze: Sampler-Matched Training for Self-Correcting Masked Diffusion Language Models" | Abstract, §3, Appendix B (Algorithms 1–2, B.1–B.3) read |
| PAPL | arXiv 2509.23405v3, "Planner Aware Path Learning in Diffusion Language Models Training" | Abstract and §3.4 stabilised loss read |
| CDLM paper | arXiv 2512.15596v3, "Corrective Diffusion Language Models" | Title/version confirmed. Implementation details taken from the repo README, which says the implementation (not Eq. 1) is authoritative |
| DMax | arXiv 2604.08302v3, "DMax: Aggressive Parallel Decoding for dLLMs" | Title/version confirmed only (related work; no required arm) |
| ReMDM | arXiv 2503.00307v4, "Remasking Discrete Diffusion Models with Inference-Time Scaling" | Title/version confirmed only. Algorithm not yet audited (needed for the training-free remasking reference) |
| TinyGSM | arXiv 2312.09241v1 | Title confirmed only (optional reference) |
| SCOPE ResearchGate mirror, GSM8K, DeepMind Mathematics, PyTorch MPS notes | URLs in the build prompt | **Not opened in milestone 1** |

SCOPE code: the arXiv HTML text names no code repository. None was found or used,
so a SCOPE adapter will be a clean re-implementation from Appendix B and labelled
as an adaptation.

## 2. Backbone findings (Open-dCoder-0.5B)

### 2.1 Architecture and attention

- `config.json`: `Qwen2ForCausalLM`, 24 layers, hidden 896, 14 query heads / 2 KV
  heads (GQA), RoPE θ = 1e6 with no scaling, RMSNorm eps 1e-6, tied embeddings,
  vocabulary 151,936, checkpoint dtype bf16, `use_sliding_window: false`.
  `transformers_version: 4.51.3`.
- **The model card's loading path does not load the diffusion model.** `config.json`
  has no `auto_map`, so `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
  silently loads stock, causal `transformers` Qwen2. This was verified by execution:
  it returned class `transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM`
  without `diffusion_generate`. Its logits differed from the bidirectional reference
  by up to 30.67, and a right-context perturbation had exactly 0 effect, meaning it
  is causal. Open-dLLM's `sample.py` and CDLM's `utils.py` avoid this path by
  importing the class directly.
- **The Hub `modeling_qwen2.py` is not runnable as published.** This was verified by
  execution on transformers 4.54.1 and 5.23.0:
  1. Import fails because `@replace_return_docstrings` is applied to a `forward`
     that has no docstring (`AttributeError: 'NoneType' object has no attribute 'split'`).
  2. With that decorator neutralised, `forward` fails because the file calls
     `attention_interface(query, key, value, …)` without the leading `module`
     argument that `ALL_ATTENTION_FUNCTIONS` requires.
- **The working upstream implementation** is `veomni.models.transformers.qwen2.modeling_qwen2.Qwen2ForCausalLM`
  in Open-dLLM, which CDLM's evaluation loads for any model name containing
  `open-dcoder`. Executed with transformers 4.54.1 (CDLM's verified version):
  - SDPA, `attention_mask=None`, `is_causal=False`: **bidirectional**. Changing the
    last token changed an earlier output row by up to 5.35.
  - SDPA with an all-ones 2-D mask: bidirectional (same 5.35).
  - SDPA with a 2-D mask containing **one padded key**: **silently causal** (effect 0.0).
    `_update_causal_mask` ignores `is_causal` and builds a 4-D causal mask whenever
    the mask cannot be dropped.
  - `attn_implementation="eager"`: **causal even with `is_causal=False`** (effect 0.0),
    for the same reason.
  - Upstream generation sets `gen_attention_mask = (x != config.pad_token_id)` only if
    `config.pad_token_id` is set. `config.json` has no `pad_token_id`, so the
    published sampler runs with no mask. Batched generation with padding is therefore
    unsupported upstream. Even if it were supported, pad = EOS = 151643, so an
    ID-based mask would also hide real EOS tokens.
- **Our implementation** (`src/diffusion_correction/models/qwen2_bidir.py`) is a
  self-contained PyTorch re-implementation of the same forward pass. Attention is
  `F.scaled_dot_product_attention(..., is_causal=False)` with an explicit boolean
  key-padding mask built from lengths, never from token IDs. There is no KV cache
  and no `transformers` dependency. On the reference inputs (2 × 46 tokens,
  float32, CPU) its raw logits equal the `veomni` SDPA logits **exactly** (max |diff|
  = 0.0). Perturbing one norm weight by a factor of (1 + 1e-6) produced a
  4.8e-5 difference, so the comparison has resolution.

### 2.2 Logit alignment

- Evidence for the shift, from three independent places:
  - Hub `generation_utils.py`: `logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)`
    with the comment "Shift logits to predict the next token, aligning with training".
  - Open-dLLM training: `labels[..., 1:]` against `hidden_states[..., :-1, :]`.
  - CDLM README: "Open-dCoder keeps the autoregressive shift: the output at position
    i − 1 predicts token i. As a result, the first token of each packed chunk is
    never a target."
- Our rule (`models/alignment.py::align_logits`) is that `aligned[:, i] = raw[:, i-1]`,
  and position 0 is never a target. This is enforced by `assert_targets_predictable`
  and by requiring a non-empty prompt. Upstream fills row 0 with `raw[:, 0]`; we fill
  it with zeros so it has no gradient path. The value is unused either way.

### 2.3 Special tokens and tokenizer

| Fact | Evidence |
| --- | --- |
| Mask token is `<M>` = **151665** | `special_tokens_map.json`; `tokenizer.json` added token (special); Open-dLLM `train_torch.py` adds `<M>` and passes `tokenizer.mask_token_id`; CDLM README ("mask token (`<M>`, id 151665)"); CDLM `utils.py` hard-codes 151665 |
| `generation_config.json` says `mask_token_id: 151643` | This is `<|endoftext|>` (= pad = EOS). It contradicts every training and evaluation source above, so we treat it as stale and do not use it (D-003). A test asserts both facts |
| pad = EOS = `<|endoftext|>` = 151643; no BOS (`bos_token: null`, `add_bos_token: false`) | tokenizer files; `config.json` `eos_token_id` |
| Digits tokenize one digit per token; punctuation can merge with neighbours (`";b"` is one token) | Executed (`tests/test_open_dcoder.py::test_tokenizer_digits_are_single_tokens`). Fixed character width does not imply fixed token width, which matters for the milestone-2 layout |
| *(added in milestone 2)* Sign merges depend on the neighbour: `"=-"` (10829), `",+"` (63454), `",-"` (4999) and `"(+"` are single tokens, while `"=+"` is two; `" +"` (488) and `" -"` (481) are single tokens; `"\n+"` is `"Ċ"` + `"+"` | Executed (`tests/test_data_format.py`). The pre-tokenizer regex (`tokenizer.json`) has a ` ?[^\s\p{L}\p{N}]+` branch, which joins runs of punctuation and allows one leading space. This is why the response format puts a space before every sign (D-018) |
| *(added)* No printable single-character prompt ending merges with a following `" +"`/`" -"` | Executed (`test_leading_space_sign_never_merges_with_any_prompt_ending`) |
| *(added)* The `tokenizers` wrapper, transformers' `Qwen2TokenizerFast`, and the slow pure-Python `Qwen2Tokenizer` (vocab.json + merges.txt) give identical IDs on 83,713 strings | Executed (`scripts/reference_tokenizer.py`, pilot data). The slow tokenizer does not read `tokenizer.json`, so this is an independent check of the BPE |
| *(added)* The output vocabulary is 151,936 rows, but the tokenizer has 151,666 tokens; IDs 151,666 and above have no token | `config.json` vs. `tokenizer.get_vocab_size(with_added_tokens=True)`. The checker reports such IDs as `unknown_token_id` |

### 2.4 Upstream training objective (what we deliberately do not inherit)

Open-dLLM `Qwen2ForCausalLM.forward` (confirmed by the CDLM README):

    L = (1/|S|) Σ_{i∈S} ℓ_i  +  (1/|S|) Σ_{i∈S} sg(exp(−ℓ_i)) · ℓ_i / t_i

The mask ratio is t ~ U(0,1), clamped to [1/500, 1 − 1/500], and each token is
masked independently. There is no 1/t weight on the main term. The second ("path")
term is a detached, confidence-weighted CE with the same form as PAPL's
planner-weighted loss; PAPL shares an author with Open-dLLM. That the two are
exactly equivalent has **not** been verified.

Both terms use liger-kernel's `LigerFusedLinearCrossEntropyLoss(reduction="none")`.
The CDLM README documents that in liger-kernel 0.5.8 this kernel's backward
"scales every position's gradient by the upstream gradient of the micro-batch's
first row", so per-token weights are not applied. CDLM pins 0.5.8, and every
released CDLM 0.5B checkpoint was trained this way. Open-dLLM pins
`liger-kernel>=0.5.5`. **Unresolved:** which liger version trained the base
Open-dCoder checkpoint, and so whether its path term was effectively applied.

Our losses use plain PyTorch `log_softmax` + `gather`. Our masked loss has no path
term and no 1/t weight (D-006, clarified by D-017).

*Added in milestone 2 (read in `veomni/models/transformers/qwen2/modeling_qwen2.py`
lines 1036–1057 and `veomni/data/data_collator.py` lines 158–214, Open-dLLM `5b9ba4d`):*
- The plain term is `mdm_loss = (token_loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)`
  over the flattened (packed) micro-batch. It is therefore a **pooled** mean over all
  masked tokens, the same estimator as our L_mask. With no masked token it is 0.
- The collator draws one `mask_ratio = rand(1).clamp(1/500, 1 − 1/500)` per packed
  sequence and masks each token independently (`rand_like(ids) < mask_ratio`). Labels
  are kept only where the input is `<M>`. A sequence can therefore draw zero masked
  tokens.
- The upstream path without labels (`labels=None`) returns `lm_head(hidden_states)`
  logits with no loss. That path was used for the gradient comparison (D-016).

### 2.5 Upstream numeric casts (added in milestone 2)

Upstream `Qwen2RMSNorm.forward` always casts hidden states to float32, and
`Qwen2RotaryEmbedding.forward` always computes cos/sin in float32 (lines 278–283 and
386–405), whatever the activation dtype. Ours does the same for bf16/fp16 and float32,
but keeps float64 in float64. This is the only cause of the 2.7e-6 to 5.4e-6
float64 gradient differences: matching both casts gives 3.4e-15 (V, "Milestone-1
follow-ups").

## 3. Prior methods: published rules vs. our adaptations

### 3.1 ProSeCo

Published (paper Algorithm 1 and code `diffusion.py`):
- An MDM pass on z_t with loss weight α̇_t/(1−α_t) (continuous time; code
  `dsigma / expm1(sigma)`, which is 1/t for the log-linear schedule).
- The corrector input is `sg(argmax x_θ(z_t))`. The paper sets it at **all**
  positions (Alg. 1 line 9). The code default (`use_model_outputs_as_corrector_input: False`)
  instead takes argmax at masked positions and keeps x₀ at visible ones. These agree
  under MDLM's SUBS "copy-over" parameterisation but **not** for Open-dCoder, which
  has no copy-over (decision deferred to the adapter milestone, Q-4).
- A second forward pass with gradient. Self-correction CE at all positions, with the
  same 1/t weight, and the mask logit set to −∞ on the corrector output.
- The "error-prioritized" variant (paper App. C.3 Eq. 11, code flag
  `corrector_loss_errors_upweighted`): wrong draft positions get −log p(x₀), and
  already-correct positions get −p(x₀) (probability, **not** log probability). The
  paper used this only for unconditional OWT text. The basic all-position-CE
  variant and this variant must keep distinct names.
- Sampler (`llada/generate.py`): corrector loops every ω steps, with up to S
  fixed-point iterations of argmax over the active region. Each iteration counts as
  one NFE.

Our adaptation (training still not implemented; inference sampler is D-049):
Open-dCoder backbone, our tasks, our prompt/response layout and shift; Open-dCoder's
U(0,1)-clamped mask ratio unless the adapter milestone decides otherwise. Online
draft construction makes the training objective a **family-B** comparison, not the
shared-draft control. The milestone-6 sampler (`sampling/proseco.py`) is the shared
inference decoder only. It is not Family A and not `proseco_gen_conf`.

ProSeCo's released sampler is LLaDA, not Open-dCoder. `llada/generate.py` at the
pinned commit loads `GSAI-ML/LLaDA-8B-Instruct`. LLaDA uses the SUBS copy-over
parameterisation: at a visible position the model predicts the token that is
already there. Open-dCoder has no copy-over and keeps the Qwen2 shift (position
i−1 predicts token i). D-050 records why this project stays on Open-dCoder. The
milestone-6 sampler copies ProSeCo's two-sequence structure (masked main state,
temporary complete draft, fixed-point argmax) and changes the schedule, the
allowed-token argmax, and the lack of EOS blocks, as listed in D-049.

### 3.2 SCOPE + D3IM

Published (Appendix B):
- D3IM (Alg. 1): at each step, predict all active positions, set x₀ = argmax and
  c = max softmax, keep the top-K(t) = ⌊L·t/T⌋ positions, and mask the rest
  ("clean slate"). Semi-AR blocks freeze earlier blocks. During non-final steps,
  EOS/EOT confidence is set to −∞. At the final step the whole block is accepted.
- SCOPE (Alg. 2, B.3): r ~ Beta(2,2) clamped to [0.1, 0.9]; prompt never masked. A
  no-grad pass draws x̂ ~ Cat(softmax(z/τ)) with τ = 1.5; confidence comes from
  τ = 1 logits. Commit the top-ρ (ρ = 0.3) of masked positions per sample with
  K_b = max(1, ⌊ρ|M_b|⌋); the rest stay masked. Loss is the mean CE over C ∪ M
  (clean visible positions excluded). With probability 0.5 the step is replaced by
  standard MDM training.
- **Ambiguity Q-5:** Alg. 2 line 4 says only "confidence(ẑ) // temperature = 1 logits".
  It does not say whether this is the max probability (as in D3IM) or the
  probability of the sampled x̂. No code was found to resolve it.
- D3IM is that paper's sampler, not a separate model. The audit does not record
  a SCOPE training backbone and does not identify it with LLaDA. Our inference
  path (D-049) runs D3IM on Open-dCoder. D-050.

### 3.3 CDLM (related work; reference only)

The mixture objective and the documented gradient bug are as summarised in §2.4.
Original released checkpoints and corrected-gradient retraining are different
baselines (CDLM's own `corrected_gradient/README.md` reports both). We use neither
as a required arm.

### 3.4 PAPL (related; informs the confidence-weighting discussion)

Stabilised loss: masked CE with weights (1/(L−k))(1 + α w_i), where w_i ∝ a
softmax-temperature planner over denoiser confidence, with planner gradients
detached. PAPL weights *masked-position denoising* by the *current* model's
confidence. Our proposal weights *visible draft errors* by *saved generation-time*
confidence. These are different, and the report must cite PAPL as related
confidence-weighted training.

## 4. Requirements implemented, and where

| Requirement (build prompt §) | Implementation | Status |
| --- | --- | --- |
| Source audit (§2) | this file | done |
| Explicit bidirectional attention, padding mask, no causal mask (§4) | `models/qwen2_bidir.py` | TESTED LOCALLY + TESTED ON REAL BACKBONE |
| Single logits→target mapping (§4) | `models/alignment.py::align_logits` | TESTED LOCALLY + REAL |
| Token IDs from pinned config, none hard-coded unverified (§4) | `models/open_dcoder.py::resolve_token_spec` | TESTED ON REAL BACKBONE |
| Explicit prompt/response/pad/editable masks (§4) | `models/alignment.py::SequenceLayout`, `build_layout` | TESTED LOCALLY |
| Inference never receives labels; losses outside the model (§4) | `DiffusionAdapter.aligned_logits` has no label argument | IMPLEMENTED (by construction) |
| No KV cache (§4) | `qwen2_bidir.py` | IMPLEMENTED |
| Tiny CPU backbone with the same contract (§4) | `models/tiny.py` | TESTED LOCALLY |
| doctor: env, device, memory, precision, revisions, real forward/loss/backward/step (§4) | `cli.py::cmd_doctor` | TESTED ON REAL BACKBONE (CPU fp32, MPS fp32) |
| Unreduced plain-PyTorch CE; no fused path (§2, §7) | `losses/token_ce.py` | TESTED LOCALLY |
| L = L_mask + λL_error + μL_keep with safe empty sets (§7) | `losses/correction.py`, `losses/masked.py` | TESTED LOCALLY |
| Rank weights, bins, midranks, normalisation, all 6 rules, diagnostics (§7) | `losses/weighting.py` | TESTED LOCALLY |
| Gate B (§11) | `tests/test_losses.py`, `tests/test_weighting.py` | passed except items listed in verification.md |
| *(follow-up)* Gradient equivalence with upstream under one external loss; complete loading and tying | `scripts/reference_gradients.py`; `test_checkpoint_loads_completely_and_embeddings_stay_tied` | TESTED ON REAL BACKBONE (D-016) |

### Milestone 2 (§5, §11 Gate A data items, §12 inspect-example)

| Requirement | Implementation | Status |
| --- | --- | --- |
| Arithmetic chains, 2/4/6 ops, 1–3 digits, +/−/×small, explicit bounds, exact values (§5) | `data/arithmetic.py` | TESTED LOCALLY |
| Restricted program language plus deterministic interpreter, independently tested; no execution of model output (§5) | `data/programs.py` | TESTED LOCALLY |
| Fixed-slot numeric format verified on the real tokenizer; token→value spans; layout independent of answers (§5) | `data/format.py`, `data/tasks.py` | TESTED LOCALLY (real tokenizer) |
| Checker: token vs. value levels separate; malformed counted; no zipping or dropping (§5) | `data/checker.py` | TESTED LOCALLY |
| Five splits, de-duplicated by normalized problem plus inputs, explicit held-out compositions, hashed manifest (§5) | `data/splits.py`, `configs/data/*.yaml` | TESTED LOCALLY |
| Verification rebuilds composition signature, held-out status and problem identity from source (advisor review; D-024) | `data/structure.py`, `splits.verify_dataset` | TESTED LOCALLY |
| inspect-example (§12) | `data/inspect.py`, `dlmc inspect-example` | TESTED LOCALLY (fields from milestones 4–5 print n/a) |
| Difficulty-calibration command (§5) | `evaluation/calibrate.py`, `dlmc calibrate` (milestone 3) | TESTED ON REAL BACKBONE |

### Milestone 3 (§4 runtime pilot, §6 warmup, §11 Gate C, §5 calibration)

| Requirement | Implementation | Status |
| --- | --- | --- |
| Mask policy recorded before training; per-example clamped-uniform t, Bernoulli response-only masking, pooled CE, zero-mask skip (D-028) | `training/masking.py`, `training/trainer.py::train_step` | TESTED LOCALLY + used on real backbone |
| Separate reproducible RNG streams (data order, train masks, eval) | `training/rng.py` | TESTED LOCALLY + TESTED ON REAL BACKBONE (eval at update 0 leaves the training losses unchanged) |
| Training from verified dataset files only | `training/data.py::load_verified_examples`, `run.prepare_trainer` (`verify_dataset`) | TESTED LOCALLY + REAL |
| Full fine-tuning vs. LoRA, bf16 vs. fp32, timing/memory pilot (§4) | `training/lora.py`, `evaluation/benchmark.py`, `evaluation/precision.py` | TESTED ON REAL BACKBONE (MPS) |
| Resumable atomic checkpoints, compatibility checks, durable copy (§6) | `training/checkpoint.py`, `Trainer.save/resume`, `find_resume_checkpoint` | TESTED LOCALLY + TESTED ON REAL BACKBONE |
| Monotone sampler with call counting, traces, observer hook (D-029) | `sampling/monotone.py` | TESTED LOCALLY + TESTED ON REAL BACKBONE |
| Gate C overfit and resume trajectory (§11) | `configs/train/overfit32.yaml`, `resume_check.yaml`, `evaluation/compare.py` | passed (verification.md, "Gate C") |
| Colab launcher using the same package | `notebooks/colab_runner.ipynb` | IMPLEMENTED (not run) |

Not reused: the DeepMind Mathematics generator, GSM8K and TinyGSM (build prompt §5
lists them as optional references). Our generators are written from scratch, so there
is no third-party data or code to attribute.

## 5. Unresolved questions

- **Q-1** Which liger-kernel version trained the base Open-dCoder checkpoint? This
  determines whether the base model's path term was effectively applied. It does not
  affect our training (we use plain CE) but matters for describing the starting point.
- **Q-2** *(resolved by D-018)* Open-dCoder's EOS = pad = 151643. Responses have a
  public fixed length and contain no EOS. An EOS/pad ID inside a response is a
  malformed special token, and attention masks stay length-based.
- **Q-8** *(resolved by D-028)* Precise mask-rate sampling policy and zero-mask
  handling for L_mask (D-017). Upstream's clamped-uniform rate is kept, but drawn per
  example with Bernoulli masking of response tokens only. Zero-mask examples are kept,
  and all-zero minibatches skip the optimizer step.
- **Q-3** *(resolved for MPS by D-030)* Precision and trainable parameters. bf16
  autocast passes the screening tolerances on MPS but is slower; pure bf16 parameters
  fail argmax agreement. The choice is fp32 full fine-tuning. The A100 is still
  unmeasured, and any precision change there needs its own comparison and entry.
- **Q-4** ProSeCo corrector input on a non-copy-over backbone: argmax at all
  positions (paper) vs. argmax only at masked positions (code default)? Still open
  for training (milestone 7). D-049 specifies the inference sampler only: the
  corrector rewrites every editable position of a temporary complete draft.
- **Q-5** SCOPE commit confidence: max probability vs. probability of the sampled token?
  Still open for SCOPE training. D-049's D3IM sampler scores the new allowed
  candidate by its float32 full-vocabulary probability. That does not resolve Q-5.
- **Q-6** ReMDM adaptation to Open-dCoder (schedule, σ_t choice) is not yet audited.
  D-049's `extra_unmask` control is not that adaptation.
- **Q-7** Whether the pretrained model's visible-token copying (L_keep ≈ 6e-5 and
  L_error ≈ 15.8 on one doctor example) is typical. This is a single example, not
  evidence. Measure it in the post-warmup signal diagnostic.
