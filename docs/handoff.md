# Handoff for a fresh session

As of 2026-11-17. **Milestones 1, 2 and 3 are complete.** Milestone 3 (masked warmup
and difficulty calibration) was completed on the local M2 Max: its software is
TESTED ON REAL BACKBONE, Gate C passed, the bounded common warmup ran to its cap, and
the validation-only calibration is done. The common warmup **reached its 1,000-update
cap while still improving**. The user chose to continue it on TACC Lonestar6 (D-033).
Local preparation for that is complete: code, scripts, transfer manifest and runbook
`docs/warmup_continuation.md`. **Nothing has run on TACC yet.** The transfer, Ranch
backup, environment check, CUDA pilot, continuation and final comparison are pending,
and the user submits them. Milestone 4 has **not** been started.
No experimental arm has been trained, and no test split has been evaluated. This
file should be enough to continue. The detailed records are in `docs/decisions.md`,
`docs/verification.md`, `docs/source_audit.md`, `docs/implementation_plan.md` and
`docs/data_format.md`; if they disagree with this file, they win.

## 1. Inputs (full spec, stored in the repo)

- Build prompt: the approved implementation and experiment protocol, revised
  16 Nov 2026. It is in `docs/inputs/Cursor_Diffusion_LM_Build_Prompt.txt`, tracked
  unchanged since D-025. No other tracked document contains the complete protocol.
- Proposal (scientific scope, 2 pages):
  `docs/inputs/Diffusion_LM_Error_Correction_Project_Proposal.pdf`. Cursor's PDF
  reader fails on it, so extract the text with `pypdf` in a throwaway environment.

Both are byte-identical copies of the originals (sha256 values in D-013). Read the
build prompt before writing code. Sections 8 (draft bank), 9 (signals), 11 (Gate D)
and 12 govern milestone 4.

## 2. Project summary

Working title: *Does Generation-Time Confidence Improve Diffusion Self-Correction?*
The author is Ayuj Verma (CS 395T).

**Question.** When training drafts are held fixed, does weighting wrong tokens by
their saved generation-time confidence improve correction beyond uniform
weighting, generation-time correct-target difficulty, and current correction
difficulty?

**Signals at a wrong draft token i:**
- c_i: the frozen generator's probability of the wrong token it committed.
- q_i: the same distribution's probability of the correct token at that step;
  d_gen = −log q_i.
- d_now_i: the current corrector's cross-entropy on the correct token.

These are four different things (uniform, c, d_gen, d_now) and must stay distinct.

**Design.**
1. Masked fine-tuning warmup of the pretrained Open-dCoder-0.5B.
2. A frozen copy of the warmed model builds an immutable draft bank.
3. Every correction arm starts from the same checkpoint with identical drafts and
   hyperparameters; only the error-weighting rule changes.

**Required arms:**
1. compute-matched ordinary masked fine-tuning;
2. uniform;
3. high c;
4. current loss;
5. ProSeCo error-prioritized, adapted;
6. SCOPE with D3IM, adapted;
7. low c;
8. shuffled c;
9. d_gen.

Arms 5 and 6 form a separate comparison family. The primary outcome is
final-answer accuracy at a validation-selected fixed model-call budget. Extensions
stay optional.

**Rules.**
- Do not change the question or drop arms; record conflicts in `decisions.md`.
- Do not launch draft collection, correction training, the grid, a longer warmup,
  or paid hardware without the user's go-ahead. Do not invent results. The 40–85%
  band is a heuristic, not a target; never alter weights to manufacture errors.
- Never evaluate `test_id`, `test_long` or `test_unseen` for any selection decision
  (`dlmc calibrate` refuses them).
- Label every result IMPLEMENTED / TESTED LOCALLY / TESTED ON REAL BACKBONE /
  EXPERIMENT COMPLETED. Keep software verification separate from scientific results.
- No credentials in the repo.
- Do not record human review that has not happened. **None has happened so far.**
- Keep `decisions.md`, `verification.md` and `ai_usage.md` current. Supersede
  decisions with new entries rather than editing them.
- Ask the user before git commits.

## 3. Milestone status

| Milestone | Status |
| --- | --- |
| 1: source audit, adapter, losses, gradient checks | DONE |
| Milestone-1 advisor follow-ups | DONE (D-016, D-017) |
| 2: tasks, format, checker, splits, inspect-example | DONE and closed as software (D-018 to D-021, D-024). Pilot grid still labelled provisional (D-019, D-032) |
| 3: masked warmup and difficulty calibration | **DONE** (D-028 to D-032). Software TESTED ON REAL BACKBONE; Gate C passed; warmup ran to its 1,000-update cap (not converged); validation calibration done; pilot dataset kept |
| Warmup continuation on Lonestar6 (D-033) | **PREPARED, NOT RUN.** Code and scripts TESTED LOCALLY; every TACC step pending (`docs/warmup_continuation.md`) |
| 4 to 8 | not started |

**Milestone-3 completed items:**
- Mask policy (D-028), sampler policy (D-029), precision and trainable parameters
  (D-030) and the calibration protocol (D-031), each recorded before its code or run.
- Trainer with atomic, resumable checkpoints and a verified durable copy.
- Precision check, timing pilot, and LoRA vs. full fine-tuning feasibility runs.
- Gate C: the overfit reached 100% trace accuracy at update 250; the resume
  trajectory falls within the device noise floor, including a restore from the
  durable copy.
- Common warmup `warmup-pilot-v1`: 1,000 updates, 5,790 s.
- Calibration: an 18-setting screen, then 10 settings × 200 val problems × 5 budgets.
- `paper/methods.md` draft (no results).

**Milestone-3 incomplete or deferred items:**
- The warmup is not converged. The user chose to continue it (D-033); the continuation
  is prepared but has not run.
- The final primary budget is not chosen (deferred to milestone 4/5 by D-031).
- The Colab launcher has not been run; no A100 measurement exists.
- No human review.

## 4. Next steps

**First: the warmup continuation (D-033).** Follow `docs/warmup_continuation.md` §4 in
order. The user authenticates and submits every job:
1. stage and copy the transfer from the Mac;
2. verify and install on ls6;
3. back up checkpoint 1000 to Ranch;
4. build the environment;
5. run `env_check.slurm`;
6. run `pilot.slurm`;
7. run `continuation.slurm` from the original checkpoint 1000;
8. run `final_eval.slurm`.

Each step's pass condition is in the runbook (§10 lists those required before the
continuation). Record CUDA results in `verification.md` only after the job outputs
exist. After the final comparison, the user decides which checkpoint becomes the
milestone-4 generator, and a new decisions entry records it with its sha256.

**Then milestone 4: immutable draft bank and signal diagnostics.** Full list in
`docs/implementation_plan.md`. Do not start it before that generator decision. Then:
1. Freeze a generator copy. Collect drafts with the D-029 sampler through its
   observer hook (`sampling.monotone.monotone_unmask(..., observer=...)`): per
   committed token, the commit call, the mask fraction, and float32 log c and log q
   from the **same** forward pass. The observer receives the aligned float32 logits
   before the commit, and its return value is ignored.
2. Gate D observer checks: changing reference targets changes only analysis fields,
   never tokens or calls.
3. Bank validation and hashes; separate validation and test banks.
4. Signal diagnostic: c, −log q and step-0 correction loss by budget, stage, position
   and difficulty.
The budget used for bank collection must be chosen and recorded first. The
calibration table (`verification.md`) is the input to that choice.

## 5. Files

| File | Responsibility |
| --- | --- |
| `src/diffusion_correction/models/alignment.py` | `align_logits` (the only shift rule: `aligned[:, i] = raw[:, i−1]`), `SequenceLayout`, `build_layout` (masks from lengths, never token IDs) |
| `models/qwen2_bidir.py` | Own bidirectional Qwen2 forward pass (SDPA `is_causal=False`, key-padding mask, no KV cache). Apache-2.0 attribution in the header |
| `models/adapter.py` | `DiffusionAdapter.raw_logits` / `.aligned_logits(layout)`; takes no labels |
| `models/open_dcoder.py` | Pinned revision and sha256 values, `download`, `verify_files`, `resolve_token_spec`, `load_open_dcoder` (strict keys, tied embeddings), `load_tokenizer` |
| `models/tiny.py` | Random tiny backbone for software checks |
| `losses/` | Plain-PyTorch CE; L_mask (pooled, D-017); L_error, L_keep; `combined_objective`; rank weights and diagnostics |
| `data/arithmetic.py` | Arithmetic chains: `ArithmeticSpec`, `make_problem`, `sample_chain`, exact evaluation |
| `data/programs.py` | Restricted language: lexer, parser, validator, interpreter (`run_program`), `normalized_key`, `sample_program` |
| `data/format.py` | `ResponseFormat` (sign + 4 digits per value, no EOS), `TokenizedFormat` (IDs from the tokenizer, `parse_slot`), `encode_and_verify` |
| `data/tasks.py` | `render_prompt` (value-free fields only), `Example` record, `build_example` |
| `data/checker.py` | `check_response`, `aggregate`: token, value, final and trace levels; malformed-output policy |
| `data/splits.py` | `DataConfig`, `generate_splits`, `write_splits` (manifest), `load_split`, `verify_dataset`, `assert_grouped_by_problem` |
| `data/structure.py` | `derive_problem`: rebuilds a record's problem from its source, enforcing the generated structure; verification derives signature, held-out status and identity from it (D-024) |
| `data/fixtures.py`, `data/inspect.py` | Hand-checked fixtures; the `inspect-example` table |
| `cli.py`, `env.py` | `dlmc doctor`, `download-backbone`, `generate-data`, `verify-data`, `inspect-example`; provenance |
| `configs/data/smoke.yaml`, `pilot.yaml` | Data configs (the pilot is provisional) |
| `scripts/reference_logits.py`, `reference_gradients.py`, `reference_tokenizer.py` | Upstream / transformers references; run in `.venv-ref` |
| `training/rng.py`, `masking.py` | Named CPU RNG streams (data_order, train_mask, eval); D-028 mask sampling |
| `training/data.py` | `load_verified_examples` (manifest and split hashes), `select_subset`, `select_per_setting`, `EpochSampler`, `collate_targets` |
| `training/lora.py` | `LoRALinear`, `apply_lora` (freezes the base), adapter state I/O |
| `training/checkpoint.py` | Atomic checkpoint directories with `COMPLETE.json` hashes, `latest`, pruning, `copy_to_durable` (verified) |
| `training/config.py`, `model_io.py`, `run.py`, `trainer.py` | Config defaults and resume-compatibility hash; model building and loading; `Trainer` (step, eval, save, resume, train loop) |
| `sampling/monotone.py` | D-029 sampler: `reveal_schedule`, `allowed_token_mask`, `monotone_unmask` (observer, traces, call counts) |
| `evaluation/generation.py`, `calibrate.py` | Grouped generation plus checker scoring; validation-only calibration with traces |
| `evaluation/benchmark.py`, `precision.py`, `compare.py` | Timing/memory pilot; bf16 vs. fp32 check; run-trajectory comparison |
| `configs/train/*.yaml` | `warmup_pilot.yaml` (common warmup), `overfit32.yaml` (Gate C), `resume_check.yaml` (Gate C) |
| `notebooks/colab_runner.ipynb` | Thin CUDA launcher for the same commands (not run) |
| `scripts/gatec_inspect.py`, `select_calibration_settings.py`, `mutation_check_m3.py`, `mutation_check_d033.py` | Gate C inspection; D-031 stage-2 selection; milestone-3 and D-033 mutation checks |
| `configs/train/warmup_continuation.yaml`, `configs/eval/monitor_val_stage2_first50.json` | D-033 continuation config (resume-compatible with `warmup_pilot.yaml`); the 500-problem monitoring subset |
| `scripts/tacc/` | Lonestar6 runbook scripts (`docs/warmup_continuation.md`): `tacc_env.sh` (shared settings and checks), `stage_transfer.sh` (Mac), `install_transfer.sh`, `setup_env.sh`, `backup_to_ranch.sh`, `env_check.slurm`, `pilot.slurm`, `continuation.slurm`, `final_eval.slurm`; Python helpers `preflight_checkpoint.py`, `cuda_fixture.py`, `check_expected_batches.py`, `compare_preflight.py`, `continuation_status.py`, `pilot_report.py`, `compare_calibrations.py`, `make_monitor_subset.py`; Linux lock `requirements-tacc-lock.txt` (`compile_lock.sh`) |
| `paper/methods.md` | Methods draft from verified code |

Tests (225 in total with MPS available; 224 inside the sandbox):

| File | Tests | Notes |
| --- | --- | --- |
| `test_alignment.py` | 8 | |
| `test_losses.py` | 13 | |
| `test_weighting.py` | 11 | |
| `test_open_dcoder.py` | 7 | marker `real_backbone`; 6 without MPS |
| `test_data_tasks.py` | 39 | |
| `test_data_format.py` | 48 | |
| `test_data_structure.py` | 34 | hand-specified signatures, structure rejections, identity |
| `test_data_splits.py` | 20 | includes 9 tamper tests with re-hashed manifests |
| `test_masking.py`, `test_sampler.py`, `test_trainer.py`, `test_sampler_real.py` | 36 | milestone 3; `test_sampler_real.py` is `real_backbone` (8 such tests in total) |
| `test_continuation.py`, `test_tacc_scripts.py` | 9 | D-033 trainer rules; resume guard and pilot report |

68 tests carry the `tokenizer` marker.

## 6. Environment and commands

The machine is a MacBook Pro M2 Max (64 GB), macOS 26.6.2, Python 3.14.7 and torch
2.15.0. A Colab A100 is available later, with the user's approval. Cursor's sandbox
hides MPS and blocks the Hub, so use full permissions for those commands.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # pinned: requirements-lock.txt; no transformers (D-008)
.venv/bin/dlmc download-backbone                              # -> runs/hf/open-dcoder-0.5B, sha256-verified
.venv/bin/python -m pytest -q                                 # expect 225 passed (224 inside the sandbox: no MPS case)
.venv/bin/python -m pytest -q -m "not real_backbone and not tokenizer"   # 149, no assets needed
.venv/bin/dlmc doctor --backbone open-dcoder --device mps     # ~10 GB of memory
.venv/bin/dlmc generate-data --config configs/data/pilot.yaml  # -> runs/data/pilot-seed20261116 (verifies itself)
.venv/bin/dlmc verify-data --data-dir runs/data/pilot-seed20261116
.venv/bin/dlmc inspect-example --fixture arith_17_28 --draft-values 46,135,116 --pad 2
.venv/bin/dlmc inspect-example --data-dir runs/data/pilot-seed20261116 --split val --index 0
```

Reference environment (optional; transformers 4.54.1):

```bash
python3 -m venv .venv-ref && .venv-ref/bin/pip install -r requirements-ref-lock.txt
git clone https://github.com/pengzhangzhi/Open-dLLM runs/sources/Open-dLLM
git -C runs/sources/Open-dLLM checkout 5b9ba4d12c92be0010b99c61fc1129a89deeec04
PYTHONPATH=runs/sources/Open-dLLM .venv-ref/bin/python scripts/reference_logits.py
PYTHONPATH=runs/sources/Open-dLLM:src .venv-ref/bin/python scripts/reference_gradients.py
PYTHONPATH=src .venv-ref/bin/python scripts/reference_tokenizer.py --data-dir runs/data/pilot-seed20261116
```

Milestone-3 commands. MPS needs full permissions. Run long jobs as tool-managed
background commands: `&`/`nohup` inside a sandboxed shell call were killed silently.

```bash
H=990ecd1dd2341b81bc742fa91189b129877cd9b48b196fbdbc208e8586c81536
# Gate C
.venv/bin/dlmc train --config configs/train/overfit32.yaml --device mps
.venv/bin/python scripts/gatec_inspect.py --checkpoint runs/train/gatec-overfit32/checkpoints/ckpt-0000250 --device mps
# common warmup (as run); resume after any interruption with the same command plus --resume auto
.venv/bin/dlmc train --config configs/train/warmup_pilot.yaml --device mps --run-dir runs/train/warmup-pilot-v1 [--resume auto]
# The continuation (D-033) runs on Lonestar6 with configs/train/warmup_continuation.yaml through
# scripts/tacc/continuation.slurm; see docs/warmup_continuation.md, not a local command.
# calibration (validation only)
.venv/bin/dlmc calibrate --checkpoint runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000 --data-dir runs/data/pilot-seed20261116 \
    --manifest-sha256 $H --split val --settings all --n-per-setting 32 --budgets 1,2,4,8,L --out runs/calibration/screen-warmup1000
.venv/bin/python scripts/select_calibration_settings.py runs/calibration/screen-warmup1000/summary.json
```

Note that `--max-wall-seconds` is cumulative across resumes: `train_wall_seconds` is
stored in the checkpoint, and the first run used 5,790 s. On Colab, use
`notebooks/colab_runner.ipynb`; `--durable-dir` copies each checkpoint to Drive, and
`--resume auto` copies the latest one back if the local disk was wiped.

Hardware limits (measured): fp32 full fine-tuning at batch 32 uses about 30 GB of
Metal driver memory with the per-update cache release, and about 5.0 s per update
(5.8 s/update including evaluation and checkpoints). Each full checkpoint is about
5.9 GB (weights plus AdamW state) and takes about 10 s to save. Generation at
T = L is the slowest evaluation: about 1.5 s per call for a batch of 50 with L = 30
(about 3 minutes per 200 problems).

## 7. Evidence (software verification only; no scientific results exist)

Details and tables are in `docs/verification.md`.

**Tests.** `pytest`: 225 passed outside the sandbox. Inside it, 224 pass, because
MPS is hidden and the MPS case is not collected. The milestone-3 mutation check
catches 12/12 mutants and the D-033 check 7/7.

**Continuation preparation (details in `verification.md`, "Warmup-continuation
preparation"; all on the Mac):**
- The preflight of `ckpt-0001000` passed 14/14 checks; its predicted batches were
  validated on `ckpt-0000900` (all 100 recorded batches reproduced).
- The CPU reference fixture for the CUDA pilot was made and self-checked.
- A real-CLI CPU smoke run resumed a clone of checkpoint 1000, reproduced batches
  1001–1002 exactly, and stopped gracefully on SIGUSR1.
- No CUDA result exists.

**Milestone 3 (details and tables in `verification.md`, "Milestone 3"):**
- Precision on MPS: bf16 autocast passes the tolerances but is slower; pure bf16
  fails argmax agreement (0.977 < 0.98). Decision: fp32 (D-030).
- Timing: full fine-tuning at batch 32 on the longest setting runs at 5.88 s/update
  (5.44 examples/s); LoRA at batch 32 at 4.61 s/update. LoRA has 8,798,208 trainable
  parameters, verified.
- Gate C overfit: 32 examples. At update 250, masked loss 0.0007 and 100% trace
  accuracy at T = L and T = 2. Predictions inspected (p(target) ≥ 0.9977). All 290
  tensors move under full fine-tuning. One LoRA step changes all 168 `lora_B` tensors
  and leaves all 290 base tensors bitwise unchanged.
- Gate C resume: uninterrupted vs. interrupted-and-resumed (restored from the durable
  copy after the local checkpoints were deleted). Example IDs, masks and learning
  rates are identical. Loss relative difference 4.1e-7 and parameter divergence
  2.1e-5 of the update norm, both below the uninterrupted-repeat noise floor (3.9e-6,
  4.2e-5).
- Common warmup: 1,000 updates, 5,790 s, no zero-mask batches. Monitor (144 val
  problems) at T = L: final 0% → 38.9%, malformed 100% → 0%. Val masked loss 3.878 →
  0.591, still falling. The gradient was clipped on every update.
- Calibration (200 val per selected setting): final accuracy ranges from 0.91
  (arithmetic 2/1, T = L) down to 0.02 (programs 6/3, T = L). More calls always help.
  The only malformed reason is `negative_zero`.

**Forward pass.** Our raw logits equal upstream `veomni` exactly (max |diff| 0.0,
fp32, CPU).

**Gradients (D-016).** The same external aligned loss was applied to both models,
on a masked input and on a flawed-draft input:
- fp32: worst per-parameter relative error 3.1e-5 (tolerance 1e-4).
- fp64 with upstream's float32 RoPE/RMSNorm casts matched: 3.4e-15 (tolerance 1e-9).
- fp64 without the matched casts: about 5e-6. This is information only, and the
  cause is identified.
- Controls are detected at relative error 2.9 to 24.7: no logit shift in our loss,
  and upstream run causally.
- Artifact: `runs/reference/gradient_equivalence.json` (sha256 `cac20388…cb54`).

**Loading and tying.**
- All 291 checkpoint tensors load bit-exactly: 290 parameters plus the `lm_head`
  alias.
- `lm_head` and `embed_tokens` share storage after the move to CPU and to MPS, and
  still after an AdamW step.

**Data (pilot):**
- 27,900 records.
- Labels re-derived by the interpreter for every record.
- Composition signature, held-out status, normalized key and problem ID are rebuilt
  from each record's source and match the stored metadata exactly (D-024).
- Splits disjoint by the rebuilt problem ID and normalized key.
- Held-out compositions appear only in `test_unseen`, judged on derived signatures.
- Every record re-tokenizes to its stored IDs, with no merge across the
  prompt/response boundary.
- Manifest byte-identical on regeneration.
- `verify-data` passes.

**Tokenizer.** Our wrapper, transformers' fast tokenizer and the slow pure-Python
Qwen2 tokenizer agree on 83,713 strings with 0 mismatches.

**Mutation check.** 7 of 8 injected data/checker bugs were caught. The survivor is
an equivalent mutant.

**Real inspect-example output.** The fixture `arith_17_28` with a HYPOTHETICAL draft
46,135,116 gives tokens 14/15 correct, values 2/3, final answer correct, trace wrong.
It is stored in `runs/inspect/arith_17_28_hypothetical_draft.txt` and excerpted in
`verification.md`.

**Artifacts.** These are git-ignored under `runs/`. Recreate them with the commands
above.

| Path | Contents |
| --- | --- |
| `runs/hf/open-dcoder-0.5B/` | weights and tokenizer |
| `runs/sources/{Open-dLLM,CDLM,proseco}` | upstream clones |
| `runs/reference/open_dcoder_reference.{pt,json}`, `reference_run.log` | upstream logits |
| `runs/reference/gradient_equivalence.json`, `gradient_run.log` | gradient comparison |
| `runs/data/smoke-seed0/` | smoke dataset; manifest sha256 `527404808937a52446627e576965a0bd37a8cdb3cdb862e577fa278c51b1b2a0` |
| `runs/data/pilot-seed20261116/` | pilot dataset; manifest sha256 `990ecd1dd2341b81bc742fa91189b129877cd9b48b196fbdbc208e8586c81536`; includes `tokenizer_reference_check.json` |
| `runs/inspect/*.txt`, `*.json` | inspect-example outputs |
| `runs/doctor/*.json` | doctor records |
| `runs/benchmark/precision_mps.json`, `pilot_mps.{json,log}` | precision check, timing pilot |
| `runs/train/feas-full-20`, `feas-lora-20` | 20-update feasibility runs (5.5 GB and 0.1 GB) |
| `runs/train/gatec-overfit32/` | Gate C overfit; `ckpt-0000250` model sha256 `e278dcfd090101f555cd86762a8b81be049996a9e0a235e3ce32f5a8b07313b7` |
| `runs/gatec/inspect/`, `runs/gatec/compare-*.txt` | Gate C inspection and resume comparisons |
| `runs/train/resume-{A,A2,B}`, `runs/durable/resume-B` | Gate C resume runs and the durable copy |
| `runs/train/warmup-pilot-v1/` | **common warmup**; `ckpt-0001000` model sha256 `f48abe457d483b19b854e1fdf6be4bb294c702f949f9d76d7c44a2274b0d5a6a`, optimizer `ae55b39481da056d6b4180cd0197a732efecaece7b17c0d0eab52aa0fbfa99af`; `ckpt-0000900` also kept; trained on pilot train, monitored on pilot val |
| `runs/train/warmup-pilot-v1.{oom,swap}-attempt` | failed attempts (17 and 59 updates, no checkpoints), kept for the record |
| `runs/calibration/screen-warmup1000/` | stage-1 screen (18 settings × 32 × 5 budgets), selection output |
| `runs/calibration/stage2-warmup1000/` | stage-2 calibration (10 settings × 200 × 5 budgets): `summary.{json,md}`, `records.jsonl`, `traces.txt` |
| `runs/calibration/stage2-warmup1000.swap-attempt` | stopped stage-2 attempt (identical accuracies; inflated timings) |

There is no backup of these runs: no durable directory was configured locally, and
only the Gate C test copy in `runs/durable/` exists. That copy is not a backup of the
warmup checkpoint. A Ranch backup of `ckpt-0001000` is step 4 of the runbook and has
not been made. `runs/preflight/`, `runs/reference/cuda_pilot_fixture.*`,
`runs/cli_smoke/` (a smoke run's clone and its 5.9 GB `ckpt-0001002`, disposable) and
`runs/tacc_transfer/` (the staged manifest and bundle) are continuation preparation. Together they take about 66 GB
(`runs/train` 55 GB, `runs/durable` 11 GB).
The resume-check and feasibility runs can be deleted once they are no longer needed
as evidence.

**Pinned revisions.**

| Item | Pin |
| --- | --- |
| Backbone | HF `fredzzp/open-dcoder-0.5B` @ `d0d86d5b99960c05258bb1f8265dd91564dbac67` (file hashes in `open_dcoder.py::PINNED_SHA256`) |
| Open-dLLM (`veomni` reference) | `pengzhangzhi/Open-dLLM` @ `5b9ba4d12c92be0010b99c61fc1129a89deeec04` |
| CDLM | `zhangshuibai/CDLM` @ `86d108e2bb6dbc3f491d217aae04de4fc7c26ce3` |
| ProSeCo | `kuleshov-group/proseco` @ `43ce881779f06f475b56d90a152acd65ddc74fe4` |
| Token IDs | mask `<M>` = 151665 (the generation config's 151643 is stale, D-003); pad = EOS = 151643; no BOS; vocabulary 151,936 (tokenizer 151,666) |
| Response format | `" +"` = 488, `" -"` = 481, digits `"0"`–`"9"` = 15–24 |

## 8. Risks, unresolved choices, and checks not run

**Risks:**
- **The warmup is not converged** (D-032). It is being continued on Lonestar6
  (D-033). The CUDA trajectory will not match MPS bit for bit. Only data order,
  masks, learning rates and restored state must match exactly, and the pilot gates
  check them.
- **The torch 2.15.0 build cannot run on Lonestar6.** Environment-check job 3591308
  (user ayuj, allocation ASC26098) found driver 570.195.03; torch 2.15.0+cu132 needs
  R580. The transfer and a hash-verified Ranch backup of checkpoint 1000 exist (as
  reported by the user). The fallback torch 2.14.1+cu126 is prepared as D-034 and was
  approved by the user; it is not installed yet, and a fresh environment check and the
  CUDA pilot must pass before the continuation (runbook §11). SU balance is unconfirmed.
- **The hardest pilot settings are near 0%** (length 6 with 3 digits: ≤ 2.5% at every
  budget), and wrong responses there contain about 4.8 wrong values. Those settings
  may give the correction arms little to work with. The pilot grid stays provisional
  (D-019); any regeneration needs a new manifest and entry.
- **The value-only format has no variable names in the response.** A model must
  track slot order. If this turns out to be the main failure mode, revisit it through
  a superseding decision, not an edit.
- **Response-only masking differs from upstream** (D-026). This is settled.
- **MPS allocator cache growth** made the machine swap until the cache was released
  per update or batch (D-032). Any new long-running loop that varies shapes must call
  `env.release_cached_memory`.
- **Normalized keys miss commuted or semantically equal programs** (a documented
  limitation).

**Unresolved choices:**
- After the D-033 continuation: which checkpoint becomes the milestone-4 generator;
  then the bank-collection budget and the primary budget.
- D-011: β = 0.5 and the bin edges.
- D-012: λ and μ.
- Q-1: which liger version trained the base checkpoint.
- Q-4: the ProSeCo corrector input.
- Q-5: the SCOPE commit confidence.
- Q-6: the ReMDM adaptation.
- Q-7: whether near-zero L_keep is typical (measure in the milestone-4 diagnostic).
(Resolved in milestone 3: Q-8 by D-028; Q-3 for MPS by D-030; sampler schedule and ties
by D-029.)

**Checks not run:**
- Gate D (observer log c / log q same-pass, reference-independence of the bank).
- ProSeCo/SCOPE stop-gradient fixtures.
- Any CUDA/A100 run, including the Colab launcher and every TACC step of the D-033
  runbook (transfer, backup, environment check, pilot, continuation, final evaluation).
- Any test-split evaluation, draft collection or experimental arm.
- No human review of anything.

## 9. Git state

- Parent of the milestone-2 work: `8e00216` ("Add research docs, proposal, and
  milestone-1 plan.").
- Milestone 2 is split into reviewable commits on `main` after that parent, in
  order:
  1. milestone-1 follow-ups (gradients, loading/tying, wording);
  2. arithmetic and program task generators;
  3. fixed-slot format, checker and inspect-example;
  4. splits, verification-from-source, data CLI and configs;
  5. documentation, tracked protocol inputs, and milestone-2 closure notes.
- `docs/handoff.md`, `docs/ai_usage.md` and the build prompt are tracked
  (D-023, D-025). Everything under `runs/` stays git-ignored: weights, datasets,
  reference and doctor outputs.
- Milestone 3 is committed on `main` on top of `35b3282` as two commits: the code,
  configs, notebook, scripts and tests; then the documentation.
- The D-033 continuation preparation is one commit on top of `52991f4`: code,
  configs, `scripts/tacc/`, tests and documentation. The transfer bundle is built
  from that commit.
- Run `git log --oneline 8e00216..HEAD` for the current hashes.
