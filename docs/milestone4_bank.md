# Milestone 4: immutable draft bank (runbook)

**Status (2026-11-20): DONE.** Job `3597054` produced bank `m4-ckpt8000-v1` (D-043).
This file remains the runbook and schema reference; do not re-collect under the same
bank name.

Decisions: D-037 (continuation outcome), D-038 (generator: checkpoint 8000), D-039
(collection protocol), D-040 (signals, schema, checks), D-041 (approved scope: the full
`m4_ckpt8000_v1` config), D-042 (rank score = float32 raw log probability; preflight
compute separate), D-043 (collection completed).
Code: `src/diffusion_correction/drafts/`. Only the user runs commands on TACC.

## 1. What the job produces

One Slurm job (`scripts/tacc/bank_collect.slurm`) on one A100 of a full `gpu-a100`
node, with no trial job:

1. GPU selection and checks; checkpoint 8000's files against the identity recorded on
   the login node; the 3595133 validation records against their pinned hash.
2. In the collection process itself, the startup gates (D-040): Gate D, exact
   same-forward recompute, independent single-example check (2e-3), exact
   reproduction of job 3595133's validation drafts, record invariants, RNG unchanged.
   A failure stops the job before any shard exists; the report stays in
   `retained/bank/<name>.jobs/job-<id>/preflight/`.
3. Collection, shard by shard, until done or the deadline (job end − 45 min).
4. If complete: validation and finalization (`MANIFEST.json`, `BANK_COMPLETE`), an
   integrity re-check, step-0 signals, diagnostics and the examples report.

```
$WORK/dlmc/retained/bank/<name>/
  IDENTITY.json plan.json status.json events.jsonl
  shards/<split>/b<budget>/<family>-n<len>-d<digits>/part-NNN.{jsonl,json}
  validation.json MANIFEST.json BANK_COMPLETE
  derived/step0/ derived/diagnostics/{diagnostics.json,.md} derived/report/{examples.md,.html,.json,report_verification.json}
$WORK/dlmc/retained/bank/<name>.jobs/job-<id>/   preflight/, status snapshots, Slurm log copy
```

## 2. Record schema (`dlmc.draft.v1`, one JSON line per draft)

| Part | Fields |
|---|---|
| identity | `split`, `problem_id`, `family`, `length`, `operand_digits`, `setting`, `budget` ("1", "2", "4", "8", "L"), `calls`, `response_length`, `prompt_ids`, `bank`, `shard`, `batch_index`, `batch_slot`, `batch_size`, `identity_sha256`, `generator_model_sha256`, `tokenizer_json_sha256`, `sampler`, `seed` (null: no RNG) |
| `generation` (target-independent) | `draft_ids`, `schedule`, `commit_call`, `commit_order`, `mask_fraction_before`, `log_c_gen`, `rank_score`, `policy_log_prob`, `raw_max_log_prob` |
| `analysis` (target-dependent) | `reference_ids`, `values`, `log_q_gen`, `token_correct`, `parsed_values`, `slot_status`, `value_correct`, `final_correct`, `trace_correct`, `well_formed`, `n_token_errors`, `n_value_errors` |

Per-token arrays are indexed by response position. `log_c_gen` and `log_q_gen` are
float32 natural-log probabilities over the full output vocabulary (151,936 rows),
temperature 1, from the forward pass that committed the token. `rank_score` is the
sampler's own sort key: the float32 full-vocabulary **log** probability of the selected
candidate (`CommitEvent.raw_logp` from `monotone_unmask`). The implemented D-029 sampler
ranks by that log value (never by `exp` of it); see D-042. It equals `log_c_gen` bitwise
because the observer recomputes the same `log_softmax`. `policy_log_prob` is renormalized
over the D-029 allowed set (it is what the greedy choice maximizes) and is not `c`.
The mask fraction is the global response fraction still masked before the commit call.

Sidecars hold the shard's sha256, counts, problem-ID hash, per-batch timing and
forwards (`batched_forward_calls`, `example_model_calls`, `model_seconds_with_signals`),
rank-score mismatches, RNG check, job ID and memory (including CUDA allocator retries and
OOMs). Those counts cover collection only.

**Startup verification compute is separate.** Same-forward recomputations and the
independent single-example forwards run only on the four configured preflight batches
(50 problems each), before any shard is written. Their wall times and batched-forward
counts live in the job directory's `preflight/preflight.json` (`seconds`,
`batched_forwards`, `preflight_seconds`) and are never added to shard sidecars or
`validation.json` collection totals. Full-bank collection does not save pre-commit states
and does not re-forward them.

## 3. Scope options and provisional estimates (D-039; the full scope was approved in D-041)

Counts come from the plan built on the real dataset. Times are **extrapolations** from
job 3595133's measured CUDA generation (batch 50, no observer): about 0.248, 0.323 and
0.459 s per batched forward at L = 10, 20, 30. Observer overhead was not measured on
CUDA; 10–30% is assumed. Storage comes from the local smoke bank (about 150 bytes per
response token plus about 1.1 KB per record).

| | Recommended: all 18 settings | Alternative: 10 stage-2 settings |
|---|---|---|
| config | `configs/bank/m4_ckpt8000_v1.yaml` | `configs/bank/m4_ckpt8000_stage2_v1.yaml` |
| problems (train / val) | 18,000 / 3,600 | 10,000 / 2,000 |
| drafts (5 budgets) | 108,000 | 60,000 |
| example-calls / batched forwards | 756,000 / 15,120 | 384,000 / 7,680 |
| token records | 2,160,000 | 1,020,000 |
| collection (model) | 92 min + overhead ≈ 1.7–2.0 h | 42 min + overhead ≈ 0.8–0.9 h |
| step-0 + preflight + validation + diagnostics + report | ≈ 25–35 min | ≈ 15–25 min |
| suggested `-t` / expected SU (3 SU/node-hour) | 05:00:00 / ≈ 6–7.5 (≤ 15) | 03:00:00 / ≈ 3–4 (≤ 9) |
| bank storage | ≈ 0.45–0.6 GB (+ ≈ 0.1 GB step-0) | ≈ 0.22–0.3 GB (+ ≈ 0.05 GB) |

**Expected errors (proxy only).** On validation (10 stage-2 settings, 2,000 problems,
34,000 tokens per budget), checkpoint 8000 made 5,093 / 3,108 / 1,674 / 968 / 743 token
errors at T = 1 / 2 / 4 / 8 / L, i.e. 11,586 wrong tokens of 170,000 (6.8%), and 2,798
wrong final answers of 10,000 drafts. Train error rates are unknown and probably lower:
checkpoint 8000 saw each train problem about 14 times (8,000 × 32 / 18,000). If train
matched validation, the stage-2 train bank would hold about 58,000 wrong tokens; at a
third of that rate, about 19,000. The full scope roughly doubles either number and adds
the 8 settings not used in calibration. Correct and easy drafts are kept in both options.

## 4. TACC commands (the user runs them)

The update bundle is staged in `runs/tacc_update/9cf1a6e..<head>/` on the Mac; its
`COMMITS` and `SHA256SUMS` files hold the source revision and hashes (a commit cannot
record its own hash). Paths: WORK `/work/10692/ayuj/ls6`, repo
`/work/10692/ayuj/ls6/dlmc/repo`.

### 4.1 Copy and apply the code update (Mac, then login node)

Printed by `DLMC_UPDATE_BASE=9cf1a6e DLMC_REMOTE=ayuj@ls6.tacc.utexas.edu
DLMC_REMOTE_ROOT=/work/10692/ayuj/ls6/dlmc scripts/tacc/stage_update.sh` (§11.3–11.4 of
`docs/warmup_continuation.md`): one `tar | ssh` copy, then on the login node
`sha256sum --strict -c SHA256SUMS`, base-commit and clean-tree checks, `git bundle
verify`, `fetch`, `merge --ff-only`, and `UPDATE APPLIED <head>`. The environment is
not rebuilt: the package is installed in editable mode and the lock is unchanged.

### 4.2 Verify and record the generator (login node; hashing only)

```bash
cd /work/10692/ayuj/ls6/dlmc/repo
bash -c 'source scripts/tacc/tacc_env.sh && check_env_build && "$DLMC_CLI" collect-drafts --help >/dev/null \
  && "$DLMC_PY" scripts/tacc/generator_identity.py --checkpoint "$DLMC_GEN" --model-sha256 "$DLMC_GEN_MODEL_SHA256" \
     --update "$DLMC_GEN_UPDATE" --eval-records "$DLMC_GEN_EVAL_RECORDS" \
     --eval-records-sha256 "$DLMC_GEN_EVAL_RECORDS_SHA256" --out "$DLMC_GEN_IDENTITY"'
```

It must end with `GENERATOR VERIFIED AND RECORDED`. Anything else: stop and report.

### 4.3 Ranch backup of checkpoint 8000 (login node)

```bash
cd /work/10692/ayuj/ls6/dlmc/repo && bash scripts/tacc/backup_to_ranch.sh ckpt8000
```

It re-verifies the identity, archives checkpoint 8000, the continuation logs, the final
evaluation records and the identity file, and prints `BACKUP VERIFIED` only after the
sha256 computed on Ranch matches. The record is
`retained/backups/dlmc-ckpt8000-<stamp>.tar.record`.

### 4.4 Submit (after the scope is approved)

```bash
cd /work/10692/ayuj/ls6/dlmc/repo
DLMC_BANK_CONFIG=configs/bank/m4_ckpt8000_v1.yaml sbatch -A ASC26098 scripts/tacc/bank_collect.slurm
# alternative scope:
DLMC_BANK_CONFIG=configs/bank/m4_ckpt8000_stage2_v1.yaml sbatch -A ASC26098 -t 03:00:00 scripts/tacc/bank_collect.slurm
```

### 4.5 Monitor (login node)

```bash
squeue -u ayuj
tail -f /work/10692/ayuj/ls6/dlmc/repo/runs/slurm/dlmc-bank-<jobid>.out
cd /work/10692/ayuj/ls6/dlmc/repo && $SCRATCH/dlmc/venv/bin/python scripts/tacc/bank_status.py \
    /work/10692/ayuj/ls6/dlmc/retained/bank/<name>
```

Stop early and keep every verified shard: `scancel --signal=USR1 --batch <jobid>`.

### 4.6 Resume (only if the job ended with `COLLECTION STOPPED` or step-0 incomplete)

```bash
cd /work/10692/ayuj/ls6/dlmc/repo
DLMC_BANK_CONFIG=<same config> DLMC_BANK_RESUME=1 sbatch -A ASC26098 scripts/tacc/bank_collect.slurm
```

The resumed job re-runs the startup gates on its node, refuses any identity change
(config, plan, generator, code commit, environment), keeps verified shards unchanged,
moves partial files to `incomplete/<stamp>/`, and stops on a corrupt shard. A preflight
or collection failure is not retried: report it.

### 4.7 Back up and download the finished bank

```bash
# login node
cd /work/10692/ayuj/ls6/dlmc/repo && bash scripts/tacc/backup_to_ranch.sh bank:<name>
# Mac, from the repository root: records, diagnostics, report (small)
mkdir -p runs/tacc_results/bank
ssh ayuj@ls6.tacc.utexas.edu "cd /work/10692/ayuj/ls6/dlmc/retained/bank && tar -cf - \
  <name>/IDENTITY.json <name>/plan.json <name>/status.json <name>/events.jsonl <name>/validation.json \
  <name>/MANIFEST.json <name>/BANK_COMPLETE <name>/derived <name>.jobs" | tar -xf - -C runs/tacc_results/bank
# Mac: the full bank (about 0.5 GB), then verify every shard against the manifest
ssh ayuj@ls6.tacc.utexas.edu "cd /work/10692/ayuj/ls6/dlmc/retained/bank && tar -cf - <name>/shards" \
  | tar -xf - -C runs/tacc_results/bank
.venv/bin/python scripts/tacc/bank_status.py runs/tacc_results/bank/<name> --verify
```

## 5. Signals and the confidence question

**ProSeCo (arXiv 2602.11590v4, primary source).** Algorithm 1 trains the MDM loss on
masked positions and a self-correction loss on a second pass over
y = sg(one_hot(argmax x_θ(z_t))) at **all** positions, both weighted by
α̇_t / (1 − α_t) (Eq. 6; λ_t is set equal to the MDM weight). For unconditional OWT text
only, Appendix C.3 (Eq. 11) replaces the correction term on positions where the
denoiser was already right (y = x) by the probability ⟨x_θ(y), x⟩ without the
logarithm, keeping the log on wrong positions, "so that gradients for tokens which the
original denoiser model incorrectly predicts are steeper". The released code's
`corrector_loss_errors_upweighted` flag implements the same split. Neither form uses the
generator's confidence in its own wrong token.

**Does softmax cross-entropy already depend on confidence?** Yes, on the *current*
model's confidence. For logits z with p = softmax(z) and target y,
∂(−log p_y)/∂z = p − e_y, so the target-logit gradient has magnitude 1 − p_y. For C.3's
−p_y term it is p_y(1 − p_y). Both depend only on p_y = p_now(target | full draft), which
changes as the corrector trains.

The confidence weight studied here uses a different quantity: c = p_gen(committed token |
partially masked context at commit time). It differs from what CE sees in four ways:
- the context is the partially masked state at the commit step, not the complete draft;
- the model is the frozen generator, not the evolving corrector;
- it is the probability of the wrong committed token, not of the target;
- it is fixed per draft token.

At step 0 the corrector equals the generator, but the contexts differ, so c cannot be
read off the step-0 logits. The diagnostics measure how close the two are (Spearman
correlations of log c with step-0 CE and with the step-0 probability of the draft token,
overall and within budget × stage groups, with denominators). They also report how much
of log c's variance stage, budget and setting explain (η²), so it is visible if
"confidence" mostly re-encodes difficulty or stage. These are descriptive. A correlation
does not show that a weight helps, and no novelty or improvement claim follows from it.

Note: on wrong tokens c + q ≤ 1, so high c forces a low q; the correlation of log c with
−log q is partly mechanical.

## 6. Local evidence (Mac)

See `docs/verification.md`, "Milestone 4: draft bank": targeted tiny-backbone tests, and a
real-backbone smoke on MPS with checkpoint 1000 (not the generator). In that smoke the
startup gates passed, the 300 comparable validation drafts were identical to checkpoint
1000's CUDA records, the bank finalized, and step-0, diagnostics and the report ran.
