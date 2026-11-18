# Warmup continuation on TACC Lonestar6 (D-033)

Plan and runbook for continuing the common warmup `warmup-pilot-v1` from
`ckpt-0001000` on Lonestar6 (`ls6`), after a CUDA pilot. The user performs
authentication and submits every job; nothing here has been run on TACC.

**Status (2026-11-17).** Local preparation is complete: code, configuration,
scripts, transfer manifest, Linux lock file, reference fixture and Mac-side checks.
Everything on TACC is **PENDING**: transfer, environment build, environment check,
pilot, backup, continuation and final evaluation. No CUDA result exists. Milestone 4
(draft bank) and correction training are out of scope and have not been started.

**Update (2026-11-17, reported by the user).** The transfer, the Ranch backup of checkpoint 1000
(hash-verified) and the torch 2.15.0 environment build were done. Environment-check job
3591308 stopped at the driver guard: the node has driver 570.195.03, and torch 2.15.0+cu132
needs R580. No pilot or continuation has run. The fallback torch 2.14.1+cu126 is prepared
(§11, D-034) but **not approved and not installed**; its remote validation is **PENDING**.

## 1. What the continuation does

| Item | Setting |
| --- | --- |
| Start | the original `runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000` (model sha256 `f48abe45…5a6a`, optimizer `ae55b394…99af`), resumed with model, optimizer, scheduler, sampler and the CPU RNG streams. Never a pilot copy or pilot-updated weights |
| Config | `configs/train/warmup_continuation.yaml`. Every key that enters the resume-compatibility hash equals `warmup_pilot.yaml` (preflight check `immutable_key_differences: []`): data, masking, fp32 full fine-tuning, AdamW (lr 3e-5, betas 0.9/0.999, eps 1e-8, wd 0.01), 20-update warmup then constant LR, clipping 1.0, batch 32, seed 20261117 |
| Update cap | 10,000 **total** updates (at most 9,000 more) |
| Training-time cap | 7,200 s of training-update time, counted from update 1000 across all jobs (§5) |
| Early stopping | 3 consecutive scheduled evaluations without a strictly higher score (§5) |
| Device | one NVIDIA A100 40 GB, `gpu-a100-small` queue, TF32 off; torch 2.14.1 + CUDA 12.6 if D-034 is approved (§11; the D-033 build, torch 2.15.0 + CUDA 13.2, cannot run on the R570 driver) |

The learning rate stays 3e-5 for the whole continuation: the schedule has no decay
horizon, so neither the update cap nor resumes change it (preflight check).

## 2. Configuration status

Resolved: system Lonestar6; Ranch for the backup; the layout in §3; queue
`gpu-a100-small` (1 node, 48 h max, 1.5 SU per node-hour, minimum charge 0.25 h per
job, at most 3 running jobs per user; Lonestar6 user guide, queue table).

**Missing (the user supplies these; scripts stop with an error if they are needed and absent):**

| Item | Where it is needed |
| --- | --- |
| TACC username and the `$WORK` path on ls6 (`echo $WORK` after login) | `DLMC_REMOTE`, `DLMC_REMOTE_DIR` in `stage_transfer.sh` |
| Allocation (project) ID and SU balance (`/usr/local/etc/taccinfo`) | `sbatch -A <allocation>` on every submission |
| Ranch access (`$ARCHIVER`/`$ARCHIVE` set; 2 TB default user quota) | `backup_to_ranch.sh` |
| GPU count and driver on `gpu-a100-small` nodes | job 3591308 saw one A100-PCIE-40GB and driver 570.195.03; `env_check.slurm` checks each job (one A100 visible, driver ≥ 560.35.05 for the CUDA 12.6 build) |
| Network access from compute nodes | `setup_env.sh` in an `idev` session; if it has none, run it on a login node instead |

## 3. TACC layout

| Path | Contents | Policy |
| --- | --- | --- |
| `$WORK/dlmc/transfer/` | the transferred files at their repo-relative paths, made read-only after `sha256sum -c`. **This is the persistent copy of `ckpt-0001000` on TACC** | `$WORK`: not purged, not backed up, 1 TB quota |
| `$WORK/dlmc/repo/` | `git clone` of the transferred bundle; `runs/hf`, `runs/data`, `runs/train/warmup-pilot-v1`, calibration, fixture and preflight are symlinks into `transfer/`, so all recorded relative paths and hashes are unchanged | |
| `$WORK/dlmc/retained/` | env-check and pilot reports; continuation checkpoints (`warmup-pilot-v1-cont/checkpoints`) and logs; final evaluation; backup records | |
| `$SCRATCH/dlmc/venv`, `python/`, `uv-cache/` | Python environment. The Lonestar6 guide says virtual environments go on `$SCRATCH`, not `$WORK` | `$SCRATCH`: purged after 10 days without access; rebuild with `setup_env.sh` and re-run the env check |
| `$SCRATCH/dlmc/runs/warmup-pilot-v1-cont`, `pilot/<job>` | job working directories, local checkpoints | |
| Ranch `$ARCHIVE/dlmc/` | tar of `ckpt-0001000` with its logs, data, calibration, fixture and manifest | |

The scripts refuse to run elsewhere, refuse a modified checkout, and never delete
anything. Where a target exists they stop and ask you to move it aside.

## 4. Order of work

Each step names where it runs. "Login" means an ls6 login node (file operations
only). Every model execution runs inside a Slurm job.

1. **Mac — stage the transfer** (after the commit that contains this document; file list in §13):
   ```bash
   DLMC_REMOTE=<user>@ls6.tacc.utexas.edu DLMC_REMOTE_DIR=<$WORK on ls6>/dlmc/transfer \
       scripts/tacc/stage_transfer.sh
   ```
   This checks the tree is clean, re-checks the checkpoint and manifest hashes, and
   writes `runs/tacc_transfer/` (`FILES.txt`, `MANIFEST.sha256`, `SIZES.txt`,
   `dlmc.bundle`, `COMMIT`). It then prints the single copy command:
   `tar … -T runs/tacc_transfer/FILES.txt | ssh … "tar -xf - -C …"`. That is one ssh
   session, so one MFA prompt. Re-run the copy if it is interrupted.
2. **Mac — copy:** run the printed command (about 7.25 GB).
3. **Login — verify and install:**
   ```bash
   cd $WORK/dlmc/transfer && sha256sum --quiet -c runs/tacc_transfer/MANIFEST.sha256 && echo TRANSFER VERIFIED
   git clone -b main $WORK/dlmc/transfer/runs/tacc_transfer/dlmc.bundle $WORK/dlmc/repo
   bash $WORK/dlmc/repo/scripts/tacc/install_transfer.sh
   ```
4. **Login — Ranch backup of checkpoint 1000** (needs a Ranch allocation):
   `bash $WORK/dlmc/repo/scripts/tacc/backup_to_ranch.sh ckpt1000`. It re-verifies the
   manifest, tars about 6.0 GB (the Hugging Face backbone is excluded: public and pinned),
   copies the tar with `scp`, and computes sha256 **on Ranch** immediately (the file is
   still on disk; never re-checksum later, which would recall it from tape). It prints
   `BACKUP VERIFIED` and writes `$WORK/dlmc/retained/backups/<tar>.record` only on a match.
5. **Compute (interactive, cheap) — environment:**
   ```bash
   idev -p vm-small -N 1 -n 1 -t 00:30:00      # 0.143 SU/node-hour, 0.25 h minimum
   bash $WORK/dlmc/repo/scripts/tacc/setup_env.sh
   ```
   This installs uv 0.13.9 (tarball sha256 `a7fbaff2…71d5`) and uv-managed CPython
   3.14.7. It then installs every package from `scripts/tacc/requirements-tacc-lock.txt`
   with `--require-hashes --no-deps`, and the project with `--no-index
   --no-build-isolation`. Nothing imports torch here.
6. **GPU job — environment check** (≤ 30 min, ≤ 0.75 SU):
   ```bash
   cd $WORK/dlmc/repo && sbatch -A <allocation> scripts/tacc/env_check.slurm
   ```
   It checks the GPU (A100, compute capability 8.0, `sm_80` in the build, ≥ 39 GB,
   one visible GPU), the driver and the exact torch, CUDA and cuDNN versions of the
   lock (§11), that the venv was built from the lock in the checkout, and that TF32
   is off. It also checks that an fp32 GPU matmul matches CPU float64 within 1e-6.
   It then runs the full pytest suite and `dlmc verify-data`, plus `dlmc doctor` on
   CUDA (real backbone forward, loss, backward and AdamW step). Last, it re-hashes
   `ckpt-0001000` and re-runs the preflight on it, which must match the Mac preflight
   field by field (`compare_preflight.py`; floats within 1e-9 relative). It writes
   `ENV_CHECK_PASSED` only if everything passes.
7. **GPU job — pilot** (≤ 1.5 h, ≤ 2.25 SU):
   ```bash
   DLMC_ENV_CHECK=$WORK/dlmc/retained/env_check/<jobid> sbatch -A <allocation> scripts/tacc/pilot.slurm
   ```
   Read `$WORK/dlmc/retained/pilot/<jobid>/pilot_report.md`. Continue only if it says
   PASSED, then re-derive the continuation time limit from its estimates (§8).
8. **GPU job — continuation** (default `-t 05:00:00`; adjust from the pilot):
   ```bash
   DLMC_RESUME_FROM=$WORK/dlmc/transfer/runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000 \
   DLMC_PILOT=$WORK/dlmc/retained/pilot/<pilot jobid> \
   DLMC_BACKUP_RECORD=$WORK/dlmc/retained/backups/<tar>.record \
   sbatch -A <allocation> -t <HH:MM:SS> scripts/tacc/continuation.slurm
   ```
9. **Resume** only if a job ended by its deadline or a signal, not by a stopping rule
   (§6).
10. **GPU job — final evaluation**, after the continuation has stopped by a rule:
    `sbatch -A <allocation> scripts/tacc/final_eval.slurm` (§7).

## 5. Limits and stopping policy

**Training-update timer.** `timers.train_update_seconds` is stored in every checkpoint
and restored on resume. For each loop iteration it adds the wall time from drawing the
batch indices to the end of the update's bookkeeping (metrics record, memory snapshot,
logging). The update itself is device-synchronized. Zero-mask batches count too.
Evaluations and checkpoint saves (including the durable copy and pruning) are timed
separately (`eval_seconds`, `checkpoint_seconds`), as are model loading and process
start-up, which are not timed at all. The cap is checked before each update, so it is
exceeded by at most one update.

The timer's origin is fixed at update 1000 (`train_seconds_origin_update`). Checkpoint
1000 predates the timer, so resuming it starts the timer at 0. A checkpoint that
already carries a timer refuses a different origin ("cannot be re-based"). The
continuation job's Slurm limit is therefore **larger** than two hours: two hours of
training plus evaluations, checkpoints, start-up and a stop margin. The old
`max_wall_seconds` (which also counts evaluation, and already held 5,688 s) is null
for the continuation.

Two scope notes on the cap:
- It counts training-update time on the trajectory that is kept. Work lost after the
  last checkpoint, when a job dies without saving, is redone and charged only once.
  It still costs SUs.
- `checkpoint_seconds` stored inside a checkpoint excludes that checkpoint's own
  save. This is reporting only and does not affect the cap.

**Monitoring subset.** `configs/eval/monitor_val_stage2_first50.json` (ID-list sha256
`324d5ca8…3dc94`): the first 50 validation problems, in file order, of each of the 10
stage-2 calibration settings. That is 500 problems, all drawn from the existing
calibration IDs (val split sha256 `ae3333a7…d566`; stage-2 summary `ade11fe8…6978`).
The trainer verifies the hash, that every ID is in val, and that there are no
duplicates. The identity is stored with the selection state, and a different subset
on resume is refused. Test splits are never loaded.

**Scheduled evaluations.** These run at update 1000 (the CUDA baseline of the original
checkpoint, before any continuation update) and then at every 1,000 total updates
(2000, 3000, …). Each runs greedy D-029 generation at T = L on the 500 problems,
batch 50. It also computes masked validation loss on the same problems with fixed
masks: the eval RNG stream is re-seeded for every evaluation, so the masks are
identical each time.

**Score.** Final-answer accuracy at T = L. The mean is taken over settings within each
family, then over the two families: equal weight per family and per setting
(`evaluation.generation.selection_score`). Each evaluation also records, in
`evals.jsonl` and `eval_records/`, per-family and per-setting results: final, value,
token and trace accuracy, malformed counts and reasons, model calls and masked loss.

**Rule.** A strictly higher score becomes the new best. A tie or a lower score adds 1
to `evals_without_improvement`, so ties keep the earlier checkpoint. Training stops
when the counter reaches 3. The best score, best update and checkpoint, the counter
and the full history are stored in every checkpoint and in `selection.json`, and
restored on resume. Changing the budget, patience, interval or monitoring subset on
resume is refused ("cannot be reset or changed").

**Stop reasons**, checked before each update in this order:
1. `max_updates` (10,000);
2. `early_stop_patience`;
3. `max_train_seconds` (7,200);
4. `deadline` (the job's `--stop-at-unix`);
5. `signal_SIGUSR1` / `signal_SIGTERM`.

The first three end the continuation. A deadline or signal stop is graceful: the
current update finishes, a checkpoint is saved, copied to `$WORK` and verified, and
the process exits without an unscheduled evaluation.

**Eligibility.** Only scheduled evaluations (updates 1000, 2000, …, 10000) can be
selected. If the time cap or the update cap ends training between scheduled points
(e.g. at 5,812), that final state is evaluated once, unscheduled, for reporting only;
it is never selected. Its checkpoint is kept. So the selected checkpoint is
1000 or a multiple of 1000 not later than the stop.

**Checkpoints.** A checkpoint is saved every 500 updates. Each one is copied to
`$WORK/dlmc/retained/warmup-pilot-v1-cont/checkpoints` and verified, and the local
copy stays in `$SCRATCH`. Pruning (`keep_last 2`) never removes:
- the current checkpoint;
- 1500 (`protect_updates`);
- any multiple of 1000 (`protect_every`), i.e. every scheduled-evaluation checkpoint;
- the best checkpoint.

The final stopping state is the newest checkpoint. Checkpoint 1000 itself is the
read-only original in `transfer/`, also on Ranch.

## 6. Running, monitoring, cancelling, resuming

`continuation.slurm` runs one attempt per submission. It never retries, it is
submitted with `--no-requeue`, and it has no automatic resubmission. Before training it
checks the pilot's `PILOT_PASSED`, the Ranch record's `BACKUP VERIFIED`, the GPU, and
the original checkpoint 1000's hashes. It then runs the resume guard
(`continuation_status.py --check-resume`), which accepts the named checkpoint only if:
- it verifies against its `COMPLETE.json`; and
- either it is the original checkpoint 1000 and no continuation checkpoint exists yet,
- or it is the **newest** continuation checkpoint (timer origin 1000) and no stopping
  rule has been reached.

Resuming an older checkpoint is refused, since it would roll back the update counter,
the cap and the early-stopping state.

The job derives `--stop-at-unix` from Slurm's end time minus
`DLMC_STOP_MARGIN_SECONDS` (default 1,500 s). That margin covers an evaluation and a
checkpoint started just before the deadline. `--signal=B:USR1@900` is a backstop:
the batch shell forwards SIGUSR1 to the trainer, which stops gracefully. Logs are
copied to `$WORK/dlmc/retained/warmup-pilot-v1-cont/logs` when the job exits.

| Action | Command (login node) |
| --- | --- |
| Queue state | `squeue -u $USER` |
| Live log | `tail -f $WORK/dlmc/repo/runs/slurm/dlmc-cont-<jobid>.out` |
| Progress, timers, selection, newest checkpoint | `$SCRATCH/dlmc/venv/bin/python scripts/tacc/continuation_status.py --run-dir $SCRATCH/dlmc/runs/warmup-pilot-v1-cont --durable $WORK/dlmc/retained/warmup-pilot-v1-cont --origin $WORK/dlmc/transfer/runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000` (reads JSON only) |
| Stop gracefully (checkpoint, copy, exit) | `scancel --signal=USR1 --batch <jobid>` |
| Stop immediately | `scancel <jobid>`. SIGTERM triggers the same graceful path, but Slurm may kill the process before the save finishes; the last completed checkpoint stays valid (atomic) |
| Resume | resubmit with `DLMC_RESUME_FROM=$WORK/dlmc/retained/warmup-pilot-v1-cont/checkpoints/ckpt-XXXXXXX`, naming the newest checkpoint printed by the status command. Same `DLMC_PILOT` and `DLMC_BACKUP_RECORD` |

If a job fails rather than stopping (non-zero exit without a `stop_reason`, NaN loss,
out-of-memory error, hash error), do **not** resubmit. Inspect the log, record what
happened, and decide explicitly.

## 7. Final comparison (after the continuation stops by a rule)

`final_eval.slurm` gets the verified best checkpoint from `continuation_status.py
--best`, which refuses while the continuation is unfinished. It then runs the full
D-031 stage-2 protocol on CUDA for checkpoint 1000 and for that checkpoint (the
original-1000 run is reused if the best is 1000). The protocol is:
- the 10 settings × 200 validation problems;
- budgets 1, 2, 4, 8 and L;
- batch 50;
- 2 trace examples per family.

`compare_calibrations.py` checks that both runs used the same problems, budgets,
sampler and dataset. Per setting and budget it reports final-answer, value, token
and trace accuracy with numerators and denominators, malformed outputs, model calls,
model seconds and wall time. Per budget it adds the equal-weight family/setting
aggregate. It also reports how checkpoint 1000 on CUDA agrees with the Mac MPS stage-2
run (identical responses, changed correctness). The final primary correction-comparison
budget stays unselected; test splits stay untouched.

## 8. Pilot (CUDA, disposable copy)

All pilot training runs on `$SCRATCH/dlmc/pilot/<job>/copy/ckpt-0001000`, a verified
copy. The original is re-hashed before and after. Pilot work is counted separately
and never enters the continuation's counters:
- `dlmc benchmark`: 3 counted warmup steps and 20 synchronized timed updates
  (fresh pretrained weights);
- 1 fixture step;
- 10 updates each in runs U, U2 and R.

| Part | What it shows |
| --- | --- |
| `cuda_fixture.py check` | Against the Mac CPU reference fixture (`runs/reference/cuda_pilot_fixture.*`, made from checkpoint 1000 on CPU fp32), on a fixed 18-example batch with fixed masks: loss, per-token CE, logsumexp, top-5 logits, argmax, pre-clip gradient norm, per-tensor gradient norms, then one clipped AdamW step from the checkpoint's optimizer state (per-tensor update norms), and greedy T = L generation for one problem per setting |
| `dlmc benchmark full,32,longest` | batch shapes, s/update, examples/s, tokens/s, CUDA allocated/reserved/peak memory, optimizer-state bytes and device |
| run M | baseline monitoring evaluation at 1000 (cost, CUDA baseline score), no update |
| runs U, U2 | uninterrupted 1000 → 1010; U also copies its checkpoints to `$WORK` (copy timing). U2 repeats U to measure the CUDA run-to-run noise floor |
| run R | 1000 → 1005, process exits, resume from R's own checkpoint → 1010 |
| `dlmc compare-runs U R` | example order, problem IDs, mask digests, mask rates, learning rates (exact); losses; final parameters |
| `check_expected_batches.py` | U, U2 and R reproduce the batches 1001–1010 that the Mac preflight predicted from checkpoint 1000's CPU streams (indices, problem IDs, masks, mask rates, shapes, LR 3e-5) |

**Tolerances, declared before any CUDA result.** These are fixed in `cuda_fixture.py`
(`TOLERANCES`, also stored in the fixture and checked to be unchanged) and in
`pilot.slurm`.

| Quantity | Tolerance |
| --- | --- |
| fixture loss | relative 1e-4 |
| per-token CE, logsumexp | absolute 1e-3 |
| top-5 logits (IDs must match) | absolute 1e-2 |
| argmax flips | none where the reference top-1/top-2 margin ≥ 1e-3 |
| pre-clip gradient norm | relative 1e-3 |
| per-tensor gradient and update norms (tensors ≥ 1e-3 of the largest) | relative 1e-2 |
| greedy generation | identical, unless the reference's smallest committed margin < 1e-3 |
| resume U vs R | exact data and mask fields; loss rtol 1e-4; parameter atol 1e-5; ‖θ_R − θ_U‖ / ‖θ_U − θ_1000‖ ≤ 1e-3 (the Gate C tolerances) |

**What may differ from MPS.**
- CUDA kernels, reduction order and SDPA backend differ from MPS, so losses and
  weights will not follow the MPS trajectory bit for bit. No identical MPS-to-CUDA
  trajectory is promised.
- What must carry over exactly is everything driven by the CPU RNG streams and the
  restored state: example order, mask positions, mask rates, the learning rate,
  counters and the selection state.
- TF32 is disabled and fp32 matmul precision is "highest" on CUDA (D-033), so
  numerics stay fp32 as on the Mac.
- The accelerator RNG is not restored across devices (logged as
  `accelerator_rng_restored: false`). Nothing in training uses it.

**Gates** (`pilot_report.py`; all must pass for `PILOT_PASSED`):
- every step exits 0;
- the fixture check passes;
- U vs R passes;
- U, U2 and R reproduce the predicted batches;
- the baseline evaluation records a selection score at 1000;
- checkpoint 1000 is unchanged.

U vs U2 is reported, not gated. If U vs R fails only on the parameter tolerance and U
vs U2 shows a similar gap, that is CUDA nondeterminism, not a resume defect. It
still needs an explicit decision and a decisions entry before continuing; the
tolerance is not loosened silently.

## 9. Estimates (PROVISIONAL until the pilot report exists)

These are A100 guesses scaled from the Mac (MPS 5.0 s/update, 5.8 s with
evaluations and checkpoints; a 5.9 GB checkpoint in about 10 s). The pilot report
replaces them with measured numbers.

| Item | Provisional estimate |
| --- | --- |
| fp32 full fine-tuning update, batch 32 | ~1.5 s → the 2-hour cap binds at roughly 4,800 more updates (update ~5,800), unless early stopping comes first |
| monitoring evaluation (500 problems, T = L) | 1–3 min (MPS estimate 191 s of model time) |
| checkpoint save + verified copy to `$WORK` | ~1–2 min per save, ≤ 18 saves |
| continuation job | ~3.5–4.5 h of wall time; 5 h requested → ≤ 7.5 SU |
| environment check / pilot / final evaluation | ≤ 0.75 / ≤ 2.25 / ≤ 3 SU (charged for time used, 0.25 h minimum) |
| total | ≲ 14 SU |
| retained checkpoints | ≤ 11 × 5.93 GB ≈ 65 GB in `$WORK` (and similar in `$SCRATCH`), plus ~12 GB of pilot durable copies |
| Ranch | one ~6.0 GB tar |

`pilot_report.py` recomputes, from the measured iteration time, evaluation cost and
save-plus-copy time: the updates within the cap, the number of evaluations and saves,
the job work time, a recommended `-t` (work plus the stop margin, +15%), SUs,
retained storage, and the final-evaluation cost.

## 10. Before submitting the continuation, all of these must hold

1. `TRANSFER VERIFIED` on ls6 (all 41 data files plus the bundle and commit match the
   Mac manifest).
2. `BACKUP VERIFIED` for the Ranch tar of checkpoint 1000. The Gate C copy in
   `runs/durable/` is a test copy, not a backup of the warmup checkpoint.
3. `ENV_CHECK_PASSED`, including the preflight matching the Mac, produced with the
   environment built from the current lock (§11.2).
4. `PILOT_PASSED`: fixture, CUDA save/reload/resume, expected batches, baseline
   evaluation, original checkpoint unchanged; run with that same environment.
5. The continuation time limit is re-derived from the pilot's measurements.

The continuation script enforces 2–4 itself (1 is a precondition of 2 and 3).

## 11. Driver too old, or other environment failure

torch 2.15.0 is published for Linux only as a CUDA 13.2 build, which needs driver
R580 or newer. If `env_check` reports an older driver, stop. Two options need an
explicit decision and a new decisions entry before anything else runs:
- torch 2.14.1 + cu126, which changes the torch version from the Mac;
- asking TACC about the driver.

Do not switch silently. The same applies to any failed gate: record it, do not retry
or loosen a tolerance without a decision.

### 11.1 Observed failure (job 3591308, reported by the user)

At commit `944905e`, account ASC26098, `gpu-a100-small`, node
`v330-007.ls6.tacc.utexas.edu`: one NVIDIA A100-PCIE-40GB, driver 570.195.03. The
installed environment was CPython 3.14.7, torch 2.15.0, CUDA 13.2. The R580 guard
stopped the job before the torch, test, doctor and preflight checks. No pilot or
continuation has run. Checkpoint 1000 has a hash-verified Ranch backup.

### 11.2 Proposed fallback (D-034): PROPOSED, NOT APPROVED, remote validation PENDING

Only the TACC environment changes; the Mac environment and every setting in §1, §5
and §8 are unchanged, including all tolerances.

| | D-033 (failed on R570) | D-034 fallback |
| --- | --- | --- |
| torch | 2.15.0 (CUDA 13.2), PyPI | 2.14.1+cu126, official wheel `torch-2.14.1+cu126-cp314-cp314-manylinux_2_28_x86_64.whl` by direct URL from `download-r2.pytorch.org/whl/cu126` (the link `download.pytorch.org/whl/cu126/torch/` publishes), sha256 `eb7564fdaccd2920ed78b4e3c08ca4a180fcaf5d4b3ce3068b1968bce4f5834b` |
| Driver guard | major ≥ 580 | full version ≥ 560.35.05 (CUDA 12.6 Update 3 minimum, NVIDIA release notes); job 3591308's 570.195.03 passes |
| triton | 3.9.0 | 3.8.0 (torch requires `~=3.8.0`) |
| CUDA libraries | CUDA 13.2 set, cuDNN 9.26.0.51 (`-cu13`) | CUDA 12.6.3 set, cuDNN 9.10.2.21 (`-cu12`); nvJitLink held at 12.6.85 |
| Shared packages | Mac versions | Mac versions (unchanged) |
| Sources | PyPI | torch from the PyTorch URL above; everything else PyPI only |

`env_check.slurm` now requires torch exactly `2.14.1+cu126`, `torch.version.cuda ==
"12.6"` and cuDNN 91002, besides the existing GPU, TF32, matmul, test, doctor and
preflight checks. Every job also refuses a venv whose `$WORK/dlmc/env/BUILD.txt` does not
record the sha256 of the lock in the checkout. `ENV_CHECK_PASSED` records that sha256;
the pilot accepts only a marker with the current one and keeps a copy
(`ENV_CHECK_PASSED.used`), and the continuation accepts only a pilot whose copy matches.
So the old venv, a marker from another environment, or a pilot run under another
environment cannot pass.

Specific risk, tested by the preflight comparison: torch 2.14.1 must load the
torch 2.15.0 checkpoint (optimizer state, RNG states) and reproduce the Mac-predicted
batches and RNG streams field by field. If it does not, the fallback fails; record it and
stop.

### 11.3 Commands, only after D-034 is approved

`<$WORK>` is the output of `echo $WORK` on ls6; `<update>` is the commit printed by
step 1. Steps 2–4 run on a login node (file operations only); step 5 on a `vm-small`
compute node; steps 6–7 are Slurm jobs.

1. **Mac — stage the code-only update** (no data, checkpoint or 7.24 GB bundle):
   ```bash
   DLMC_REMOTE=ayuj@ls6.tacc.utexas.edu DLMC_REMOTE_ROOT=<$WORK>/dlmc scripts/tacc/stage_update.sh
   ```
   It writes `runs/tacc_update/944905e..<update>/` (`dlmc-update.bundle`, `COMMITS`,
   `SHA256SUMS`, `DIFFSTAT.txt`) and prints the copy command and the apply commands
   with the exact commit IDs. Run the printed copy command (one ssh session).
2. **Login — preserve the failed job's logs and the old environment** (nothing is
   deleted; each `mv` refuses an existing target):
   ```bash
   J=3591308; D=$WORK/dlmc
   mkdir -p $D/retained/env_check/$J
   cp -p $D/repo/runs/slurm/dlmc-envcheck-$J.out $D/retained/env_check/$J/
   chmod -R a-w $D/retained/env_check/$J
   test ! -e $D/env.torch2150-cu132 && mv $D/env $D/env.torch2150-cu132 && chmod -R a-w $D/env.torch2150-cu132
   test ! -e $SCRATCH/dlmc/venv.torch2150-cu132 && mv $SCRATCH/dlmc/venv $SCRATCH/dlmc/venv.torch2150-cu132
   ```
   The old venv stays on `$SCRATCH` (purged after 10 days without access). It is
   rebuildable from the lock at `944905e`; to keep a copy anyway:
   `tar -C $SCRATCH/dlmc -cf $D/venv.torch2150-cu132.tar venv.torch2150-cu132`.
3. **Login — verify and apply the update** (the commands printed in step 1):
   ```bash
   U=$WORK/dlmc/updates/944905e..<update>; R=$WORK/dlmc/repo
   cd $U && sha256sum --strict -c SHA256SUMS
   test "$(git -C $R rev-parse HEAD)" = 944905e071572a8f82bb432526ea4cfe9950ed73
   test -z "$(git -C $R status --porcelain --untracked-files=no)"
   git -C $R bundle verify $U/dlmc-update.bundle
   git -C $R fetch $U/dlmc-update.bundle main:refs/remotes/update/main
   git -C $R merge --ff-only <update full sha>
   test "$(git -C $R rev-parse HEAD)" = <update full sha> && echo "UPDATE APPLIED"
   ```
   The original transfer (`$WORK/dlmc/transfer`, read-only), checkpoint 1000 and the
   Ranch backup are not touched.
4. **Login — re-check the transferred files** (unchanged; 41 files plus bundle):
   `cd $WORK/dlmc/transfer && sha256sum --quiet --strict -c runs/tacc_transfer/MANIFEST.sha256 && echo TRANSFER VERIFIED`
5. **Compute (`vm-small`) — rebuild the environment:**
   ```bash
   idev -p vm-small -N 1 -n 1 -t 00:30:00
   bash $WORK/dlmc/repo/scripts/tacc/setup_env.sh
   ```
   It needs network access to PyPI and `download-r2.pytorch.org`.
6. **GPU job — fresh environment check:**
   `cd $WORK/dlmc/repo && sbatch -A ASC26098 scripts/tacc/env_check.slurm`
7. **GPU job — the existing CUDA pilot**, only if step 6 wrote `ENV_CHECK_PASSED`:
   `DLMC_ENV_CHECK=$WORK/dlmc/retained/env_check/<step-6 jobid> sbatch -A ASC26098 scripts/tacc/pilot.slurm`

The continuation (§4 step 8) is submitted only after `PILOT_PASSED` from step 7 and a
time limit re-derived from its report. If step 6 or 7 fails, record it and stop.

## 12. Local checks performed (Mac, 2026-11-17)

- Preflight of checkpoint 1000 (`runs/preflight/ckpt-0001000-mac.json`): 14/14 checks.
  Batch predictions validated by running the same preflight on `ckpt-0000900`, which
  predicted all 100 recorded batches 901–1000 exactly (`check_expected_batches.py`).
- CUDA reference fixture made on CPU and self-checked (zero difference).
- Real-CLI smoke run on CPU, from an APFS clone of checkpoint 1000 (monitoring
  disabled, `runs/cli_smoke/`). Cross-device resume worked; batches 1001–1002 matched
  the preflight exactly; SIGUSR1 during update 1002 gave `stop_reason
  signal_SIGUSR1`, a verified `ckpt-0001002`, and the timer starting at origin 1000.
  The original checkpoint's hashes were unchanged afterwards.
- Tests, mutation checks, lock compile, and shell syntax and `shellcheck`: see
  `docs/verification.md`, "Warmup-continuation preparation".

## 13. Transfer manifest

Source: the Mac repository root. Destination: `$WORK/dlmc/transfer/` on ls6, at the
same relative path. `install_transfer.sh` links the directories into
`$WORK/dlmc/repo/` at that path again. Only these files move. The Mac virtual
environment, the rest of `runs/` (feasibility, Gate C, resume checks, the earlier
`ckpt-0000900`, failed attempts, upstream sources) and any credentials stay local.

| Path | Bytes | sha256 |
| --- | --- | --- |
| `runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000/COMPLETE.json` | 387 | `145dea5b78ac29e7a618574402e333a9a510a1617289de7f0ee72a8297452001` |
| `runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000/model.safetensors` | 1976163448 | `f48abe457d483b19b854e1fdf6be4bb294c702f949f9d76d7c44a2274b0d5a6a` |
| `runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000/optimizer.pt` | 3952497739 | `ae55b39481da056d6b4180cd0197a732efecaece7b17c0d0eab52aa0fbfa99af` |
| `runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000/rng.pt` | 30317 | `6c20ac394897297b28f854083791b72e4e247caefe708eec1919d76af5dcec75` |
| `runs/train/warmup-pilot-v1/checkpoints/ckpt-0001000/state.json` | 230372 | `3bca74fad6781f3cfcd8fb38d3abe5953f105beec6a1492235b261431efb2c0b` |
| `runs/train/warmup-pilot-v1/config.resolved.json` | 1477 | `df176973a32c312d4fcd3650ca08a3b8bd60b662d44008de1065455f44aafe12` |
| `runs/train/warmup-pilot-v1/config.sha256` | 65 | `ceb870ba3ff0302a2f0a735fce2192af27194adb2e7f2f6b8020d05cd946b877` |
| `runs/train/warmup-pilot-v1/metrics.jsonl` | 2016937 | `57ec6abdee1c647af936d5c11b11377bbd09e28fe797236eaef5b940773e2f36` |
| `runs/train/warmup-pilot-v1/evals.jsonl` | 4307 | `2b72fa3a215720b482d6001a35adbf97830afb634e5663eb28f6bdb153e445f8` |
| `runs/train/warmup-pilot-v1/events.jsonl` | 4564 | `68eace1b6e4a62e15ea9ea5a5d7a30bd1611373a0d1b2684ed83d49a1a34783c` |
| `runs/train/warmup-pilot-v1/summary.json` | 2464 | `e5189bb3f8f86b3f3aaa72a77c2d4f6a2de785c75c103d1d94a2f89e66a25317` |
| `runs/hf/open-dcoder-0.5B/README.md` | 1704 | `ae09bff0b21dd43db2cde645df3386aec5cc6d78326b85aa9bfd9c193c19e2c0` |
| `runs/hf/open-dcoder-0.5B/__init__.py` | 600 | `ced14be479d27e864cf3557c37d8abe00d2204ba6c030620bf48ea85e25a5921` |
| `runs/hf/open-dcoder-0.5B/added_tokens.json` | 622 | `8ce9ccd426d4ac0e822dca98ae1a0f94099c63c77af73c3bc0d390f8bbf41e98` |
| `runs/hf/open-dcoder-0.5B/config.json` | 723 | `fcd627396dda21d8f93425a6c18f1be20cd6bfe219f5e939fd33ba8357d7a1d9` |
| `runs/hf/open-dcoder-0.5B/generation_config.json` | 146 | `c269052c80cbb710726e30ad35c24172486bd1be17c64681a2230cafc5057119` |
| `runs/hf/open-dcoder-0.5B/generation_utils.py` | 10750 | `f9f092c97da1e54d3024f62d6c5c21542f994a1e26a7cfc2dd7fd726a44ba701` |
| `runs/hf/open-dcoder-0.5B/merges.txt` | 1671853 | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |
| `runs/hf/open-dcoder-0.5B/model.safetensors` | 1260367448 | `59c1a005f4b672bdd3bbdab6258b283dff3b87bab5cc4bbc0d479e8388f77b6e` |
| `runs/hf/open-dcoder-0.5B/modeling_qwen2.py` | 30939 | `7194b1fbabecf0ebdadebf14b345cc64c4ca0d16166f563877d5d62f81d719b1` |
| `runs/hf/open-dcoder-0.5B/special_tokens_map.json` | 753 | `6e57d1b59798c99b30b53fa9536e328f8af2aa5616d2c03bea5f0f83f676989b` |
| `runs/hf/open-dcoder-0.5B/tokenizer.json` | 11422076 | `4f3ead5ffbb2abfb9e51757b104af5174f59c994ff3b2ca715d344dc66ee1f4e` |
| `runs/hf/open-dcoder-0.5B/tokenizer_config.json` | 7478 | `fd348836e1a0a9444e4ceca15c26169f95576ca65b570154a0a5dcc9111c2633` |
| `runs/hf/open-dcoder-0.5B/vocab.json` | 2776833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `runs/data/pilot-seed20261116/manifest.json` | 7510 | `990ecd1dd2341b81bc742fa91189b129877cd9b48b196fbdbc208e8586c81536` |
| `runs/data/pilot-seed20261116/provenance.json` | 681 | `b3b7502ed1d1ef181b32e24428b99de186838aec727794d560bfaea343d99312` |
| `runs/data/pilot-seed20261116/train.jsonl` | 20241509 | `215fdb4a8116f16925f0236f5d96718b51d5f22f81e4ef8d58fe94677ee2e18a` |
| `runs/data/pilot-seed20261116/val.jsonl` | 4040871 | `ae3333a7a9175ecd6b6b9dbc2c199def19580d0491e03f830f212955fac6d566` |
| `runs/data/pilot-seed20261116/test_id.jsonl` | 4055132 | `e7dc645c26461428ec80b3c1e5303530ffd85115dad7181d53572d1d99ef1d74` |
| `runs/data/pilot-seed20261116/test_long.jsonl` | 1636742 | `9d6095e84269380c1c3518b40fad8b0d7e5d09383881565b8cf03b4e9154ea58` |
| `runs/data/pilot-seed20261116/test_unseen.jsonl` | 2026544 | `fa8475baa5265729841acd4467dbb8623b1bc96462a8eb16cc7ba4216f645129` |
| `runs/data/pilot-seed20261116/tokenizer_reference_check.json` | 684 | `01f78252100aaa705f88fa5ae2b19d3c157a6be2af3f7056b1fe95bec5178114` |
| `runs/data/pilot-seed20261116/tokenizer_reference_run.log` | 684 | `01f78252100aaa705f88fa5ae2b19d3c157a6be2af3f7056b1fe95bec5178114` |
| `runs/calibration/stage2-warmup1000/summary.json` | 107226 | `ade11fe8ec6003be98ac4882d7302ce81b97aa5d82cbad0d0a65cf5a097d6978` |
| `runs/calibration/stage2-warmup1000/summary.md` | 5018 | `679d77d3ef18b0fefc38571d56d28884bf9686086748367114383cf46da952fb` |
| `runs/calibration/stage2-warmup1000/records.jsonl` | 4273219 | `b0c8edff3f4833be8d9b8db79fb6954984f969b40e9ff1a564517c84d27719c6` |
| `runs/calibration/stage2-warmup1000/traces.txt` | 3532 | `98c2d96ac4050de92ff98bc9be6094068d66dfe67420d7160155e55a571b9f81` |
| `runs/calibration/screen-warmup1000/stage2_selection.txt` | 1927 | `48e57dc01c9d523495f61609da05c2900d73a739a0f0b46be017c37ec9b2bfaf` |
| `runs/reference/cuda_pilot_fixture.pt` | 19168 | `b5dc70811254bdf4394499693836e5cf6e38dc0b0f38e31af11872e6a8ce3e74` |
| `runs/reference/cuda_pilot_fixture.json` | 20956 | `0f3c48557274ac74e862686ab56d13c5cc6669c35da5ea5cd7caa7db56a6fd24` |
| `runs/preflight/ckpt-0001000-mac.json` | 16631 | `a09068e309e334d39489d71bfe4438733cea4095f1add1caff8bcba7524faf5c` |
| `runs/tacc_transfer/dlmc.bundle` | written by `stage_transfer.sh` | git bundle of `main` at the transfer commit |
| `runs/tacc_transfer/COMMIT` | written by `stage_transfer.sh` | commit ID and branch |
| `runs/tacc_transfer/MANIFEST.sha256` | written by `stage_transfer.sh` | sha256 of every file above |

The 41 data files total 7,243,702,036 bytes (7.24 GB): checkpoint 1000 5.93 GB, backbone
1.28 GB, dataset 32 MB, calibration, logs and reference files 6.5 MB. The bundle adds a
few MB. The three test-split files are transferred only because the dataset manifest
covers them; nothing loads them. `tokenizer_reference_check.json` and
`tokenizer_reference_run.log` have identical content, as on the Mac.
