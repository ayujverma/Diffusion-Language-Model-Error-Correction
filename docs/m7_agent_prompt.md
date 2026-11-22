# Milestone 7 agent brief (fresh session)

Copy everything below the line into a new Cursor agent chat. The file is also in
the repo, so the agent can re-read it.

---

You are continuing the CS 395T project “Does Generation-Time Confidence Improve Diffusion Self-Correction?” (Ayuj Verma). Fresh session — do not rely on prior chat memory.

You will implement Milestone 7 (Family B training adapters), record the decisions that lock it, update the docs, write Slurm scripts the user will submit, and at the end explain the Mac→TACC transfer. You will not run anything on TACC. You will not push. You will split the work into small commits authored only by Ayuj Verma.

════════════════════════════════════════════════════════════════════════
0. FIRST ACTIONS (read before writing code)
════════════════════════════════════════════════════════════════════════

Read, in order:

1. `docs/handoff.md` — status through M6. If it disagrees with a later doc, the later doc wins.
2. `docs/implementation_plan.md` — Milestone 6 (done as software) and Milestone 7 (your job). Milestone 8 is out of scope.
3. `docs/decisions.md` — especially D-002, D-005, D-010, D-011, D-017, D-026, D-028, D-029, D-030, D-044, D-045, D-046, D-047, D-049, D-050. Supersede by adding a new D-xxx. Do not edit old decision text.
4. `docs/source_audit.md` — §3.1 ProSeCo, §3.2 SCOPE+D3IM, §5 open questions Q-4, Q-5, Q-6.
5. `docs/verification.md` — Milestone 5 EXPERIMENT COMPLETED, and the Milestone 6 TESTED LOCALLY section.
6. `docs/inputs/Cursor_Diffusion_LM_Build_Prompt.txt` — §7 loss, §8 prior-method adaptations, §9 sampling and call accounting, §10 evaluation, Gate B and Gate D. This is the protocol.
7. `docs/data_format.md` and `docs/milestone4_bank.md` — response layout and the frozen bank. Family B does not train on that bank. Family A already did.
8. Code you must reuse, not reinvent:
   - `src/diffusion_correction/models/alignment.py` — the only logit shift
   - `src/diffusion_correction/models/adapter.py` — `aligned_logits`, no labels
   - `src/diffusion_correction/sampling/proseco.py`, `d3im.py`, `nfe.py`, `monotone.py`
   - `src/diffusion_correction/evaluation/call_budget.py`, `evaluation/repair.py`
   - `src/diffusion_correction/training/correction_trainer.py` and `configs/correction/m5_*.yaml` — Family A pattern for checkpoints, CLI, and Slurm. Do not change those arms.
   - `src/diffusion_correction/losses/` — CE, L_mask, L_error, L_keep, rank weights
   - `runs/sources/proseco/` (gitignored local clone, pin `43ce881`) — `llada/generate.py` and the corrector loss in `diffusion.py`. Read it. Do not copy it onto Open-dCoder and call it a reproduction.
9. Upstream notes already in the audit. SCOPE has no pinned code repository. Do not invent a repo.

Then run `.venv/bin/python -m pytest -q` once so you know the tree you started from. The M6 check was 300 passed. Do not start training code until Q-4 and Q-5 are new decision entries.

════════════════════════════════════════════════════════════════════════
1. PROJECT, GLOBALLY
════════════════════════════════════════════════════════════════════════

Author: Ayuj Verma. Working title: Does Generation-Time Confidence Improve Diffusion Self-Correction?

Backbone: Open-dCoder-0.5B only (`fredzzp/open-dcoder-0.5B` @ `d0d86d5`). Bidirectional Qwen2. Logits are shifted: raw position i−1 predicts token i. Every loss and every sampler must use `align_logits` / `DiffusionAdapter.aligned_logits`. Prompt and pad are immutable. There is no EOS in a response (D-018). There is no LLaDA copy-over: a visible token’s prediction can differ from the token sitting there (D-050).

Why not LLaDA: the approved protocol fixed Open-dCoder. ProSeCo’s released sampler loads LLaDA-8B. Full fp32 fine-tuning of 8B does not fit one A100 40GB. Family A, the warmup, and the bank already exist on Open-dCoder. Do not switch backbones. Do not describe results as reproductions of the ProSeCo or SCOPE papers.

Generator: continuation checkpoint 8000 (D-038), model sha256 `a5ceb4ffbf70ecc49e9fb48d7a0cdd01ccafa4ce81a6b2f91c73e0bd1b755341`. Bank `m4-ckpt8000-v1` is immutable (D-043): identity `e3e810bef987753ebdb8c9cf002de0f113eb9c4e5c61843094893ca5b22203ea`, manifest `c5901617bbc6467483d1a91d8679c881dc055d62bbe993aa03fe181092e71ac4`. Family A trained from that checkpoint on the bank. Family B also starts from that checkpoint, but builds new drafts online. Do not retrain, delete, or retune the six Family A arms.

Two families, and they are not the same algorithm:

Family A — DONE (D-047). Fixed saved drafts. Loss is L_mask + λ L_error + μ L_keep with λ = μ = 1. Only the weight on wrong tokens changes: uniform, high_conf, gen_difficulty (−log q), current_loss, low_conf, shuffled_conf. Screen: 1000 updates, seed 20261120, batch 16 (D-045), β = 0.5. Held-out metric so far is one greedy repair on val drafts at T = 4. High-c did not beat uniform (wrong tokens after repair: low_conf 3175, uniform 3180, high_conf 3314, shuffled 3367, gen_difficulty 3483, current_loss 3571; @0 was 6589). That is a screen, not the primary metric, and not a ProSeCo result.

Family B — YOUR JOB. Online drafts from the model being trained. Four names, kept distinct:

1. `proseco_basic` — ProSeCo’s two-pass corrector with cross-entropy on the draft positions, adapted.
2. `proseco_error_prioritized` — the appendix Eq. 11 variant: CE on wrong draft positions, and the probability (not the log probability) on already-correct positions. Different name. Do not fold this into basic or into gen_conf.
3. `proseco_gen_conf` (D-046) — same online draft as basic, but the CE on error positions is rank-weighted by the stop-gradient log-probability of the committed draft token from the draft-construction forward. CE stays on both error and keep positions. This is the user’s “weight ProSeCo by generation confidence” arm. It is not Eq. 11. It is not Family A. Family A’s high-c result is a reason to be honest, not a reason to drop the arm.
4. SCOPE, evaluated with the existing D3IM sampler. SCOPE is training. D3IM is sampling. SCOPE does not have its own sampler. D3IM does not have its own training loss.

Primary metric for every method, including the six Family A checkpoints: final-answer accuracy at one validation-chosen model-call budget, same NFE rules (D-049). Do not select that budget, and do not touch `test_id`, `test_long`, or `test_unseen`. M5 greedy repair stays available and is not the comparison.

Signals that must not be mixed up:

- c: probability of the token that was committed, at the forward that committed it.
- q: probability of the gold token at that same forward. d_gen = −log q.
- d_now: the current model’s CE on the gold token.
- For `proseco_gen_conf`, the weight uses log c of the committed draft token from the first forward of that training step, stop-grad. Not q. Not the second forward. Not the frozen bank.

════════════════════════════════════════════════════════════════════════
2. WHAT THE SAMPLERS ALREADY DO (do not rebuild them)
════════════════════════════════════════════════════════════════════════

Milestone 6 is software, tested on the tiny backbone (D-049, D-050). One NFE is one `aligned_logits` call. A call past the cap does not run.

Monotone unmasking (`sampling/monotone.py`, D-029): start from a masked response. Each call reveals new tokens. A token that has been written never changes. S = 0 on the ProSeCo-style sampler matches this.

ProSeCo-style sampler (`sampling/proseco.py`): two sequences. The main sequence still has masks. On steps divisible by ω, a temporary complete draft is built by filling those masks, then up to S argmax passes run on that draft. Visible main-sequence tokens can change. Masks stay until the unmask quota is written from the last logits. Provisional defaults ω = 2, S = 4, from the pinned ProSeCo code, not a selected budget. Differences from LLaDA’s `llada/generate.py` are listed in D-049 (aligned logits, allowed-token argmax, no Gumbel, no EOS blocks, D-029 reveal schedule). Do not “fix” this back toward LLaDA’s sampler.

D3IM (`sampling/d3im.py`): one forward per step. Re-predict the active block, rank the new candidate by its float32 full-vocabulary probability, keep the top K, put every other active position back to mask. A visible token can become mask. K = floor(L·t/T) before the block’s last step, and the whole block on the last step. Optional earlier blocks freeze. This is not a confidence threshold on monotone unmasking. `extra_unmask` is the other control: it spends leftover calls revising the current token and does not remask. It is not ReMDM (Q-6 stays open; do not implement ReMDM in M7).

Eval entry points that already exist: `dlmc evaluate-calls`, `dlmc evaluate-repair` (`--method greedy` is M5; `--method proseco` is the multi-step repair). Test splits are refused.

════════════════════════════════════════════════════════════════════════
3. WHAT MILESTONE 7 IS
════════════════════════════════════════════════════════════════════════

Train the four Family B adapters on Open-dCoder, with stop-gradient fixtures, configs, CLI, tests, and Slurm the user can submit. Wire their checkpoints into the M6 call-budget harness as consumers. Keep Family A loadable by the same scripts.

Order, and do not skip ahead:

1. Decision entry (next free id after D-050; check the file) that closes Q-4 for training, before any ProSeCo trainer code.
2. `proseco_basic` loss + trainer step + stop-gradient fixture.
3. `proseco_error_prioritized` as a separate loss + fixture. A test must fail if the keep term is CE, or if the error term is probability.
4. Decision is already D-046 for the math. Implement `proseco_gen_conf` only after the basic and Eq. 11 fixtures pass. A test must fail if the weight depends on the gold token, on the second forward, or if gradient flows into the rank weights or the argmax.
5. Decision entry that closes Q-5, before any SCOPE trainer code.
6. SCOPE training step + fixture (which positions get loss, which pass is no_grad, clean context excluded, MDM mix).
7. Eval path: a trained Family B checkpoint can be scored with `evaluate-calls` under monotone, proseco, and d3im, on val only. Do not run the full val job yourself.
8. Configs, CLI (`dlmc train-family-b` or the smallest extension that fits `cli.py`), unit tests on the tiny backbone.
9. Slurm scripts modeled on `scripts/tacc/correction_screen.slurm` and `repair_eval.slurm`. User submits. Include D-048’s one-line `squeue` EndTime fix if a script computes a stop deadline.
10. Docs: implementation_plan, handoff, verification, source_audit Q-4/Q-5 status, ai_usage. Label IMPLEMENTED / TESTED LOCALLY. Do not write EXPERIMENT COMPLETED. No TACC result exists until the user runs the jobs and the artifacts are checked.

Q-4 choice to record, unless you find a concrete contradiction in the pinned ProSeCo code or D-046:

Training draft ŷ is stop-grad argmax over the D-029 allowed set at every editable response position, from aligned logits. Prompt and pad stay as they are. This matches ProSeCo Algorithm 1 (argmax at all positions) and the M6 temporary complete draft. The ProSeCo code default (keep the visible token, argmax only where masked) is the alternative that copy-over makes equivalent on LLaDA. State that alternative in the decision and say it is not the training rule on Open-dCoder. Do not feed gold tokens into the positions that are masked in z_t.

Q-5 choice to record, unless the paper text you re-read forces the other reading:

SCOPE ranks commits by the temperature-1 probability of the sampled token x̂, not by the max probability. D3IM’s sampler confidence stays what D-049 already says (probability of the new allowed argmax). Write both definitions in the decision so a later reader cannot mix them. The paper line is only “confidence(ẑ) // temperature = 1 logits”.

If you believe either choice is unsafe, stop and write the decision as two explicit options with the consequence of each. Do not write the trainer while the choice is still open.

════════════════════════════════════════════════════════════════════════
4. TRAINING RULES (do not drift)
════════════════════════════════════════════════════════════════════════

Same interface as the rest of the project:

- Full fine-tune, fp32, AdamW, the D-030 hyperparameters, gradient clip 1.0, batch 16 (D-045: batch 32 OOMs on an A100 40GB when a step does two forwards with grad). If a Family B step does more forwards, say so in the Slurm comment and keep one batch size for every Family B arm. Do not silently shrink one arm.
- Init: checkpoint 8000. Do not start from a Family A `ckpt-0001000`.
- Masked term: D-028 / D-017. Response-only Bernoulli mask, clamped-uniform t, pooled CE, no 1/t on L_mask, no Open-dLLM path term, zero-mask minibatches skip the optimizer. Family B’s extra time weight a(t), if you use one, applies to the ProSeCo corrector term only, and the decision must say whether a(t) is 1/t or the paper’s α̇/(1−α). Pick one, test it, and use it for basic, Eq. 11, and gen_conf.
- Online draft: sample the mask from the gold response (labels are allowed in the training step). First forward builds ŷ with stop-grad. Second forward is the one that receives gradient, conditioned on ŷ. The checker and the gold ids must not be an input to a sampler at eval time. Training may use gold to build the mask and the loss. Eval may not.
- Rank weights for gen_conf: D-011, β = 0.5, upward on log ĉ over error positions, groups that you define in the decision. Online steps may not have the bank’s (budget, stage) pair. Say what the group is (for example the mask rate bin of that step, within the minibatch). Do not reuse saved bank log_c. Weights are float64 on CPU, no gradient, then cast.
- Empty E or K: graph-connected zero, and log the emptiness. Same as D-010.
- Stop-gradient fixtures, tiny backbone, before any long run:
  - Permuting the gold ids changes the loss and must not change ŷ or the first-forward tokens when the RNG is fixed.
  - No gradient into the argmax tokens or the rank weights. A finite-difference or autograd check on a 1-layer or the tiny model is enough.
  - Eq. 11 keep-term gradient is not the CE gradient. A test with an analytic derivative on one logit row must catch a CE implementation.
  - gen_conf: increasing log ĉ of one error, holding the others fixed, increases that row’s weight and scales its logit gradient. Weighting by q, or by the second-forward probability, fails a test.
  - SCOPE: loss positions are committed-this-step ∪ still-masked. Originally visible clean tokens are excluded. The draft pass is no_grad. A 0.5 MDM mix is an explicit Bernoulli with its own RNG stream, not Python’s random.
- Checkpoints: the existing atomic checkpoint format (`training/checkpoint.py`). Resume must refuse a changed arm name or loss. One seed for the screen scripts (`20261120` unless the decision says otherwise). Three seeds are Milestone 8.
- Screen length: 1000 updates, matching M5, unless a decision explains a shorter smoke. The Slurm script should support `DLMC_MAX_UPDATES` so the user can smoke-test with a handful of updates first.
- Memory: call `release_cached_memory` on any loop whose shapes vary (D-032). TF32 off on CUDA, matching D-033.

Do not add a SCOPE-plus-confidence-weight arm. That is not in the plan. Do not drop low_conf or shuffled_conf from the project. Do not retune λ, μ, or β.

════════════════════════════════════════════════════════════════════════
5. EVAL AND SLURM (user runs TACC)
════════════════════════════════════════════════════════════════════════

Scripts live in `scripts/tacc/`. Follow `correction_screen.slurm` and `call_budget_eval.slurm`: `gpu-a100`, one GPU, `require_job`, `select_gpu` once before any Python, no `srun` / multi-GPU. Account ASC26098. Paths: repo `/work/10692/ayuj/ls6/dlmc/repo`, retained under `$WORK/dlmc/retained`. Data dir and manifest sha are in `scripts/tacc/tacc_env.sh`.

Prepare, do not submit:

1. A training array, one task per Family B arm: `proseco_basic`, `proseco_error_prioritized`, `proseco_gen_conf`, `scope`. Same batch, seed, update cap, and init checkpoint. Val split only inside the trainer’s own monitor if you add one. Refuse test splits in the training CLI the same way `calibrate` does.
2. An eval script that calls `dlmc evaluate-calls` on val for those checkpoints and, optionally, a Family A arm via `DLMC_M6_ARM` or an explicit checkpoint. Methods `monotone`, `proseco`, and `d3im`. Budgets as a comma list defaulting to `1,2,4,8`, with the comment that this is not a selected primary budget. Default `DLMC_M6_N=32` for a smoke; `all` for the full val split. The user chooses.
3. Header comments: NOT SUBMITTED, does not train Family A, does not read test splits, ω and S are the D-049 defaults.

At the end of your session, print the exact Mac and login-node commands for a code-only update. The pattern is `scripts/tacc/stage_update.sh`. It refuses a dirty tree, so commits come first. The user pushes. Typical shape, with the real base filled in only after you know TACC’s HEAD:

```bash
# login node, to learn the base:
git -C /work/10692/ayuj/ls6/dlmc/repo rev-parse HEAD

# Mac, repo root, after the commits are pushed or the bundle is the transfer:
DLMC_UPDATE_BASE=<tacc HEAD> \
DLMC_REMOTE=ayuj@ls6.tacc.utexas.edu \
DLMC_REMOTE_ROOT=/work/10692/ayuj/ls6/dlmc \
scripts/tacc/stage_update.sh
```

Then tell the user to run the `tar | ssh` line the script prints, and the login-node `sha256sum`, `git fetch`, and `git merge --ff-only` lines it prints. No weight transfer, no bank transfer, no venv rebuild (editable install, lock unchanged unless you actually change dependencies — do not). Do not claim the update was applied.

Local M6 commits already on this machine, if TACC does not have them yet, are `fb00b61` through `4affc74`. Verify with `git log`; do not assume TACC’s HEAD.

════════════════════════════════════════════════════════════════════════
6. COMMITS
════════════════════════════════════════════════════════════════════════

Small commits, one concern each, imperative subject, one sentence on why. Suggested cut: Q-4 decision; proseco_basic; Eq. 11; gen_conf; Q-5 decision; SCOPE; CLI and configs; tests; Slurm; docs. Adjust if a cut would leave a broken import.

Author must be Ayuj Verma `<ayuj@utexas.edu>`, taken from `git log -1`. No `Co-authored-by`. No Cursor trailer.

In this environment, `git commit` is rewritten to add a Cursor co-author trailer and can be blocked. Do not use that path. Do not change git config. Do not use `--no-verify`. Build each commit with the index and `git commit-tree`, then move HEAD:

```bash
author_name=$(git log -1 --format='%an')
author_email=$(git log -1 --format='%ae')
export GIT_AUTHOR_NAME="$author_name" GIT_AUTHOR_EMAIL="$author_email"
export GIT_COMMITTER_NAME="$author_name" GIT_COMMITTER_EMAIL="$author_email"
git add <paths for this commit only>
tree=$(git write-tree)
parent=$(git rev-parse HEAD)
new=$(git commit-tree "$tree" -p "$parent" -m "$(cat <<'EOF'
Subject line.

Why, in one sentence.
EOF
)")
git update-ref HEAD "$new"
```

After all commits, `git log -1 --format='%B'` and a scan of `git log <base>..HEAD` must show no `Co-authored-by` and no Cursor. `git status` must be clean. Do not push. Tell the user the hashes and that they push.

════════════════════════════════════════════════════════════════════════
7. DONE WHEN
════════════════════════════════════════════════════════════════════════

- New decision entries close Q-4 and Q-5 and state a(t) and the gen_conf grouping.
- Four trainers exist, names distinct, init checkpoint 8000, batch shared.
- Stop-gradient and “wrong term” tests pass on the tiny backbone.
- `pytest` passes. Say the count. Do not claim a real-backbone training run you did not do.
- Slurm files are written and not submitted.
- Docs updated. No EXPERIMENT COMPLETED. No test-split leakage.
- The final message explains the transfer commands and lists the commits.
- Handoff says Milestone 8 is next (cost estimate and the multi-seed grid), and that Family B training has not been run on TACC until the user says so.

════════════════════════════════════════════════════════════════════════
8. OUT OF SCOPE
════════════════════════════════════════════════════════════════════════

Milestone 8’s three-seed grid, bootstrap, paper plots, and a selected primary budget. ReMDM / Q-6. A SCOPE arm that also rank-weights by confidence. Retraining Family A. Switching to LLaDA. Running or submitting Slurm. Pushing. Inventing remote results. Editing old D-xxx text in place.
