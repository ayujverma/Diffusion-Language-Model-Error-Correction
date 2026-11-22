# Milestone 6 agent brief (fresh session)

Copy everything below the line into a new Cursor agent chat.

---

You are continuing the CS 395T project **“Does Generation-Time Confidence Improve Diffusion Self-Correction?”** (Ayuj Verma). Fresh session — do not rely on prior chat memory.

════════════════════════════════════════════════════════════════════════
0. FIRST ACTIONS (read before writing code)
════════════════════════════════════════════════════════════════════════

Read, in order:

1. `docs/handoff.md` (current status through M5; next = M6)
2. `docs/implementation_plan.md` — **Milestone 6** and **Milestone 7** (know the boundary)
3. `docs/decisions.md` — at least D-022, D-029, D-011, D-044–D-048, **D-046**
4. `docs/verification.md` — Milestone 5 EXPERIMENT COMPLETED section
5. `docs/source_audit.md` §3.1 ProSeCo sampler notes; §5 open questions (Q-4, Q-5, Q-6)
6. `docs/milestone4_bank.md` — bank schema / paths (eval may load bank drafts)
7. `docs/inputs/Cursor_Diffusion_LM_Build_Prompt.txt` — especially sampler / call-budget
   sections (shared ProSeCo-style correction sampler; fixed total model-call budget;
   repair of complete drafts; no test-split selection)
8. Existing code: `src/diffusion_correction/sampling/monotone.py`,
   `src/diffusion_correction/evaluation/repair.py`,
   `src/diffusion_correction/evaluation/benchmark.py` (and calibrate/compare if present),
   `src/diffusion_correction/cli.py`

Do **not** start Milestone 7 training adapters unless the user explicitly expands scope.
Do **not** retrain Family A arms. Do **not** invent EXPERIMENT COMPLETED claims.

════════════════════════════════════════════════════════════════════════
1. PROJECT CONTEXT (what already exists)
════════════════════════════════════════════════════════════════════════

**Question (Family A):** When training drafts are held fixed, does weighting wrong
tokens by saved generation-time confidence improve correction beyond uniform /
d_gen / current loss?

**Family A (shared bank) — DONE:** Six arms on immutable bank `m4-ckpt8000-v1` from
generator `ckpt-0008000`: `uniform`, `high_conf`, `gen_difficulty`, `current_loss`,
`low_conf`, `shuffled_conf`. Trained 1000 updates (D-044/D-045/D-047). Screen finding:
correction training helps a lot vs @0; **high-c weighting did not beat uniform**;
`low_conf` ≈ `uniform` best on net wrong tokens. Keep **all six** checkpoints for
later matched-budget eval (user decision; SU available).

**Family B (prior methods) — NOT DONE:** ProSeCo basic, ProSeCo error-prioritized,
**`proseco_gen_conf` (D-046)**, SCOPE+D3IM. These are **Milestone 7**, not M6.
M6 only builds the **inference samplers + NFE/call accounting** those methods (and
Family A) will use.

**Important distinction the user cares about:**
- Family A ≠ “improved ProSeCo.” It is fixed-bank loss reweighting.
- `proseco_gen_conf` (D-046) is the planned train-time ProSeCo improvement (online
  draft + rank-weight error CE by stop-grad gen confidence). Implement training in
  **M7**. M6 may implement the **ProSeCo-style correction sampler** those checkpoints
  will share at eval time.

**Primary scientific metric (build prompt / handoff):** final-answer accuracy at a
**validation-selected fixed total model-call (NFE) budget**, same accounting for every
method. M5’s one-forward greedy repair is a screen metric only.

════════════════════════════════════════════════════════════════════════
2. WHAT MILESTONE 6 IS (and is not)
════════════════════════════════════════════════════════════════════════

### M6 IS
Software for **generation / correction at inference** and a shared **eval harness**:

1. **ProSeCo-style correction sampler**  
   - Scheduled unmasking with correction every ω steps.  
   - Up to S fixed-point argmax correction iterations on the active region.  
   - Distinguish temporary complete drafts vs main masked state (per ProSeCo / build
     prompt).  
   - **Count every model forward** (ordinary unmask steps, each correction iteration,
     final calls) as NFEs / model calls.

2. **Repair-of-complete-draft mode beyond M5**  
   - M5 `evaluation/repair.py` = one greedy forward rewriting all editable tokens.  
   - M6: multi-step / budgeted repair using the correction sampler (and keep the M5
     one-shot path for regression).

3. **D3IM (and related) inference path** as needed for later SCOPE eval and
   compute-matched controls (build prompt: extra-unmasking / ReMDM-style controls).

4. **Call / NFE accounting + eval harness**  
   - One interface: run method X on a problem set with a **max call budget** (and/or
     record full accuracy-vs-calls curves).  
   - Log final-answer / value / trace / malformed (D-020 checker), plus calls used and
     wall time.  
   - Validation-only for any budget **selection**; refuse test splits for selection
     (`dlmc calibrate` already refuses tests — same rule).  
   - CLI entry points + configs + unit tests (tiny backbone).  
   - Timing hooks for Milestone 8 cost estimates.

5. **Compute-matched ordinary MFT / extra-unmask controls** at inference (same call
   budget), so Family A correctors and baselines are comparable.

### M6 IS NOT
- Training ProSeCo, SCOPE, or `proseco_gen_conf` (→ M7).
- Retraining or dropping Family A arms.
- Running the full multi-seed main grid (→ M8).
- Touching test_id / test_long / test_unseen for selection.
- SSH/sbatch on TACC unless the user asks you to prepare scripts only; **user runs**
  remote jobs. No credentials in the repo.
- Claiming EXPERIMENT COMPLETED without verified artifacts.

════════════════════════════════════════════════════════════════════════
3. DESIGN CONSTRAINTS (must follow)
════════════════════════════════════════════════════════════════════════

- Open-dCoder alignment: bidirectional, shift so output i−1 predicts target i; same
  interface as training (`DiffusionAdapter.aligned_logits`). See D-002 / adapter.
- Prompt/pad immutable; never feed gold answers, error masks, or checker signals into
  the sampler to choose edits (build prompt).
- Candidate restrictions: explicit allowed-token mask (as in D-029 / repair).
- Stop-gradient / discrete argmax details that affect **training** adapters are M7
  (Q-4); for **sampling**, document how correction logits / remasking work and record
  any ambiguity in a **new D-xxx** before locking behavior.
- Supersede decisions with new entries; do not silently edit old D-xxx text.
- Ask before git commits. Prefer no Cursor co-author trailer if the user requests
  Ayuj-only commits (user has asked this before).
- Match existing code style; minimal diffs; tests for NFE counting and sampler
  invariants on the tiny backbone.

════════════════════════════════════════════════════════════════════════
4. SUGGESTED IMPLEMENTATION ORDER
════════════════════════════════════════════════════════════════════════

1. **Decision entry (D-049 or next free):** freeze M6 sampler/eval contracts before
   large code: ω/S defaults (provisional), what counts as one NFE, how budgets stop
   mid-trajectory, relationship to D-029 monotone unmask, how complete-draft repair
   uses the new sampler vs M5 greedy.
2. **NFE / call counter** primitive used by all samplers (impossible to bolt on later).
3. **ProSeCo-style correction sampler** on top of / beside `monotone_unmask`.
4. **Budgeted eval runner** (val problems or bank drafts): accuracy vs calls; fixed
   budget cutoff.
5. **Complete-draft repair** via correction sampler (multi-step) + CLI.
6. **D3IM / extra-unmask control** stubs or full paths as scoped in the decision.
7. **Configs + CLI** (`evaluate-...` commands) + TACC script stubs (user submits).
8. **Docs:** update `implementation_plan.md` / `handoff.md` / `verification.md` /
   `ai_usage.md` with IMPLEMENTED / TESTED LOCALLY; no fake TACC results.
9. **Local tests:** `.venv/bin/python -m pytest -q` (or project’s usual filter).

Wire Family A checkpoints into the harness as **consumers** (load `ckpt-0001000` from
each `m5-*-v1` run) only for smoke/local or user-approved TACC eval — not a new
training run.

════════════════════════════════════════════════════════════════════════
5. USER PREFERENCES FOR THIS PROJECT (honor these)
════════════════════════════════════════════════════════════════════════

- Keep **all six** Family A arms through later milestones for comparison.
- Primary story still includes answering the confidence-weighting question (Family A)
  **and** later improving ProSeCo via **`proseco_gen_conf` (D-046)** at train time in
  M7 — M6 must not paint Family A as “the ProSeCo method.”
- Eval comparisons must be **call-budget matched** once the harness exists.
- Plenty of TACC SU: prefer correctness and clear accounting over premature pruning
  of methods.
- Do not mark scientific success from M5 greedy repair alone when discussing M6+.
- User-only remote execution; prepare exact commands when TACC is needed.

════════════════════════════════════════════════════════════════════════
6. DONE CRITERIA FOR M6
════════════════════════════════════════════════════════════════════════

- [ ] New decision(s) recorded for sampler/NFE/budget contracts.
- [ ] ProSeCo-style correction sampler implemented with explicit NFE counting.
- [ ] Eval harness can run at least: ordinary unmask baseline vs correction sampler
      under a fixed call budget on val (tiny and/or real backbone locally as feasible).
- [ ] Complete-draft repair path that uses the M6 sampler (M5 greedy preserved).
- [ ] D3IM / control path present at the level promised in the M6 decision (full or
      clearly stubbed with follow-ups named).
- [ ] CLI + unit tests green; docs updated; no test-split leakage; no invented results.
- [ ] Clear handoff note: M7 starts ProSeCo/SCOPE/`proseco_gen_conf` **training**
      adapters next.

════════════════════════════════════════════════════════════════════════
7. OUT OF SCOPE REMINDERS
════════════════════════════════════════════════════════════════════════

Milestone 7 trains: `proseco_basic`, `proseco_error_prioritized`, `proseco_gen_conf`
(D-046 math), SCOPE+D3IM. Milestone 8 is the multi-seed grid and paper analysis.
Do not collapse those into M6.
