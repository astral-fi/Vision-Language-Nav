# Motion head: design choices

System 2 = frozen FutureNav-4B. System 1 = the motion head, the part we train. This document lists
every design choice: what our own results already settle, and what is still open. Each open choice has
a recommendation.

## Decisions (2026-10-30)
- **Role:** full policy (1b). The head decides motion, route deviations and STOP; FutureNav's action
  probabilities are an input.
- **Fast vision:** VGGT tokens of the current frame plus the last 3 frames (5a), taken from the per-step
  `advance_vggt` call.
- **Generator:** rectified flow main model + regression baseline on the same network (3).
- **Route information:** L18/22/26/final + FutureNav's action now; instruction-token states only if
  bend-side accuracy stays at FutureNav's level (4).
- **Robot:** tb3 waffle first. It is differential drive and turns in place like the sim. JetRacer
  (Ackermann) comes later, via Nav2 smoothing or Ackermann data.
- **STOP:** the head predicts p(stop) and the remaining route distance (m). Rule: stop when
  p(stop) > 0.5, or when remaining < τ and FutureNav says STOP. τ and the rule are tuned on the holdout
  buildings. Guards: no STOP in the first steps; stop if circling near the goal for k steps.
- **DAgger:** expert intervention. The head drives; if it is > 3 m off-route for > 10 steps, the
  expert drives it back onto the route and hands over. Every step is labelled.
- **Fork consistency:** K = 8 flow samples per step. Keep the one closest to the previous plan unless
  all samples changed mode; blend only within the kept mode. The sample spread is an uncertainty signal
  and can trigger an early FutureNav refresh.
- **Also settled:**
  - Output and tracker: poses + nearest-motion tracker (pick the move whose analytic next pose is
    closest to the predicted one).
  - Staleness: train with context age 0–7 (free from the R=4 cache), so R can change at test time.
  - H = 16, with the loss weighted toward the first 8 steps.
  - Oversample decision steps: bends, forks, off-route states, near-goal steps, recoveries.
  - Evaluation adds a perturbation benchmark: the same injected detours and wrong starts on holdout,
    FutureNav vs the head.

## What the evidence already settles

| Fact | Source | Consequence |
|---|---|---|
| The upcoming bend side is most readable at L22–26 and fades by the final layer | #27–#29 | Read middle layers, not only the final one |
| The side signal that drives behaviour is spread across L18–26 | #31 | Read several layers |
| The look-ahead beyond ~1 m comes from the instruction text, not the view | #29 | Route information reaches the head only through System 2 states, or from instruction tokens |
| FutureNav's own ~2 m plan is readable (0.39 m median error at 8 steps) | #27 | A horizon of 8–16 steps (2–4 m) is supported |
| October 27: diffusing ±1 action codes zigzags; zero-noise ≈ sampling; separate STOP did best | Oct-27 runs | Predict continuous poses; STOP outside the generative part |
| Sparse VGGT (only re-run frames enter the geometry memory) degrades FutureNav to 88.7% agreement with itself | Oct-27 bridge diagnosis | Don't do that |
| **A VGGT-only update every step + an LLM call every R steps = full FutureNav exactly** (diff 0.0, 9/9 actions); 0.18 s vs 0.75 s | 2026-10-30 test, `advance_vggt` | System 2 is undegraded; ~0.32 s/step at R=4 |
| The path-following expert recovers from wrong starts, detours and real FutureNav failures (100%) | A2 | Recovery labels are trustworthy |

## Open choices

### 1. What the head owns
- **(a) Motion only.** FutureNav decides the route and STOP; the head turns that into safe, smooth motion
  and recovers small errors.
- **(b) Full policy.** The head also decides STOP and can overrule FutureNav's route choice.
- *Recommendation:* (b) in the architecture, with FutureNav's own action **given to the head as an
  input**. The head can copy FutureNav when FutureNav is right and learn when to deviate. If it never
  learns to deviate, it falls back to (a) by itself.

### 2. Output
- Time-indexed poses for the next H steps: (x, y, sin dyaw, cos dyaw) in the current agent frame.
  Labels come from the expert's rollout (`chunk_pose`), so every target is reachable with our discrete
  moves.
- Sim: a tracker turns the first pose into an action (turn if the heading error > 7.5°, else forward).
  It is exact on expert labels.
- Robot: the poses become a Nav2 path.
- *Alternative:* discrete chunk via cross-entropy. That was worse in October 27 (0.31 success).
- *Recommendation:* poses.

### 3. Generative model
- **Rectified flow** (multimodal: it can choose one branch at a fork instead of averaging), against
  **direct regression** (simpler, deterministic) on the same network, behind one flag.
- *Recommendation:* build both. Flow is the main model, regression the baseline. Report both, and fork
  episodes separately.

### 4. What the head reads from FutureNav (refreshed every R steps)
| Input | Size | Why |
|---|---|---|
| Last-token residual at L18, L22, L26 + final | 4 × 2560 | Where the plan and side signal live |
| FutureNav's action probabilities at the call | 4 | Its decision, including STOP |
| Steps since the refresh (0…R−1) | 1 | How stale the context is |
| *Optional:* instruction-token states at L22, projected to 256-d | ~60 × 256 | Direct access to route words if (1b) needs more route information |

- *Recommendation:* the first three now. Add instruction tokens only if the head's bend-side accuracy
  stays at FutureNav's level.

### 5. What the head sees every step (fast path)
- **(a) VGGT tokens of the current frame.** They are already computed every step by `advance_vggt` and
  are geometry-aware, so obstacle and free-space information comes at no extra cost. Pool to an ~8×6
  grid.
- **(b) A separate small encoder** (October 27's 16 × 384 tokens).
- **(c) Depth** (sim only; upper bound).
- Short memory: current frame + the last k = 3 frames' pooled tokens, for motion and collision cues.
- *Recommendation:* (a) with k = 3. It needs a check that the VGGT tokens can be taken out of the
  aggregator call; they are `aggregated_tokens[-2]` there.

### 6. Timing
- R (FutureNav LLM refresh) = 4. VGGT runs every step.
- The head re-plans **every step** (it is cheap) and blends overlapping chunks: average the predicted
  pose for each future time over the chunks that cover it, with exponential weights favouring newer
  chunks.
- *Alternative:* execute 2 of 8, as on October 27.
- *Recommendation:* re-plan every step with blending. Later, try R = 2 and R = 8 as the
  speed-versus-quality trade-off.

### 7. Horizon
- Train with H = 16 (up to 4 m); the loss weights the first 8 steps more.
- Sweep H ∈ {8, 16}.

### 8. STOP
- The head predicts p(stop) (BCE on expert STOP labels).
- Final rule, to compare: head only / FutureNav only / either / both.
- *Recommendation:* pick the rule on the holdout buildings. Default "head only", because FutureNav's
  action is one of its inputs.

### 9. Auxiliary outputs (cheap, label-backed)
- Progress along the route (0–1) and off-route distance.
- Both give the model an explicit "where am I / am I lost" signal.
- The off-route output is also a recovery trigger for the robot.

### 10. Data
- **Round 1:** perturbed expert data (A4: ~9.7k episodes, clean + wrong-start heading + jitter +
  detours).
- **Round 2 (DAgger):** System 2 + the head drive on ~1–2k train episodes; the expert relabels every
  step. That puts the head's own mistakes in the data.
- System 2 features are cached at the same time: the rollout computes them anyway.
- The October 27 head trained on 512 clean episodes; this is about 20× the data, with recoveries.

### 11. Losses and size
- Flow-matching (or L1) on poses + BCE stop + L1 auxiliaries.
- Transformer denoiser ~5–10M parameters, as in October 27: 4 layers, width 256, cross-attending to the
  System 2 and vision tokens.
- Training takes minutes to an hour on the 4090; it is not the bottleneck.

### 12. Evaluation
- **Development:** 100 holdout episodes (train buildings not trained on) + 300 val_unseen episodes, all
  with existing FutureNav baselines.
- **Metrics:** success rate, SPL, nDTW, collisions, zigzag (left/right reversals), recovery rate after
  injected detours, compute per step.
- **Final:** full 1,407-episode pool.

## Cost of the plan (for the week)

| Item | Estimate |
|---|---|
| Finish A4 collection | ~1.5 h (1,047 of 9,744 done) |
| Cache System 2 features for ~3,000 A4 episodes (dense VGGT, R=4) | ~0.32 s/step × ~215k steps ≈ 19 h; ~10 h for 1,500 episodes |
| Head + tracker + closed-loop agent | ~1.5 days to build |
| Round-2 DAgger rollouts, 1k episodes | ~7 h |
| Evaluation, 400 episodes | ~3 h each |

The System 2 feature cache dominates. With dense VGGT, the cacher must step VGGT on every frame, not
only every R-th.
