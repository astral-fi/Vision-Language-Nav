# Stage A — data engine

Goal: training data that teaches a policy to follow the *described* route and to get back onto it
after a mistake. Stage A produces the expert, the perturbations and two datasets: expert+noise, and
FutureNav on-policy rollouts relabelled by the expert. No model training happens in Stage A.

The sim settings match our eval exactly (`eval/config/vlnce_task_futurenav.yaml`):
- habitat 0.1.7
- 0.25 m forward steps, 15° turns, sliding on
- RGB 640×480 with a 79° field of view, camera at 1.25 m
- success radius 3 m

## Checklist

### A0. Setup
- [x] Stop the steering server to free the GPU.
- [x] Create the `experiments/data_engine/` layout. Outputs go to `outputs/data_engine/`.
- [x] Split R2R-CE train by building: hold out 6 of the 61 train buildings as **train-holdout**. Stage B/C
      checks then never touch val_unseen. *Done (`split.py`): 9,744 train / 1,075 holdout episodes; 1,760
      use stairs (flagged, not dropped).*
- [x] Stairs: same 0.5 m rule as the val pool, flagged in `split.json` instead of filtered.

### A1. Path-following expert (`expert.py`)
- [x] Route = the dense `locations` from `train_gt` (one point every 0.25 m).
- [x] Progress tracker: nearest route point within a forward window of the last match, so loops and
      parallel corridors can't make it jump ahead. It gives a progress index *i* and the off-route
      distance.
- [x] Target = the next route point (0.25 m) past *i*. A 1 m lookahead cut corners (84% agreement). The action comes from habitat's
      `ShortestPathFollower` toward that target on the navmesh.
- [x] STOP when within 0.2 m of the last route point. At 0.5 m it stopped a step early. Taking the success radius instead would teach
      sloppy stops.
- [x] Off-route fallback: if more than 3 m from the route ahead, target the nearest route point ahead
      and flag the step.
- [x] Chunk labels: simulate the expert H = 16 steps ahead without rendering (navmesh `try_step` for
      forward moves, yaw maths for turns), then restore the state. Store the poses in the agent frame.
- [x] Per-step label: action, chunk poses, stop flag, progress fraction, off-route distance, fallback
      flag.

### A2. Expert validation (300 train episodes)
- [x] **Clean start:** matches the GT `actions` on >90% of steps, success ~100%, nDTW ~1.
- [x] **Random start heading:** turns to the route and follows it; success >95%.
- [x] **Teleport 1–2 m off-route mid-episode:** rejoins the route instead of shortcutting; nDTW stays high.
- [x] **Recorded failures:** on FutureNav wrong-fork cases, the label points to the correct branch.
- [x] Report the failure cases (navmesh islands, stairs, loops) and fix or exclude them.

**A2 result** (`validate.py`, 300 episodes per check, `outputs/data_engine/validate/summary.json`):

| Check | Success | Route coverage | nDTW |
|---|---|---|---|
| clean (99.8% agreement with the native expert's actions) | 1.00 | 1.00 | 0.999 |
| random start heading | 1.00 | 1.00 | 0.981 |
| teleport 1–2 m off-route (6 skipped) | 1.00 | 0.978 | 0.838 |
| FutureNav failure states, val_unseen | 1.00 | 0.999 | 0.62 (path starts mid-route) |

There were no fallbacks or follower errors. One bug was found and fixed: routes that double back (36
of 10,819 train paths) made the tracker loop forever. Ties now resolve to the point at or after the
current progress.

### A3. Perturbations (`perturb.py`)
- [ ] Random start heading, uniform 0–360°, aimed at wrong starts.
- [ ] Mid-route teleport to a navigable point 1–2 m off-route.
- [ ] DART-style action noise: with probability ε, run a random or "wrong-branch" action for k
      steps, while the label stays the expert's.
- [ ] Each episode records its perturbation, so later we can measure the effect of each type.

### A4. Expert + noise collection (sim only, no FutureNav)
- [ ] All train episodes minus train-holdout, about 9.8k, with perturbations sampled per episode.
- [ ] Save per step: JPEG frame, pose, instruction id and the A1 labels. Shard by building, and write
      a manifest.
- [ ] Expected size: about 0.8M frames, ~35 GB.

### A5. FutureNav on-policy rollouts, relabelled
- [ ] Run FutureNav through the existing server and `eval/run.py`, adding a `--relabel` hook that
      calls the A1 expert every step.
- [ ] About 1,000 train episodes with A3 perturbations. FutureNav probably trained on these buildings,
      so it errs less on clean starts; the perturbations are what make errors appear.
- [ ] Save the same fields as A4, plus FutureNav's action, so we have the disagreement points.
- [ ] Byproduct: FutureNav's failure stats on train, with and without perturbations.

### A6. Dataset QA
- [ ] Label statistics: action mix, stop rate, off-route fraction, fallback rate, chunk-pose ranges.
- [ ] Videos of 20 random episodes with labels drawn on top.
- [ ] Final manifest plus a `README` with the schema.

## Done when
The expert passes A2. A4 and A5 data are on disk with a manifest. QA shows sane labels.

## Estimate

| Step | My build time | Compute (wall clock) |
|---|---|---|
| A0 setup | 0.5 h | — |
| A1 expert | 3–4 h | — |
| A2 validation | 1.5 h | ~1 h |
| A3 perturbations | 1.5 h | — |
| A4 expert+noise collection | 1.5 h | 3–5 h (4 sim processes) |
| A5 FutureNav rollouts | 2 h | ~14 h (1k episodes × ~100 steps × ~0.5 s; GPU-bound) |
| A6 QA | 1.5 h | ~0.5 h |
| **Total** | **~12 h** | **~20 h**, mostly A5 overnight |

That's about 2–3 days of wall clock. The val_unseen baseline needs no new run: capture_pool and the
wrong-start review already cover all 1,407 stairs-free episodes.
