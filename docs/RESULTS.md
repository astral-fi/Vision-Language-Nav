# Results summary

Setting: R2R VLN-CE val_unseen in Habitat, stair-free episodes. Paired comparisons use exact McNemar tests on
success, and CIs are bootstrapped.

The interpretability and failure-analysis results (probes, derailment, steering, self-diagnosis) are part of a
paper in preparation. They will be added here after publication.

---

## 1. Sim-to-real feasibility

| Setting | SR | Δ vs baseline |
|---|---|---|
| Released model, camera at 1.25 m (n = 150) | 49.3% | — |
| Camera at 0.50 m | 40–42% | −7 to −9 pp |
| Camera at 0.12 m, pitched 20° (JetRacer height) | 45.3% | −4 pp |
| **Ackermann arcs (R = 0.65 m) + 0.12 m camera** | **35.3%** | **−14 pp** |

A car can't turn in place. Each 15° turn also moves it ~0.17 m forward, which the policy can't sense. I wrote a
custom Habitat action space ([`sim/ackermann.py`](../sim/ackermann.py)) to measure this before going to
hardware. It is the largest single sim-to-real penalty. The model can't run on the car's own hardware either
(the feasibility table is in [ARCHITECTURE.md](ARCHITECTURE.md)), so I built the thin-client stack.

## 2. Better motion: waypoint head, continuous executor, System 1

| Model (n = 300, val_unseen) | SR | SPL |
|---|---|---|
| FutureNav (released) | 0.547 | 0.490 |
| LoRA, text plan | 0.573 | 0.535 |
| **LoRA waypoint head + continuous pose executor** | **0.627** | **0.565** |

- Continuous vs released: 57 vs 33 paired wins, **p = 0.015**. Continuous vs LoRA text plan: p = 0.093.
- The continuous executor ([`sim/waypoint_exec.py`](../sim/waypoint_exec.py)) applies (forward, lateral, yaw) in
  one step through the navmesh filter. Executed motion matches the commanded motion to within 1 mm and 0.05°.
- The data engine ([`research/data_engine`](../research/data_engine)) generates training data with a
  path-following expert plus perturbations. All validation checks pass.
- **System 1** ([`sim/s1_local.py`](../sim/s1_local.py)) is a LiDAR local planner (local A\* + DWA-style arc
  sampling with swept clearance). It follows the VLA's plan and re-queries the slow model only on triggers: plan
  used up, > 0.5 m off plan, blocked, or 16 steps since the last query. Its behaviour is covered by CPU tests in
  CI.

Waypoint heads I tried and dropped: a rectified-flow action-chunk head (SR 0.44 vs 0.50, oscillates) and a
discrete-executor waypoint head (0.532 on a 79-episode subset).
