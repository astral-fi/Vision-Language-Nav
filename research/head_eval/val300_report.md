# Waypoint head, discrete vs continuous execution: val300 benchmark (n=79)

| arm | SR [95% CI] | SPL [95% CI] | oracle SR | NE (m) | path (m) | steps |
|---|---|---|---|---|---|---|
| FutureNav | 0.595 [0.481, 0.696] | 0.517 [0.418, 0.618] | 0.633 | 5.29 | 10.44 | 88 |
| LoRA (text plan) | 0.557 [0.443, 0.671] | 0.517 [0.412, 0.620] | 0.620 | 5.50 | 9.37 | 72 |
| waypoint head, discrete exec | 0.532 [0.418, 0.646] | 0.489 [0.387, 0.591] | 0.582 | 5.34 | 10.73 | 94 |
| waypoint head, continuous exec | 0.608 [0.494, 0.709] | 0.543 [0.444, 0.642] | 0.684 | 5.06 | 11.05 | 105 |

## Paired (exact McNemar on success, bootstrap on SPL difference)

| A vs B | only A wins | only B wins | p | ΔSPL (A−B) [95% CI] |
|---|---|---|---|---|
| continuous vs FutureNav | 12 | 11 | 1.000 | +0.026 [-0.072, +0.125] |
| continuous vs LoRA (text plan) | 14 | 10 | 0.541 | +0.026 [-0.082, +0.135] |
| continuous vs waypoint head, discrete exec | 12 | 6 | 0.238 | +0.054 [-0.033, +0.142] |

## Behaviour (per episode, from traces; mean over episodes that have a trace)

| arm | heading change (deg) | zig-zag reversals | blocked steps | sideways travel (m) | step-cap endings |
|---|---|---|---|---|---|
| FutureNav | 567 | 15.1 | 7.3 | 0.00 | 1% |
| LoRA (text plan) | 437 | 6.6 | 4.1 | 0.00 | 1% |
| waypoint head, discrete exec | 667 | 13.1 | 6.2 | 0.00 | 3% |
| waypoint head, continuous exec | 608 | 11.4 | 20.3 | 0.16 | 9% |
