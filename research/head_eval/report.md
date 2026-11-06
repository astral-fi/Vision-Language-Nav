# Four-model comparison, stair-free val_unseen subset (n=400 of 400)

## 1. Headline metrics (mean, 95% bootstrap CI)

| arm | SR | SPL | oracle SR | distance to goal (m) | path length (m) | steps |
|---|---|---|---|---|---|---|
| futurenav | 0.560 [0.510, 0.608] | 0.503 [0.457, 0.549] | 0.627 | 5.46 | 10.27 | 82 |
| lora | 0.557 [0.507, 0.608] | 0.520 [0.475, 0.566] | 0.610 | 5.39 | 9.61 | 68 |
| discrete_head | 0.515 [0.465, 0.562] | 0.453 [0.408, 0.498] | 0.603 | 5.66 | 10.82 | 93 |
| lora_waypoint_head | 0.595 [0.547, 0.642] | 0.538 [0.493, 0.584] | 0.645 | 5.14 | 10.67 | 91 |

## 2. Paired comparisons (same episodes)

| A vs B | SR A | SR B | A only | B only | exact McNemar p | dSPL (A-B) [95% CI] |
|---|---|---|---|---|---|---|
| futurenav vs lora | 0.560 | 0.557 | 65 | 64 | 1 | -0.017 [-0.067, +0.034] |
| futurenav vs discrete_head | 0.560 | 0.515 | 72 | 54 | 0.13 | +0.050 [+0.001, +0.100] |
| futurenav vs lora_waypoint_head | 0.560 | 0.595 | 56 | 70 | 0.247 | -0.035 [-0.084, +0.013] |
| lora vs discrete_head | 0.557 | 0.515 | 68 | 51 | 0.142 | +0.067 [+0.020, +0.116] |
| lora vs lora_waypoint_head | 0.557 | 0.595 | 52 | 67 | 0.199 | -0.018 [-0.066, +0.029] |
| discrete_head vs lora_waypoint_head | 0.515 | 0.595 | 53 | 85 | 0.00808 | -0.085 [-0.135, -0.037] |

## 3. Who solves what

- solved by all four: 97; by none: 71; by exactly one: 66
- only futurenav succeeds: 14
- only lora succeeds: 16
- only discrete_head succeeds: 13
- only lora_waypoint_head succeeds: 23

## 4. How each model ends (counts per arm)

| arm | success | stopped far (>3 m, STOP issued) | stopped near but failed | never stopped (hit step cap) | oracle reached but failed |
|---|---|---|---|---|---|
| futurenav | 224 | 142 | 0 | 4 | 27 |
| lora | 223 | 127 | 0 | 3 | 21 |
| discrete_head | 206 | 117 | 0 | 8 | 35 |
| lora_waypoint_head | 238 | 123 | 0 | 11 | 20 |

## 5. Action mix and efficiency

| arm | forward % | left % | right % | turns per forward | success-episode steps | path length / reference geodesic |
|---|---|---|---|---|---|---|
| futurenav | 58.1 | 22.8 | 19.1 | 0.72 | 67 | 1.12 |
| lora | 62.7 | 19.7 | 17.7 | 0.60 | 54 | 1.06 |
| discrete_head | 51.6 | 24.9 | 23.5 | 0.94 | 75 | 1.20 |
| lora_waypoint_head | 54.7 | 22.5 | 22.9 | 0.83 | 73 | 1.13 |

## 6a. By goal distance (geodesic terciles)

| slice | n | futurenav SR | lora SR | discrete_head SR | lora_waypoint_head SR |
|---|---|---|---|---|---|
| short | 134 | 0.619 | 0.545 | 0.560 | 0.597 |
| medium | 133 | 0.549 | 0.617 | 0.504 | 0.632 |
| long | 133 | 0.511 | 0.511 | 0.481 | 0.556 |

## 6b. By instruction length (terciles)

| slice | n | futurenav SR | lora SR | discrete_head SR | lora_waypoint_head SR |
|---|---|---|---|---|---|
| short | 141 | 0.489 | 0.546 | 0.504 | 0.504 |
| medium | 136 | 0.640 | 0.581 | 0.574 | 0.691 |
| long | 123 | 0.553 | 0.545 | 0.463 | 0.593 |

## 6c. By building

| slice | n | futurenav SR | lora SR | discrete_head SR | lora_waypoint_head SR |
|---|---|---|---|---|---|
| 2azQ1b91cZZ | 64 | 0.547 | 0.531 | 0.547 | 0.578 |
| 8194nk5LbLH | 5 | 0.800 | 0.400 | 0.400 | 0.400 |
| EU6Fwq7SyZv | 26 | 0.615 | 0.462 | 0.577 | 0.615 |
| QUCTc6BB5sX | 71 | 0.549 | 0.493 | 0.437 | 0.479 |
| TbHJrupSAjP | 51 | 0.647 | 0.588 | 0.412 | 0.765 |
| X7HyMhZNoso | 25 | 0.520 | 0.520 | 0.640 | 0.600 |
| Z6MFQCViBuw | 45 | 0.467 | 0.667 | 0.489 | 0.511 |
| oLBMNvg9in8 | 11 | 0.545 | 0.909 | 0.636 | 0.727 |
| pLe4wQe7qrG | 5 | 0.400 | 0.400 | 0.600 | 0.200 |
| x8F5xyUWy9e | 12 | 0.417 | 0.500 | 0.250 | 0.583 |
| zsNo4HB9uLZ | 85 | 0.588 | 0.576 | 0.600 | 0.659 |

## 7. Head-server bookkeeping

- discrete_head: 37048 steps, 14606 head queries (2.5 steps per query), 1682 queue drops (collisions), 0 invalid outputs
- lora_waypoint_head: 36303 steps, 13388 head queries (2.7 steps per query), 2632 queue drops (collisions), 0 invalid outputs
