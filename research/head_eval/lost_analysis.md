# Why the LoRA waypoint head loses episodes other models solve

Stair-free val_unseen subset (400, `ids_sf400.txt`). The head fails 162 episodes; **91** of them are solved by at
least one other model (52 by LoRA, its own backbone). Scripts: `analyze_lost.py` (per-episode features →
`lost_episodes.json`), `analyze_lost2.py` (plan / collision / stop follow-ups).

LoRA and the head share backbone, generation and execution rules (≤4 actions, re-query after 30°), so their
trajectories are identical until the first differing executed action — the **fork**.

## Numbers

| | head lost, others won (91) | of which LoRA won (52) | head won (238) |
|---|---|---|---|
| ending: stop far / step cap / stopped after passing goal | 51 / 24 / 15 | 29 / 12 / 11 | – |
| left the route for good (>2 m, never back <1 m) | 62, median step 29 | 36, median step 24 | 7 |
| executed action = geodesic oracle | 27% | 27% | 57% |
| blocked forward steps per episode | 14.2 | 11.6 | 0.8 |
| head first action overrides a correct LoRA first action | 11% of queries | 13% | 10% |

Fork against LoRA (52 LoRA-won episodes): median **step 2**; at the fork LoRA matches the oracle 36×, the head 9×.

## Causes, in order of impact

1. **Turn commitments at decision points (mostly the start).** When LoRA's plan is a big turn (`LLLLLLLL`), the
   head picks either a straight "compromise" anchor (`FRFLFFRF`, `FFFFFRFF`, `FFRFLFFF`) or the mirror-image
   turn (anchor 234 `RRRRRRRF`, 7 episodes). When LoRA turns, the head turns *less* in 52% of lost-episode queries
   (36% in won ones) and the opposite way in 11% (6%). Examples: ep 1148 needs ~120° right, LoRA turns 8×R, the
   head 4×R then walks forward; ep 323 the head turns right instead of left and never recovers.
   Likely mechanism: **minimum-expected-anchor-distance decoding averages modes** — with probability split
   between left- and right-turn anchors, the risk minimiser is a central, straight anchor. Only 16 of 256 anchors
   turn ≥90°, so large turns are rare classes that the min-risk rule pulls away from.
2. **Zig-zag anchors.** 118 of 256 anchors decode to plans with both L and R (`FFLFRFFF`, `FRFFFLFR`); they serve
   **65% of all queries**, and there is a single pure `FFFFFFFF` anchor. The anchors are k-means centres
   (blurred trajectories), so small heading drift rounds to a turn and back. The most common disagreement in
   every group is LoRA `FFFF` → head `FFLF`/`FRFF`. This wastes steps, adds lateral drift, and — because the 30°
   budget counts opposite turns — triggers extra re-queries. Off-route, the head wobbles `FLFR…` instead of
   committing (eps 190, 1148). Part of this is the anchor → primitive conversion in `head_eval_server.py`, not
   only the head.
3. **Wall pressing.** In lost episodes the head drives into an obstacle 14 times per episode, with blocked runs
   of median 26 steps (max 140); 24 lost episodes end at the step cap. After a blocked forward, the next query
   says forward again for both the head and LoRA's own plan (oracle says turn), so the *backbone* is blind to
   collisions too — but the LoRA run only gets there 4×/episode on the same episodes: the head's zig-zags put it
   against walls more often. The head's previous-action input says "forward" whether or not it moved.
4. **Wrong stops are mostly inherited.** Head STOP plans in lost episodes happen at a median 8.6 m from the goal
   (0.95 m in won ones); 43 of 70 coincide with a STOP in LoRA's own plan. Most stop-far endings follow an early
   wrong fork, so fixing 1–3 removes many of them.

## Test 0 result: decoding or head? (`test0.sh`, `test0_analyze.py`)

The 52 LoRA-won episodes were re-run with the 256 anchor probabilities logged. The run was deterministic (SR 0, as before).
Decodings are compared offline, only on states the min-risk policy visited.

| first action = oracle | min-risk (ran) | argmax | two-stage | LoRA plan |
|---|---|---|---|---|
| fork queries (52) | 42% | 54% | 42% | 77% |
| forks where LoRA's plan is a big turn (31) | 26% | 42% | 23% | 81% |
| all queries before departure (1,532) | 28% | 32% | 26% | 36% |

- **Mostly the head.** At forks the oracle's side is the heaviest side only 53% of the time. The best decoding (argmax) closes about a third of the gap to LoRA.
- **Big turns: a near 50/50 left/right split with a right bias.** In 12 of the 31 big-turn forks the mass is about 0.4 big-left vs 0.55 big-right, with almost nothing in between. Examples: eps 137, 323, 324, 545, 1070, 1171, 1433, 1819. The head knows it should turn hard but not which way, and it turns right the long way round. Across big-turn forks the mean mass is bigR 0.35 vs bigL 0.24, while LoRA's plan is bigL 23 vs bigR 8. The anchors do not explain this: there are 29 bigL and 34 bigR anchors, 7 vs 9 at 90° or more.
- **Small and straight forks (21) are unimodal and confidently wrong.** For example, LoRA goes straight while the head puts 0.56–0.59 on smallL. All three decodings agree, so decoding cannot fix them.
- **Two-stage does not help.** It commits to the heavier side, and the heavier side is often wrong.

## Layer probe: does a mid-layer input help with turn side? (`layer_probe.py` → `layer_probe_results.json`)

These numbers are offline, on the head's own cache: 1,300 expert episodes, LoRA last-prompt-token states. There are 318 big-turn test states (|net turn over the first 4 actions| ≥ 45°), about 50% left.

| Side accuracy on big-turn states (test) | L18 | L22 | L26 | final |
|---|---|---|---|---|
| Linear probe | 0.69 | 0.82 | 0.81 | 0.82 |

| Same states | min-risk | argmax |
|---|---|---|
| Shipped head (final layer) | 0.81 | 0.77 |
| Retrained on L22 | 0.79 | 0.77 |
| Retrained on L22+L26+final | 0.79 | 0.80 |
| Retrained on L18+L22+L26+final | 0.80 | 0.80 |
| LoRA's generated text plan | 0.78 | – |

- The turn side is readable from the hidden state, at about 82% from L22 onward.
- The shipped head already reaches that level, and it matches or beats LoRA's text. Broken down by turn size, head vs LoRA text: 45–60° turns 0.89 vs 0.84; 60° turns 0.77 vs 0.75; start states 0.75 vs 0.80, but there are only 20 start states.
- Reading earlier layers gives no gain. Retraining the final-layer head reproduces the shipped checkpoint exactly.
- So the 50/50 splits in test 0 are a **selection effect**: those 52 episodes were picked because the head failed and LoRA succeeded. Each model gets the side wrong about 20% of the time, on partly different states.
- Another decoder or input layer will not fix side errors. The ceiling is roughly 82% for every layer.

## What to try (cheapest first)

1. **Decoding, no retraining:** argmax, or two-stage (pick the turn direction / net-turn bin by summed
   probability, then the best anchor inside it). Re-run only the 91 lost + a matched set of won episodes.
2. **Exact-primitive anchors:** use the 256 most frequent expert 8-action *sequences* instead of k-means centres
   (or snap each centre to its nearest exact sequence). Removes zig-zags and the conversion error entirely.
3. **Collision input:** a 6th previous-action token "forward, blocked" in training data and at inference, or at
   least mask forward-anchors right after a blocked forward.
4. **Rebalance large turns:** up-weight anchors with |net turn| ≥ 60° / more anchors in that region, since the
   decisive errors are at big turns.
5. **Fall back to LoRA's own plan** when it is confident (all-turn plans at episode start) — LoRA wins the fork
   36:9 in exactly these cases.
