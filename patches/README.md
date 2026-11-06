# Changes to upstream FutureNav

`futurenav.patch` holds my changes to four upstream files. Apply it to a clean checkout of
[linglingxiansen/FutureNav](https://github.com/linglingxiansen/FutureNav):

```bash
cd FutureNav && git apply /path/to/futurenav.patch
```

**`eval/run.py`** gets new evaluation flags:
- `--server` runs the model in a separate process or on a separate machine.
- `--episode-ids` and `--instruction-override` select episodes and swap their instructions (counterfactual replay).
- `--dump-hidden` and `--probe-spec` record hidden states and run probes online.
- `--capture-dir` and `--capture-profile full|lite` capture activations on every step.
- `--derail-spec` runs scripted derailments (walk, teleport, memory wipe, rotation).
- `--ackermann-radius` and `--ackermann-substeps` turn in arcs like a car.
- `--cam-pitch` and `--cam-height` set the robot camera's height and pitch.
- `--waypoint-exec` and `--s1` select the continuous executor and the System 1 local planner.

**`eval/agent.py`** adds attention-backend selection (`FUTURENAV_ATTN`) and VGGT cache window control
(`FUTURENAV_VGGT_RECENT`).

**`modeling_qwen3_vl_obshead.py`** adds a `vggt_feature_reuse(capture=, inject=)` context manager. It can capture
the geometry tensor, or inject one so that VGGT is skipped and no frame enters the spatial memory. This is used
for counterfactual re-queries. The patch also fixes the auxiliary-head sizes to match the released checkpoint and
makes cache trimming happen in place.

**`modeling_qwen3_vl.py`** fixes an out-of-memory bug in the VGGT KV-cache trim. The original code built every
layer's trimmed K/V before releasing the old ones, so on the step the trim first fired, two full caches were
resident at once (an extra ~8.9 GB at the default 8+48 frames). That pushed peak memory to 23.2 GB on a 24 GB card.
The fix trims each layer in place, which brings the transient down to one layer.
