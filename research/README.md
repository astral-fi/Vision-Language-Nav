# Research code

This is the code behind the waypoint-head work summarised in [../docs/RESULTS.md](../docs/RESULTS.md).

In the working checkout this directory is `experiments/` inside FutureNav, which is why the scripts refer to
`experiments/...` and `outputs/...` paths. To re-run something, link it in and set the root:

```bash
ln -s "$(pwd)" "$FUTURENAV_ROOT/experiments"
export FUTURENAV_ROOT=~/FutureNav
```

Raw outputs, episode-ID lists from earlier runs, datasets and weights aren't included. The interpretability
and failure-analysis experiments are part of a paper in preparation and will be released with it.

| Directory | Topic |
|---|---|
| `data_engine/` | Path-following expert, perturbations (teleports, detours), validation and split for waypoint-head training data |
| `s2_lora/` | LoRA fine-tuning of the frozen backbone for text plans and waypoint prediction, plus offline evaluation |
| `pose_head/`, `motion_head/` | Alternative heads (pose regression, rectified-flow action chunks) and their benchmarks |
| `head_eval/` | Closed-loop benchmarks (sf400, val300) with paired statistics, and an analysis of the episodes the head loses |
| `vggt_depth/` | Can VGGT's depth head give metric obstacle distances? (Shape yes, scale mostly no.) |
