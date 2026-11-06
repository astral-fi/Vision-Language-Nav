# Inference server and deployment tools

This directory corresponds to `deploy/` in a FutureNav checkout. Link it there so the scripts' relative imports
resolve:

```bash
ln -s "$(pwd)" "$FUTURENAV_ROOT/deploy"
```

| File | Role |
|---|---|
| `server.py`, `futurenav_core.py` | HTTP inference server. One episode per session, with a monotonic step counter that protects the append-only VGGT cache |
| `head_server.py`, `head_eval_server.py`, `chunk_server.py` | Servers for the LoRA waypoint head (`--exec discrete|continuous|plan`) and the action-chunk head |
| `camera_adapter.py`, `calibrate_camera.py` | Fisheye → 79° pinhole rectification, so robot frames match the training distribution |
| `motion.py`, `calibrate_motion.py` | Ackermann motion primitives and their calibration (coast, arc radius) |
| `jetracer_client.py` | Standalone HTTP client for the JetRacer (the Zenoh client in [`../robot`](../robot) replaced it) |
| `capture.py` | Hidden-state and attention capture used by the probes |
| `verify_fusion.py`, `smoke_test.py`, `check_data.py` | Checks that the 2D/3D fusion contract, the server and the data layout behave as expected |
| `analyze_*.py`, `compare_*.py` | Trace analysis and run comparison |
| `setup_*_env.sh`, `fetch_*.sh`, `run_eval.sh` | Environment setup, data and checkpoint download, evaluation launcher |

See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the model breakdown, compute budget and hardware
feasibility analysis.
