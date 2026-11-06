# Robot client

A Zenoh client that drives a real robot with FutureNav running on a remote GPU. It connects to a Zenoh router
(`zenoh-bridge-ros2dds` on the robot exposes the ROS 2 topics), grabs camera frames, sends them to the
inference server and executes the returned action as a closed-loop motion primitive.

## Supported robots ([`robots.py`](robots.py))

| Profile | Drive | Camera | Feedback |
|---|---|---|---|
| `jetracer` (namespace `bot1`) | Ackermann, turn radius 0.31 m | CSI IMX219-160 fisheye, rectified to a 64.5° pinhole and color-corrected | wheel odometry |
| `waffle` (namespace `tb3`) | differential, pivot turns | FLIR, undistorted with its factory calibration | open loop, speed limits checked before driving |

Each profile carries its topics, calibration files and motion limits. A command the robot would clip is refused
before it's sent, because in open loop a clipped speed just produces a short step that nobody notices.

## Main pieces

| File | Role |
|---|---|
| `client.py` | Main loop: preflight checks, frame freshness checks, server round trip, action execution, failure limits |
| `motion.py` | Motion primitives: 0.25 m forward and 15° turns, closed loop on odometry, Ackermann arc turns |
| `transport.py`, `config.py` | Zenoh session, topic wiring, CLI and config |
| `camera_feed.py`, `camera_color.py`, `view.py` | MJPEG preview, color correction, view adapters |
| `recording.py`, `recording_server.py` | Per-run recording: `camera.mp4`, observations, `actions.jsonl`, and every decision's activations (see below) |
| `inspect_capture.py` | Load and verify a recorded run |
| `fit_targetless_lens.py`, `fit_camera_color.py`, `check_lens_fit.py` | Fisheye lens fit without a calibration target, color fit, fit validation |
| `measure_primitive.py`, `diagnose_yaw.py`, `diagnose_waffle.py` | Hardware characterisation (top speed, heading drift, actuator checks) |
| `check_guards.py`, `check_capture_*.py` | Offline checks of the safety guards and the capture pipeline |

## Activation capture

Every run records, for every decision: the exact input JPEG, the prompt token IDs, the image grids and history
frame IDs, the residual stream / attention / MLP activations at every layer, attention rows for the generated
tokens, full-vocabulary logits and the merged VGGT geometry features. Tensors go to a bounded background writer
so compression never blocks an action. Steps that are already committed survive a server crash, and
`provenance.json` pins the weight hashes, source hashes and package versions. This is what makes robot runs
analysable with the same probes as the simulation experiments in [`../research`](../research).

## Running

```bash
export FUTURENAV_ROOT=~/FutureNav          # upstream checkout with weights
python start_server.py                     # starts the recording inference server on :8642 (GPU machine)
bash dry_run.sh                            # full loop, motors disabled
bash run_bot1.sh "Turn right through the doorway and stop at the red door." --max-steps 60
bash run_waffle.sh "Go forward past the chair and stop at the door."
```

Both launchers share an inference lock (exit code 75 means another run is active). Set `ZENOH_ROUTER` to point
at your router (default `127.0.0.1:7447`) and `ROBOT_PYTHON` to choose the interpreter.
