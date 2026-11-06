# Habitat extensions

| File | What it does |
|---|---|
| [`ackermann.py`](ackermann.py) | A Habitat action space where turns drive an arc (configurable radius and substeps), so a car's inability to turn in place can be measured in simulation. Enabled with `run.py --ackermann-radius R`. |
| [`waypoint_exec.py`](waypoint_exec.py) | Continuous pose executor. It overrides `move_forward` so one step applies (forward, lateral, yaw) through the navmesh sliding filter. Commanded and executed motion agree to within 1 mm and 0.05°. Enabled with `run.py --waypoint-exec`. |
| [`s1_local.py`](s1_local.py) | **System 1** local planner: a 2D LiDAR (navmesh ray casts in sim, a real scan on the robot), local A\* on the scan grid toward the plan point 1.5 m ahead, and DWA-style arc sampling with swept-clearance checks. It decides when to re-query the slow VLA ("System 2"). |
| [`check_s1_local.py`](check_s1_local.py) | Offline checks of the planner in a kinematic 2D world, run in CI via [`../tests`](../tests). |
| [`agent_remote.py`](agent_remote.py), [`eval_r2r_remote.sh`](eval_r2r_remote.sh) | Habitat evaluation against a remote inference server (`run.py --server`), so the simulator and the 23 GB model can run in separate environments or on separate machines. |

These files go into an upstream FutureNav checkout's `eval/` directory, next to the patched `run.py`
(see [`../patches`](../patches)).
