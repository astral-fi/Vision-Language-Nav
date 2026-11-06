"""Continuous pose execution for Habitat: MOVE_FORWARD can carry an arbitrary planar motion for one step.

Importing this module replaces habitat-sim's "move_forward" move_fn with one that reads two optional extra
fields of the actuation spec: `lateral` (metres, + = left) and `yaw` (degrees, + = left). A plain
ActuationSpec has neither, so every existing caller (the default action space, the greedy geodesic follower
behind SHORTEST_PATH_SENSOR) gets exactly the stock 0.25 m forward step.

`set_motion(sim, dx, dy, dyaw_deg)` swaps the agent's MOVE_FORWARD spec for one step; `clear(sim)` restores
the stock spec. eval/run.py calls them around env.step() when --waypoint-exec is given and the server's
response carries a "motion" (deploy/head_eval_server.py --exec continuous).

Why through the move_fn and not by setting the agent state: ObjectControls.action() applies the navmesh
`move_filter_fn` to the translation after the move_fn runs, so a commanded pose collides and slides exactly
like a normal MOVE_FORWARD, and habitat's measures, step count and episode termination stay correct
(same reasoning as eval/ackermann.py). The motion is translate (in the start frame) then rotate, matching
how the waypoint head's pose increments are defined: position of the next pose in the previous pose's
frame, then the heading change.
"""
import math

import attr
import magnum as mn

import habitat_sim
from habitat_sim.agent.controls.controls import ActuationSpec, SceneNodeControl
from habitat_sim.registry import registry as sim_registry

FORWARD = 0.25


@attr.s(auto_attribs=True)
class MotionSpec(ActuationSpec):
    """`amount` = forward metres; `lateral` = left metres; `yaw` = degrees, + = left."""
    lateral: float = 0.0
    yaw: float = 0.0


@sim_registry.register_move_fn(name="move_forward", body_action=True)
class MoveForwardOrPose(SceneNodeControl):
    def __call__(self, scene_node, spec) -> None:
        # Same axis handling as habitat_sim's _move_along (see the trap noted in eval/ackermann.py):
        # transformation[k].xyz is the node's local axis k in world coordinates; forward = -Z, left = -X.
        fwd = scene_node.transformation[2].xyz
        right = scene_node.transformation[0].xyz
        scene_node.translate_local(fwd * -spec.amount + right * -getattr(spec, "lateral", 0.0))
        yaw = getattr(spec, "yaw", 0.0)
        if yaw:
            scene_node.rotate_y_local(mn.Rad(math.radians(yaw)))
            scene_node.rotation = scene_node.rotation.normalized()


def _forward_key(sim):
    space = sim.get_agent(0).agent_config.action_space
    keys = [k for k, v in space.items() if v.name == "move_forward"]
    if len(keys) != 1:
        raise RuntimeError("expected exactly one move_forward action, found %r" % keys)
    return space, keys[0]


def set_motion(sim, dx, dy, dyaw_deg):
    space, k = _forward_key(sim)
    space[k] = habitat_sim.ActionSpec("move_forward", MotionSpec(amount=float(dx), lateral=float(dy), yaw=float(dyaw_deg)))


def clear(sim):
    space, k = _forward_key(sim)
    space[k] = habitat_sim.ActionSpec("move_forward", ActuationSpec(amount=FORWARD))
