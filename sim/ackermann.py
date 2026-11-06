"""Ackermann-steering action space for Habitat, to simulate a JetRacer.

Importing this module registers an ACTION_SPACE_CONFIG named "ackermann".
Select it with `--ackermann-radius R` in eval/run.py, which also sets the
config key.

WHAT IT CHANGES
---------------
Habitat's TURN_LEFT / TURN_RIGHT rotate the agent in place: heading changes by
TURN_ANGLE, position does not change at all. A car cannot do that. An Ackermann
platform turns by driving an arc, so every turn also translates the vehicle.

For a turn of theta along a circle of radius R, the displacement in the
vehicle's own frame at the start of the manoeuvre is

    forward = R * sin(theta)
    lateral = R * (1 - cos(theta))          (towards the turn centre)

With the JetRacer's measured R = 0.65 m and Habitat's TURN_ANGLE = 15 deg:

    forward = 0.168 m      lateral = 0.022 m

So a single turn carries the car about two thirds of a MOVE_FORWARD down the
corridor. The policy never asked for that and has no pose sense to notice it.
Over an episode with ~40 turns it accumulates to metres of unintended travel,
and it is the failure mode that alternating TURN_LEFT/TURN_RIGHT -- the "stand
still and look around" behaviour VLN policies commonly use -- turns into a
slalom that creeps forward.

HOW IT INTEGRATES
-----------------
As a registered habitat_sim move_fn plus a habitat-lab ActionSpaceConfiguration,
NOT by intercepting env.step(). That matters: ObjectControls.action() applies
`move_filter_fn` (the simulator's navmesh step filter) to the translation after
the control runs, so arcs collide and slide exactly like a normal MOVE_FORWARD,
and habitat's own measures, step counting and episode termination all stay
correct. Bypassing env.step() would have silently broken all of that.

The arc is executed as `substeps` alternating rotate/translate increments rather
than one jump, so the vehicle actually follows the curve and collides where the
curve collides. The navmesh filter takes only (start, end) points and so tests a
straight chord; with 5 substeps of 3 deg each the chord deviates from the true
arc by R*(1-cos(1.5deg)) ~ 0.2 mm, which is irrelevant.

NOT MODELLED
------------
Minimum-radius asymmetry, steering slew rate, understeer, and the reverse
K-turn primitive in deploy/motion.py. This is the arc strategy only, which is
what motion.py uses by default.
"""

import math

import attr
import magnum as mn

import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import ActionSpaceConfiguration
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.agent.controls.controls import ActuationSpec, SceneNodeControl
from habitat_sim.registry import registry as sim_registry

DEFAULT_TURN_RADIUS_M = 0.65    # JetRacer full-lock radius, measured
DEFAULT_SUBSTEPS = 5


@attr.s(auto_attribs=True)
class AckermannActuationSpec(ActuationSpec):
    """`amount` is the yaw change in degrees, as for a normal turn."""
    radius: float = DEFAULT_TURN_RADIUS_M
    substeps: int = DEFAULT_SUBSTEPS


def _arc(scene_node, spec: AckermannActuationSpec, sign: int) -> None:
    """Drive an arc that yaws `spec.amount` degrees. sign +1 = left, -1 = right.

    Translation uses translate_local along the node's own forward axis, so it
    follows the heading as it changes. habitat_sim.geo.FRONT is -Z.
    """
    total = math.radians(abs(spec.amount))
    n = max(1, int(spec.substeps))
    d_theta = total / n
    arc_len = spec.radius * total          # s = R * theta
    d_len = arc_len / n

    # Two traps here, both verified empirically:
    #
    # 1. mn.Rad, not habitat_sim.geo.Rad -- geo has no Rad in habitat-sim 0.1.7.
    #
    # 2. translate_local() does NOT rotate its argument by the node's rotation.
    #    Passing a constant geo.FRONT moves along world -Z no matter how the node
    #    is oriented, so every sub-step translates the same way and the "arc"
    #    comes out as a straight line of length R*theta with zero lateral offset.
    #    Habitat's own controls avoid this by reading the local axis out of the
    #    transformation matrix first (see _move_along in default_controls.py);
    #    transformation[2].xyz is the local Z axis expressed in world coords, and
    #    forward is -Z.
    _Z = 2
    for _ in range(n):
        # Half-step rotate, translate, half-step rotate: a midpoint rule, which
        # tracks the true arc more closely than rotating then translating.
        scene_node.rotate_y_local(mn.Rad(sign * d_theta / 2.0))
        scene_node.rotation = scene_node.rotation.normalized()
        fwd_world = scene_node.transformation[_Z].xyz
        scene_node.translate_local(fwd_world * -d_len)
        scene_node.rotate_y_local(mn.Rad(sign * d_theta / 2.0))
        scene_node.rotation = scene_node.rotation.normalized()


@sim_registry.register_move_fn(body_action=True)
class ArcTurnLeft(SceneNodeControl):
    def __call__(self, scene_node, actuation_spec) -> None:
        _arc(scene_node, actuation_spec, +1)


@sim_registry.register_move_fn(body_action=True)
class ArcTurnRight(SceneNodeControl):
    def __call__(self, scene_node, actuation_spec) -> None:
        _arc(scene_node, actuation_spec, -1)


@registry.register_action_space_configuration(name="ackermann")
class AckermannActionSpaceConfiguration(ActionSpaceConfiguration):
    """Same as v0 except the turns drive arcs instead of pivoting."""

    def get(self):
        radius = float(getattr(self.config, "TURN_RADIUS", DEFAULT_TURN_RADIUS_M))
        substeps = int(getattr(self.config, "TURN_SUBSTEPS", DEFAULT_SUBSTEPS))
        theta = math.radians(self.config.TURN_ANGLE)
        print(f"[ackermann] arc turns: R={radius:.3f} m, {self.config.TURN_ANGLE} deg/turn "
              f"-> forward {radius*math.sin(theta):.3f} m, "
              f"lateral {radius*(1-math.cos(theta)):.3f} m per turn "
              f"({substeps} substeps)")
        return {
            HabitatSimActions.STOP: habitat_sim.ActionSpec("stop"),
            HabitatSimActions.MOVE_FORWARD: habitat_sim.ActionSpec(
                "move_forward",
                habitat_sim.ActuationSpec(amount=self.config.FORWARD_STEP_SIZE),
            ),
            HabitatSimActions.TURN_LEFT: habitat_sim.ActionSpec(
                "arc_turn_left",
                AckermannActuationSpec(amount=self.config.TURN_ANGLE,
                                       radius=radius, substeps=substeps),
            ),
            HabitatSimActions.TURN_RIGHT: habitat_sim.ActionSpec(
                "arc_turn_right",
                AckermannActuationSpec(amount=self.config.TURN_ANGLE,
                                       radius=radius, substeps=substeps),
            ),
        }
