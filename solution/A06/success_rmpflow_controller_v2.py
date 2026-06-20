import asyncio

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

TARGET_POSITION = np.array([0.6, 0.0, 0.3])       # pick pose above the cylinder (world frame)
DESCEND_DEPTH   = 0.11                             # how far to move down before grasping (m)
LIFT_HEIGHT     = 0.3                              # how far to move up after grasping (m)
DROP_POSITION   = np.array([0.6, -0.3, 0.3])      # right of the tray (robot -Y), release pose
POS_TOL         = 0.02                             # m, end-effector "reached" tolerance
SETTLE_FRAMES   = 60                              # frames to hold while the gripper actuates

# Franka "ready" arm pose (panda_joint1..7) commanded before any task-space planning so
# RMPflow starts from a sensible upright (hand-down) config instead of the USD all-zeros pose.
READY_JOINTS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
JOINT_TOL    = 0.05                                # rad, "ready reached" tolerance

CALLBACK_NAME = "pickplace_step"


def _log(message: str) -> None:
    carb.log_info(f"fti: {message}")


# ---------------------------------------------------------------------------
# World helper
# ---------------------------------------------------------------------------

async def get_world():
    world = World.instance()
    if world is None:
        _log("creating new world instance")
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
    else:
        _log("world exists, reusing instance")

    if not world.is_playing():
        await world.play_async()

    return world


# ---------------------------------------------------------------------------
# RMPFlowController pick-and-place demo
# ---------------------------------------------------------------------------

class FrankaPickPlaceDemo:
    """Pick a cylinder and place it beside the tray using ``RMPFlowController``.

    Motion generation is delegated to the high-level ``RMPFlowController`` wrapper
    (which owns the ``RmpFlow`` policy, its ``ArticulationMotionPolicy`` and the robot
    base pose); the gripper is driven through ``Franka.gripper`` (ParallelGripper).

    The sequence is a small state machine:
        ready     -> home the arm to READY_JOINTS
        move      -> reach the pick pose above the cylinder
        descend   -> lower onto the cylinder
        grasp     -> close the gripper and let it settle
        lift      -> raise the cylinder clear of the tray
        transport -> carry it to the drop pose
        done      -> released, holding station
    """

    def __init__(
        self,
        franka_prim_path: str = "/World/franka",
        target_position: np.ndarray = TARGET_POSITION,
    ):
        self._franka_prim_path = franka_prim_path
        self._target_position = np.array(target_position)

        self._franka = None
        self._controller = None

        # EE orientation held throughout the sequence; locked to the ready pose's actual
        # orientation (see _step_ready) so the first move is pure translation, no wrist spin.
        self._grasp_orient = np.array([0.0, 1.0, 0.0, 0.0])  # top-down quaternion (wxyz)

        self._reset_sequence()

    # --- setup ------------------------------------------------------------

    def setup(self, world):
        # Wrap the existing Franka prim (or reference the default asset if missing).
        self._franka = Franka(self._franka_prim_path, name="franka")
        if world.scene.get_object("franka"):
            _log("franka already in scene")
        else:
            _log("franka not in scene, adding it")
            world.scene.add(self._franka)

        # Acquire the physics handle (also initializes the gripper). Absolute open/closed
        # targets (no deltas) let a single open()/close() command hold.
        self._franka.initialize()
        self._franka.gripper.set_action_deltas(None)

        # High-level RMPflow controller: builds the Franka RMPflow config, the RmpFlow
        # policy and the ArticulationMotionPolicy, and captures the static base pose.
        self._controller = RMPFlowController(
            name="rmpflow_controller", robot_articulation=self._franka
        )

    # --- helpers ----------------------------------------------------------

    @property
    def _arm(self):
        """The arm-only joint subset (excludes the gripper fingers)."""
        return self._controller.get_articulation_motion_policy().get_active_joints_subset()

    def _ee_reached(self, target) -> bool:
        """True when the end-effector is within POS_TOL of ``target``."""
        ee_pos, _ = self._controller.get_motion_policy().get_end_effector_pose(
            self._arm.get_joint_positions()
        )
        return bool(np.linalg.norm(np.array(ee_pos).flatten() - target) < POS_TOL)

    def _reset_sequence(self):
        self._phase = "ready"
        self._active_target = self._target_position
        self._settle = 0

    # --- run loop ---------------------------------------------------------

    def step(self, dt):
        # Pressing Stop in the GUI invalidates the articulation physics handles while this
        # callback persists. On the next Play, re-acquire the handles and restart the
        # sequence instead of feeding RMPflow a None joint state.
        if not self._franka.handles_initialized:
            _log("simulation (re)started, re-initializing franka")
            self._franka.initialize()
            self._franka.gripper.set_action_deltas(None)
            self._controller.reset()
            self._reset_sequence()
            return

        if self._phase == "ready":
            self._step_ready()
            return

        # Plan one step toward the active target, and keep clamping the grip while carrying
        # so the cylinder does not slip out as the arm moves.
        self._franka.apply_action(self._controller.forward(self._active_target, self._grasp_orient))
        if self._phase in ("grasp", "lift", "transport"):
            self._franka.gripper.close()

        self._advance_phase()

    def _step_ready(self):
        # Smoothly drive the arm to the upright (hand-down) ready pose via the joint
        # controller -- no teleport -- before any task-space planning, so the wrist does
        # not swing through a weird orientation on the first move.
        arm = self._arm
        arm.apply_action(joint_positions=READY_JOINTS)
        self._franka.gripper.open()

        if np.max(np.abs(arm.get_joint_positions() - READY_JOINTS)) >= JOINT_TOL:
            return

        # Ready pose reached: re-seed RMPflow's integrator (reset() also re-applies the
        # base pose) and lock in this pose's actual EE orientation for the whole sequence.
        self._controller.reset()
        _, ready_rot = self._controller.get_motion_policy().get_end_effector_pose(READY_JOINTS)
        self._grasp_orient = rot_matrices_to_quats(ready_rot)
        _log("ready pose reached, starting move")
        self._phase = "move"

    def _advance_phase(self):
        if self._phase == "move" and self._ee_reached(self._active_target):
            _log("pick pose reached, opening gripper")
            self._franka.gripper.open()
            self._active_target = self._target_position - np.array([0.0, 0.0, DESCEND_DEPTH])
            self._phase = "descend"
        elif self._phase == "descend" and self._ee_reached(self._active_target):
            _log("descend complete, closing gripper to grasp")
            self._settle = SETTLE_FRAMES
            self._phase = "grasp"
        elif self._phase == "grasp":
            # Hold position while the gripper closes around the cylinder.
            self._settle -= 1
            if self._settle <= 0:
                _log("cylinder grasped, lifting up")
                self._active_target = self._active_target + np.array([0.0, 0.0, LIFT_HEIGHT])
                self._phase = "lift"
        elif self._phase == "lift" and self._ee_reached(self._active_target):
            _log("lift complete, moving to the right of the tray")
            self._active_target = DROP_POSITION
            self._phase = "transport"
        elif self._phase == "transport" and self._ee_reached(self._active_target):
            _log("drop pose reached, opening gripper to release")
            self._franka.gripper.open()
            self._phase = "done"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_demo = None  # module-level reference so the callback target isn't garbage-collected


async def run():
    global _demo

    world = await get_world()

    _demo = FrankaPickPlaceDemo()
    _demo.setup(world)

    # Register the run-loop callback, replacing any stale registration.
    if world.physics_callback_exists(CALLBACK_NAME):
        _log("removing existing pickplace_step callback before re-registering")
        world.remove_physics_callback(CALLBACK_NAME)

    world.add_physics_callback(CALLBACK_NAME, _demo.step)


asyncio.ensure_future(run())
