import asyncio

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    load_supported_motion_policy_config,
)


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

TARGET_POSITION = np.array([0.6, 0.0, 0.3])       # first RMPflow goal pose (world frame)
DESCEND_DEPTH   = 0.11                              # how far to move down after opening (m)
LIFT_HEIGHT     = 0.3                              # how far to move up after grasping (m)
DROP_POSITION   = np.array([0.6, -0.3, 0.3])      # right of the tray (robot -Y), drop pose
GRASP_ORIENT    = np.array([0.0, 1.0, 0.0, 0.0])  # top-down EE quaternion (wxyz)
POS_TOL         = 0.02                             # m, "reached" tolerance
SETTLE_FRAMES   = 60                               # frames to hold while gripper actuates

# Franka "ready" arm pose (panda_joint1..7) used before the first move so RMPflow
# plans from a sensible upright (hand-down) config instead of the USD's all-zeros pose.
READY_JOINTS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
JOINT_TOL    = 0.05                                # rad, "ready reached" tolerance

CALLBACK_NAME = "pickplace_step"


# ---------------------------------------------------------------------------
# World helper
# ---------------------------------------------------------------------------

async def get_world():
    world = World.instance()
    if world is None:
        carb.log_info("fti: motion_gen:Creating new world instance")
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
    else:
        carb.log_info("fti: World exists. return world instance")

    if not world.is_playing():
        await world.play_async()

    return world


# ---------------------------------------------------------------------------
# RMPflow + gripper-open demo
# ---------------------------------------------------------------------------

class FrankaRmpFlowDemo:
    """Drive a Franka arm to a single target pose via RMPflow and open its gripper.

    Minimal checkpoint: the arm is planned by RMPflow while the gripper is held open
    through the high-level ``Franka.gripper`` (ParallelGripper).
    """

    def __init__(
        self,
        franka_prim_path: str = "/World/franka",
        target_position: np.ndarray = TARGET_POSITION,
    ):
        self._franka_prim_path = franka_prim_path
        self._target_position = np.array(target_position)

        self._franka = None
        self._rmpflow = None
        self._articulation_rmpflow = None

        # Phases: "ready" -> home the arm, "move" -> reach target, "descend" -> move
        # down, "grasp" -> close gripper, "lift" -> move up, "transport" -> move right
        # of tray, "drop" -> open gripper, "done"
        self._phase = "ready"
        self._active_target = self._target_position
        self._settle = 0

        # EE orientation held throughout the sequence. Set from the ready pose's actual
        # orientation (see ready phase) so the first move needs no wrist rotation.
        self._grasp_orient = GRASP_ORIENT

    def setup(self, world):
        # Wrap the existing Franka prim (or reference the default asset if missing)
        self._franka = Franka(self._franka_prim_path, name="franka")
        if not world.scene.get_object("franka"):
            carb.log_info("fti: franka not in scene. Adding to scene.")
            world.scene.add(self._franka)
        else:
            carb.log_info("fti: franka already exists in the scene.")

        # Acquire the physics handle (also initializes the gripper)
        self._franka.initialize()
        # Use absolute open/closed targets so a single open() command holds
        self._franka.gripper.set_action_deltas(None)

        # Load the built-in RMPflow config for Franka and connect it to the arm
        rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")
        self._rmpflow = RmpFlow(**rmp_config)
        self._articulation_rmpflow = ArticulationMotionPolicy(self._franka, self._rmpflow)

    def _ee_reached(self, target) -> bool:
        """True when the end-effector is within POS_TOL of target."""
        joint_positions = self._articulation_rmpflow.get_active_joints_subset().get_joint_positions()
        ee_pos, _ = self._rmpflow.get_end_effector_pose(joint_positions)
        return bool(np.linalg.norm(np.array(ee_pos).flatten() - target) < POS_TOL)

    def step(self, dt):
        # Pressing Stop in the GUI invalidates the articulation physics handles while
        # this callback persists. On the next Play, re-acquire the handles and restart
        # the sequence instead of feeding RMPflow a None joint state.
        if not self._franka.handles_initialized:
            carb.log_info("fti: simulation (re)started, re-initializing franka")
            self._franka.initialize()
            self._franka.gripper.set_action_deltas(None)
            self._rmpflow.reset()
            self._phase = "ready"
            self._active_target = self._target_position
            self._settle = 0
            return

        if self._phase == "ready":
            # Smoothly drive the arm to a known upright (hand-down) pose using the joint
            # position controller -- no teleport -- before any RMPflow task-space planning.
            # This avoids the wrist swinging through a weird orientation on the first move.
            arm = self._articulation_rmpflow.get_active_joints_subset()
            arm.apply_action(joint_positions=READY_JOINTS)
            self._franka.gripper.open()

            if np.max(np.abs(arm.get_joint_positions() - READY_JOINTS)) < JOINT_TOL:
                # Re-seed RMPflow's integrator from the (now reached) ready pose so the
                # first move starts smoothly from here instead of from a stale state.
                self._rmpflow.reset()
                # Hold the ready pose's actual EE orientation for the whole sequence, so
                # the first move is pure translation with no weird wrist spin.
                robot_base_pos, robot_base_ori = self._franka.get_world_pose()
                self._rmpflow.set_robot_base_pose(robot_base_pos, robot_base_ori)
                _, ready_rot = self._rmpflow.get_end_effector_pose(READY_JOINTS)
                self._grasp_orient = rot_matrices_to_quats(ready_rot)
                carb.log_info("fti: ready pose reached, starting move")
                self._phase = "move"
            return

        # RMPflow planning: drive the arm one step toward the active target pose
        self._rmpflow.set_end_effector_target(self._active_target, self._grasp_orient)
        self._rmpflow.update_world()
        robot_base_pos, robot_base_ori = self._franka.get_world_pose()
        self._rmpflow.set_robot_base_pose(robot_base_pos, robot_base_ori)
        action = self._articulation_rmpflow.get_next_articulation_action(dt)
        self._franka.apply_action(action)

        # While carrying the cylinder, keep commanding the gripper closed every frame so the
        # grip force is maintained as the arm moves (otherwise the cylinder slips out).
        if self._phase in ("grasp", "lift", "transport"):
            self._franka.gripper.close()

        if self._phase == "move" and self._ee_reached(self._active_target):
            # Reached the target: open the gripper, then descend
            carb.log_info("fti: target reached, opening gripper")
            self._franka.gripper.open()
            self._active_target = self._target_position - np.array([0.0, 0.0, DESCEND_DEPTH])
            self._phase = "descend"
        elif self._phase == "descend" and self._ee_reached(self._active_target):
            # Reached grasp height: start closing the gripper and hold while it actuates
            carb.log_info("fti: descend complete, closing gripper to grasp")
            self._settle = SETTLE_FRAMES
            self._phase = "grasp"
        elif self._phase == "grasp":
            # Hold position while the gripper closes around the cylinder
            self._settle -= 1
            if self._settle <= 0:
                carb.log_info("fti: cylinder grasped, lifting up")
                self._active_target = self._active_target + np.array([0.0, 0.0, LIFT_HEIGHT])
                self._phase = "lift"
        elif self._phase == "lift" and self._ee_reached(self._active_target):
            # Lifted clear: carry the cylinder to the right of the tray
            carb.log_info("fti: lift complete, moving to the right of the tray")
            self._active_target = DROP_POSITION
            self._phase = "transport"
        elif self._phase == "transport" and self._ee_reached(self._active_target):
            # Over the drop spot: open the gripper to release the cylinder
            carb.log_info("fti: drop position reached, opening gripper to release")
            self._franka.gripper.open()
            self._phase = "done"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_demo = None  # module-level reference so the callback target isn't garbage-collected


async def run():
    global _demo

    world = await get_world()

    _demo = FrankaRmpFlowDemo()
    _demo.setup(world)

    # Run-loop callback (replace any stale registration)
    if world.physics_callback_exists(CALLBACK_NAME):
        carb.log_info("fti: removing existing pickplace_step callback before re-registering.")
        world.remove_physics_callback(CALLBACK_NAME)

    world.add_physics_callback(CALLBACK_NAME, _demo.step)


asyncio.ensure_future(run())
