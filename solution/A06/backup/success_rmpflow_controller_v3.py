import asyncio

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

FRANKA_PRIM_PATH = "/World/franka"

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


# ---------------------------------------------------------------------------
# Module-level handles and sequence state (persist across physics callbacks)
# ---------------------------------------------------------------------------

franka = None        # Franka articulation
controller = None    # RMPFlowController wrapping the RmpFlow policy
state = {}           # mutable sequence state: phase / target / settle / grasp_orient


def reset_sequence():
    """Restart the pick-and-place state machine from the beginning."""
    state["phase"] = "ready"
    state["target"] = TARGET_POSITION
    state["settle"] = 0
    # EE orientation held throughout the sequence; re-locked to the ready pose's actual
    # orientation (see "ready" step) so the first move is pure translation, no wrist spin.
    state["grasp_orient"] = np.array([0.0, 1.0, 0.0, 0.0])  # top-down quaternion (wxyz)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

async def get_world():
    world = World.instance()
    if world is None:
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()

    if not world.is_playing():
        await world.play_async()

    return world


def setup_franka(world):
    """Wrap the Franka prim and build the RMPflow controller."""
    global franka, controller

    franka = Franka(FRANKA_PRIM_PATH, name="franka")
    if not world.scene.get_object("franka"):
        world.scene.add(franka)

    # Acquire the physics handle (also initializes the gripper). Absolute open/closed
    # targets (no deltas) let a single open()/close() command hold.
    franka.initialize()
    franka.gripper.set_action_deltas(None)

    # High-level RMPflow controller: builds the Franka RMPflow config, the RmpFlow policy
    # and the ArticulationMotionPolicy, and captures the static robot base pose.
    controller = RMPFlowController(name="rmpflow_controller", robot_articulation=franka)


def arm_subset():
    """The arm-only joint subset (excludes the gripper fingers)."""
    return controller.get_articulation_motion_policy().get_active_joints_subset()


def ee_reached(target) -> bool:
    """True when the end-effector is within POS_TOL of ``target``."""
    ee_pos, _ = controller.get_motion_policy().get_end_effector_pose(
        arm_subset().get_joint_positions()
    )
    return bool(np.linalg.norm(np.array(ee_pos).flatten() - target) < POS_TOL)


# ---------------------------------------------------------------------------
# Physics callback: one pick-and-place step per simulation frame
# ---------------------------------------------------------------------------

def physics_step(dt):
    # Pressing Stop in the GUI invalidates the articulation physics handles while this
    # callback persists. On the next Play, re-acquire the handles and restart the sequence
    # instead of feeding RMPflow a None joint state.
    if not franka.handles_initialized:
        franka.initialize()
        franka.gripper.set_action_deltas(None)
        controller.reset()
        reset_sequence()
        return

    phase = state["phase"]

    # Step 1 - ready: drive the arm to the upright (hand-down) pose via the joint controller
    # before any task-space planning, so the wrist does not swing through a weird orientation.
    if phase == "ready":
        arm = arm_subset()
        arm.apply_action(joint_positions=READY_JOINTS)
        franka.gripper.open()
        if np.max(np.abs(arm.get_joint_positions() - READY_JOINTS)) < JOINT_TOL:
            # Re-seed RMPflow (reset() also re-applies the base pose) and lock in this pose's
            # actual EE orientation for the rest of the sequence.
            controller.reset()
            _, ready_rot = controller.get_motion_policy().get_end_effector_pose(READY_JOINTS)
            state["grasp_orient"] = rot_matrices_to_quats(ready_rot)
            state["phase"] = "move"
        return

    # Every later step plans one RMPflow step toward the active target, and keeps clamping
    # the grip while carrying so the cylinder does not slip out as the arm moves.
    franka.apply_action(controller.forward(state["target"], state["grasp_orient"]))
    if phase in ("grasp", "lift", "transport"):
        franka.gripper.close()

    # Step 2 - move: reach the pick pose above the cylinder, then open and start descending.
    if phase == "move" and ee_reached(state["target"]):
        franka.gripper.open()
        state["target"] = TARGET_POSITION - np.array([0.0, 0.0, DESCEND_DEPTH])
        state["phase"] = "descend"

    # Step 3 - descend: lower onto the cylinder, then begin the grasp settle countdown.
    elif phase == "descend" and ee_reached(state["target"]):
        state["settle"] = SETTLE_FRAMES
        state["phase"] = "grasp"

    # Step 4 - grasp: hold position while the gripper closes, then lift the cylinder clear.
    elif phase == "grasp":
        state["settle"] -= 1
        if state["settle"] <= 0:
            state["target"] = state["target"] + np.array([0.0, 0.0, LIFT_HEIGHT])
            state["phase"] = "lift"

    # Step 5 - lift: once clear of the tray, head to the drop pose.
    elif phase == "lift" and ee_reached(state["target"]):
        state["target"] = DROP_POSITION
        state["phase"] = "transport"

    # Step 6 - transport: reach the drop pose, then open the gripper to release.
    elif phase == "transport" and ee_reached(state["target"]):
        franka.gripper.open()
        state["phase"] = "done"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run():
    world = await get_world()

    setup_franka(world)
    reset_sequence()

    # Register the run-loop callback, replacing any stale registration.
    if world.physics_callback_exists(CALLBACK_NAME):
        world.remove_physics_callback(CALLBACK_NAME)

    world.add_physics_callback(CALLBACK_NAME, physics_step)


asyncio.ensure_future(run())
