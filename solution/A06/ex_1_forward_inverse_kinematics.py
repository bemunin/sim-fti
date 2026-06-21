import asyncio

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)

# forward | inverse | clear
EXECUTION = "inverse"

# End-effector frame the Lula solver reasons about for the Franka.
END_EFFECTOR_FRAME = "right_gripper"

# Max joint speed (rad/s). Solved targets are rate-limited to this so the arm
# eases toward the goal instead of snapping to it in a single step.
MAX_JOINT_SPEED = np.deg2rad(400.0)

# Forward kinematic Config:

# radians in the implementation. (The Franka has 7 arm DOF + 2 finger DOF.)
ARM_JOINT_INDICES = np.array([0, 1, 2, 3, 4, 5, 6])
FK_JOINT_TARGET = np.array([0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0])


# Inverse Kinematics Config
# Orientation as euler angles (roll, pitch, yaw) in DEGREES; converted to a
# quaternion in the implementation. [180, 0, 0] points the gripper straight down.
TARGET_ORIENTATION = np.array([180.0, 0.0, 0.0])
TARGET_POSITION = np.array([0.4, 0.0, 0.6])


async def get_world():
    world = World.instance()
    if world is None:
        carb.log_info("fti: Creating new world instance")
        world = World(stage_units_in_meters=1.0)
        world.clear()
        await world.initialize_simulation_context_async()
        await world.reset_async()
    else:
        carb.log_info("fti: World exists. Return Existing")
        carb.log_info("fti: Reset and clear everything including stage")
        world.clear_all_callbacks()  # World API: clears scene, callbacks, and resets world state

    return world

def create_franka(world):
    franka = Franka(prim_path="/World/Franka_01", name="franka")
    franka.initialize()
    world.scene.add(franka)
    return franka

def rate_limited_action(franka, target_positions, joint_indices, dt):
    # Step the commanded joints at most MAX_JOINT_SPEED * dt toward the target,
    # so the arm moves smoothly instead of jumping straight to the solution.
    current = franka.get_joint_positions()
    if joint_indices is not None:
        current = current[joint_indices]
    max_delta = MAX_JOINT_SPEED * dt
    step = np.clip(np.asarray(target_positions) - current, -max_delta, max_delta)
    return ArticulationAction(
        joint_positions=current + step, joint_indices=joint_indices
    )

def create_kinematics_solver(franka):
    # Load the bundled Lula config for the supported "Franka" robot, then wrap it
    # so it drives this articulation through the chosen end-effector frame.
    kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config(
        "Franka"
    )
    kinematics_solver = LulaKinematicsSolver(**kinematics_config)
    articulation_solver = ArticulationKinematicsSolver(
        franka, kinematics_solver, END_EFFECTOR_FRAME
    )
    return kinematics_solver, articulation_solver

async def run():
    # Ensure configured values are floats so downstream APIs (scipy, Lula) don't
    # choke on integer dtypes.
    global TARGET_ORIENTATION, TARGET_POSITION, MAX_JOINT_SPEED, FK_JOINT_TARGET
    TARGET_ORIENTATION = np.asarray(TARGET_ORIENTATION, dtype=float)
    TARGET_POSITION = np.asarray(TARGET_POSITION, dtype=float)
    MAX_JOINT_SPEED = float(MAX_JOINT_SPEED)
    FK_JOINT_TARGET = np.asarray(FK_JOINT_TARGET, dtype=float)

    world = await get_world()

    if EXECUTION == "clear":
        world.clear()
        await world.reset_async()
        carb.log_info("fti: World cleared")
        return

    franka = world.scene.get_object("franka")
    if franka is None:
        world.scene.add_default_ground_plane()
        franka = create_franka(world)
    else:
        franka.initialize()

    kinematics_solver, articulation_solver = create_kinematics_solver(franka)
    joint_target = np.deg2rad(FK_JOINT_TARGET) 
    target_orientation = euler_angles_to_quats(TARGET_ORIENTATION, degrees=True)

    def physic_step(dt):
        base_translation, base_orientation = franka.get_world_pose()
        kinematics_solver.set_robot_base_pose(base_translation, base_orientation)

        if EXECUTION == "forward":
            franka.apply_action(
                rate_limited_action(franka, joint_target, ARM_JOINT_INDICES, dt)
            )
            current_joints = franka.get_joint_positions()[ARM_JOINT_INDICES]
            if np.allclose(current_joints, joint_target, atol=0.001):
                ee_position, ee_rotation = articulation_solver.compute_end_effector_pose()
                carb.log_info(f"fti: FK end-effector position = {ee_position}")
                carb.log_info(f"fti: FK end-effector rotation =\n{ee_rotation}")
                world.remove_physics_callback("kinematics_step")
        else:
            action, success = articulation_solver.compute_inverse_kinematics(
                target_position=TARGET_POSITION,
                target_orientation=target_orientation,
            )
            if not success:
                carb.log_warn("fti: IK failed to find a solution for the target pose")
                return

            franka.apply_action(
                rate_limited_action(
                    franka, action.joint_positions, action.joint_indices, dt
                )
            )

            ee_position, _ = articulation_solver.compute_end_effector_pose()
            if np.linalg.norm(ee_position - TARGET_POSITION) < 0.001:
                carb.log_info("fti: IK target reached")
                world.remove_physics_callback("kinematics_step")

    world.add_physics_callback("kinematics_step", physic_step)


asyncio.ensure_future(run())
