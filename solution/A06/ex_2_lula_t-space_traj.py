import asyncio
import os

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot_motion.motion_generation import (
    ArticulationTrajectory,
    LulaTaskSpaceTrajectoryGenerator,
)

# URDF link the task-space generator drives toward.
END_EFFECTOR_FRAME = "panda_hand"
PHYSICS_DT = 1.0 / 60.0  # physics rate the action sequence is sampled at (60 Hz)

# Cartesian waypoints for the gripper: a closed rectangular loop in front of the robot.
POSITION_TARGETS = np.array(
    [
        [0.3, -0.3, 0.3],
        [0.3, 0.3, 0.3],
        [0.3, 0.3, 0.5],
        [0.3, -0.3, 0.5],
        [0.3, -0.3, 0.3],
    ]
)
# Orientation quaternions (w, x, y, z); [0, 1, 0, 0] points the gripper straight down.
ORIENTATION_TARGETS = np.tile(np.array([0, 1, 0, 0]), (len(POSITION_TARGETS), 1))

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
        world.clear_all_callbacks()  

    return world

def create_franka(world):
    franka = Franka(prim_path="/World/Franka_01", name="franka")
    franka.initialize()
    world.scene.add(franka)
    return franka

def create_trajectory_generator():
    # Config files for supported robots ship with the motion_generation extension
    # under "/motion_policy_configs".
    mg_extension_path = get_extension_path_from_name(
        "isaacsim.robot_motion.motion_generation"
    )
    rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

    return LulaTaskSpaceTrajectoryGenerator(
        robot_description_path=rmp_config_dir
        + "/franka/rmpflow/robot_descriptor.yaml",
        urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf",
    )

def create_action_sequence(franka):
    generator = create_trajectory_generator()
    trajectory = generator.compute_task_space_trajectory_from_points(
        POSITION_TARGETS, ORIENTATION_TARGETS, END_EFFECTOR_FRAME
    )
    if trajectory is None:
        carb.log_warn("fti: No task-space trajectory could be computed")
        return []

    # Convert the trajectory into ArticulationActions applied one per physics step.
    articulation_trajectory = ArticulationTrajectory(franka, trajectory, PHYSICS_DT)
    return articulation_trajectory.get_action_sequence()

def teleport_to_action(franka, action):
    # Snap the arm to the first action's joint configuration so it does not jump
    # from the home pose when playback starts.
    positions = np.zeros(franka.num_dof)
    positions[action.joint_indices] = action.joint_positions
    franka.set_joint_positions(positions)
    franka.set_joint_velocities(np.zeros_like(positions))

async def run():
    world = await get_world()

    franka = world.scene.get_object("franka")
    if franka is None:
        world.scene.add_default_ground_plane()
        franka = create_franka(world)
        action_sequence = create_action_sequence(franka)
        teleport_to_action(franka, create_action_sequence(franka)[0])  # snap to first pose before starting
    else:
        carb.log_info("fti: Franka exists. Return Existing")
        action_sequence = create_action_sequence(franka)
    
    action_index = 0

    def physic_step(dt):
        nonlocal action_index

        if not action_sequence:
            world.remove_physics_callback("trajectory_step")
            return

        if action_index < len(action_sequence):
            franka.apply_action(action_sequence[action_index])
            action_index += 1
        else:
            carb.log_info("fti: Task-space trajectory complete")
            world.remove_physics_callback("trajectory_step")

    world.add_physics_callback("trajectory_step", physic_step)


asyncio.ensure_future(run())
