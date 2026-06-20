import asyncio

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

# cube | cylinder
TARGET_OBJECT = "cylinder"
GRASP_APPROACH = 0.02  # offset for grasping approach in meters

async def get_world():
    world = World.instance()
    if world is None:
        carb.log_info("fti: Creating new world instance")
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
        await world.reset_async()
    else:
        carb.log_info("fti: World exists. Return Existing")
        carb.log_info("fti: Reset and clear everything including stage")
        world.clear()  # World API: clears scene, callbacks, and resets world state
        await world.reset_async()

    return world

def create_franka(world):
    franka = Franka(prim_path="/World/Franka_01", name="franka")
    franka.initialize()
    world.scene.add(franka)
    return franka

def create_object(world):
    if TARGET_OBJECT == "cylinder":
        height = 1.0
        scale = np.array([0.04, 0.04, 0.2])
        object_height = height * scale[2]  # effective standing height
        object = DynamicCylinder(
            prim_path="/World/random_object",
            name="object",
            position=np.array([0.5, 0.3, object_height / 2.0]),
            scale=scale,
            radius=0.5,
            height=height,
            mass=0.1,
            color=np.array([0, 0, 1.0]),
        )
    else:
        size = 1.0
        scale = np.array([0.0515, 0.0515, 0.0515])
        object_height = size * scale[2]  # effective standing height
        object = DynamicCuboid(
            prim_path="/World/random_object",
            name="object",
            position=np.array([0.5, 0.3, object_height / 2.0]),
            scale=scale,
            size=size,
            color=np.array([0, 0, 1.0]),
        )
    world.scene.add(object)
    return object, object_height

def create_controller(franka):
    controller = PickPlaceController(
        name="pick_place_controller",
        gripper=franka.gripper,
        robot_articulation=franka,
    )
    franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
    return controller

async def run():
    world = await get_world()

    world.scene.add_default_ground_plane()
    franka = create_franka(world)
    _, object_height = create_object(world)
    controller = create_controller(franka)

    def physic_step(dt):
        object_pos, _ = world.scene.get_object("object").get_world_pose()
        # Pick at the top of the object, pressing in slightly for a secure grip.
        picking_position = np.array(
            [object_pos[0], object_pos[1], object_height - GRASP_APPROACH]
        )
        placing_position = np.array([0.5, -0.3, object_height])

        actions = controller.forward(
            picking_position=picking_position,
            placing_position=placing_position,
            current_joint_positions=franka.get_joint_positions(),
        )
        franka.apply_action(actions)

        if controller.is_done():
            carb.log_info("fti: Pick and place done")
            world.remove_physics_callback("franka_step")
            world.pause()

    world.add_physics_callback("franka_step", physic_step)
    await world.play_async()


asyncio.ensure_future(run())
