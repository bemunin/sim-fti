import asyncio

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

    
async def run():
    # Initialize World
    world = World.instance()
    if world is None:
        carb.log_info("Creating new world instance")
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
        await world.reset_async()
    else:
        # clear all existing tasks and callbacks
        carb.log_info("World exists. Reseting world")
        world.clear_all_callbacks()
        world.scene.clear()
        await world.reset_async()
        carb.log_info("World reset done")

    # Setup scene
    world.scene.add_default_ground_plane()
    franka = Franka(prim_path="/World/Franka_01", name="franka")
    franka.initialize()

    cube = DynamicCuboid(
        prim_path="/World/random_cube",
        name="cube",
        position=np.array([0.5, 0.3, 0.15]),
        scale=np.array([0.0515, 0.0515, 0.0515]),
        color=np.array([0, 0, 1.0]),
    )
    world.scene.add(franka)
    world.scene.add(cube)
    
    controller = PickPlaceController(
      name="pick_place_controller",
      gripper=franka.gripper,
      robot_articulation=franka,
    ) 

    franka.gripper.set_joint_positions(franka.gripper.joint_opened_positions)
    
    # Execution
    def physic_step(dt):
      cube_pos, _ = world.scene.get_object("cube").get_world_pose()
      goal_pos = np.array([0.5, -0.3, 0.0515 / 2.0])
      current_robot_joints = franka.get_joint_positions()
      actions = controller.forward(
          picking_position=cube_pos,
          placing_position=goal_pos,
          current_joint_positions=current_robot_joints,
      )
      franka.apply_action(actions)
      if controller.is_done():
          carb.log_info("Pick and place done")
          world.remove_physics_callback("franka_step")
          world.pause()

    world.add_physics_callback("franka_step", physic_step)
    await world.play_async()

asyncio.ensure_future(run())
