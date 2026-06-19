import asyncio
from enum import Enum, auto

import carb
from isaacsim.robot.manipulators.examples.franka.franka import Franka
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim

async def get_world():
    world = World.instance()
    if world is None:
        carb.log_info("fti: motion_gen:Creating new world instance")
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
        
    return world


async def run():
    world = await get_world()
    
    # #initialize
    # if not world.scene.get_object("franka"):
    #     carb.log_info("fti: franka not exists. performing setup.")
    #     # franka, rmpflow, articulation_rmpflow = setup_robot("/World/franka")
    #     franka = Franka(prim_path="/World/franka", name="franka")
    #     franka.initialize()
    #     world.scene.add(franka)
    # else:
    #     carb.log_info("fti: franka already exists in the scene. Skipping setup.")

    # if not world.scene.get_object("cylinder"):
    #     cylinder = SingleXFormPrim("/World/Cylinder", name="cylinder")
    #     world.scene.add(cylinder)
    #     carb.log_info("fti: /World/Cylinder added to scene")
    # else:
    #     cylinder = world.scene.get_object("cylinder")


    # franka = world.scene.get_object("franka")
       


asyncio.ensure_future(run())