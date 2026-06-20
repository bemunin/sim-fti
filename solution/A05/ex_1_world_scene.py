import asyncio
import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

"add | get | remove | clear"
EXECUTION="add"

async def get_world():
    world = World.instance()
    if world is None:
        carb.log_info("Creating new world instance")
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
        await world.reset_async()
    else:
        carb.log_info("World exists. Return Existing")
    
    return world

async def run():
    world = await get_world()
    
    if EXECUTION == "add":
        world.scene.add_default_ground_plane()
        world.scene.add(DynamicCuboid(
                prim_path="/World/Cube",
                name="my_cube",
                position=np.array([0, 0, 1.0]),
        ))
    
    elif EXECUTION == "get":
        cube = world.scene.get_object("my_cube")
        carb.log_info(f"fti: Get Object: {cube.name} at position {cube.get_world_pose()}")
        
    elif EXECUTION == "remove":
        world.scene.remove_object("my_cube")
        
    elif EXECUTION == "clear":
        # CLEAR (remove everything)
        world.scene.clear()


asyncio.ensure_future(run())