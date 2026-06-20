import asyncio
import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid

START_DROP_HEIGHT = 10.0 # meters

async def get_world():
    world = World.instance()
    if world is None:
        carb.log_info("Creating new world instance")
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
        await world.reset_async()
    else:
        carb.log_info("World exists. Return Existing")
        await world.reset_async()
        world.scene.clear()
        world.clear_all_callbacks()

    return world


# runs every physics step; step_size = physics dt (seconds)
def on_physics_step(step_size):
    cube = World.instance().scene.get_object("my_cube")
    position, _ = cube.get_world_pose()
    carb.log_info(f"fti: dt={step_size:.4f} | cube z={position[2]:.3f}")


async def run():
    world = await get_world()

    world.scene.add_default_ground_plane()
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="my_cube",
            position=np.array([0, 0, START_DROP_HEIGHT]),
        )
    )

    # ADD physics callback
    world.add_physics_callback("phys_cb", on_physics_step)
    carb.log_info(f"fti: exists: {world.physics_callback_exists('phys_cb')}")


asyncio.ensure_future(run())