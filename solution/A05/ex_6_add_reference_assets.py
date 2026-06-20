import asyncio
import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage

OBJECT_POSITION = [1.0, 0.0, 0.0] # meters
OBJECT_ORIENTATION = [0.0, 0.0, 45.0] #degrees


async def get_world():
    world = World.instance()
    if world is None:
        carb.log_info("Creating new world instance")
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
        await world.reset_async()
    else:
        carb.log_info("World exists. Resetting world")
        await world.reset_async()
        world.scene.clear()
        world.clear_all_callbacks()

    return world


async def run():
    world = await get_world()

    # Scene setup
    world.scene.add_default_ground_plane()

    assets_root = get_assets_root_path()
    container_usd = (
        assets_root
        + "/Isaac/Props/PackingTable/props/container_h20/container_h20_base.usd"
    )
    container_path = "/World/container_h20_base"
    add_reference_to_stage(container_usd, container_path)

    # Apply only translation and orientation (no rigid body, no collision)
    SingleXFormPrim(
        container_path,
        name="container_h20_base",
        translation=np.array(OBJECT_POSITION),
        orientation=euler_angles_to_quats(np.array(OBJECT_ORIENTATION)),
    )

    carb.log_info("fti: Scene generation complete")


asyncio.ensure_future(run())