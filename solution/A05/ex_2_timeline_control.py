import asyncio
import carb
from omni.isaac.core import World

"reset | play | pause | stop"
EXECUTION="play"

async def get_world():
    world = World.instance()
    if world is None:
        carb.log_info("Creating new world instance")
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
        await world.reset_async()
        world.scene.add_default_ground_plane()
    else:
        # clear all existing tasks and callbacks
        carb.log_info("World exists. Return Existing")
    
    return world


async def run():
    world = await get_world()

    if EXECUTION == "reset":
        await world.reset_async()

    elif EXECUTION == "play":
        await world.play_async()

    elif EXECUTION == "pause":
        world.pause()

    elif EXECUTION == "stop":
        await world.stop_async()

    carb.log_info(f"fti: playing: {world.is_playing()}")
    carb.log_info(f"fti: stopped: {world.is_stopped()}")
    carb.log_info(f"fti: ---")

asyncio.ensure_future(run())