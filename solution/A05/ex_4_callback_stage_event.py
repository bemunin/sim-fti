import asyncio
import carb
import omni.usd
from omni.isaac.core import World


async def get_world():
    world = World.instance()
    if world is None:
        carb.log_info("Creating new world instance")
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
        await world.reset_async()
        world.scene.add_default_ground_plane()
    else:
        carb.log_info("World exists. Return Existing")
        await world.reset_async()
        world.scene.clear()
        world.clear_all_callbacks()

    return world


def on_stage_event(event):
    # event.type is an int from omni.usd.StageEventType
    if event.type == int(omni.usd.StageEventType.SAVED):
        carb.log_info("fti: Stage: SAVED")
    elif event.type == int(omni.usd.StageEventType.CLOSED):
        carb.log_info("fti: Stage: CLOSED")
    elif event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
        carb.log_info("fti: Stage: SELECTION_CHANGED")


async def run():
    world = await get_world()

    world.add_stage_callback("my_stage_cb", on_stage_event)


asyncio.ensure_future(run())