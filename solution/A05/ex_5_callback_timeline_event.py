import asyncio
import carb
import omni.timeline
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


def on_timeline_event(event):
    # event.type is an int from omni.timeline.TimelineEventType
    if event.type == int(omni.timeline.TimelineEventType.PLAY):
        carb.log_info("fti: Timeline: PLAY")
    elif event.type == int(omni.timeline.TimelineEventType.PAUSE):
        carb.log_info("fti: Timeline: PAUSE")
    elif event.type == int(omni.timeline.TimelineEventType.STOP):
        carb.log_info("fti: Timeline: STOP")


async def run():
    world = await get_world()

    # ADD timeline callback
    world.add_timeline_callback("my_timeline_cb", on_timeline_event)


asyncio.ensure_future(run())