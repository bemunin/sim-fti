import asyncio
import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.materials import OmniPBR
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, DynamicSphere
from isaacsim.core.utils.prims import define_prim


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

def create_materials():
    define_prim("/World/Looks", "Scope")
    omnipbr_red = OmniPBR(
        prim_path="/World/Looks/OmniPBR_red",
        name="OmniPBR_red",
        color=np.array([0.9266409, 0.00715554, 0.00715554]),
    )
    omnipbr_green = OmniPBR(
        prim_path="/World/Looks/OmniPBR_green",
        name="OmniPBR_green",
        color=np.array([0.04518124, 0.8687259, 0.04024986]),
    )
    omnipbr_blue = OmniPBR(
        prim_path="/World/Looks/OmniPBR_blue",
        name="OmniPBR_blue",
        color=np.array([0.08720798, 0.30587226, 0.9034749]),
    )
    return omnipbr_red, omnipbr_green, omnipbr_blue

def create_cube(world, material):
    cube = DynamicCuboid(
        prim_path="/World/Cube",
        name="Cube",
        position=np.array([1.0, 1.0, 0.25]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        scale=np.array([0.5, 0.5, 0.5]),
        size=1.0,
    )
    cube.apply_visual_material(material, weaker_than_descendants=True)
    world.scene.add(cube)

def create_cylinder(world, material):
    cylinder = DynamicCylinder(
        prim_path="/World/Cylinder",
        name="Cylinder",
        position=np.array([1.0, -1.0, 0.5]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        scale=np.array([0.5, 0.5, 1.0]),
        radius=0.5,
        height=1.0,
    )
    cylinder.apply_visual_material(material, weaker_than_descendants=True)
    world.scene.add(cylinder)

def create_sphere(world, material):
    sphere = DynamicSphere(
        prim_path="/World/Sphere",
        name="Sphere",
        position=np.array([-1.0, 1.0, 0.3]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        scale=np.array([0.6, 0.6, 0.6]),
        radius=0.5,
    )
    sphere.apply_visual_material(material, weaker_than_descendants=True)
    world.scene.add(sphere)

async def run():
    world = await get_world()

    world.scene.add_default_ground_plane()
    omnipbr_red, omnipbr_green, omnipbr_blue = create_materials()

    create_cube(world, omnipbr_red)
    create_cylinder(world, omnipbr_green)
    create_sphere(world, omnipbr_blue)

    carb.log_info("fti: Scene 1 generation complete")


asyncio.ensure_future(run())
