import asyncio
from pathlib import Path

import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.materials import OmniPBR, VisualMaterial
from isaacsim.core.api.objects import DynamicCylinder
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.prims import define_prim, is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import Gf, Sdf, UsdLux, UsdShade  # raw USD for cases with no isaacsim wrapper


def create_carpet_material(mdl_path):
    """Author the Carpet_Berber_Gray MDL material and return it as a VisualMaterial.

    isaacsim has no wrapper for arbitrary MDL materials, so the shader is authored with
    UsdShade here and wrapped in VisualMaterial so it can be bound via apply_visual_material.
    """
    stage = get_current_stage()
    mat_path = Sdf.Path("/World/Looks/Carpet_Berber_Gray")
    material = UsdShade.Material.Define(stage, mat_path)

    shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
    shader.GetImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(mdl_path), sourceType="mdl")
    shader.SetSourceAssetSubIdentifier("Carpet_Berber_Gray", sourceType="mdl")
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(50.0, 50.0)
    )

    shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    shader_output.SetRenderType("material")
    material.CreateOutput("mdl:displacement", Sdf.ValueTypeNames.Token).ConnectToSource(
        shader_output
    )
    material.CreateOutput("mdl:surface", Sdf.ValueTypeNames.Token).ConnectToSource(
        shader_output
    )
    material.CreateOutput("mdl:volume", Sdf.ValueTypeNames.Token).ConnectToSource(
        shader_output
    )

    return VisualMaterial(
        name="Carpet_Berber_Gray",
        prim_path=str(mat_path),
        prim=material.GetPrim(),
        shaders_list=[shader],
        material=material,
    )


async def run():
    # Initialize World (Isaac Sim Script Editor pattern)
    world = World.instance()
    if world is None:
        carb.log_info("Creating new world instance")
        world = World(stage_units_in_meters=1.0)
        await world.initialize_simulation_context_async()
    else:
        carb.log_info("World exists. Resetting world")
        world.clear_all_callbacks()
        world.scene.clear()

    # Resolve asset paths from the local Isaac assets root.
    assets_root = get_assets_root_path()
    if assets_root is None:
        carb.log_error("Could not resolve Isaac assets root path")
        return

    env_usd = assets_root + "/Isaac/Environments/Grid/default_environment.usd"
    franka_usd = (
        assets_root
        + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    )
    container_usd = (
        assets_root
        + "/Isaac/Props/PackingTable/props/container_h20/container_h20_base.usd"
    )
    # Materials live outside the versioned Assets tree (.../isaacsim_assets/Materials/...)
    carpet_mdl = str(
        Path(assets_root).parents[2]
        / "Materials/2023_1/Base/Carpet/Carpet_Berber_Gray.mdl"
    )

    # --- Materials (created first so bindings can reference them) ---
    define_prim("/World/Looks", "Scope")
    omnipbr = OmniPBR(
        prim_path="/World/Looks/OmniPBR",
        name="OmniPBR",
        color=np.array([0.8069498, 0.1277411, 0.1277411]),
    )
    carpet = create_carpet_material(carpet_mdl)

    # --- default_environment (referenced) + carpet binding ---
    add_reference_to_stage(env_usd, "/World/default_environment")
    geom_path = "/World/default_environment/Environment/Geometry"
    if is_prim_path_valid(geom_path):
        SingleXFormPrim(geom_path).apply_visual_material(
            carpet, weaker_than_descendants=True
        )
    else:
        carb.log_warn(f"Environment geometry not found at {geom_path}; carpet not bound")

    # Boost the environment's SphereLight intensity to match the target scene.
    light_path = "/World/default_environment/SphereLight"
    if is_prim_path_valid(light_path):
        sphere_light = UsdLux.SphereLight(
            get_current_stage().GetPrimAtPath(light_path)
        )
        sphere_light.GetIntensityAttr().Set(180000.0)
    else:
        carb.log_warn(f"{light_path} not found; sphere light intensity not set")

    # --- franka (referenced) with Gripper/Mesh variants ---
    franka_prim = add_reference_to_stage(franka_usd, "/World/franka")
    for vset_name, vsel in (("Gripper", "Default"), ("Mesh", "Quality")):
        vset = franka_prim.GetVariantSet(vset_name)
        if vsel in vset.GetVariantNames():
            vset.SetVariantSelection(vsel)

    # --- Cylinder (dynamic rigid body) + OmniPBR binding ---
    cylinder = DynamicCylinder(
        prim_path="/World/Cylinder",
        name="Cylinder",
        position=np.array([0.6, 0.0, 0.11]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        scale=np.array([0.04, 0.04, 0.2]),
        radius=0.5,
        height=1.0,
        mass=0.1
    )
    cylinder.apply_visual_material(omnipbr, weaker_than_descendants=True)
    world.scene.add(cylinder)

    # Refinement overrides for a smoother cylinder (no isaacsim wrapper for these).
    cyl_prim = cylinder.prim
    cyl_prim.CreateAttribute(
        "refinementEnableOverride", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(True)
    cyl_prim.CreateAttribute(
        "refinementLevel", Sdf.ValueTypeNames.Int, custom=True
    ).Set(2)

    # --- container_h20_base (referenced) ---
    container_path = "/World/container_h20_base"
    add_reference_to_stage(container_usd, container_path)
    # SingleRigidPrim applies RigidBodyAPI and resets xform ops to a canonical
    # translate/orient/scale stack, so the target pose is authored cleanly.
    SingleRigidPrim(
        container_path,
        name="container_h20_base",
        translation=np.array([0.6, 0.0, 0.0]),
        orientation=euler_angles_to_quats(np.array([0.0, 0.0, 0.0])),
        scale=np.array([1.0, 1.0, 1.0]),
    )

    # Enable convex-decomposition collision on the instanced child mesh.
    inst_path = container_path + "/container_h20_inst"
    if is_prim_path_valid(inst_path):
        inst_geom = SingleGeometryPrim(inst_path, collision=True)
        inst_geom.set_collision_approximation("convexDecomposition")
        inst_geom.set_collision_enabled(True)
    else:
        carb.log_warn(
            f"{inst_path} not found; collision override skipped"
        )

    carb.log_info("Scene generation complete")


asyncio.ensure_future(run())
