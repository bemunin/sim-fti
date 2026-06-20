from enum import Enum, auto

import isaacsim.robot.surface_gripper._surface_gripper as _sg
import numpy as np
import omni.usd
from isaacsim.core.api.objects.cuboid import DynamicCuboid
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    get_supported_robot_policy_pairs,
    load_supported_motion_policy_config,
)
from pxr import UsdShade
from usd.schema.isaac import robot_schema


class TaskState(Enum):
    STOPPING = auto()
    STANDBY = auto()
    APPROACH = auto()
    PICK = auto()
    MOVE_TO_PLACE = auto()
    RELEASE = auto()


class UR10PickPlaceTask:
    """
    Boilerplate pick-and-place task for a UR10 robot.

    Mirrors the structure of UR10TrajectoryGenerationExample so that the
    same extension lifecycle (setup → update per physics step → reset) can
    be used to drive a pick-and-place behaviour.

    Methods
    -------
    load_example_assets()
        Load and return assets to be registered with the World scene.
    setup()
        Initialise controllers / solvers once physics handles are ready.
    update(step)
        Advance the task by one physics step.
    reset()
        Return the task to its initial state.
    """

    def __init__(self):
        self._articulation = None
        self._target = None
        self._cube = None
        self._rmpflow = None
        self._articulation_rmpflow = None
        self._action_sequence = []
        self._action_sequence_index = 0
        self._cube_initial_position = np.array([1, 0, 0.8])
        self._state = None
        self._stop_done_fn = None
        self._standby_joint_positions = None
        self._gripper_iface = None
        self._surface_gripper_path = "/World/Robots/ur10/ee_link/SurfaceGripper"
        self._pick_phase = 0
        self._release_phase = 0
        self._standby_pos = np.array([1.25, 0.4, 1.32])
        self._place_pos = np.array([2.0, 1.1, 1.3])

    def load_example_assets(self) -> list:
        """Load assets onto the stage and return them for World registration.

        Returns
        -------
        list
            Objects to be passed to ``world.scene.add()``.
        """
        robot_prim_path = "/World/Robots/ur10"
        self._articulation = SingleArticulation(robot_prim_path)

        add_reference_to_stage(
            get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd",
            "/World/ur10_target",
        )
        self._target = SingleXFormPrim("/World/ur10_target", scale=[0.04, 0.04, 0.04])

        self._cube = DynamicCuboid(
            prim_path="/World/pick_cube",
            name="pick_cube",
            position=self._cube_initial_position,
            scale=np.array([0.1, 0.1, 0.1]),
        )
        stage = omni.usd.get_context().get_stage()
        material_prim = stage.GetPrimAtPath("/World/Looks/Aluminum_Anodized_Red")
        if material_prim.IsValid():
            material = UsdShade.Material(material_prim)
            UsdShade.MaterialBindingAPI(self._cube.prim).Bind(material)

        return [self._articulation, self._target, self._cube]

    def init(self) -> None:
        """Start the pick-place cycle by entering the Standby state."""
        self._stop_done_fn = None
        self._state = TaskState.STANDBY
        if self._target is not None:
            self._target.set_world_pose(
                self._standby_pos, euler_angles_to_quats([0, np.pi / 2, 0])
            )

    def setup(self) -> None:
        """Initialise controllers and planners after physics handles are ready."""
        print(
            "Supported Robots with a Provided RMPflow Config:",
            list(get_supported_robot_policy_pairs().keys()),
        )
        rmp_config = load_supported_motion_policy_config("UR10", "RMPflow")
        self._rmpflow = RmpFlow(**rmp_config)

        # Initialize an RmpFlow object
        # Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Franka robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(
            self._articulation, self._rmpflow
        )

        # Set up the SurfaceGripper prim if not already present (Short_Suction variant)
        stage = omni.usd.get_context().get_stage()
        gripper_prim = stage.GetPrimAtPath(self._surface_gripper_path)
        if not gripper_prim.IsValid():
            robot_schema.CreateSurfaceGripper(stage, self._surface_gripper_path)
            gripper_prim = stage.GetPrimAtPath(self._surface_gripper_path)
            suction_scope = stage.GetPrimAtPath(
                "/World/Robots/ur10/ee_link/suction_cup"
            )
            if suction_scope.IsValid():
                joint_paths = [p.GetPath() for p in suction_scope.GetChildren()]
                if joint_paths:
                    gripper_prim.GetRelationship(
                        robot_schema.Relations.ATTACHMENT_POINTS.name
                    ).SetTargets(joint_paths)
            gripper_prim.GetAttribute(
                robot_schema.Attributes.MAX_GRIP_DISTANCE.name
            ).Set(0.02)
            gripper_prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).Set(
                1.0
            )
            gripper_prim.GetAttribute(
                robot_schema.Attributes.SHEAR_FORCE_LIMIT.name
            ).Set(5.0)
            gripper_prim.GetAttribute(
                robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name
            ).Set(0.005)

        self._gripper_iface = _sg.acquire_surface_gripper_interface()

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _standby(self) -> None:
        """Hold the end-effector at the standby pose and wait for run() to be called."""
        self._target.set_world_pose(
            self._standby_pos, euler_angles_to_quats([0, np.pi / 2, 0])
        )

        joint_positions = (
            self._articulation_rmpflow.get_active_joints_subset().get_joint_positions()
        )
        ee_pos, _ = self._rmpflow.get_end_effector_pose(joint_positions)
        if np.linalg.norm(ee_pos - self._standby_pos) < 0.05:
            self._standby_joint_positions = self._articulation.get_joint_positions()

    def _stopping(self) -> None:
        """Drive to standby, then invoke the stop callback."""
        self._target.set_world_pose(
            self._standby_pos, euler_angles_to_quats([0, np.pi / 2, 0])
        )
        joint_positions = (
            self._articulation_rmpflow.get_active_joints_subset().get_joint_positions()
        )
        ee_pos, _ = self._rmpflow.get_end_effector_pose(joint_positions)
        if np.linalg.norm(ee_pos - self._standby_pos) < 0.05:
            self._standby_joint_positions = self._articulation.get_joint_positions()
            self._state = None
            if self._stop_done_fn is not None:
                self._stop_done_fn()
                self._stop_done_fn = None

    def _approach(self) -> None:
        """Move the end-effector to 0.4 m above the cube centre."""
        cube_pos, _ = self._cube.get_world_pose()
        approach_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.4])
        self._target.set_world_pose(
            approach_pos, euler_angles_to_quats([0, np.pi / 2, 0])
        )

    def _pick(self) -> None:
        """Execute pick sequence: descend → grasp → lift."""
        if self._pick_phase == 0:
            # Phase 0: descend to 5 cm above cube centre
            cube_pos, _ = self._cube.get_world_pose()
            descend_pos = np.array([cube_pos[0], cube_pos[1], cube_pos[2] + 0.215])
            self._target.set_world_pose(
                descend_pos, euler_angles_to_quats([0, np.pi / 2, 0])
            )
            joint_positions = self._articulation_rmpflow.get_active_joints_subset().get_joint_positions()
            ee_pos, _ = self._rmpflow.get_end_effector_pose(joint_positions)
            if np.linalg.norm(ee_pos - descend_pos) < 0.001:
                self._pick_phase = 1

        elif self._pick_phase == 1:
            # Phase 1: close suction gripper via SurfaceGripper API
            self._gripper_iface.close_gripper(self._surface_gripper_path)
            self._pick_phase = 2

        elif self._pick_phase == 2:
            # Phase 2: lift to z = 1.4 while gripper holds cube via D6 joint
            target_pos, _ = self._target.get_world_pose()
            flat_pos = np.array(target_pos).flatten()
            lift_pos = np.array([flat_pos[0], flat_pos[1], 1.4])
            self._target.set_world_pose(
                lift_pos,
                euler_angles_to_quats([0, np.pi / 2, 0]),
            )
            joint_positions = self._articulation_rmpflow.get_active_joints_subset().get_joint_positions()
            ee_pos, _ = self._rmpflow.get_end_effector_pose(joint_positions)
            if np.linalg.norm(np.array(ee_pos).flatten() - lift_pos) < 0.05:
                self._state = TaskState.MOVE_TO_PLACE

    def _move_to_place(self) -> None:
        """Carry the cube to above the bin at z=1.4, then release."""
        if not omni.usd.get_context().get_stage().GetPrimAtPath("/World/Bin").IsValid():
            return
        bin_xform = SingleXFormPrim("/World/Bin")
        bin_pos, _ = bin_xform.get_world_pose()
        self._target.set_world_pose(
            self._place_pos,
            euler_angles_to_quats([0, np.pi / 2, 0]),
        )
        joint_positions = (
            self._articulation_rmpflow.get_active_joints_subset().get_joint_positions()
        )
        ee_pos, _ = self._rmpflow.get_end_effector_pose(joint_positions)
        if np.linalg.norm(np.array(ee_pos).flatten() - self._place_pos) < 0.05:
            self._release_phase = 0
            self._state = TaskState.RELEASE

    def _release(self, step: float) -> None:
        """Descend 0.2 m below the place position, open gripper, then return to Standby."""
        if self._release_phase == 0:
            descend_pos = np.array(
                [self._place_pos[0], self._place_pos[1], self._place_pos[2] - 0.3]
            )
            self._target.set_world_pose(
                descend_pos, euler_angles_to_quats([0, np.pi / 2, 0])
            )
            joint_positions = self._articulation_rmpflow.get_active_joints_subset().get_joint_positions()
            ee_pos, _ = self._rmpflow.get_end_effector_pose(joint_positions)
            if np.linalg.norm(np.array(ee_pos).flatten() - descend_pos) < 0.05:
                self._release_phase = 1
        elif self._release_phase == 1:
            self._gripper_iface.open_gripper(self._surface_gripper_path)
            self._release_phase = 0
            self._state = TaskState.STANDBY

    def run(self) -> None:
        """Transition from Standby to Approach."""
        if self._state == TaskState.STANDBY:
            self._state = TaskState.APPROACH

    def run_pickplace(self) -> None:
        """Transition from Approach to Pick."""
        if self._state == TaskState.APPROACH:
            self._pick_phase = 0
            self._state = TaskState.PICK

    def stop(self, done_fn) -> None:
        """Enter STOPPING state; pause is triggered once standby is reached."""
        self._stop_done_fn = done_fn
        self._state = TaskState.STOPPING
        if self._target is not None:
            self._target.set_world_pose(
                self._standby_pos, euler_angles_to_quats([0, np.pi / 2, 0])
            )

    def reset_cube(self) -> None:
        """Teleport the cube back to its initial position."""
        self._pick_phase = 0
        self._release_phase = 0
        if self._gripper_iface is not None:
            self._gripper_iface.open_gripper(self._surface_gripper_path)
        if self._cube is not None:
            self._cube.set_world_pose(self._cube_initial_position)
        self._action_sequence = []
        self._action_sequence_index = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def update(self, step: float) -> None:
        """Advance the task by one physics step.

        Parameters
        ----------
        step:
            Duration of the current physics step in seconds.
        """
        if self._rmpflow is None or self._target is None or self._state is None:
            return

        # Dispatch to the active state handler
        if self._state == TaskState.STOPPING:
            self._stopping()
        elif self._state == TaskState.STANDBY:
            self._standby()
        elif self._state == TaskState.APPROACH:
            self._approach()
        elif self._state == TaskState.PICK:
            self._pick()
        elif self._state == TaskState.MOVE_TO_PLACE:
            self._move_to_place()
        elif self._state == TaskState.RELEASE:
            self._release(step)

        # Drive the arm toward the current target each step
        target_position, target_orientation = self._target.get_world_pose()
        self._rmpflow.set_end_effector_target(target_position, target_orientation)
        self._rmpflow.update_world()
        robot_base_translation, robot_base_orientation = (
            self._articulation.get_world_pose()
        )
        self._rmpflow.set_robot_base_pose(
            robot_base_translation, robot_base_orientation
        )
        action = self._articulation_rmpflow.get_next_articulation_action(step)
        self._articulation.apply_action(action)

    def reset(self) -> None:
        """Reset the task to its initial state."""
        self._state = None
        self._stop_done_fn = None
        self._pick_phase = 0
        self._release_phase = 0
        if self._gripper_iface is not None:
            self._gripper_iface.open_gripper(self._surface_gripper_path)
        if self._standby_joint_positions is not None:
            self._articulation.set_joint_positions(self._standby_joint_positions)
        if self._target is not None:
            self._target.set_world_pose(
                self._standby_pos, euler_angles_to_quats([0, np.pi / 2, 0])
            )
        if self._cube is not None:
            self._cube.set_world_pose(self._cube_initial_position)
        self._action_sequence = []
        self._action_sequence_index = 0
