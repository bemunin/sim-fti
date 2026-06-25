#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/planning_scene.hpp>
#include <moveit_msgs/msg/planning_scene_components.hpp>
#include <moveit_msgs/msg/allowed_collision_matrix.hpp>
#include <moveit_msgs/msg/allowed_collision_entry.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <shape_msgs/msg/mesh.hpp>
#include <geometric_shapes/shapes.h>
#include <geometric_shapes/mesh_operations.h>
#include <geometric_shapes/shape_operations.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("scene_pick_place");

class ScenePickPlace : public rclcpp::Node
{
public:
  explicit ScenePickPlace(const rclcpp::NodeOptions& options);

  void initializeMoveGroup();

private:
  bool waitForRobotState(double timeout_sec = 2.0);

  void setupPlanningScene();

  // Plan and move the arm to a named pose defined in the SRDF (e.g. "ready").
  void moveTo(const std::string& target);

  // Plan and move the EE to target_pose with the Pilz planner (default LIN, a
  // straight Cartesian line), then restore the OMPL pipeline used elsewhere.
  void moveTo(const geometry_msgs::msg::Pose& target_pose, const std::string& planner_id = "LIN");

  // Plan and move the EE to target_pose using the OMPL pipeline.
  void moveToPoseOmpl(const geometry_msgs::msg::Pose& target_pose);

  // Drive the gripper to a named SRDF state ("open" or "close").
  void setGripper(const std::string& state);

  // Drive the gripper to an explicit finger-joint value, in metres
  // (panda_finger_joint1, range ~0.0 closed .. 0.04 open).
  void setGripper(double joint_value);

  // Toggle collisions in the ACM between a collision object and the given
  // links/objects. With allowed=true (default) it whitelists them, so e.g. the
  // gripper can close onto an object without the plan being rejected; with
  // allowed=false it re-enables collision checking, so the planner avoids them
  // again. Reads the current ACM, applies the change, and writes it back as a
  // planning-scene diff.
  void allowCollision(const std::string& object_id, const std::vector<std::string>& links,
                      bool allowed = true);

  // Main loop. Program the pick-and-place sequence here.
  void timerCallback();

  rclcpp::CallbackGroup::SharedPtr timer_group_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> mg_arm_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> mg_gripper_;
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface_;

  // Cylinder to grasp, recorded by setupPlanningScene() and read by
  // timerCallback() to derive the top-down approach pose (so the grasp follows
  // the scene, not hardcoded values).
  geometry_msgs::msg::Pose cylinder_pose_;

  std::size_t step_{ 0 };
  // Guards against the reentrant timer starting a new step while move() blocks.
  std::atomic<bool> busy_{ false };
};

// ============================ Method definitions ============================

ScenePickPlace::ScenePickPlace(const rclcpp::NodeOptions& options)
  : Node("scene_pick_place", options)
{
  // Reentrant so the timer can run while MoveGroup callbacks are in flight.
  timer_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&ScenePickPlace::timerCallback, this),
      timer_group_);

  RCLCPP_INFO(this->get_logger(), "ScenePickPlace created. Waiting for MoveGroup init...");
}

void ScenePickPlace::initializeMoveGroup()
{
  mg_arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "panda_arm");
  mg_gripper_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "hand");

  // The arm plans with OMPL throughout (free-space, collision-aware goals).
  mg_arm_->setPlanningPipelineId("ompl");
  mg_arm_->setPlanningTime(10.0);
  mg_arm_->setNumPlanningAttempts(10);
  mg_arm_->setMaxVelocityScalingFactor(0.3);
  mg_arm_->setMaxAccelerationScalingFactor(0.3);

  // The gripper also plans with OMPL.
  mg_gripper_->setPlanningPipelineId("ompl");
  mg_gripper_->setMaxVelocityScalingFactor(0.3);
  mg_gripper_->setMaxAccelerationScalingFactor(0.3);

  RCLCPP_INFO(this->get_logger(), "Waiting for robot state...");
  if (!waitForRobotState(2.0))
  {
    RCLCPP_ERROR(this->get_logger(), "Failed to receive robot state");
  }
  else
  {
    RCLCPP_INFO(this->get_logger(), "Robot state ready. Planning frame: %s, EE link: %s",
                mg_arm_->getPlanningFrame().c_str(),
                mg_arm_->getEndEffectorLink().c_str());
  }
  
  setupPlanningScene();
}

bool ScenePickPlace::waitForRobotState(double timeout_sec)
{
  auto start = this->now();
  rclcpp::Duration timeout = rclcpp::Duration::from_seconds(timeout_sec);

  while ((this->now() - start) < timeout)
  {
    if (mg_arm_->getCurrentState(0.1))
    {
      return true;
    }
    rclcpp::sleep_for(std::chrono::milliseconds(50));
  }
  return false;
}

void ScenePickPlace::setupPlanningScene()
{
  // Wipe any collision objects left in move_group from a previous run of this
  // node so the scene is rebuilt from scratch (move_group outlives the node).
  std::vector<std::string> known = planning_scene_interface_.getKnownObjectNames();
  if (!known.empty())
  {
    planning_scene_interface_.removeCollisionObjects(known);
    RCLCPP_INFO(this->get_logger(), "Cleared %zu stale collision object(s)", known.size());
  }


  // Ground floor: a large thin slab with its top face at z = 0, so the planner
  // keeps the arm from dipping below the robot base.
  moveit_msgs::msg::CollisionObject ground;
  ground.header.frame_id = mg_arm_->getPlanningFrame();
  ground.id = "ground";

  shape_msgs::msg::SolidPrimitive ground_box;
  ground_box.type = ground_box.BOX;
  ground_box.dimensions = { 4.0, 4.0, 0.02 };  // x, y, z (metres)

  geometry_msgs::msg::Pose ground_pose;
  ground_pose.orientation.w = 1.0;
  ground_pose.position.x = 0.0;
  ground_pose.position.y = 0.0;
  ground_pose.position.z = -0.02;  // top face sits at z = 0

  ground.primitives.push_back(ground_box);
  ground.primitive_poses.push_back(ground_pose);
  ground.operation = ground.ADD;

  // Container mesh loaded from the package's meshes/ folder.
  moveit_msgs::msg::CollisionObject container;
  container.header.frame_id = mg_arm_->getPlanningFrame();
  container.id = "container_h20_base";

  shapes::Mesh* mesh = shapes::createMeshFromResource(
      "package://b06/meshes/container_h20_base.obj");
  shapes::ShapeMsg mesh_msg_tmp;
  shapes::constructMsgFromShape(mesh, mesh_msg_tmp);
  shape_msgs::msg::Mesh mesh_msg = boost::get<shape_msgs::msg::Mesh>(mesh_msg_tmp);
  container.meshes.push_back(mesh_msg);

  geometry_msgs::msg::Pose mesh_pose;
  mesh_pose.position.x = 0.0;
  mesh_pose.position.y = 0.0;
  mesh_pose.position.z = 0.0;
  
  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, 0.0);  // roll, pitch, yaw
  mesh_pose.orientation = tf2::toMsg(q);

  container.mesh_poses.push_back(mesh_pose);
  container.operation = container.ADD;

  // Cylinder object to grasp.
  moveit_msgs::msg::CollisionObject cylinder;
  cylinder.header.frame_id = mg_arm_->getPlanningFrame();
  cylinder.id = "cylinder";

  shape_msgs::msg::SolidPrimitive cylinder_shape;
  cylinder_shape.type = cylinder_shape.CYLINDER;
  cylinder_shape.dimensions = { 0.2, 0.02 };  // height, radius (metres)

  geometry_msgs::msg::Pose cylinder_pose;
  cylinder_pose.orientation.w = 1.0;
  cylinder_pose.position.x = 0.6;
  cylinder_pose.position.y = 0.0;
  cylinder_pose.position.z = 0.1;

  cylinder.primitives.push_back(cylinder_shape);
  cylinder.primitive_poses.push_back(cylinder_pose);
  cylinder.operation = cylinder.ADD;

  // Cube obstacle (thin vertical wall panel).
  moveit_msgs::msg::CollisionObject cube;
  cube.header.frame_id = mg_arm_->getPlanningFrame();
  cube.id = "cube";

  shape_msgs::msg::SolidPrimitive cube_box;
  cube_box.type = cube_box.BOX;
  cube_box.dimensions = { 0.05, 0.7, 0.7 };  // x (length), y (width), z (height)

  geometry_msgs::msg::Pose cube_pose;
  cube_pose.orientation.w = 1.0;
  cube_pose.position.x = 0.2;
  cube_pose.position.y = -0.6;
  cube_pose.position.z = 0.35;

  cube.primitives.push_back(cube_box);
  cube.primitive_poses.push_back(cube_pose);
  cube.operation = cube.ADD;

  // Apply all objects in a single batch so the scene updates atomically.
  planning_scene_interface_.applyCollisionObjects({ ground, container, cylinder, cube });

  // Record the cylinder so timerCallback() can aim the top-down approach at it.
  cylinder_pose_ = cylinder_pose;

  RCLCPP_INFO(this->get_logger(), "Planning scene initialized (added '%s', '%s', '%s', '%s')",
              ground.id.c_str(), container.id.c_str(), cylinder.id.c_str(), cube.id.c_str());
}

void ScenePickPlace::moveTo(const std::string& target)
{
  // Named-target moves (e.g. "ready") plan with OMPL; set it explicitly in case
  // a prior Pilz move left another pipeline selected.
  mg_arm_->setPlanningPipelineId("ompl");
  mg_arm_->setStartStateToCurrentState();
  bool ok = mg_arm_->setNamedTarget(target);
  if (!ok)
  {
    RCLCPP_ERROR(this->get_logger(), "Failed to set named target '%s'", target.c_str());
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Moving to '%s'...", target.c_str());
  auto result = mg_arm_->move();

  if (result)
    RCLCPP_INFO(this->get_logger(), "Reached target '%s'", target.c_str());
  else
    RCLCPP_ERROR(this->get_logger(), "Failed to reach '%s'", target.c_str());
}

void ScenePickPlace::moveTo(const geometry_msgs::msg::Pose& target_pose, const std::string& planner_id)
{
  // Switch to the Pilz pipeline for a Cartesian straight-line move; the rest of
  // the sequence plans with OMPL, so restore it once this move completes.
  mg_arm_->setPlanningPipelineId("pilz_industrial_motion_planner");
  mg_arm_->setPlannerId(planner_id);
  mg_arm_->setStartStateToCurrentState();
  mg_arm_->setPoseTarget(target_pose);

  RCLCPP_INFO(this->get_logger(),
              "Moving to pose [%.3f, %.3f, %.3f] (%s)...",
              target_pose.position.x, target_pose.position.y, target_pose.position.z,
              planner_id.c_str());
  auto result = mg_arm_->move();
  mg_arm_->clearPoseTargets();

  // Restore the OMPL pipeline used by the rest of the sequence.
  mg_arm_->setPlanningPipelineId("ompl");

  if (result)
    RCLCPP_INFO(this->get_logger(), "Reached pose target");
  else
    RCLCPP_ERROR(this->get_logger(), "Failed to reach pose target");
}

void ScenePickPlace::setGripper(const std::string& state)
{
  if (state != "open" && state != "close")
  {
    RCLCPP_ERROR(this->get_logger(),
                 "Unknown gripper state '%s' (expected 'open' or 'close')", state.c_str());
    return;
  }

  mg_gripper_->setStartStateToCurrentState();
  if (!mg_gripper_->setNamedTarget(state))
  {
    RCLCPP_ERROR(this->get_logger(), "Failed to set gripper named target '%s'", state.c_str());
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Setting gripper '%s'...", state.c_str());
  auto result = mg_gripper_->move();

  if (result)
    RCLCPP_INFO(this->get_logger(), "Gripper '%s' done", state.c_str());
  else
    RCLCPP_ERROR(this->get_logger(), "Failed to set gripper '%s'", state.c_str());
}

void ScenePickPlace::setGripper(double joint_value)
{
  mg_gripper_->setStartStateToCurrentState();
  if (!mg_gripper_->setJointValueTarget("panda_finger_joint1", joint_value))
  {
    RCLCPP_ERROR(this->get_logger(),
                 "Gripper joint value %.4f is out of bounds", joint_value);
    return;
  }

  RCLCPP_INFO(this->get_logger(), "Setting gripper to %.4f m...", joint_value);
  auto result = mg_gripper_->move();

  if (result)
    RCLCPP_INFO(this->get_logger(), "Gripper at %.4f m", joint_value);
  else
    RCLCPP_ERROR(this->get_logger(), "Failed to set gripper to %.4f m", joint_value);
}


void ScenePickPlace::allowCollision(const std::string& object_id,
                                    const std::vector<std::string>& links,
                                    bool allowed)
{
  // Read the current full ACM (incl. SRDF self-collision allowances) so we can
  // add to it rather than overwrite it.
  auto client = this->create_client<moveit_msgs::srv::GetPlanningScene>("get_planning_scene");
  if (!client->wait_for_service(std::chrono::seconds(2)))
  {
    RCLCPP_ERROR(this->get_logger(), "get_planning_scene service unavailable; cannot allow collision");
    return;
  }

  auto request = std::make_shared<moveit_msgs::srv::GetPlanningScene::Request>();
  request->components.components = moveit_msgs::msg::PlanningSceneComponents::ALLOWED_COLLISION_MATRIX;
  auto future = client->async_send_request(request);
  if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
  {
    RCLCPP_ERROR(this->get_logger(), "Timed out reading ACM; cannot allow collision");
    return;
  }

  // Toggle the object against each link, then write the full matrix back as a diff.
  collision_detection::AllowedCollisionMatrix acm(future.get()->scene.allowed_collision_matrix);
  acm.setEntry(object_id, links, allowed);

  moveit_msgs::msg::PlanningScene scene;
  scene.is_diff = true;
  acm.getMessage(scene.allowed_collision_matrix);
  planning_scene_interface_.applyPlanningScene(scene);

  RCLCPP_INFO(this->get_logger(), "%s collision between '%s' and %zu link(s)/object(s)",
              allowed ? "Allowed" : "Disallowed", object_id.c_str(), links.size());
}

void ScenePickPlace::moveToPoseOmpl(const geometry_msgs::msg::Pose& target_pose)
{
  mg_arm_->setStartStateToCurrentState();
  mg_arm_->setPoseTarget(target_pose);
  mg_arm_->setPlanningTime(10.0);
  mg_arm_->allowReplanning(true);
  mg_arm_->setGoalTolerance(0.03);  // metres
  RCLCPP_INFO(this->get_logger(),
              "Moving to pose [%.3f, %.3f, %.3f] (OMPL)...",
              target_pose.position.x, target_pose.position.y, target_pose.position.z);
  
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool planned = static_cast<bool>(mg_arm_->plan(plan));

  if (!planned)
  {
    RCLCPP_ERROR(this->get_logger(), "OMPL planning failed; not executing");
    mg_arm_->clearPoseTargets();
    return;
  }

  RCLCPP_INFO(this->get_logger(), "OMPL plan found; executing...");
  bool result = static_cast<bool>(mg_arm_->execute(plan));
  mg_arm_->clearPoseTargets();

  if (result)
    RCLCPP_INFO(this->get_logger(), "Reached pose target");
  else
    RCLCPP_ERROR(this->get_logger(), "Failed to execute plan to pose target");
}

void ScenePickPlace::timerCallback()
{
  if (!mg_arm_)
  {
    RCLCPP_WARN(this->get_logger(), "MoveGroupInterface not initialized yet.");
    return;
  }

  // Only one step may run at a time; later ticks bail out while move() blocks.
  if (busy_.exchange(true))
    return;

  switch (step_)
  {
    case 0:
      moveTo("ready");
      ++step_;
      break;

    case 1:      
      setGripper("open");
      ++step_;
      break;

    case 2:
    {
      // Top-down approach: hover above the cylinder with the gripper pointing
      // straight down (-z), mirroring blind_pickplace step 1 (Pilz PTP).
      geometry_msgs::msg::Pose target;
      target.position.x = cylinder_pose_.position.x+0.005;  // follow the scene
      target.position.y = cylinder_pose_.position.y;
      target.position.z = 0.4;  // hover above the cylinder (top at ~0.21 m)

      // Rotate pi about x so the EE approach axis (+z) faces world -z; the
      // -pi/4 yaw cancels panda_hand's mount offset, leaving the fingers level.
      tf2::Quaternion q;
      q.setRPY(M_PI, 0.0, -M_PI / 4);
      target.orientation = tf2::toMsg(q);

      moveTo(target, "PTP");
      ++step_;
      break;
    }

    case 3:
    {
      // Descend straight down onto the cylinder (Pilz LIN).
      geometry_msgs::msg::Pose target = mg_arm_->getCurrentPose().pose;
      target.position.z -= 0.13;
      moveTo(target, "LIN");
      ++step_;
      break;
    }

    case 4:
    {
      // 1. Allow the gripper to intersect the cylinder for the grasp
      allowCollision("cylinder", { "panda_leftfinger", "panda_rightfinger", "panda_hand" });
      
      // 2. Allow the cylinder to intersect the table/container while resting on it
      allowCollision("cylinder", { "container_h20_base", "ground" });

      setGripper("close");

      // 3. Attach the object, explicitly ignoring the wrist links as well
      mg_arm_->attachObject("cylinder", mg_arm_->getEndEffectorLink(),
                      { "panda_leftfinger", "panda_rightfinger", "panda_hand"});

      ++step_;
      rclcpp::sleep_for(std::chrono::milliseconds(1000));
      break;
    }

    case 5:
    {
      // Small lift straight up so the cylinder clears the container before the
      // OMPL carry plans around it (Pilz LIN, Cartesian straight line).
      geometry_msgs::msg::Pose target = mg_arm_->getCurrentPose().pose;
      target.position.z += 0.1;
      moveTo(target, "LIN");
      ++step_;
      rclcpp::sleep_for(std::chrono::milliseconds(500));
      break;
    }

    case 6:
    {
      // Re-enable cylinder<->container collision so the OMPL carry plans a path
      // that avoids the container instead of cutting through it.
      allowCollision("cylinder", { "container_h20_base" }, false);
      
      // Carry the cylinder to the drop area (OMPL), keeping the gripper pointing
      // straight down (-z) at the goal.
      geometry_msgs::msg::Pose target;
      target.position.x = -0.3;
      target.position.y = -0.2;
      target.position.z = 0.4;

       // EE approach axis (+z) faces world -z
      tf2::Quaternion q;
      q.setRPY(M_PI, 0.0, -M_PI / 4); 
      target.orientation = tf2::toMsg(q);

      moveToPoseOmpl(target);
      ++step_;
      
      break;
    }

    case 7:
    {
      // Lower the cylinder straight down into the drop area (Pilz LIN).
      geometry_msgs::msg::Pose target = mg_arm_->getCurrentPose().pose;
      target.position.z -= 0.13;
      moveTo(target, "LIN");
      ++step_;
      break;
    }

    case 8:
    {
      // Settle, then open the gripper and release the cylinder into the scene.
      rclcpp::sleep_for(std::chrono::milliseconds(800));
      setGripper("open");
      mg_arm_->detachObject("cylinder");
      rclcpp::sleep_for(std::chrono::milliseconds(300));
      allowCollision("cylinder", { "panda_leftfinger", "panda_rightfinger", "panda_hand" }, false);
      ++step_;
      break;
    }

    case 9:
    {
      rclcpp::sleep_for(std::chrono::milliseconds(800));
      // Retreat to the ready pose.
      moveTo("ready");
      ++step_;
      break;
    }

    default:
      // Sequence complete; stop firing.
      timer_->cancel();
      break;
  }

  busy_.store(false);
}

// =================================== main ===================================

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  // Auto-declare parameters supplied as overrides by the launch file so that
  // MoveGroupInterface can read robot_description_semantic (SRDF) etc.
  rclcpp::NodeOptions options;
  options.automatically_declare_parameters_from_overrides(true);

  auto node = std::make_shared<ScenePickPlace>(options);
  node->initializeMoveGroup();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
