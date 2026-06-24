#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("blind_pick_place");

// Minimal MoveGroupInterface scaffold for a blind (no planning scene) pick-and-place
// routine on the Panda, planned with the Pilz industrial motion planner.
// The main loop lives in timerCallback(); motion logic is left for later.
class BlindPickPlace : public rclcpp::Node
{
public:
  explicit BlindPickPlace(const rclcpp::NodeOptions& options);

  // Must be called after construction (needs a valid shared_from_this()).
  void initializeMoveGroup();

private:
  bool waitForRobotState(double timeout_sec = 2.0);

  // Plan and move the arm to a named pose defined in the SRDF (e.g. "ready").
  void moveTo(const std::string& target);

  // Plan and move the arm so its end-effector reaches target_pose
  // (expressed in the planning frame). planner_id selects the Pilz command:
  // "PTP" (point-to-point) or "LIN" (Cartesian straight line).
  void moveTo(const geometry_msgs::msg::Pose& target_pose, const std::string& planner_id = "PTP");

  // Drive the gripper to a named SRDF state ("open" or "close").
  void setGripper(const std::string& state);

  // Drive the gripper to an explicit finger-joint value, in metres
  // (panda_finger_joint1, range ~0.0 closed .. 0.04 open).
  void setGripper(double joint_value);

  // Main loop. Program the blind pick-and-place sequence here.
  void timerCallback();

  rclcpp::CallbackGroup::SharedPtr timer_group_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> mg_arm_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> mg_gripper_;

  // Current step of the pick-and-place sequence (see timerCallback()).
  std::size_t step_{ 0 };
  // Guards against the reentrant timer starting a new step while move() blocks.
  std::atomic<bool> busy_{ false };
};

// ============================ Method definitions ============================

BlindPickPlace::BlindPickPlace(const rclcpp::NodeOptions& options)
  : Node("blind_pick_place", options)
{
  // Reentrant so the timer can run while MoveGroup callbacks are in flight.
  timer_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  timer_ = this->create_wall_timer(
      std::chrono::milliseconds(100),
      std::bind(&BlindPickPlace::timerCallback, this),
      timer_group_);

  RCLCPP_INFO(this->get_logger(), "BlindPickPlace created. Waiting for MoveGroup init...");
}

void BlindPickPlace::initializeMoveGroup()
{
  mg_arm_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "panda_arm");
  mg_gripper_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "hand");

  // --- Pilz planner setup ---
  // PlannerId: "PTP" / "LIN" / "CIRC"
  mg_arm_->setPlanningPipelineId("pilz_industrial_motion_planner");
  mg_arm_->setPlannerId("PTP");
  mg_arm_->setMaxVelocityScalingFactor(0.3);
  mg_arm_->setMaxAccelerationScalingFactor(0.3);

  // The gripper plans with OMPL: Pilz PTP/LIN expect a serial chain with a
  // configured tip/IK solver and abort on the single-DOF "hand" group.
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
}

bool BlindPickPlace::waitForRobotState(double timeout_sec)
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

void BlindPickPlace::moveTo(const std::string& target)
{
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

void BlindPickPlace::moveTo(const geometry_msgs::msg::Pose& target_pose, const std::string& planner_id)
{
  mg_arm_->setStartStateToCurrentState();
  mg_arm_->setPlannerId(planner_id);
  mg_arm_->setPoseTarget(target_pose);

  RCLCPP_INFO(this->get_logger(),
              "Moving to pose [%.3f, %.3f, %.3f] (%s)...",
              target_pose.position.x, target_pose.position.y, target_pose.position.z,
              planner_id.c_str());
  auto result = mg_arm_->move();

  if (result)
    RCLCPP_INFO(this->get_logger(), "Reached pose target");
  else
    RCLCPP_ERROR(this->get_logger(), "Failed to reach pose target");
}

void BlindPickPlace::setGripper(const std::string& state)
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

void BlindPickPlace::setGripper(double joint_value)
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

void BlindPickPlace::timerCallback()
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
      // Home the arm at the SRDF "ready" pose.
      moveTo("ready");
      ++step_;
      break;

    case 1:
    {
      // Move above the work area with the gripper pointing straight down (-z).
      geometry_msgs::msg::Pose target;
      target.position.x = 0.6;
      target.position.y = 0.0;
      target.position.z = 0.4;

      // Rotate pi about x so the EE approach axis (+z) faces world -z.
      tf2::Quaternion q;
      q.setRPY(M_PI, 0.0, -M_PI/4);
      target.orientation = tf2::toMsg(q);

      moveTo(target);
      ++step_;
      break;
    }

    case 2:
      // Open the gripper before descending onto the object.
      setGripper("open");
      ++step_;
      break;

    case 3:
    {
      // Descend straight down 0.2 m (Cartesian LIN) to the grasp height.
      geometry_msgs::msg::Pose target = mg_arm_->getCurrentPose().pose;
      target.position.z -= 0.12;
      moveTo(target, "LIN");
      ++step_;
      break;
    }

    case 4:
      // Close the gripper to grasp.
      setGripper("close");
      ++step_;
      break;

    case 5:
    {
      // Lift straight up to z = 0.6 (Cartesian LIN), keeping the grasp.
      geometry_msgs::msg::Pose target = mg_arm_->getCurrentPose().pose;
      target.position.z = 0.6;
      moveTo(target, "LIN");
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

  auto node = std::make_shared<BlindPickPlace>(options);
  node->initializeMoveGroup();

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
