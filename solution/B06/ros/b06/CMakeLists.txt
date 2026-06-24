cmake_minimum_required(VERSION 3.8)
project(b06)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

add_executable(blind_pick_place src/blind_pick_place.cpp)
ament_target_dependencies(blind_pick_place
  rclcpp
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  tf2
  tf2_geometry_msgs
)

install(TARGETS
  blind_pick_place
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  rviz2
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
