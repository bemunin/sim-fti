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
find_package(shape_msgs REQUIRED)
find_package(geometric_shapes REQUIRED)

add_executable(blind_pickplace src/blind_pickplace.cpp)
ament_target_dependencies(blind_pickplace
  rclcpp
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  tf2
  tf2_geometry_msgs
)

add_executable(scene_pickplace src/scene_pickplace.cpp)
ament_target_dependencies(scene_pickplace
  rclcpp
  geometry_msgs
  moveit_msgs
  moveit_ros_planning_interface
  tf2
  tf2_geometry_msgs
  shape_msgs
  geometric_shapes
)

install(TARGETS
  blind_pickplace
  scene_pickplace
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  rviz2
  config
  meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
