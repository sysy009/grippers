cmake_minimum_required(VERSION 3.8)
project(grippers_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MissionState.msg"
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/MissionSpec.msg"
  "srv/SetGripper.srv"
  "srv/GetLoad.srv"
  "srv/GetArmState.srv"
  "srv/MonitorClearance.srv"
  "srv/ConfirmGrasp.srv"
  "srv/ObserveTarget.srv"
  "srv/OffsetBaseYaw.srv"
  "srv/Parse.srv"
  "srv/ConfirmPhrase.srv"
  "srv/GetPolicyState.srv"
  "action/ReorientArm.action"
  "action/MoveToCartesian.action"
  "action/MoveToFloorPose.action"
  "action/ExecuteJointChunk.action"
  "action/RunVlaGrasp.action"
  DEPENDENCIES geometry_msgs action_msgs
)

find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
