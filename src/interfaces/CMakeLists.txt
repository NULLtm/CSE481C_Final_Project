cmake_minimum_required(VERSION 3.8)
project(interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Add this line to find the dependency
find_package(builtin_interfaces REQUIRED)

set(srv_files
  "srv/AlgNotation.srv"
  "srv/Align.srv"
  "srv/Move.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${srv_files}
  DEPENDENCIES builtin_interfaces
)

ament_package()