cmake_minimum_required(VERSION 3.8)
project(corn)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)


# PCL 1.13 강제로 /usr/local에서 찾기
find_package(PCL 1.13 REQUIRED
  PATHS /usr/local/share/pcl-1.13
  NO_DEFAULT_PATH
)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(/usr/local/lib ${PCL_LIBRARY_DIRS})

add_executable(parking_node src/parking_node.cpp)
ament_target_dependencies(parking_node
  rclcpp sensor_msgs visualization_msgs pcl_conversions pcl_msgs geometry_msgs
)
target_link_libraries(parking_node ${PCL_LIBRARIES})

install(TARGETS
  parking_node
  DESTINATION lib/${PROJECT_NAME})
ament_package()
