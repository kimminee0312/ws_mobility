cmake_minimum_required(VERSION 3.8)
project(velodyne_segmentation)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters segmentation search)
find_package(pcl_conversions REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ================================= node1 ===================================
add_executable(lidar_segmentation_node src/lidar_segmentation_node.cpp)
ament_target_dependencies(lidar_segmentation_node
  rclcpp sensor_msgs visualization_msgs pcl_conversions)
target_link_libraries(lidar_segmentation_node ${PCL_LIBRARIES})

#================================= node2 ===================================
#add_executable(lidar_segmentation_node2 src/lidar_segmentation_node2.cpp)
#ament_target_dependencies(lidar_segmentation_node2
#  rclcpp sensor_msgs visualization_msgs pcl_conversions)
#target_link_libraries(lidar_segmentation_node2 ${PCL_LIBRARIES})

install(TARGETS
  lidar_segmentation_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
