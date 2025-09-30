from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='velodyne_segmentation',
            executable='lidar_segmentation_node',
            name='lidar_segmentation',
            parameters=[{
                'input_topic': '/velodyne_points',
                'voxel_leaf': 0.1,
                'z_min': -2.0, 'z_max': 2.0,
                'ground_dist_thresh': 0.15,
                'ground_eps_angle_deg': 10.0,
                'cluster_tolerance': 0.35,
                'cluster_min_size': 20,
                'cluster_max_size': 25000,
                # cone rules
                'cone_min_height': 0.25,
                'cone_max_height': 1.1,
                'cone_max_xy_diameter': 0.5,
                'cone_height_over_xy_min_ratio': 1.0,
                'cone_min_points': 30,
                'cone_max_points': 3000,
                'cone_use_intensity': False
            }]
        )
    ])
