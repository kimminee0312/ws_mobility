from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    parking_pre_node = Node(
        package='corn',                   
        executable='parking_pre_node',     
        name='parking_pre_node',
        output='screen',
        parameters=[{
            'slot_width_nom': 2.4,
            'slot_length_nom': 5.0,
            'line_width': 0.05
        }]
    )

    rect_fitter_node = Node(
        package='corn',                   
        executable='rect_fitter_node',     
        name='rect_fitter_node',
        output='screen',
        parameters=[{
            'extension_len': 1.0,
            'extension_len_ext5': 12.0,
            'max_edge_len_strict': 5.0,
            'cluster_radius': 0.5,
            'min_cluster_size': 1
        }]
    )

    return LaunchDescription([
        parking_pre_node,
        rect_fitter_node
    ])
