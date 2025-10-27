from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    parking_final_node = Node(
        package='corn',                   
        executable='parking_final_node',     
        name='parking_final_node',
        output='screen',
        # parameters=[{
        #     'slot_width_nom': 2.4,
        #     'slot_length_nom': 5.0,
        #     'line_width': 0.05
        # }]
    )

    cone_find_node = Node(
        package='corn',                   
        executable='cone_find_node',     
        name='cone_find_node',
        output='screen',
        # parameters=[{
        #     'extension_len': 1.0,
        #     'extension_len_ext5': 12.0,
        #     'max_edge_len_strict': 5.0,
        #     'cluster_radius': 0.5,
        #     'min_cluster_size': 1
        # }]
    )

    return LaunchDescription([
        parking_final_node,
        cone_find_node
    ])
