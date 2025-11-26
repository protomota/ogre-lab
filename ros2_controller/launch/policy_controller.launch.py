"""Launch file for Ogre Policy Controller node."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('ogre_policy_controller')

    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(pkg_dir, 'models', 'policy.onnx'),
        description='Path to trained policy model'
    )

    model_type_arg = DeclareLaunchArgument(
        'model_type',
        default_value='onnx',
        description='Model type: onnx or jit'
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=os.path.join(pkg_dir, 'config', 'policy_controller_params.yaml'),
        description='Path to parameters file'
    )

    # Policy controller node
    policy_controller_node = Node(
        package='ogre_policy_controller',
        executable='policy_controller',
        name='ogre_policy_controller',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'model_path': LaunchConfiguration('model_path'),
                'model_type': LaunchConfiguration('model_type'),
            }
        ],
        remappings=[
            # Remap topics if needed
            # ('/cmd_vel', '/nav2_cmd_vel'),
        ]
    )

    return LaunchDescription([
        model_path_arg,
        model_type_arg,
        params_file_arg,
        policy_controller_node,
    ])
