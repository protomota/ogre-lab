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
    use_policy_arg = DeclareLaunchArgument(
        'use_policy',
        default_value='true',
        description='Use learned policy (true) or pass-through mode (false)'
    )

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

    output_mode_arg = DeclareLaunchArgument(
        'output_mode',
        default_value='twist',
        description='Output mode: twist (for Nav2) or wheel_velocities (for direct motor control)'
    )

    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/policy_cmd_vel_in',
        description='Input velocity command topic'
    )

    output_topic_arg = DeclareLaunchArgument(
        'output_topic',
        default_value='/cmd_vel',
        description='Output topic (Twist or Float32MultiArray depending on mode)'
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
                'use_policy': LaunchConfiguration('use_policy'),
                'model_path': LaunchConfiguration('model_path'),
                'model_type': LaunchConfiguration('model_type'),
                'output_mode': LaunchConfiguration('output_mode'),
                'input_topic': LaunchConfiguration('input_topic'),
                'output_topic': LaunchConfiguration('output_topic'),
            }
        ],
        remappings=[
            # Remap topics if needed
            # ('/cmd_vel', '/nav2_cmd_vel'),
        ]
    )

    return LaunchDescription([
        use_policy_arg,
        model_path_arg,
        model_type_arg,
        output_mode_arg,
        input_topic_arg,
        output_topic_arg,
        params_file_arg,
        policy_controller_node,
    ])
