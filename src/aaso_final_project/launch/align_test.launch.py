import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    detect_aruco_markers = Node(
        package='aaso_final_project',
        executable='align_test',
        output='screen',
        )

    return LaunchDescription([
        detect_aruco_markers,
        ])