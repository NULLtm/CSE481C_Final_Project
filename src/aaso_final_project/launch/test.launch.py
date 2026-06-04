from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Helper to find package directories
    stretch_core_dir = get_package_share_directory('stretch_core')
    aaso_project_dir = get_package_share_directory('aaso_final_project')

    return LaunchDescription([
        # 1. ros2 launch stretch_core d405_basic.launch.py
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(stretch_core_dir, 'launch', 'd405_basic.launch.py')
            )
        ),

        # 2. ros2 launch stretch_core stretch_driver.launch.py
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(stretch_core_dir, 'launch', 'stretch_driver.launch.py')
            )
        ),

        # 3. ros2 launch aaso_final_project aruco_test.launch.py
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(aaso_project_dir, 'launch', 'aruco_test.launch.py')
            )
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(aaso_project_dir, 'launch', 'align_test.launch.py')
            )
        )
    ])