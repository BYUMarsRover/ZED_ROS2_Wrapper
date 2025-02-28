from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

def generate_launch_description():
    # Get the path to the params folder in your package
    object_detection_dir = get_package_share_directory('object_detection')
    params_file = os.path.join(object_detection_dir, 'params', 'config.yaml')

    return LaunchDescription([
        # Launch the object_detection_node with parameters from the YAML file
        Node(
            package='object_detection',
            executable='object_detection',
            name='object_detection',
            namespace='zed',
            # output='screen',
            parameters=[params_file],
        ),

        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(os.path.join( 
        #         get_package_share_directory('gpsd_client'), 'launch', 'gpsd_client-launch.py'))
        # ),
        # Node(
        #      package='tf2_ros',
        #      executable='static_transform_publisher',
        #      arguments = ['--x', '0', '--y', '0', '--z', '0', '--yaw', '0', '--pitch', '0', '--roll', '0', '--frame-id', 'base_link', '--child-frame-id', 'zed_imu_link']
        # ),
    ])
