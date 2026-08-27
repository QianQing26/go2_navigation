"""Bring up the whole perception chain in one command:

    ros2 launch seanav_deploy perception.launch.py

    sllidar_node (RPLIDAR A2M12) -> /scan
    scan_to_rays                 -> /rays  (41 rays, base_link)
    lidar_odom  (BreezySLAM)     -> /pose  (world-frame x, y, theta)

All parameters come from ``seanav_deploy/config.py``. The sllidar driver is an
external package, so its parameters are injected here; the seanav_deploy nodes
read the config module themselves.
"""

from launch import LaunchDescription
from launch_ros.actions import Node

from seanav_deploy.config import LidarCfg


def generate_launch_description():
    lidar = Node(
        package="sllidar_ros2",
        executable="sllidar_node",
        name="sllidar_node",
        output="screen",
        parameters=[
            {
                "channel_type": "serial",
                "serial_port": LidarCfg.port,
                "serial_baudrate": LidarCfg.baudrate,
                "frame_id": LidarCfg.frame_id,
                "inverted": False,
                "angle_compensate": True,
                "scan_mode": LidarCfg.scan_mode,
            }
        ],
    )

    scan_to_rays = Node(
        package="seanav_deploy",
        executable="scan_to_rays",
        name="scan_to_rays",
        output="screen",
    )

    lidar_odom = Node(
        package="seanav_deploy",
        executable="lidar_odom",
        name="lidar_odom",
        output="screen",
    )

    return LaunchDescription([lidar, scan_to_rays, lidar_odom])
