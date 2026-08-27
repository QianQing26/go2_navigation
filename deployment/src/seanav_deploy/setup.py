import os
from glob import glob

from setuptools import find_packages, setup

package_name = "seanav_deploy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    description="SEA-Nav real-robot deployment: /rays and /pose publishers plus the controller entry.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "scan_to_rays = seanav_deploy.scan_to_rays_node:main",
            "lidar_odom = seanav_deploy.lidar_odom_node:main",
            "rays_monitor = seanav_deploy.rays_monitor_node:main",
            "controller = seanav_deploy.controller:main",
        ],
    },
)
