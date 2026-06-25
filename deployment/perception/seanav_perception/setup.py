from setuptools import find_packages, setup

package_name = "seanav_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="liuhy25",
    maintainer_email="liuhy25@example.com",
    description="SEA-Nav real-robot perception publishers for quad_deploy",
    license="MIT",
    entry_points={
        "console_scripts": [
            "scan_to_rays_node = seanav_perception.scan_to_rays_node:main",
            "breezy_lidar_odom_node = seanav_perception.breezy_lidar_odom_node:main",
            "rays_monitor_node = seanav_perception.rays_monitor_node:main",
        ],
    },
)
