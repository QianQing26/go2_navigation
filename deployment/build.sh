#!/usr/bin/env bash
# Build the deployment workspace (run once, and again after pulling new code):
#
#     bash build.sh
#
#   1. fetch the RPLIDAR driver source into src/ (first run only)
#   2. colcon build with --symlink-install, so later edits to python sources
#      and config.py take effect on restart without rebuilding
#
# No `set -u`: sourcing ROS2 setup.bash trips nounset.
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d src/sllidar_ros2 ]; then
    echo "[build] cloning sllidar_ros2 ..."
    git clone https://github.com/Slamtec/sllidar_ros2.git src/sllidar_ros2
fi

source env.sh
colcon build --symlink-install

echo
echo "[build] done. Verify with:"
echo "    source env.sh && ros2 pkg executables seanav_deploy"
