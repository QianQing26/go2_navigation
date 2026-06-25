#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-time (or after-sync) build of the perception workspace.
#
#   1. clone sllidar_ros2 into ${SEANAV_WS}/src   (RPLIDAR A2M12 driver)
#   2. symlink this repo's deployment/perception/seanav_perception into it
#      (the repo stays the single source of truth - edits apply on rebuild)
#   3. colcon build both packages
#
# Re-run it after you `git pull` / `rsync` new perception code.
# ---------------------------------------------------------------------------
# NOTE: no `set -u` here on purpose - sourcing ROS2 setup.bash trips nounset.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/config.sh"

PKG_SRC="$(cd "${SCRIPT_DIR}/../perception/seanav_perception" >/dev/null 2>&1 && pwd)"
if [ -z "${PKG_SRC}" ] || [ ! -f "${PKG_SRC}/package.xml" ]; then
    echo "[build] ERROR: cannot find seanav_perception package next to this script." >&2
    exit 1
fi

mkdir -p "${SEANAV_WS}/src"

# 1) RPLIDAR driver -----------------------------------------------------------
if [ ! -d "${SEANAV_WS}/src/sllidar_ros2" ]; then
    echo "[build] cloning sllidar_ros2 ..."
    git clone https://github.com/Slamtec/sllidar_ros2.git "${SEANAV_WS}/src/sllidar_ros2"
else
    echo "[build] sllidar_ros2 already present, skipping clone."
fi

# 2) link our package ---------------------------------------------------------
LINK="${SEANAV_WS}/src/seanav_perception"
if [ -L "${LINK}" ] || [ ! -e "${LINK}" ]; then
    ln -sfn "${PKG_SRC}" "${LINK}"
    echo "[build] linked ${LINK} -> ${PKG_SRC}"
else
    echo "[build] WARN: ${LINK} exists and is not a symlink; leaving it untouched."
fi

# 3) source env + colcon build ------------------------------------------------
source "${SCRIPT_DIR}/env.sh"
cd "${SEANAV_WS}"
echo "[build] colcon build (sllidar_ros2 + seanav_perception) ..."
colcon build --symlink-install --packages-select sllidar_ros2 seanav_perception

echo
echo "[build] done. Verify with:"
echo "    ros2 pkg executables seanav_perception"
echo "New terminals pick this up automatically (env.sh sources ${SEANAV_WS}/install/setup.bash)."
