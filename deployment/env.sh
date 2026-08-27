#!/usr/bin/env bash
# SEA-Nav real-robot environment. `source` this file in every terminal:
#
#     source env.sh
#
# Machine-level settings - edit the variables below directly if your robot
# differs (all runtime parameters live in src/seanav_deploy/seanav_deploy/config.py).
CONDA_ENV=sea
ROS_SETUP=/opt/ros/foxy/setup.bash
CYCLONEDDS_WS=~/cyclonedds_ws
DOMAIN_ID=1
RMW=rmw_cyclonedds_cpp

_DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# 1) conda
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV}" 2>/dev/null \
        || echo "[env] WARN: could not 'conda activate ${CONDA_ENV}'."
else
    echo "[env] WARN: conda not on PATH; assuming the intended Python is already active."
fi

# 2) ROS2 + CycloneDDS. ROS setup files reference unbound variables, so relax
# nounset before sourcing them.
set +u
unset ROS_DISTRO ROS_VERSION 2>/dev/null || true
if [ -f "${ROS_SETUP}" ]; then
    source "${ROS_SETUP}"
else
    echo "[env] WARN: ROS setup '${ROS_SETUP}' not found; edit ROS_SETUP in env.sh."
fi
CYCLONEDDS_WS="${CYCLONEDDS_WS/#\~/${HOME}}"
if [ -f "${CYCLONEDDS_WS}/install/setup.bash" ]; then
    source "${CYCLONEDDS_WS}/install/setup.bash"
fi
export RMW_IMPLEMENTATION="${RMW}"
export ROS_DOMAIN_ID="${DOMAIN_ID}"

# 3) this workspace's overlay (sllidar_ros2 + seanav_deploy)
if [ -f "${_DEPLOY_DIR}/install/setup.bash" ]; then
    source "${_DEPLOY_DIR}/install/setup.bash"
else
    echo "[env] NOTE: ${_DEPLOY_DIR}/install not found - run 'bash build.sh' first."
fi

echo "[env] conda=${CONDA_DEFAULT_ENV:-?}  ROS_DISTRO=${ROS_DISTRO:-?}  RMW=${RMW_IMPLEMENTATION}  ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
