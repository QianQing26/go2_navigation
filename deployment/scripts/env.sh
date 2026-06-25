#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Common environment for every SEA-Nav real-robot terminal.
#
#   conda env (sea)  ->  ROS2 Foxy + CycloneDDS (ROS_DOMAIN_ID=1)  ->  seanav_ws
#
# This file is meant to be `source`d (the per-terminal scripts do it for you):
#
#     source env.sh
#
# It is intentionally tolerant: a missing optional piece prints a hint instead
# of aborting, so a half-configured machine still gets you a usable shell.
# ---------------------------------------------------------------------------

_SEANAV_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# shellcheck source=config.sh
source "${_SEANAV_SCRIPTS_DIR}/config.sh"

# 1) Conda -------------------------------------------------------------------
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${SEANAV_CONDA_ENV}" 2>/dev/null \
        || echo "[env] WARN: could not 'conda activate ${SEANAV_CONDA_ENV}'."
else
    echo "[env] WARN: conda not on PATH; assuming the intended Python is already active."
fi

# 2) ROS2 (Foxy) + CycloneDDS ------------------------------------------------
# ROS2 setup.bash files reference unbound vars (e.g. AMENT_TRACE_SETUP_FILES);
# disable nounset so sourcing them never aborts a caller that ran `set -u`.
set +u
unset ROS_DISTRO ROS_VERSION 2>/dev/null || true
if [ -f "${SEANAV_ROS_SETUP}" ]; then
    source "${SEANAV_ROS_SETUP}"
else
    echo "[env] WARN: ROS setup '${SEANAV_ROS_SETUP}' not found; set SEANAV_ROS_SETUP in config.sh."
fi
# CycloneDDS overlay (skipped if the workspace is absent / already on system).
if [ -f "${SEANAV_CYCLONEDDS_WS}/install/setup.bash" ]; then
    source "${SEANAV_CYCLONEDDS_WS}/install/setup.bash"
fi
# Lock the DDS / domain the whole stack agrees on for ROS2 topics.
export RMW_IMPLEMENTATION="${SEANAV_RMW}"
export ROS_DOMAIN_ID="${SEANAV_ROS_DOMAIN_ID}"

# 3) seanav_ws overlay (sllidar_ros2 + seanav_perception) --------------------
if [ -f "${SEANAV_WS}/install/setup.bash" ]; then
    source "${SEANAV_WS}/install/setup.bash"
else
    echo "[env] NOTE: ${SEANAV_WS}/install not found - run build_perception.sh first."
fi

echo "[env] conda=${CONDA_DEFAULT_ENV:-?}  ROS_DISTRO=${ROS_DISTRO:-?}  RMW=${RMW_IMPLEMENTATION:-?}  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-?}"
