#!/usr/bin/env bash
# Terminal 2 - /scan -> /rays. Down-samples to the 41-ray base_link scan the
# SEA-Nav navigation policy consumes (ranges[20] = straight ahead).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "[rays] scan_to_rays_node: yaw_offset=${LASER_YAW_OFFSET} invert=${RAYS_INVERT_ANGLE} range=[${RAYS_RANGE_MIN}, ${RAYS_RANGE_MAX}]"
exec ros2 run seanav_perception scan_to_rays_node --ros-args \
    -p input_topic:=/scan \
    -p output_topic:=/rays \
    -p output_frame:=base_link \
    -p laser_yaw_offset_rad:="${LASER_YAW_OFFSET}" \
    -p invert_angle:="${RAYS_INVERT_ANGLE}" \
    -p range_min:="${RAYS_RANGE_MIN}" \
    -p range_max:="${RAYS_RANGE_MAX}"
