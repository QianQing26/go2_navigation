#!/usr/bin/env bash
# Terminal 3 - /scan -> /pose via BreezySLAM lidar odometry.
# Keep the robot STILL until the "origin fixed" line appears in this terminal.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "[pose] breezy_lidar_odom_node: yaw_offset=${LASER_YAW_OFFSET} signs=(x=${POSE_X_SIGN}, y=${POSE_Y_SIGN}, yaw=${POSE_YAW_SIGN})"
echo "[pose] keep the robot still until you see 'origin fixed'."
exec ros2 run seanav_perception breezy_lidar_odom_node --ros-args \
    -p input_topic:=/scan \
    -p output_topic:=/pose \
    -p map_size_pixels:=800 \
    -p map_size_meters:=20.0 \
    -p distance_no_detection_m:=4.0 \
    -p breezy_scan_size:=360 \
    -p scan_rate_hz:=10.0 \
    -p scan_qos_depth:=1 \
    -p min_valid_bins:="${POSE_MIN_VALID_BINS}" \
    -p skip_first_scans:=3 \
    -p origin_settle_scans:=30 \
    -p random_seed:=0 \
    -p sigma_xy_mm:="${POSE_SIGMA_XY_MM}" \
    -p sigma_theta_degrees:="${POSE_SIGMA_THETA_DEG}" \
    -p max_search_iter:="${POSE_MAX_SEARCH_ITER}" \
    -p debug_log_hz:="${POSE_DEBUG_LOG_HZ}" \
    -p laser_yaw_offset_rad:="${LASER_YAW_OFFSET}" \
    -p laser_x_in_base_m:=0.0 \
    -p laser_y_in_base_m:=0.0 \
    -p pose_x_sign:="${POSE_X_SIGN}" \
    -p pose_y_sign:="${POSE_Y_SIGN}" \
    -p yaw_sign:="${POSE_YAW_SIGN}" \
    -p use_motion_prior:="${POSE_USE_MOTION_PRIOR}" \
    -p motion_prior_alpha:="${POSE_MOTION_PRIOR_ALPHA}" \
    -p motion_prior_max_fwd_mm:="${POSE_MOTION_PRIOR_MAX_FWD_MM}" \
    -p motion_prior_max_dtheta_deg:="${POSE_MOTION_PRIOR_MAX_DTHETA_DEG}"
