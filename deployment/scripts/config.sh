#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# SEA-Nav real-robot deployment - central configuration.
#
# `source`d by env.sh (and therefore by every terminal script). Every value can
# be overridden from the outer shell, e.g.:
#
#     LIDAR_PORT=/dev/ttyUSB1 GOAL_X=2.0 bash 5_controller.sh
#
# so you normally never edit the launch scripts themselves - only this file or
# the environment.
# ---------------------------------------------------------------------------

# ---- Conda / ROS environment --------------------------------------------
# Conda env that has rclpy + onnxruntime + numpy + quad_deploy (+ ros_base,
# unitree_sdk2py) + BreezySLAM installed. On the validated robot this is `sea`.
: "${SEANAV_CONDA_ENV:=sea}"
# ROS2 distro setup. SEA-Nav real-robot deployment is validated on ROS2 Foxy.
: "${SEANAV_ROS_SETUP:=/opt/ros/foxy/setup.bash}"
# CycloneDDS overlay workspace (the RMW the whole stack uses). Leave empty if
# CycloneDDS is already on the system / provided by the ROS distro.
: "${SEANAV_CYCLONEDDS_WS:=${HOME}/cyclonedds_ws}"
# ROS2 topics (/scan /rays /pose) live on this domain + RMW. The Unitree SDK
# uses a *separate* DDS channel (domain 0 on eth0), configured inside
# sea_nav_run_sdk.py - do not confuse the two.
: "${SEANAV_ROS_DOMAIN_ID:=1}"
: "${SEANAV_RMW:=rmw_cyclonedds_cpp}"

# ---- Workspaces / repos --------------------------------------------------
# Independent overlay workspace for our packages (sllidar_ros2 +
# seanav_perception). Keeps the shared ~/ros2_ws untouched.
: "${SEANAV_WS:=${HOME}/seanav_ws}"
: "${QUAD_DEPLOY_DIR:=${HOME}/Projects/quad_deploy}"
# ONNX models: must contain nav_model/model.onnx AND loco_model/model.onnx.
: "${SEANAV_DATA:=${HOME}/Data/onboard_data/onnx_models/sea_nav}"

# ---- RPLIDAR A2M12 (Terminal 1: /scan) -----------------------------------
: "${LIDAR_PORT:=/dev/ttyUSB0}"
: "${LIDAR_BAUD:=256000}"
: "${LIDAR_FRAME:=laser}"
: "${LIDAR_SCAN_MODE:=Sensitivity}"

# ---- /scan -> /rays (Terminal 2) -----------------------------------------
# Mounting correction: ranges[20] must point straight ahead. On the validated
# robot the lidar is mounted backwards, hence the 180-degree (pi) yaw offset.
: "${LASER_YAW_OFFSET:=3.1415926}"
: "${RAYS_INVERT_ANGLE:=false}"
: "${RAYS_RANGE_MIN:=0.1}"
: "${RAYS_RANGE_MAX:=3.0}"

# ---- /scan -> /pose, BreezySLAM (Terminal 3) -----------------------------
# Axis signs are calibrated against the physical robot (see README section 6):
#   forward +1 m  -> pose.x increases  (validated value: -1.0)
#   strafe left   -> pose.y increases  (validated value: -1.0)
#   turn CCW       -> pose.theta +     (validated value:  1.0)
: "${POSE_X_SIGN:=-1.0}"
: "${POSE_Y_SIGN:=-1.0}"
: "${POSE_YAW_SIGN:=1.0}"
: "${POSE_MIN_VALID_BINS:=90}"
: "${POSE_SIGMA_XY_MM:=20}"
: "${POSE_SIGMA_THETA_DEG:=3}"
: "${POSE_MAX_SEARCH_ITER:=500}"
: "${POSE_DEBUG_LOG_HZ:=2.0}"

# Constant-velocity motion prior for BreezySLAM. Without wheel odometry the
# RMHC scan matcher searches only around the previous pose, so it lags fast
# forward (+X) motion. The prior predicts this frame's pose_change from the
# last estimated velocity so a small sigma stays stable AND tracks motion.
# Set POSE_USE_MOTION_PRIOR=false to fall back to the old (0,0,0) behavior.
: "${POSE_USE_MOTION_PRIOR:=true}"
: "${POSE_MOTION_PRIOR_ALPHA:=0.5}"
: "${POSE_MOTION_PRIOR_MAX_FWD_MM:=80.0}"
: "${POSE_MOTION_PRIOR_MAX_DTHETA_DEG:=15.0}"

# ---- Controller (Terminal 5) ---------------------------------------------
# Goal in the same world frame as /pose (origin = where the robot stood when
# BreezySLAM fixed its origin). 1 m straight ahead is a safe first target.
: "${GOAL_X:=1.0}"
: "${GOAL_Y:=0.0}"

# ---- tmux ----------------------------------------------------------------
: "${SEANAV_TMUX_SESSION:=seanav_perception}"
