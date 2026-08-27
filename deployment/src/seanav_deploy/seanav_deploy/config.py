"""Central configuration for the SEA-Nav real-robot deployment.

Edit the values below directly and restart the affected node (the workspace is
built with ``--symlink-install``, so no rebuild is needed). Machine-level
settings (conda env, ROS setup paths, DDS domain) live in ``deployment/env.sh``.

The ``/rays`` and ``/pose`` specs must stay consistent with
``quad_deploy.config.sea.sea_nav_agent_cfg.SEANavAgentCfg`` (41 rays,
[0.1, 3.0] m, base_link / world frames) - the control stack is not modified
for real-robot deployment.
"""


class LidarCfg:
    """RPLIDAR A2M12 driver (sllidar_ros2) and its mounting extrinsics."""

    port = "/dev/ttyUSB0"
    baudrate = 256000
    frame_id = "laser"
    scan_mode = "Sensitivity"

    # Mounting correction, shared by the /rays and /pose nodes. On the
    # validated robot the lidar is mounted backwards -> 180 deg yaw offset.
    yaw_offset_rad = 3.1415926
    # Lidar position in the base frame (meters). Zero when centered.
    x_in_base_m = 0.0
    y_in_base_m = 0.0


class RaysCfg:
    """/scan -> /rays down-sampling (scan_to_rays node)."""

    input_topic = "/scan"
    output_topic = "/rays"
    output_frame = "base_link"
    # Set True only if left/right are still mirrored after yaw_offset_rad is
    # calibrated (see the verification table in the README).
    invert_angle = False
    # Range clip. Must match SEANavAgentCfg.rays_clip_min / rays_clip_max.
    range_min = 0.1
    range_max = 3.0
    # Angular window (deg) used to pool raw returns into each of the 41 rays.
    bin_width_deg = 6.0


class PoseCfg:
    """/scan -> /pose BreezySLAM lidar odometry (lidar_odom node)."""

    input_topic = "/scan"
    output_topic = "/pose"

    # BreezySLAM map / scan model.
    map_size_pixels = 800
    map_size_meters = 20.0
    distance_no_detection_m = 4.0
    breezy_scan_size = 360
    scan_rate_hz = 10.0
    scan_qos_depth = 1
    map_quality = 50
    hole_width_mm = 600
    random_seed = 0

    # RMHC search. Small sigmas keep the standstill pose stable; the motion
    # prior below compensates for fast forward motion.
    sigma_xy_mm = 20
    sigma_theta_degrees = 3
    max_search_iter = 500

    # Scan gating / origin settling.
    min_valid_bins = 90
    skip_first_scans = 3
    origin_settle_scans = 30
    use_previous_on_bad_scan = True

    # Axis signs, calibrated on the physical robot (see the verification
    # table in the README):
    #   push forward 1 m -> pose.x ~ +1.0
    #   push left        -> pose.y increases
    #   turn CCW 90 deg  -> pose.theta ~ +1.57
    x_sign = -1.0
    y_sign = -1.0
    yaw_sign = 1.0

    # Constant-velocity motion prior. Without wheel odometry RMHC_SLAM searches
    # around the previous pose only and systematically lags forward (+X)
    # motion; the prior seeds the search with the predicted displacement.
    # Set use_motion_prior=False to fall back to the default (0, 0, 0) prior.
    use_motion_prior = True
    motion_prior_alpha = 0.5  # velocity EMA: larger = more responsive
    motion_prior_max_fwd_mm = 80.0  # per-frame forward prediction clamp
    motion_prior_max_dtheta_deg = 15.0  # per-frame turn prediction clamp
    motion_prior_max_dt_s = 0.5

    debug_log_hz = 2.0


class MonitorCfg:
    """/rays terminal monitor (rays_monitor node)."""

    topic = RaysCfg.output_topic
    print_hz = 5.0
    warn_distance = 0.6


class ControllerCfg:
    """SEA-Nav controller entry (controller node).

    The entry performs a perception handshake (waits until /rays and /pose are
    both streaming), then exec's the quad_deploy control stack.
    """

    run_module = "quad_deploy.scripts.sea.sea_run_sdk"
    # Directory that contains nav_model/model.onnx and loco_model/model.onnx.
    data_dir = "~/Data/onboard_data/onnx_models/sea_nav"

    # Navigation goal in the /pose world frame (origin = robot pose when the
    # lidar odometry logged "origin fixed"). 1 m straight ahead by default.
    goal_x = 1.0
    goal_y = 0.0

    # Perception handshake: each topic must have delivered at least
    # handshake_min_msgs messages and the newest one must be fresher than
    # handshake_fresh_s before the control stack is started.
    handshake_min_msgs = 10
    handshake_fresh_s = 1.0
    handshake_report_period_s = 2.0
