"""Lidar-only odometry for ``/pose`` using BreezySLAM.

Subscribes to the raw ``/scan``, runs ``RMHC_SLAM`` to estimate the robot's 2D
pose, and republishes it as ``geometry_msgs/Pose2D`` on ``/pose`` in a local
world frame whose origin/heading are fixed once at startup (so the SEA-Nav goal
can be expressed relative to where the robot started).

The robot must stay still until the ``origin fixed`` log line appears; before
that BreezySLAM is still converging. Mounting orientation and axis signs are
corrected with ``LidarCfg.yaw_offset_rad`` and ``PoseCfg.x_sign / y_sign /
yaw_sign`` so that the published pose matches the ``/rays`` frame.

Defaults come from :mod:`seanav_deploy.config`; every value can still be
overridden per-run with ``--ros-args -p name:=value`` for debugging.
"""

import math
import time

import numpy as np
import rclpy
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from seanav_deploy.config import LidarCfg, PoseCfg


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class LidarOdomNode(Node):
    def __init__(self):
        super().__init__("lidar_odom")

        self.declare_parameter("input_topic", PoseCfg.input_topic)
        self.declare_parameter("output_topic", PoseCfg.output_topic)
        self.declare_parameter("map_size_pixels", int(PoseCfg.map_size_pixels))
        self.declare_parameter("map_size_meters", float(PoseCfg.map_size_meters))
        self.declare_parameter("distance_no_detection_m", float(PoseCfg.distance_no_detection_m))
        self.declare_parameter("breezy_scan_size", int(PoseCfg.breezy_scan_size))
        self.declare_parameter("scan_rate_hz", float(PoseCfg.scan_rate_hz))
        self.declare_parameter("scan_qos_depth", int(PoseCfg.scan_qos_depth))
        self.declare_parameter("min_valid_bins", int(PoseCfg.min_valid_bins))
        self.declare_parameter("skip_first_scans", int(PoseCfg.skip_first_scans))
        self.declare_parameter("origin_settle_scans", int(PoseCfg.origin_settle_scans))
        self.declare_parameter("use_previous_on_bad_scan", bool(PoseCfg.use_previous_on_bad_scan))
        self.declare_parameter("map_quality", int(PoseCfg.map_quality))
        self.declare_parameter("hole_width_mm", int(PoseCfg.hole_width_mm))
        self.declare_parameter("random_seed", int(PoseCfg.random_seed))
        self.declare_parameter("sigma_xy_mm", int(PoseCfg.sigma_xy_mm))
        self.declare_parameter("sigma_theta_degrees", int(PoseCfg.sigma_theta_degrees))
        self.declare_parameter("max_search_iter", int(PoseCfg.max_search_iter))
        self.declare_parameter("debug_log_hz", float(PoseCfg.debug_log_hz))
        self.declare_parameter("laser_yaw_offset_rad", float(LidarCfg.yaw_offset_rad))
        self.declare_parameter("laser_x_in_base_m", float(LidarCfg.x_in_base_m))
        self.declare_parameter("laser_y_in_base_m", float(LidarCfg.y_in_base_m))
        self.declare_parameter("pose_x_sign", float(PoseCfg.x_sign))
        self.declare_parameter("pose_y_sign", float(PoseCfg.y_sign))
        self.declare_parameter("yaw_sign", float(PoseCfg.yaw_sign))
        # Constant-velocity motion prior: without odometry, RMHC_SLAM searches
        # around the previous pose only, so it systematically lags fast forward
        # motion (the +X / heading direction). Feeding a predicted pose_change
        # moves the search start to where the robot is expected to be, which lets
        # a small sigma stay stable while still tracking motion.
        self.declare_parameter("use_motion_prior", bool(PoseCfg.use_motion_prior))
        self.declare_parameter("motion_prior_alpha", float(PoseCfg.motion_prior_alpha))
        self.declare_parameter("motion_prior_max_fwd_mm", float(PoseCfg.motion_prior_max_fwd_mm))
        self.declare_parameter("motion_prior_max_dtheta_deg", float(PoseCfg.motion_prior_max_dtheta_deg))
        self.declare_parameter("motion_prior_max_dt_s", float(PoseCfg.motion_prior_max_dt_s))

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.map_size_pixels = int(self.get_parameter("map_size_pixels").value)
        self.map_size_meters = float(self.get_parameter("map_size_meters").value)
        self.no_detection_mm = int(1000.0 * float(self.get_parameter("distance_no_detection_m").value))
        self.no_detection_m = self.no_detection_mm / 1000.0
        self.breezy_scan_size = int(self.get_parameter("breezy_scan_size").value)
        self.scan_rate_hz = float(self.get_parameter("scan_rate_hz").value)
        self.scan_qos_depth = int(self.get_parameter("scan_qos_depth").value)
        self.min_valid_bins = int(self.get_parameter("min_valid_bins").value)
        self.skip_first_scans = int(self.get_parameter("skip_first_scans").value)
        self.origin_settle_scans = int(self.get_parameter("origin_settle_scans").value)
        self.use_previous_on_bad_scan = bool(self.get_parameter("use_previous_on_bad_scan").value)
        self.map_quality = int(self.get_parameter("map_quality").value)
        self.hole_width_mm = int(self.get_parameter("hole_width_mm").value)
        self.random_seed = int(self.get_parameter("random_seed").value)
        self.sigma_xy_mm = float(self.get_parameter("sigma_xy_mm").value)
        self.sigma_theta_degrees = float(self.get_parameter("sigma_theta_degrees").value)
        self.max_search_iter = int(self.get_parameter("max_search_iter").value)
        self.debug_log_hz = float(self.get_parameter("debug_log_hz").value)
        self.laser_yaw_offset = float(self.get_parameter("laser_yaw_offset_rad").value)
        self.laser_xy_in_base = np.array(
            [
                float(self.get_parameter("laser_x_in_base_m").value),
                float(self.get_parameter("laser_y_in_base_m").value),
            ],
            dtype=np.float32,
        )
        self.pose_x_sign = float(self.get_parameter("pose_x_sign").value)
        self.pose_y_sign = float(self.get_parameter("pose_y_sign").value)
        self.yaw_sign = float(self.get_parameter("yaw_sign").value)
        self.use_motion_prior = bool(self.get_parameter("use_motion_prior").value)
        self.motion_prior_alpha = float(self.get_parameter("motion_prior_alpha").value)
        self.motion_prior_max_fwd_mm = float(self.get_parameter("motion_prior_max_fwd_mm").value)
        self.motion_prior_max_dtheta_deg = float(self.get_parameter("motion_prior_max_dtheta_deg").value)
        self.motion_prior_max_dt_s = float(self.get_parameter("motion_prior_max_dt_s").value)

        self.slam = None
        self.origin = None
        self.scan_count = 0
        self.processed_updates = 0
        self.prev_scan_mm = None
        self.last_debug_time = 0.0

        # Motion-prior state, all in BreezySLAM's internal map frame.
        self.prev_internal = None  # (x_mm, y_mm, theta_deg) from last getpos()
        self.prev_update_time = None
        self.fwd_vel_mm_s = 0.0
        self.yaw_vel_deg_s = 0.0

        self.pub = self.create_publisher(Pose2D, self.output_topic, 10)
        self.sub = self.create_subscription(LaserScan, self.input_topic, self.callback, self.scan_qos_depth)

        self.get_logger().info(
            f"lidar_odom: {self.input_topic} -> {self.output_topic}, "
            f"scan_size={self.breezy_scan_size}, scan_rate={self.scan_rate_hz:.1f} Hz, "
            f"min_valid_bins={self.min_valid_bins}, "
            f"laser_yaw_offset={self.laser_yaw_offset:.3f}, "
            f"laser_xy_in_base=({self.laser_xy_in_base[0]:.3f}, {self.laser_xy_in_base[1]:.3f}), "
            f"pose_sign=({self.pose_x_sign:.1f}, {self.pose_y_sign:.1f}), "
            f"yaw_sign={self.yaw_sign:.1f}"
        )

    def _should_debug_log(self):
        if self.debug_log_hz <= 0.0:
            return False
        now = time.monotonic()
        if now - self.last_debug_time < 1.0 / self.debug_log_hz:
            return False
        self.last_debug_time = now
        return True

    def _init_slam(self, msg):
        # A2M12 with angle_compensate can publish more than 360 ROS samples per
        # revolution, while BreezySLAM's C extension expects the scan length to
        # match the Laser model exactly. Raw /scan is therefore binned into
        # fixed 1-degree distances and update(scan_mm) is called without
        # scan_angles_degrees.
        laser = Laser(self.breezy_scan_size, self.scan_rate_hz, 360, self.no_detection_mm)
        self.slam = RMHC_SLAM(
            laser,
            self.map_size_pixels,
            self.map_size_meters,
            map_quality=self.map_quality,
            hole_width_mm=self.hole_width_mm,
            random_seed=None if self.random_seed < 0 else self.random_seed,
            sigma_xy_mm=self.sigma_xy_mm,
            sigma_theta_degrees=self.sigma_theta_degrees,
            max_search_iter=self.max_search_iter,
        )
        self.get_logger().info(
            f"BreezySLAM initialized: laser={self.breezy_scan_size} bins, map={self.map_size_meters} m, "
            f"quality={self.map_quality}, sigma_xy={self.sigma_xy_mm:.1f} mm, "
            f"sigma_theta={self.sigma_theta_degrees:.1f} deg, max_search_iter={self.max_search_iter}"
        )

    def _scan_to_breezy_scan(self, msg):
        ranges_m = np.asarray(msg.ranges, dtype=np.float32)
        angles_rad = msg.angle_min + np.arange(len(ranges_m), dtype=np.float32) * msg.angle_increment
        angles_deg = np.mod(angles_rad * 180.0 / math.pi, 360.0)

        range_min = max(float(msg.range_min), 0.05)
        range_max = min(float(msg.range_max), self.no_detection_m)
        valid = np.isfinite(ranges_m) & (ranges_m >= range_min) & (ranges_m <= range_max)
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            return [self.no_detection_mm] * self.breezy_scan_size, 0, 0

        scan_mm = np.full(self.breezy_scan_size, self.no_detection_mm, dtype=np.int32)
        bin_idx = np.floor(angles_deg[valid] * self.breezy_scan_size / 360.0).astype(np.int32)
        bin_idx = np.clip(bin_idx, 0, self.breezy_scan_size - 1)
        distances_mm = np.rint(ranges_m[valid] * 1000.0).astype(np.int32)
        np.minimum.at(scan_mm, bin_idx, distances_mm)
        valid_bins = int(np.count_nonzero(scan_mm < self.no_detection_mm))
        return scan_mm.tolist(), valid_count, valid_bins

    def _motion_prior(self, now, used_previous):
        # Predict this frame's (forward_mm, dtheta_deg, dt_s) from the constant
        # velocity estimated over the last accepted update. Returning None makes
        # RMHC_SLAM fall back to its default (0, 0, 0) prior.
        if not self.use_motion_prior or used_previous or self.prev_update_time is None:
            return None
        dt = now - self.prev_update_time
        if dt <= 0.0 or dt > self.motion_prior_max_dt_s:
            dt = 1.0 / self.scan_rate_hz if self.scan_rate_hz > 0.0 else 0.1
        fwd_mm = self.fwd_vel_mm_s * dt
        dtheta_deg = self.yaw_vel_deg_s * dt
        fwd_mm = max(-self.motion_prior_max_fwd_mm, min(self.motion_prior_max_fwd_mm, fwd_mm))
        dtheta_deg = max(-self.motion_prior_max_dtheta_deg, min(self.motion_prior_max_dtheta_deg, dtheta_deg))
        return (fwd_mm, dtheta_deg, dt)

    def _update_velocity(self, now, x_mm, y_mm, theta_deg):
        # Estimate forward/yaw velocity from the internal pose delta so the next
        # frame can predict a pose_change. Smoothed with an EMA to reject jitter.
        if not self.use_motion_prior:
            return
        if self.prev_internal is not None and self.prev_update_time is not None:
            dt = now - self.prev_update_time
            if 1e-3 < dt <= self.motion_prior_max_dt_s:
                px, py, pth = self.prev_internal
                pth_rad = math.radians(pth)
                fwd_disp = (x_mm - px) * math.cos(pth_rad) + (y_mm - py) * math.sin(pth_rad)
                dtheta_deg = math.degrees(wrap_pi(math.radians(theta_deg - pth)))
                a = self.motion_prior_alpha
                self.fwd_vel_mm_s = (1.0 - a) * self.fwd_vel_mm_s + a * (fwd_disp / dt)
                self.yaw_vel_deg_s = (1.0 - a) * self.yaw_vel_deg_s + a * (dtheta_deg / dt)
        self.prev_internal = (x_mm, y_mm, theta_deg)
        self.prev_update_time = now

    def callback(self, msg):
        t0 = time.monotonic()
        self.scan_count += 1

        if len(msg.ranges) == 0:
            return
        if self.slam is None:
            self._init_slam(msg)

        scan_mm, valid_count, valid_bins = self._scan_to_breezy_scan(msg)
        if self.scan_count <= self.skip_first_scans:
            self.prev_scan_mm = scan_mm if valid_bins >= self.min_valid_bins else self.prev_scan_mm
            return

        used_previous = False
        if valid_bins < self.min_valid_bins:
            if self.use_previous_on_bad_scan and self.prev_scan_mm is not None:
                scan_mm = self.prev_scan_mm
                used_previous = True
            else:
                if self._should_debug_log():
                    self.get_logger().warning(f"skip bad scan: valid_bins={valid_bins}, need>={self.min_valid_bins}")
                return
        else:
            self.prev_scan_mm = scan_mm

        now = time.monotonic()
        pose_change = self._motion_prior(now, used_previous)
        self.slam.update(scan_mm, pose_change)
        self.processed_updates += 1
        x_mm, y_mm, theta_deg = self.slam.getpos()
        self._update_velocity(now, x_mm, y_mm, theta_deg)
        theta_laser_rad = self.yaw_sign * math.radians(theta_deg)
        theta_base_rad = wrap_pi(theta_laser_rad - self.laser_yaw_offset)

        lidar_xy_map = np.array([float(x_mm) / 1000.0, float(y_mm) / 1000.0], dtype=np.float32)
        cb = math.cos(theta_base_rad)
        sb = math.sin(theta_base_rad)
        laser_xy_map_from_base = np.array(
            [
                cb * self.laser_xy_in_base[0] - sb * self.laser_xy_in_base[1],
                sb * self.laser_xy_in_base[0] + cb * self.laser_xy_in_base[1],
            ],
            dtype=np.float32,
        )
        base_xy_map = lidar_xy_map - laser_xy_map_from_base

        if self.origin is None:
            if self.processed_updates < self.origin_settle_scans:
                if self._should_debug_log():
                    self.get_logger().info(
                        f"settling origin: update={self.processed_updates}/{self.origin_settle_scans}, "
                        f"raw_pose=({base_xy_map[0]:.3f}, {base_xy_map[1]:.3f}, {theta_base_rad:.3f}), "
                        f"valid_raw={valid_count}, valid_bins={valid_bins}"
                    )
                return
            self.origin = (float(base_xy_map[0]), float(base_xy_map[1]), float(theta_base_rad))
            self.get_logger().info(
                f"origin fixed at raw_pose=({base_xy_map[0]:.3f}, {base_xy_map[1]:.3f}, {theta_base_rad:.3f}) "
                f"after {self.processed_updates} Breezy updates"
            )

        x0, y0, theta0 = self.origin
        dx = float(base_xy_map[0]) - x0
        dy = float(base_xy_map[1]) - y0

        # Rotate the SLAM map frame so startup base_link heading becomes world +X.
        c = math.cos(theta0)
        s = math.sin(theta0)
        x_rel = c * dx + s * dy
        y_rel = -s * dx + c * dy
        yaw_rel = wrap_pi(theta_base_rad - theta0)

        pose = Pose2D()
        pose.x = float(self.pose_x_sign * x_rel)
        pose.y = float(self.pose_y_sign * y_rel)
        pose.theta = float(yaw_rel)
        self.pub.publish(pose)

        if self._should_debug_log():
            dt_ms = 1000.0 * (time.monotonic() - t0)
            self.get_logger().info(
                f"pose x={pose.x:.3f}, y={pose.y:.3f}, yaw={pose.theta:.3f}; "
                f"valid_raw={valid_count}, valid_bins={valid_bins}, prev={used_previous}, dt={dt_ms:.1f} ms, "
                f"vel=({self.fwd_vel_mm_s:.0f} mm/s, {self.yaw_vel_deg_s:.1f} deg/s)"
            )


def main():
    rclpy.init()
    node = LidarOdomNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
