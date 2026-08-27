"""Convert the raw 360-degree ``/scan`` into the 41-ray ``/rays`` that the
SEA-Nav navigation policy consumes.

The node down-samples the lidar into 41 rays spanning [-120, +120] deg in the
robot ``base_link`` frame (the same contract the simulator publishes). Mounting
orientation is corrected with ``LidarCfg.yaw_offset_rad`` / ``RaysCfg.invert_angle``
so that ``ranges[20]`` always points straight ahead.

No policy-specific encoding happens here: distances are published in meters and
clipped to ``[range_min, range_max]``; the ``log2`` encoding is applied later
inside the Nav agent so the input distribution matches training.

Defaults come from :mod:`seanav_deploy.config`; every value can still be
overridden per-run with ``--ros-args -p name:=value`` for debugging.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from seanav_deploy.config import LidarCfg, RaysCfg


def wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class ScanToRaysNode(Node):
    def __init__(self):
        super().__init__("scan_to_rays")

        self.declare_parameter("input_topic", RaysCfg.input_topic)
        self.declare_parameter("output_topic", RaysCfg.output_topic)
        self.declare_parameter("output_frame", RaysCfg.output_frame)
        self.declare_parameter("laser_yaw_offset_rad", float(LidarCfg.yaw_offset_rad))
        self.declare_parameter("invert_angle", bool(RaysCfg.invert_angle))
        self.declare_parameter("range_min", float(RaysCfg.range_min))
        self.declare_parameter("range_max", float(RaysCfg.range_max))
        self.declare_parameter("bin_width_deg", float(RaysCfg.bin_width_deg))

        self.input_topic = self.get_parameter("input_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.output_frame = self.get_parameter("output_frame").value
        self.laser_yaw_offset = float(self.get_parameter("laser_yaw_offset_rad").value)
        self.invert_angle = bool(self.get_parameter("invert_angle").value)
        self.range_min = float(self.get_parameter("range_min").value)
        self.range_max = float(self.get_parameter("range_max").value)
        self.bin_width = math.radians(float(self.get_parameter("bin_width_deg").value))

        # SEA-Nav Sim2Sim contract: 41 rays over [-120, +120] deg.
        self.target_angles = np.linspace(-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0, 41)

        self.pub = self.create_publisher(LaserScan, self.output_topic, 10)
        self.sub = self.create_subscription(LaserScan, self.input_topic, self.callback, 10)

        self.get_logger().info(
            f"scan_to_rays: {self.input_topic} -> {self.output_topic}, "
            f"yaw_offset={self.laser_yaw_offset:.3f}, invert={self.invert_angle}"
        )

    def callback(self, msg):
        raw_ranges = np.asarray(msg.ranges, dtype=np.float32)
        n = raw_ranges.shape[0]
        if n == 0:
            return

        raw_angles = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment
        sign = -1.0 if self.invert_angle else 1.0

        rays = []
        for base_angle in self.target_angles:
            # base_angle = laser_yaw_offset + sign * laser_angle
            laser_angle = sign * (base_angle - self.laser_yaw_offset)
            diff = np.abs(wrap_pi(raw_angles - laser_angle))
            mask = diff <= 0.5 * self.bin_width

            candidates = raw_ranges[mask]
            candidates = candidates[np.isfinite(candidates)]
            candidates = candidates[candidates > 0.0]

            if candidates.size == 0:
                # Fallback to nearest raw sample.
                nearest = int(np.argmin(diff))
                value = raw_ranges[nearest]
                if not np.isfinite(value) or value <= 0.0:
                    value = self.range_max
            else:
                # Conservative obstacle distance in this angular bin.
                value = float(np.min(candidates))

            rays.append(float(np.clip(value, self.range_min, self.range_max)))

        out = LaserScan()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.output_frame
        out.angle_min = float(self.target_angles[0])
        out.angle_max = float(self.target_angles[-1])
        out.angle_increment = float(self.target_angles[1] - self.target_angles[0])
        out.time_increment = 0.0
        out.scan_time = msg.scan_time if msg.scan_time > 0.0 else 0.1
        out.range_min = self.range_min
        out.range_max = self.range_max
        out.ranges = rays
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ScanToRaysNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
