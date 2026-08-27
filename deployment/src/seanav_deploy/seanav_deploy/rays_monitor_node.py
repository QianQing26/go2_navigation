"""Live ASCII monitor for ``/rays``.

Prints the 41-ray scan as a one-line bar plus a few sampled distances to verify
the lidar mounting direction (index 0 = -120 deg right, index 20 = straight
ahead, index 40 = +120 deg left) and to track the nearest obstacle while test
obstacles are moved around the robot.

Read-only: it never publishes, so it is safe to keep running during deployment.
"""

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from seanav_deploy.config import MonitorCfg


class RaysMonitorNode(Node):
    def __init__(self):
        super().__init__("rays_monitor")

        self.declare_parameter("topic", MonitorCfg.topic)
        self.declare_parameter("print_hz", float(MonitorCfg.print_hz))
        self.declare_parameter("warn_distance", float(MonitorCfg.warn_distance))

        self.topic = self.get_parameter("topic").value
        self.print_period = 1.0 / float(self.get_parameter("print_hz").value)
        self.warn_distance = float(self.get_parameter("warn_distance").value)
        self.last_print_t = 0.0

        self.sub = self.create_subscription(LaserScan, self.topic, self.callback, 10)
        self.get_logger().info(f"Monitoring {self.topic}")

    def _finite_ranges(self, ranges):
        out = []
        for i, value in enumerate(ranges):
            if math.isfinite(value):
                out.append((i, float(value)))
        return out

    def _sector_min(self, ranges, start, end):
        vals = [float(ranges[i]) for i in range(start, end) if math.isfinite(ranges[i])]
        return min(vals) if vals else float("nan")

    def _symbol(self, value):
        if not math.isfinite(value):
            return "?"
        if value < 0.3:
            return "X"
        if value < 0.6:
            return "#"
        if value < 1.0:
            return "*"
        if value < 2.0:
            return "."
        return "_"

    def callback(self, msg):
        now = time.monotonic()
        if now - self.last_print_t < self.print_period:
            return
        self.last_print_t = now

        ranges = list(msg.ranges)
        if not ranges:
            print("No rays received.")
            return

        finite = self._finite_ranges(ranges)
        if finite:
            nearest_i, nearest_r = min(finite, key=lambda item: item[1])
            nearest_deg = math.degrees(msg.angle_min + nearest_i * msg.angle_increment)
        else:
            nearest_i, nearest_r, nearest_deg = -1, float("nan"), float("nan")

        front_i = min(20, len(ranges) - 1)
        front = float(ranges[front_i]) if math.isfinite(ranges[front_i]) else float("nan")
        front_min = self._sector_min(ranges, max(0, front_i - 1), min(len(ranges), front_i + 2))
        right_min = self._sector_min(ranges, 0, front_i)
        left_min = self._sector_min(ranges, min(len(ranges), front_i + 1), len(ranges))

        symbols = [self._symbol(v) for v in ranges]
        if len(symbols) > front_i:
            symbols[front_i] = "|" if front >= self.warn_distance else "!"
        ray_line = "".join(symbols)

        print("\033[2J\033[H", end="")
        print("SEA-Nav /rays live monitor")
        print("idx 0=-120deg right, idx 20=front, idx 40=+120deg left")
        print("legend: X<0.3m  #<0.6m  *<1.0m  .<2.0m  _>=2.0m  | front")
        print("")
        print(f"count={len(ranges)}  range_min={msg.range_min:.2f}  range_max={msg.range_max:.2f}")
        print(f"nearest: idx={nearest_i:02d}  angle={nearest_deg:+6.1f}deg  dist={nearest_r:.3f}m")
        print(f"front idx20={front:.3f}m  front_min[19:21]={front_min:.3f}m")
        print(f"right_min[0:19]={right_min:.3f}m  left_min[21:40]={left_min:.3f}m")
        print("")
        print("right(-120)                                        left(+120)")
        print(ray_line)
        print("")
        print("sample rays:")
        for idx in (0, 5, 10, 15, 20, 25, 30, 35, 40):
            if idx < len(ranges):
                deg = math.degrees(msg.angle_min + idx * msg.angle_increment)
                print(f"  idx={idx:02d}  angle={deg:+6.1f}deg  dist={float(ranges[idx]):.3f}m")


def main():
    rclpy.init()
    node = RaysMonitorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
