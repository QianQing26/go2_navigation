"""SEA-Nav real-robot controller entry.

Performs a perception handshake first - blocks until ``/rays`` and ``/pose``
are both streaming (the same handshake pattern as the ``ros_base.BaseManager``
rules) - and only then exec's the SEA-Nav control stack::

    python -m quad_deploy.scripts.sea.sea_run_sdk --nosimrun [...]

The handshake lives on the deployment side, so the controller can be launched
together with the perception nodes (see ``launch/deploy_launch.py``) without
modifying ``quad_deploy``.

Safety: motors stay disabled (dry-run) unless ``--run`` is given.
"""

import argparse
import os
import sys
import time

import rclpy
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from seanav_deploy.config import ControllerCfg, PoseCfg, RaysCfg


class PerceptionGate(Node):
    """Counts /rays and /pose messages until both streams qualify as ready."""

    def __init__(self, rays_topic, pose_topic):
        super().__init__("seanav_controller_gate")
        self._counts = {"rays": 0, "pose": 0}
        self._last_time = {"rays": None, "pose": None}

        self.create_subscription(LaserScan, rays_topic, lambda msg: self._on_msg("rays"), 10)
        self.create_subscription(Pose2D, pose_topic, lambda msg: self._on_msg("pose"), 10)

    def _on_msg(self, key):
        self._counts[key] += 1
        self._last_time[key] = time.monotonic()

    def _stream_ready(self, key, min_msgs, fresh_s):
        last = self._last_time[key]
        if last is None or self._counts[key] < min_msgs:
            return False
        return (time.monotonic() - last) <= fresh_s

    def ready(self, min_msgs, fresh_s):
        return self._stream_ready("rays", min_msgs, fresh_s) and self._stream_ready("pose", min_msgs, fresh_s)

    def status(self):
        return f"rays={self._counts['rays']} msgs, pose={self._counts['pose']} msgs"


def wait_for_perception(rays_topic, pose_topic):
    """Block until both perception topics stream steadily.

    There is no timeout: the control stack must not start while perception is
    down, so the gate simply keeps waiting and reporting progress.
    """
    rclpy.init()
    gate = PerceptionGate(rays_topic, pose_topic)
    print(f"[handshake] waiting for perception: {rays_topic} + {pose_topic} ...")

    last_report = 0.0
    try:
        while rclpy.ok() and not gate.ready(ControllerCfg.handshake_min_msgs, ControllerCfg.handshake_fresh_s):
            rclpy.spin_once(gate, timeout_sec=0.2)
            now = time.monotonic()
            if now - last_report >= ControllerCfg.handshake_report_period_s:
                last_report = now
                print(f"[handshake] waiting ... ({gate.status()})")
        print(f"[handshake] perception ready ({gate.status()}).")
    finally:
        gate.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser(description="SEA-Nav real-robot controller (perception handshake + quad_deploy)")
    parser.add_argument("--run", action="store_true", help="Enable motors (default is dry-run: motors stay off).")
    parser.add_argument("--goal_x", type=float, default=ControllerCfg.goal_x, help="Goal x in the /pose world frame.")
    parser.add_argument("--goal_y", type=float, default=ControllerCfg.goal_y, help="Goal y in the /pose world frame.")
    parser.add_argument("--data", type=str, default=ControllerCfg.data_dir, help="ONNX model directory.")
    # ros2 run appends --ros-args; accept and ignore unknown args.
    args, _ = parser.parse_known_args()

    try:
        wait_for_perception(RaysCfg.output_topic, PoseCfg.output_topic)
    except KeyboardInterrupt:
        print("[handshake] interrupted; controller not started.")
        sys.exit(130)

    cmd = [
        sys.executable,
        "-m",
        ControllerCfg.run_module,
        "--nosimrun",
        "--data",
        args.data,
        "--goal_x",
        str(args.goal_x),
        "--goal_y",
        str(args.goal_y),
    ]
    if args.run:
        cmd.append("--nodryrun")

    mode = "LIVE (motors ON)" if args.run else "DRY-RUN (motors OFF)"
    print(f"[controller] mode={mode}  goal=({args.goal_x}, {args.goal_y})  data={args.data}")
    print(f"[controller] exec: {' '.join(cmd[1:])}")
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
