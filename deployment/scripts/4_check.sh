#!/usr/bin/env bash
# Terminal 4 (option B) - one-shot health check of the perception stack.
# Prints publisher counts, ~rates, and verifies /rays is a finite 41-vector
# inside [range_min, range_max]. Exits on its own (does not stream forever).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "================ topics ================"
ros2 topic list | grep -E '^/scan$|^/rays$|^/pose$' || echo "  (none of /scan /rays /pose found yet)"

for t in /scan /rays /pose; do
    echo
    echo "================ ${t} ================"
    ros2 topic info "${t}" 2>/dev/null || echo "  no publisher on ${t}"
    echo "--- rate (sampling ~4s) ---"
    timeout 4 ros2 topic hz "${t}" 2>/dev/null || true
done

echo
echo "================ /rays sanity ================"
python - <<'PY'
import sys
import rclpy
from sensor_msgs.msg import LaserScan

rclpy.init()
node = rclpy.create_node("seanav_rays_check")
state = {}

def cb(msg):
    state["msg"] = msg
    rclpy.shutdown()

node.create_subscription(LaserScan, "/rays", cb, 10)
# wait up to 3 s for one message
import threading, time
t0 = time.time()
while rclpy.ok() and "msg" not in state and time.time() - t0 < 3.0:
    rclpy.spin_once(node, timeout_sec=0.2)

msg = state.get("msg")
if msg is None:
    print("  FAIL: no /rays message in 3 s")
    sys.exit(0)

import math
n = len(msg.ranges)
finite = all(math.isfinite(v) for v in msg.ranges)
in_range = all(msg.range_min - 1e-3 <= v <= msg.range_max + 1e-3 for v in msg.ranges if math.isfinite(v))
print(f"  len={n} (expect 41)   finite={finite}   in[{msg.range_min:.2f},{msg.range_max:.2f}]={in_range}")
print(f"  front idx20={msg.ranges[20]:.3f} m" if n > 20 else "  (too few rays to read idx20)")
print("  OK" if (n == 41 and finite and in_range) else "  CHECK THE WARNINGS ABOVE")
PY
