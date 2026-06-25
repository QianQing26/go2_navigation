#!/usr/bin/env bash
# Terminal 4 (option A) - live ASCII monitor of /rays for direction calibration.
# idx 0 = -120 deg (right), idx 20 = straight ahead, idx 40 = +120 deg (left).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

exec ros2 run seanav_perception rays_monitor_node --ros-args \
    -p topic:=/rays \
    -p print_hz:=5.0 \
    -p warn_distance:=0.6
