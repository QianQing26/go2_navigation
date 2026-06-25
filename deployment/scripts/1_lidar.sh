#!/usr/bin/env bash
# Terminal 1 - RPLIDAR A2M12 driver. Publishes the raw 360-degree /scan.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

if [ ! -e "${LIDAR_PORT}" ]; then
    echo "[lidar] ERROR: ${LIDAR_PORT} does not exist. Is the lidar plugged in?" >&2
    exit 1
fi
if [ ! -w "${LIDAR_PORT}" ]; then
    echo "[lidar] NOTE: ${LIDAR_PORT} not writable. Grant access with:"
    echo "    sudo chmod 666 ${LIDAR_PORT}"
fi

echo "[lidar] starting sllidar_node on ${LIDAR_PORT} @ ${LIDAR_BAUD} (frame=${LIDAR_FRAME})"
exec ros2 run sllidar_ros2 sllidar_node --ros-args \
    -p channel_type:=serial \
    -p serial_port:="${LIDAR_PORT}" \
    -p serial_baudrate:="${LIDAR_BAUD}" \
    -p frame_id:="${LIDAR_FRAME}" \
    -p inverted:=false \
    -p angle_compensate:=true \
    -p scan_mode:="${LIDAR_SCAN_MODE}"
