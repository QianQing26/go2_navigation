#!/usr/bin/env bash
# Terminal 5 - SEA-Nav controller (Nav + Loco + safe-stop FSM) on the real Go2.
#
# Subscribes to /rays + /pose (ROS2, domain 1) and talks to the Go2 over the
# Unitree SDK (DDS domain 0 on eth0, configured inside sea_nav_run_sdk.py).
#
# Safety: defaults to DRY-RUN (motors disabled). Enable motors only after the
# perception checks pass and an operator is holding the joystick:
#
#     bash 5_controller.sh            # dry-run, motors OFF (default)
#     bash 5_controller.sh --run      # motors ON  (real movement!)
#     GOAL_X=2.0 GOAL_Y=0.5 bash 5_controller.sh --run
#
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"

EXTRA=()
MODE="DRY-RUN (motors OFF)"
if [ "${1:-}" = "--run" ] || [ "${MOTOR:-0}" = "1" ]; then
    EXTRA+=(--nodryrun)
    MODE="LIVE (motors ON - robot will move!)"
fi

if [ ! -f "${SEANAV_DATA}/nav_model/model.onnx" ] || [ ! -f "${SEANAV_DATA}/loco_model/model.onnx" ]; then
    echo "[controller] WARN: expected ONNX models not found under ${SEANAV_DATA}:" >&2
    echo "    ${SEANAV_DATA}/nav_model/model.onnx" >&2
    echo "    ${SEANAV_DATA}/loco_model/model.onnx" >&2
fi

cd "${QUAD_DEPLOY_DIR}" || { echo "[controller] ERROR: ${QUAD_DEPLOY_DIR} not found" >&2; exit 1; }

echo "[controller] mode=${MODE}"
echo "[controller] goal=(${GOAL_X}, ${GOAL_Y})  data=${SEANAV_DATA}"
echo "[controller] joystick: X=stand->teleop  R1=teleop->nav  R2=takeover  L2=E-stop  L1=recover"
exec python -m quad_deploy.scripts.SEA_Nav.sea_nav_run_sdk \
    --nosimrun \
    "${EXTRA[@]}" \
    --data "${SEANAV_DATA}" \
    --goal_x "${GOAL_X}" \
    --goal_y "${GOAL_Y}"
