#!/usr/bin/env bash
# One-command bring-up of the whole PERCEPTION stack in a 4-pane tmux session:
#   pane 0: 1_lidar.sh    (/scan)
#   pane 1: 2_rays.sh     (/rays)
#   pane 2: 3_pose.sh     (/pose, BreezySLAM)
#   pane 3: 4_monitor.sh  (live /rays monitor)
#
# The controller (Terminal 5) is intentionally NOT started here: launch it by
# hand with `bash 5_controller.sh [--run]` once perception is verified and an
# operator is ready on the joystick.
#
#     bash start_perception_tmux.sh
#     tmux attach -t seanav_perception
#     tmux kill-session -t seanav_perception   # stop everything
# NOTE: no `set -u` here on purpose - sourcing ROS2 setup.bash trips nounset.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/config.sh"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found. Install it (sudo apt install -y tmux) or start each n_*.sh by hand." >&2
    exit 1
fi

SESSION="${SEANAV_TMUX_SESSION}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "Session '${SESSION}' already running. Attach with: tmux attach -t ${SESSION}"
    exit 0
fi

# Target the session's active window/pane only (no hardcoded :0 indices, so
# this works regardless of the user's base-index / pane-base-index settings).
# send-keys is used instead of passing commands to split-window so a pane that
# errors out leaves an interactive shell behind for debugging.
tmux new-session -d -s "${SESSION}" -n perception
tmux send-keys -t "${SESSION}" "bash '${SCRIPT_DIR}/1_lidar.sh'" C-m
sleep 2  # let /scan come up before the consumers subscribe

tmux split-window -h -t "${SESSION}"
tmux send-keys -t "${SESSION}" "bash '${SCRIPT_DIR}/2_rays.sh'" C-m

tmux split-window -v -t "${SESSION}"
tmux send-keys -t "${SESSION}" "bash '${SCRIPT_DIR}/3_pose.sh'" C-m

tmux select-pane -t "${SESSION}" -L
tmux split-window -v -t "${SESSION}"
tmux send-keys -t "${SESSION}" "bash '${SCRIPT_DIR}/4_monitor.sh'" C-m

tmux select-layout -t "${SESSION}" tiled

echo "Started perception stack in tmux session '${SESSION}'."
echo "  attach: tmux attach -t ${SESSION}"
echo "  stop  : tmux kill-session -t ${SESSION}"
echo "Then, in a separate terminal, run the controller: bash ${SCRIPT_DIR}/5_controller.sh [--run]"
