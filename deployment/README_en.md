# SEA-Nav Real-Robot Deployment Guide (Go2 + RPLIDAR A2M12 + BreezySLAM)

This guide walks through deploying SEA-Nav autonomous navigation on a Unitree Go2 from scratch — environment setup, repository wiring, SLAM calibration, and terminal startup. All commands target the latest code on the `feat/sea-nav-agent` branch of `quad_deploy`.

> **Design principle**: The control stack inside `quad_deploy` (Nav + Loco + safety FSM) was validated in Sim2Sim and is **not modified** for real-robot deployment. All real-robot adaptation lives on the "publisher side" — converting raw lidar and SLAM output into the two topics the control stack already subscribes to: `/rays` and `/pose`. This directory provides the publisher-side code (`perception/`), one-shot launch scripts (`scripts/`), and this document.

## 0. One-minute overview

| Terminal | Script | Role | Output |
|----------|--------|------|--------|
| Terminal 1 | `1_lidar.sh` | RPLIDAR A2M12 driver | `/scan` (360° raw laser) |
| Terminal 2 | `2_rays.sh` | `/scan` → `/rays` downsampling | `/rays` (41 beams, `base_link`) |
| Terminal 3 | `3_pose.sh` | `/scan` → BreezySLAM odometry | `/pose` (world-frame `x,y,θ`) |
| Terminal 4 | `4_monitor.sh` / `4_check.sh` | Live monitor / one-shot health check | Terminal printout |
| Terminal 5 | `5_controller.sh` | SEA-Nav controller (Nav+Loco+FSM) | Go2 `rt/lowcmd` |

Terminals 1–4 are the "perception publisher side" and can be launched together with `start_perception_tmux.sh`. Terminal 5 is the controller — start it manually only after perception is verified.

---

## 1. Data flow

```text
RPLIDAR A2M12
   │  (Terminal 1: sllidar_ros2)
   ▼
 /scan  (sensor_msgs/LaserScan, 360°, ~10 Hz, frame=laser)
   ├──────────────► (Terminal 2: scan_to_rays_node) ──► /rays (LaserScan, 41 beams, base_link, 10 Hz)
   └──────────────► (Terminal 3: breezy_lidar_odom_node) ──► /pose (Pose2D, world frame, 10 Hz)

 /rays + /pose
   │  (Terminal 5: sea_nav_run_sdk.py, ROS2 domain 1 + CycloneDDS)
   ▼
 SEA-Nav Nav policy (obs 550 → vx,vy,vyaw)  →  SEA-Nav Loco policy (obs 235 → 12 joints + vel_pred)
   │  (Unitree SDK, DDS domain 0 / eth0)
   ▼
 Unitree Go2  (reads rt/lowstate, writes rt/lowcmd)
```

The control stack depends only on these two topics (spec fixed by `SEA_Nav_NavAgentCfg` — **do not modify `quad_deploy`**):

| Topic | Type | Rate | Content |
|-------|------|------|---------|
| `/rays` | `sensor_msgs/msg/LaserScan` | 10 Hz | 41 beams, `base_link`, forward 240° (±120°), range `[0.1, 3.0]` m |
| `/pose` | `geometry_msgs/msg/Pose2D` | 10 Hz | Robot pose `(x, y, θ)` in local world frame |

> **Two independent DDS channels**: `/scan`, `/rays`, `/pose` use ROS2 (`ROS_DOMAIN_ID=1` + CycloneDDS); Go2 body communication is handled separately inside `sea_nav_run_sdk.py` via `ChannelFactoryInitialize(0, "eth0")` (DDS domain 0, interface `eth0`). They do not interfere.

---

## 2. Prerequisites

**Hardware**: Unitree Go2, RPLIDAR A2M12 (USB-serial, default `/dev/ttyUSB0`, baud 256000), gamepad, ethernet cable between the compute machine and Go2 (`eth0`).

**Software (to install on the robot machine)**:

| Repo / Package | Suggested path | Purpose |
|----------------|---------------|---------|
| `quad_deploy` | `~/Projects/quad_deploy` | Main control program, runs `sea_nav_run_sdk.py` |
| `ros_base` | `~/Projects/ros_base` | Base manager/node/agent classes for `quad_deploy` |
| `unitree_sdk2_python` | `~/Projects/unitree_sdk2_python` | Reads `rt/lowstate`, writes `rt/lowcmd` |
| `cyclonedds` | `~/Projects/cyclonedds` | DDS transport dependency for `unitree_sdk2_python` |
| `sllidar_ros2` | `~/seanav_ws/src/sllidar_ros2` | RPLIDAR driver, publishes `/scan` (auto-cloned by build script) |
| `BreezySLAM` | `~/Projects/BreezySLAM` | Laser odometry for `/pose` |
| `seanav_perception` | `~/seanav_ws/src/seanav_perception` | Publisher-side adapter package from this directory (symlinked by build script) |

**Not needed on the robot**: the `SEA-Nav-Code` training repo (only the exported ONNX is needed), Isaac Gym / `legged_gym`, `unitree_mujoco_ros` (Sim2Sim only).

---

## 3. One-time environment setup

The steps below only need to be done once per machine. If code has already been synced from a dev machine (see Section 4), skip the `git clone` steps and go straight to installation.

### 3.1 Conda environment and ROS2

Use a single Conda environment on the robot (this guide calls it `sea`). It must have: `rclpy` (ROS2 Foxy Python bindings), `onnxruntime`, `numpy`, and the packages installed below (`quad_deploy`, `ros_base`, `unitree_sdk2py`, `BreezySLAM`). The ROS2 side uses **Foxy + CycloneDDS + `ROS_DOMAIN_ID=1`**, equivalent to:

```bash
conda activate sea
source /opt/ros/foxy/setup.bash
source ~/cyclonedds_ws/install/setup.bash          # CycloneDDS overlay (see 3.2)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=1
```

> **Version constraints (important)**: ROS2 Foxy's `rclpy` bindings require **Python 3.8**, so the `sea` environment must be Python 3.8, and `numpy` / `onnxruntime` cannot be too new (no 3.8 wheels available):
>
> | Package | Recommended version | Reason |
> |---------|-------------------|--------|
> | `numpy` | `>=1.20,<1.25` | numpy 1.25+ requires Python ≥3.9; 1.24.x is the last branch supporting 3.8 |
> | `onnxruntime` | `>=1.12,<1.20` | onnxruntime 1.20+ requires Python ≥3.10; 1.19.2 is the last version supporting 3.8 |
>
> Install example: `pip install "numpy>=1.20,<1.25" "onnxruntime>=1.12,<1.20"`. Neither `quad_deploy` nor `seanav_perception` pin versions in their `setup.py` — if pip pulls in something too new, downgrade per the table above.

`scripts/env.sh` encapsulates the entire setup (activate conda → source ROS2 Foxy → source CycloneDDS → set RMW and domain). All terminal scripts auto-source it, so **you do not need to run these commands manually**. All paths and names are centralized in `scripts/config.sh` (see Section 5) — edit that file if your conda env is not named `sea`, ROS2 is not at `/opt/ros/foxy`, or the CycloneDDS workspace is elsewhere. You can also override per-run with env vars:

```bash
SEANAV_CONDA_ENV=myenv SEANAV_ROS_SETUP=/opt/ros/foxy/setup.bash bash 1_lidar.sh
```

> If `python -c "import rclpy"` fails after `conda activate sea`, the environment's Python does not match ROS2 Foxy's bindings. Fix the environment until `import rclpy` works stably — do not create a second ROS environment as a workaround.

### 3.2 Install `quad_deploy` dependency chain

```bash
conda activate sea
# ros_base
cd ~/Projects && git clone https://github.com/11chens/ros_base.git
cd ~/Projects/ros_base && pip install -e .

# unitree_sdk2_python
cd ~/Projects && git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd ~/Projects/unitree_sdk2_python && pip install -e .
```

If installing `unitree_sdk2_python` reports `Could not locate cyclonedds`, build CycloneDDS first:

```bash
cd ~/Projects && git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd ~/Projects/cyclonedds && mkdir -p build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install -j"$(nproc)"

export CYCLONEDDS_HOME=~/Projects/cyclonedds/install
echo 'export CYCLONEDDS_HOME=~/Projects/cyclonedds/install' >> ~/.bashrc
cd ~/Projects/unitree_sdk2_python && pip install -e .
```

Then clone and install `quad_deploy` (branch `feat/sea-nav-agent`) and verify the SEA-Nav entry points are importable:

```bash
cd ~/Projects && git clone -b feat/sea-nav-agent https://github.com/11chens/quad_deploy.git
cd ~/Projects/quad_deploy && pip install -e .
python - <<'PY'
from quad_deploy.scripts.SEA_Nav.sea_nav_run_sdk import main
from quad_deploy.agents.SEA_Nav.SEA_Nav_loco_agent import SEA_Nav_LocoAgent
from quad_deploy.agents.SEA_Nav.SEA_Nav_nav_agent import SEA_Nav_NavAgent
from quad_deploy.config.SEA_Nav.SEA_Nav_nav_agent_cfg import SEA_Nav_NavAgentCfg
print("quad_deploy SEA-Nav import OK")
print("rays topic:", SEA_Nav_NavAgentCfg.rays_topic, "| pose topic:", SEA_Nav_NavAgentCfg.pose_topic)
PY
```

### 3.3 Prepare ONNX models

After unifying Loco, **both Nav and Loco models live in the same directory** `~/Data/onboard_data/onnx_models/sea_nav/` (the default `--data` value for `sea_nav_run_sdk.py`):

```text
~/Data/onboard_data/onnx_models/sea_nav/
├── nav_model/model.onnx     # High-level navigation, input 550, output 3 (vx, vy, vyaw)
└── loco_model/model.onnx    # Low-level locomotion, input 235, output 12 joints + vel_pred(3)
```

Sync from dev machine (adjust paths as needed):

```bash
mkdir -p ~/Data/onboard_data/onnx_models/sea_nav
rsync -av --progress \
  <dev-pc>:~/Data/onboard_data/onnx_models/sea_nav/ \
  ~/Data/onboard_data/onnx_models/sea_nav/
```

Verify dimensions (nav input should be 550, loco input 235):

```bash
python - <<'PY'
import os, onnxruntime as ort
base = os.path.expanduser("~/Data/onboard_data/onnx_models/sea_nav")
for sub in ("nav_model/model.onnx", "loco_model/model.onnx"):
    s = ort.InferenceSession(os.path.join(base, sub))
    print(sub)
    print("  in :", [(i.name, i.shape) for i in s.get_inputs()])
    print("  out:", [(o.name, o.shape) for o in s.get_outputs()])
PY
```

### 3.4 Lidar driver, BreezySLAM, and perception adapter

Install BreezySLAM first (Python/C extension, installed into the `sea` env):

```bash
sudo apt update && sudo apt install -y git build-essential python3-dev tmux
cd ~/Projects && git clone https://github.com/simondlevy/BreezySLAM.git
cd ~/Projects/BreezySLAM/python && python -m pip install .
python -c "from breezyslam.algorithms import RMHC_SLAM; print('BreezySLAM OK')"
```

Then use the build script in this directory to do everything in one shot: clone `sllidar_ros2`, symlink `seanav_perception` into `~/seanav_ws/src`, and `colcon build` both packages:

```bash
cd ~/Projects/SEA-Nav-Code/deployment/scripts
bash build_perception.sh
ros2 pkg executables seanav_perception   # should list 3 executable nodes
```

> The symlink means **this repo is the single source of truth** for publisher-side code. After editing nodes under `perception/`, just re-run `build_perception.sh` — no manual copying needed.

After building, plug in the lidar and check serial port permissions:

```bash
ls -l /dev/ttyUSB0           # confirm device exists
sudo chmod 666 /dev/ttyUSB0  # if not writable
```

---

## 4. Code sync (dev machine ↔ robot)

If there are uncommitted local changes on the dev machine, use `rsync` to sync to the robot (more suitable for repeated syncs than `scp -r`):

```bash
# Sync quad_deploy (exclude caches)
rsync -av --progress --exclude '__pycache__' --exclude '*.pyc' \
  ~/Projects/quad_deploy/  unitree@<robot_ip>:~/Projects/quad_deploy/

# Sync ros_base (if locally modified)
rsync -av --progress --exclude '__pycache__' \
  ~/Projects/ros_base/  unitree@<robot_ip>:~/Projects/ros_base/

# Sync this deployment directory (publisher code + scripts + docs)
rsync -av --progress \
  ~/Projects/SEA-Nav-Code/deployment/  unitree@<robot_ip>:~/Projects/SEA-Nav-Code/deployment/

# Sync ONNX models
rsync -av --progress \
  ~/Data/onboard_data/onnx_models/sea_nav/  unitree@<robot_ip>:~/Data/onboard_data/onnx_models/sea_nav/
```

> After syncing source, always re-run `pip install -e .` (Python packages) or `bash build_perception.sh` (ROS2 packages) on the robot, otherwise the latest code may not be loaded. Use NoMachine for GUI and small text copies; always use `rsync`/`scp` for code and models.

---

## 5. Centralized configuration (`scripts/config.sh`)

All tunable parameters are in `scripts/config.sh`. Every value can be overridden by an environment variable without editing the script. Common entries:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEANAV_CONDA_ENV` | `sea` | Conda env containing rclpy/onnxruntime/quad_deploy/BreezySLAM |
| `SEANAV_WS` | `~/seanav_ws` | Dedicated perception workspace (keeps `~/ros2_ws` clean) |
| `SEANAV_DATA` | `~/Data/onboard_data/onnx_models/sea_nav` | ONNX model directory (contains `nav_model/` and `loco_model/`) |
| `LIDAR_PORT` / `LIDAR_BAUD` | `/dev/ttyUSB0` / `256000` | Lidar serial port and baud rate |
| `LASER_YAW_OFFSET` | `3.1415926` | Lidar mounting orientation correction (currently inverted, needs 180°) |
| `RAYS_INVERT_ANGLE` | `false` | Left-right mirror (set `true` only if still mirrored after yaw offset) |
| `POSE_X_SIGN` / `POSE_Y_SIGN` / `POSE_YAW_SIGN` | `-1.0` / `-1.0` / `1.0` | `/pose` axis sign corrections (empirically calibrated) |
| `POSE_USE_MOTION_PRIOR` | `true` | Constant-velocity motion prior; fixes forward (x) underestimation and drift. Set `false` to revert to legacy behavior |
| `POSE_MOTION_PRIOR_ALPHA` | `0.5` | EMA smoothing coefficient for velocity estimate (higher = more responsive, lower = smoother) |
| `POSE_MOTION_PRIOR_MAX_FWD_MM` | `80.0` | Per-frame forward prediction cap (mm), prevents divergence |
| `POSE_MOTION_PRIOR_MAX_DTHETA_DEG` | `15.0` | Per-frame yaw prediction cap (degrees), prevents divergence |
| `GOAL_X` / `GOAL_Y` | `1.0` / `0.0` | Goal position in the same world frame as `/pose` (origin = robot position when SLAM initializes) |

Override example:

```bash
LIDAR_PORT=/dev/ttyUSB1 bash 1_lidar.sh
GOAL_X=2.0 GOAL_Y=0.5 bash 5_controller.sh --run
```

---

## 6. Startup: five terminals

Every terminal script sources `env.sh` first (activate `sea` → init ROS2 Foxy/CycloneDDS/domain 1 → source `seanav_ws`), so just run `bash n_xxx.sh` directly — **no manual environment activation needed**.

### 6.1 Recommended: launch perception with tmux

```bash
cd ~/Projects/SEA-Nav-Code/deployment/scripts
bash start_perception_tmux.sh      # launches Terminals 1/2/3/4 in 4 tmux panes
tmux attach -t seanav_perception   # attach to view
# To stop all perception: tmux kill-session -t seanav_perception
```

Terminal 5 (controller) is **not** included in the tmux session — start it manually in a separate terminal after perception is verified and the operator is holding the gamepad.

### 6.2 Or: start manually one by one

```bash
cd ~/Projects/SEA-Nav-Code/deployment/scripts
bash 1_lidar.sh      # Terminal 1
bash 2_rays.sh       # Terminal 2
bash 3_pose.sh       # Terminal 3
bash 4_monitor.sh    # Terminal 4 (live monitor)  or  bash 4_check.sh (one-shot check)
```

### 6.3 What each terminal does

- **Terminal 1 `1_lidar.sh`**: Starts the `sllidar_ros2` driver, publishing raw `/scan` (360°, ~10 Hz, `frame_id=laser`). The script checks that the serial port exists and is writable, and prompts `chmod` if needed.
- **Terminal 2 `2_rays.sh`**: `scan_to_rays_node` downsamples `/scan` into the 41-beam `/rays` required by the control stack (`base_link`, forward ±120°, range clipped to `[0.1, 3.0]`), and applies `LASER_YAW_OFFSET` so that `ranges[20]` points straight ahead.
- **Terminal 3 `3_pose.sh`**: `breezy_lidar_odom_node` runs BreezySLAM laser odometry on `/scan` and publishes world-frame `/pose`. **Keep the robot still after startup** until the log shows `origin fixed` — at that point the current heading is set as world +X and `x≈0, y≈0, θ≈0`.
- **Terminal 4 `4_monitor.sh` / `4_check.sh`**: The former prints a live ASCII bar chart of `/rays` (useful for orientation calibration); the latter runs a one-shot health check (topic count, rate, `/rays` dimensionality, finite values, range bounds) then exits.
- **Terminal 5 `5_controller.sh`**: SEA-Nav controller. See Section 8.

> **Why launch the controller directly via script / `sea` env instead of `quad_launch.py`?** The controller subscribes to `/rays` and `/pose` via ROS2, so it must be on the **same** `ROS_DOMAIN_ID=1 + CycloneDDS` as the perception side. The `launch/sea_nav_launch_cfg.yaml` is hardcoded for simulation (`ros_env` + `~/ros2_ws`) and the domain/middleware may not match. Using `5_controller.sh` (which launches inside the `sea` env) is the safest approach.

---

## 7. Orientation calibration and verification

Before enabling motors, verify that `/rays` and `/pose` directions and signs are correct. Use Terminal 4 for live monitoring and the check script to validate dimensions.

**`/rays` orientation** (index convention: `0` = right-forward −120°, `20` = straight ahead 0°, `40` = left-forward +120°):

| Test | Expected |
|------|----------|
| Place obstacle 1 m directly ahead | `ranges[20] ≈ 1.0`, nearest point near idx 20 |
| Place obstacle to the left-forward | Nearest point idx in 20–40 |
| Place obstacle to the right-forward | Nearest point idx in 0–20 |

If the straight-ahead obstacle does not land at `ranges[20]`, adjust `LASER_YAW_OFFSET` first. If it is still left-right mirrored after the yaw offset, set `RAYS_INVERT_ANGLE=true`.

**`/pose` signs** (after `origin fixed`, robot stationary should read `x≈0, y≈0, θ≈0`):

| Test | Expected | Corresponding parameter (empirical) |
|------|----------|-------------------------------------|
| Push robot forward ~1 m | `x` increases by ~+1.0 | `POSE_X_SIGN=-1.0` |
| Push robot left laterally | `y` increases | `POSE_Y_SIGN=-1.0` |
| Rotate in place ~90° CCW | `θ ≈ +1.57` | `POSE_YAW_SIGN=1.0` |

`/pose` must also satisfy: `ros2 topic info /pose -v` shows exactly **1** publisher; `ros2 topic hz /pose` is ~10 Hz; Terminal 3 logs show `valid_bins` stably above `min_valid_bins`, `prev=True` not appearing frequently, and `dt` well below 100 ms. If `/pose` keeps jumping while the robot is completely still, try more conservative values: `POSE_SIGMA_XY_MM=10`, `POSE_SIGMA_THETA_DEG=1.5`, `POSE_MAX_SEARCH_ITER=300`.

**Forward direction (x) drift / "can't keep up"**: BreezySLAM's `RMHC_SLAM` defaults to `pose_change=(0,0,0)` when none is provided (see official `algorithms.py`), meaning the random-search start point never predicts motion. The robot's primary motion direction (forward +X) is systematically underestimated each frame. This deployment defaults to **constant-velocity motion prior** (`POSE_USE_MOTION_PRIOR=true`): uses the displacement estimated by SLAM's own internal coordinates from the previous frame to compute a `pose_change` fed back into `update()`, shifting the search start point to the predicted position. This preserves small-sigma stability when stationary while tracking forward motion. Terminal 3 logs append `vel=(.. mm/s, .. deg/s)` — the forward component should be clearly nonzero while moving. If the prior causes abnormal divergence, reduce `POSE_MOTION_PRIOR_ALPHA`, tighten `POSE_MOTION_PRIOR_MAX_FWD_MM`, or temporarily set `POSE_USE_MOTION_PRIOR=false` to compare against legacy behavior.

> `/rays` being healthy **does not imply** BreezySLAM input is healthy: `/rays` is a 41-beam downsample, while BreezySLAM uses the full-circle `/scan`. Verify them separately.

---

## 8. Controller: dry-run first, then enable motors

`5_controller.sh` defaults to **dry-run** (motors disabled, safe). Add `--run` (or `MOTOR=1`) to enable motors. The underlying command is equivalent to:

```bash
python -m quad_deploy.scripts.SEA_Nav.sea_nav_run_sdk \
    --nosimrun [--nodryrun] --data ~/Data/onboard_data/onnx_models/sea_nav \
    --goal_x <GX> --goal_y <GY>
```

**Step 1: dry-run to verify the pipeline** (make sure Terminals 1/2/3 are publishing `/scan` `/rays` `/pose`):

```bash
bash 5_controller.sh        # default dry-run, motors disabled
```

The startup banner should show `sim_run = False`, `dry_run = True`, `--> MOTOR DISABLED`, `data = .../sea_nav`, and `goal = (...)`. Expected behavior:
- Receives Go2 `rt/lowstate` without stalling at `Robot Connection`
- Press gamepad `X` to enter `human_teleop`
- With `/rays` and `/pose` fresh, press `R1` to enter `navigation`; logs show `[nav] Entering navigation state.` and periodic `[Nav] ... fresh=True`

If stuck at `Robot Connection`:
```bash
ip addr show eth0
ping <Go2_IP>
```
Confirm Go2 is connected via `eth0` (the controller uses DDS domain 0 / `eth0` to communicate with Go2).

**Step 2: enable motors** (only after Step 7 calibration and Step 1 dry-run both pass):

```bash
bash 5_controller.sh --run        # banner shows MOTOR ENABLED
# Or with a custom goal:
GOAL_X=2.0 GOAL_Y=0.0 bash 5_controller.sh --run
```

Procedure: place robot in open area → operator holds gamepad ready to press `R2`/`L2` → wait for stand-up to complete → press `X` for `human_teleop` → confirm low-speed teleoperation works → press `R1` for `navigation` → start with a short 1 m run first.

---

## 9. FSM operation (gamepad buttons)

On real hardware the FSM is driven by the gamepad (keyboard is for simulation only). State transitions:

```text
cold_start ─(stand-up done + X)─► human_teleop ─(R1 + perception fresh)─► navigation
                                       ▲                                         │
                                   R2 takeover / safe_stop recovery              │
                                       └──────── safe_stop ◄── perception stale > 1s
any state ─L2─► emergency ─L1─► recovery ─(stand-up done + X)─► human_teleop
```

| Button | Action |
|--------|--------|
| `X` | Enter `human_teleop` after stand-up completes |
| `R1` | `human_teleop` → `navigation` (requires `/rays` and `/pose` both fresh) |
| `R2` | Return to human teleoperation from `navigation` / `safe_stop` |
| `L2` | Any state → `emergency` (motors off) |
| `L1` | `emergency` → `recovery` |

In `navigation`, if `/rays` or `/pose` remains stale for more than 1 s, the FSM automatically enters `safe_stop` (forces zero commands every tick, robot stays standing). After perception recovers and is stable for 0.5 s, **press `R1`** to resume navigation (it does not auto-resume).

**Safety stop fault injection test**: while in `navigation`, kill Terminals 1 and 3 (stop `/scan` → `/rays`/`/pose`). The robot should enter `safe_stop` within ~1 s. Restore publishing, then press `R1` to resume.

---

## 10. Pre-run checklist

- [ ] All terminals initialized through `env.sh` (log shows `[env] ... ROS_DOMAIN_ID=1  RMW=rmw_cyclonedds_cpp`)
- [ ] `python -c "import rclpy"` works; `ros_base` / `unitree_sdk2py` / `quad_deploy` SEA-Nav all importable
- [ ] ONNX files at `~/Data/onboard_data/onnx_models/sea_nav/{nav_model,loco_model}/model.onnx`, dims nav=550 / loco=235
- [ ] `/scan` ~10 Hz; `/rays` ~10 Hz, 41 beams, `[0.1, 3.0]`, `ranges[20]` points straight ahead
- [ ] `/pose` ~10 Hz, single publisher, forward/lateral/yaw signs correct, `origin fixed` seen in logs
- [ ] Go2 reachable via `eth0`; dry-run receives `rt/lowstate`, can enter `navigation` with `fresh=True`
- [ ] Operator familiar with `R2` takeover and `L2` emergency stop

---

## 11. Troubleshooting

| Symptom | What to check |
|---------|--------------|
| `ros2` command missing / topics invisible to each other | Did `env.sh` run? Are `ROS_DOMAIN_ID=1` and `RMW=rmw_cyclonedds_cpp` consistent across all terminals? |
| `import rclpy` fails | The `sea` env's Python doesn't match ROS2 Foxy bindings; fix the `sea` environment |
| `ModuleNotFoundError: ros_base / unitree_sdk2py` | Package not installed with `pip install -e .`, or in a different Python env |
| `Could not locate cyclonedds` | Build CycloneDDS, `export CYCLONEDDS_HOME`, then reinstall Unitree SDK (see 3.2) |
| No `/scan` | Check `/dev/ttyUSB0` permissions, wiring, baud rate (256000); follow prompts from `bash 1_lidar.sh` |
| `/scan` present but no `/rays` | `seanav_perception` not built/sourced; run `bash build_perception.sh` and reopen the terminal |
| `No executable found` | Missing `setup.cfg` means scripts weren't installed to `lib/seanav_perception/`; rebuild |
| `/rays[20]` not pointing straight ahead | Adjust `LASER_YAW_OFFSET` first; if still mirrored set `RAYS_INVERT_ANGLE=true` |
| `/pose` keeps jumping | `ros2 topic info /pose -v` must show exactly 1 publisher; kill any extra publisher processes |
| `malloc(): invalid size` | A2M12 point count exceeds Breezy's `Laser(360)`; this node already fixes it to `breezy_scan_size=360` — confirm you are using the node from this repo |
| Stuck at `Robot Connection` | Check Go2 network / `eth0` / Unitree DDS (domain 0); `ping <Go2_IP>` |
| Cannot enter `navigation` | `/rays` or `/pose` is stale; `ros2 topic hz /rays /pose` |
| Abnormal movement after entering `navigation` | Press `R2` or `L2` immediately; revert to dry-run and recheck `/rays`, `/pose`, ONNX path |

---

## Appendix A: Directory structure

```text
SEA-Nav-Code/deployment/
├── README.md                              # Chinese guide
├── README_en.md                           # This document
├── perception/
│   └── seanav_perception/                 # ROS2 publisher-side adapter package (ament_python)
│       ├── package.xml                    # Package metadata and dependencies
│       ├── setup.py                       # Registers 3 console_scripts nodes
│       ├── setup.cfg                      # ament_python install path
│       ├── resource/
│       │   └── seanav_perception          # ament resource index marker
│       └── seanav_perception/
│           ├── __init__.py
│           ├── scan_to_rays_node.py        # /scan -> /rays
│           ├── breezy_lidar_odom_node.py   # /scan -> /pose (BreezySLAM)
│           └── rays_monitor_node.py        # /rays live monitor
└── scripts/
    ├── config.sh                          # All tunable parameters (overridable via env vars)
    ├── env.sh                             # Common env: conda + ROS2 Foxy/CycloneDDS/domain1 + seanav_ws
    ├── build_perception.sh                # One-shot build: clone sllidar + symlink this package + colcon build
    ├── 1_lidar.sh                         # Terminal 1: /scan
    ├── 2_rays.sh                          # Terminal 2: /rays
    ├── 3_pose.sh                          # Terminal 3: /pose
    ├── 4_monitor.sh                       # Terminal 4: live monitor
    ├── 4_check.sh                         # Terminal 4: one-shot health check
    ├── 5_controller.sh                    # Terminal 5: SEA-Nav controller (dry-run by default)
    └── start_perception_tmux.sh           # One-shot launcher for Terminals 1–4 (tmux)
```

---

## Appendix B: Known issues and solutions

### Odometry drift: forward direction (x) "can't keep up" + y axis reversed

**Symptom**: After `origin fixed`, when the robot primarily moves forward (+X), `/pose`'s `x` is severely inaccurate and lags far behind ("can't keep up"), while `y` and `yaw` are roughly correct. Additionally, `y` is globally reversed.

**Root cause (confirmed against BreezySLAM official `python/breezyslam/algorithms.py`)**:

`RMHC_SLAM.update()` defaults to `(0, 0, 0)` when `pose_change` is not provided:

```python
def update(self, scans_mm, pose_change=None, ...):
    if not pose_change:
        pose_change = (0, 0, 0)
```

`pose_change` determines the **start point** of the RMHC hill-climbing search (advance `dxy_mm` along current heading, then rotate `dtheta`):

```python
start_pos.x_mm += dxy_mm * cos(theta)
start_pos.y_mm += dxy_mm * sin(theta)
start_pos.theta_degrees += dtheta_degrees
new_position = self._getNewPosition(start_pos)   # random search near start_pos
```

The original deployment called `update(scan_mm)` without a `pose_change` → the search start always equals the previous-frame position, **never predicting motion**. The robot mainly moves along its heading (= forward +X), so each frame relies on `sigma_xy_mm=20mm` random search to "feel out" ~30 mm of displacement (0.3 m/s ÷ 10 Hz), causing systematic underestimation and lag in the forward direction. `y` and `yaw` barely change, so the small random search locks onto them fine — hence "good enough." This also explains the tension between "small sigma for stationary stability" and "large sigma to track motion."

**Fix: constant-velocity motion prior (enabled by default)**

In `breezy_lidar_odom_node.py`, the previous frame's SLAM-internal displacement is used to estimate forward/yaw velocity (EMA smoothed), which then computes `pose_change = (forward_mm, delta_deg, dt)` fed back into `update()`, shifting the search start to the predicted position:

- Preserves small `sigma` (stationary stability) while tracking forward motion (no more x lag)
- Per-frame caps (`max_fwd_mm` / `max_dtheta_deg`) prevent feedback divergence
- All estimation is done in BreezySLAM's internal coordinate frame, then passed through the existing base_link/world-frame transform

Related parameters (defaults, all overridable in `config.sh`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `POSE_USE_MOTION_PRIOR` | `true` | Master switch; set `false` to revert to `(0,0,0)` for comparison |
| `POSE_MOTION_PRIOR_ALPHA` | `0.5` | Velocity EMA coefficient (higher = more responsive, lower = smoother) |
| `POSE_MOTION_PRIOR_MAX_FWD_MM` | `80.0` | Per-frame forward prediction cap (mm) |
| `POSE_MOTION_PRIOR_MAX_DTHETA_DEG` | `15.0` | Per-frame yaw prediction cap (degrees) |

Terminal 3 debug logs now append `vel=(.. mm/s, .. deg/s)`. The forward component should be clearly nonzero while moving, confirming the prior is active.

**y-axis reversal**: unrelated to the motion prior; this is an axis sign calibration issue. The empirically correct value is `POSE_Y_SIGN=-1.0` (set as default), meaning "push robot left → `pose.y` increases."

**Key references**:

- BreezySLAM `RMHC_SLAM` / `CoreSLAM.update` `pose_change` semantics and default `(0,0,0)`: https://github.com/simondlevy/BreezySLAM/blob/master/python/breezyslam/algorithms.py
- BreezySLAM basic interface `slam.update(scan)`: https://github.com/simondlevy/BreezySLAM
- Slamtec `sllidar_ros2` (A2M12 defaults to `angle_compensate=true`; lidar frame extrinsics must be handled separately): https://github.com/Slamtec/sllidar_ros2

---

*Based on `quad_deploy@feat/sea-nav-agent` latest code. Real-robot adapter code and scripts are maintained in this directory; the control stack itself is not modified.*
