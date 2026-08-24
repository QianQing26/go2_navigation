# SEA-Nav Real-Robot Deployment (Unitree Go2)

Colcon workspace for running SEA-Nav on a real Unitree Go2. It publishes the two topics the control stack consumes — `/rays` (lidar beams) and `/pose` (lidar odometry) — and launches the [quad_deploy](https://github.com/11chens/quad_deploy) control stack (Nav + Loco + safe-stop FSM) unmodified.

Hardware: RPLIDAR A2M12 (USB serial). Odometry: [BreezySLAM](https://github.com/simondlevy/BreezySLAM).

## Data Pipeline

| Component | Package | Subscribes | Publishes |
| --- | --- | --- | --- |
| `sllidar_node` | `sllidar_ros2` | lidar serial port | `/scan` (360°, ~10 Hz) |
| `scan_to_rays` | this package | `/scan` | `/rays` (41 beams, `base_link` frame, ±120°, [0.1, 3.0] m) |
| `lidar_odom` | this package | `/scan` | `/pose` (world-frame `x, y, θ`, origin at startup pose) |
| `rays_monitor` | this package | `/rays` | terminal visualization (read-only) |
| `controller` | this package → `quad_deploy` | `/rays`, `/pose` | Go2 `rt/lowcmd` |

The `controller` waits until `/rays` and `/pose` are streaming (perception handshake), then starts `quad_deploy.scripts.sea.sea_run_sdk`. Both topics follow the specification in `SEANavAgentCfg`.

## Dependencies

| Repository | Location | Purpose |
| --- | --- | --- |
| [quad_deploy](https://github.com/11chens/quad_deploy) | `~/Projects/quad_deploy` | SEA-Nav control stack (see its README for installation) |
| [ros_base](https://github.com/11chens/ros_base) | `~/ros_base` | manager / node / launcher base classes |
| [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | `~/unitree_sdk2_python` | Go2 low-level communication |
| [BreezySLAM](https://github.com/simondlevy/BreezySLAM) | `~/Projects/BreezySLAM` | lidar odometry |
| [sllidar_ros2](https://github.com/Slamtec/sllidar_ros2) | this workspace `src/` (fetched by `build.sh`) | lidar driver |

Before starting, install the control stack and its dependencies following the `quad_deploy` README, and place the ONNX models under `~/Data/onboard_data/onnx_models/sea_nav/` (`nav_model/model.onnx` and `loco_model/model.onnx`).

## Installation

```bash
# BreezySLAM (one-time, into the deployment conda environment)
cd ~/Projects && git clone https://github.com/simondlevy/BreezySLAM.git
cd BreezySLAM/python && python -m pip install .

# Build this workspace
cd ~/Projects/SEA-Nav-Code/deployment
bash build.sh
```

If your paths or environment names differ from the defaults, edit the variables at the top of `env.sh` first. Re-run `bash build.sh` after pulling new code.

## Configuration

- `env.sh` — machine-level environment (conda, ROS2, CycloneDDS, `ROS_DOMAIN_ID`); `source` it once per terminal.
- `src/seanav_deploy/seanav_deploy/config.py` — all runtime parameters: lidar serial port, mounting orientation (`yaw_offset_rad`), `/pose` axis signs, navigation goal (`goal_x` / `goal_y`), etc. Edits take effect after restarting the corresponding node (`--symlink-install`); no rebuild needed.

## Usage

### 1. Launch in dry-run mode

```bash
cd ~/Projects/SEA-Nav-Code/deployment
python launch/deploy_launch.py        # PERCEPTION + RAYS_MONITOR + CONTROL_DRY (motors disabled)
tmux attach -t seanav_deploy
```

Keep the robot stationary until the `lidar_odom` log prints `origin fixed` — the current pose becomes the world-frame origin and +X direction.

To debug a single component, `source env.sh` in any terminal and run it directly: `ros2 launch seanav_deploy perception.launch.py`, `ros2 run seanav_deploy rays_monitor`, or `ros2 run seanav_deploy controller [--run]`.

### 2. Verify before enabling motors

With the dry-run session up, check every item:

| Check | Method | Expected |
| --- | --- | --- |
| Topic rates | `ros2 topic hz /rays` (same for `/scan`, `/pose`) | all ~10 Hz |
| `/rays` dimension | `RAYS_MONITOR` pane | `count=41`, values within [0.1, 3.0] |
| `/rays` heading | place an obstacle 1 m straight ahead | `idx=20` reads ≈ 1.0, nearest point near idx 20 |
| `/rays` left/right | place an obstacle front-left / front-right | nearest point idx falls in 20–40 / 0–20 |
| `/pose` forward | push the robot forward ~1 m | `x` increases by ~+1.0 |
| `/pose` lateral | push the robot to the left | `y` increases |
| `/pose` heading | rotate in place 90° counterclockwise | `θ` ≈ +1.57 |
| Controller handshake | `CONTROL_DRY` pane | handshake passes, `rt/lowstate` received, `navigation` reachable |

If a check fails, edit `config.py`: adjust `LidarCfg.yaw_offset_rad` for `/rays` heading (also set `RaysCfg.invert_angle` if still mirrored), or `PoseCfg.x_sign / y_sign / yaw_sign` for `/pose` signs. Then restart the node and re-check.

### 3. Run with motors enabled

Only after every check above passes:

```bash
python launch/deploy_launch.py --enable CONTROL_REAL --disable CONTROL_DRY
```

## Gamepad Controls

```text
cold_start ─(stand complete + X)─► human_teleop ─(R1, perception fresh)─► navigation
                                       ▲                                      │
                                  R2 takeover                    perception stale > 1 s
                                       └──────────── safe_stop ◄──────────────┘
any state ─L2─► emergency ─L1─► recovery ─(stand complete + X)─► human_teleop
```

| Button | Action |
| --- | --- |
| `X` | enter `human_teleop` after standing completes |
| `R1` | `human_teleop` → `navigation` (requires fresh `/rays` and `/pose`) |
| `R2` | `navigation` / `safe_stop` → manual takeover |
| `L2` | any state → `emergency` (motors off) |
| `L1` | `emergency` → `recovery` |

During `navigation`, if perception stays stale for more than 1 s the robot enters `safe_stop` (zero command, keeps standing). After perception recovers, press `R1` to resume — navigation does not resume automatically.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Topics not visible across terminals | every terminal has sourced `env.sh` (`ROS_DOMAIN_ID` and RMW must match) |
| No `/scan` | serial port permission: `sudo chmod 666 /dev/ttyUSB0`; wiring and `LidarCfg.port` |
| `No executable found` | workspace not built or `env.sh` not sourced; re-run `bash build.sh` |
| `/pose` jitters | `ros2 topic info /pose -v` must show exactly 1 publisher; tighten `PoseCfg.sigma_*` |
| Controller stuck at `[handshake] waiting` | perception pipeline not started or not publishing; check the `PERCEPTION` pane |
| Controller stuck at `Robot Connection` | Go2 network unreachable: check the `eth0` cable, `ping` the Go2 IP |
| `import numpy` / `onnxruntime` fails | ROS2 Foxy runs Python 3.8; requires `numpy<1.25`, `onnxruntime<1.20` |

## Directory Layout

```text
deployment/                        # colcon workspace
├── README.md
├── env.sh                         # machine-level environment (to source)
├── build.sh                       # fetch lidar driver source + colcon build
├── launch/
│   ├── deploy_launch.py           # ros_base BaseLauncher entry
│   └── deploy_cfg.yaml            # tmux node orchestration (CONTROL_DRY / CONTROL_REAL)
└── src/
    └── seanav_deploy/             # ROS2 package (ament_python)
        ├── launch/perception.launch.py
        └── seanav_deploy/
            ├── config.py          # all runtime parameters
            ├── scan_to_rays_node.py
            ├── lidar_odom_node.py
            ├── rays_monitor_node.py
            └── controller.py      # perception handshake + start quad_deploy control stack
```
