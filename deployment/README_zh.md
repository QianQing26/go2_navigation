# SEA-Nav 真机部署（Unitree Go2）

本目录是 SEA-Nav 在 Unitree Go2 上的真机部署工作区（colcon workspace）：将激光雷达数据整理为控制栈订阅的 `/rays` 与 `/pose` 两个话题，并启动 `quad_deploy` 中已经 Sim2Sim 验证的 SEA-Nav 控制栈（Nav + Loco + 安全停机 FSM）。真机适配全部在本目录完成，不修改 `quad_deploy`。

传感器：RPLIDAR A2M12（USB 串口）；里程计：BreezySLAM 激光里程计。

## 数据链路

| 组件 | 所属 | 订阅 | 发布 |
| --- | --- | --- | --- |
| `sllidar_node` | `sllidar_ros2` | 雷达串口 | `/scan`（360°，~10 Hz） |
| `scan_to_rays` | 本包 | `/scan` | `/rays`（41 束，`base_link` 系，±120°，[0.1, 3.0] m） |
| `lidar_odom` | 本包 | `/scan` | `/pose`（世界系 `x, y, θ`，原点为启动位姿） |
| `rays_monitor` | 本包 | `/rays` | 终端可视化（只读，供校验） |
| `controller` | 本包 → `quad_deploy` | `/rays`、`/pose` | Go2 `rt/lowcmd` |

`controller` 启动时先与感知链路握手（等待 `/rays` 与 `/pose` 持续发布），随后运行 `quad_deploy.scripts.sea.sea_run_sdk`。`/rays`、`/pose` 的规格由 `SEANavAgentCfg` 约定，两侧保持一致。

## 依赖

| 仓库 | 位置 | 用途 |
| --- | --- | --- |
| [quad_deploy](https://github.com/11chens/quad_deploy) | `~/Projects/quad_deploy` | SEA-Nav 控制栈（安装见其 README） |
| [ros_base](https://github.com/11chens/ros_base) | `~/Projects/ros_base` | manager / node / launcher 基类 |
| [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | `~/Projects/unitree_sdk2_python` | Go2 底层通信 |
| [BreezySLAM](https://github.com/simondlevy/BreezySLAM) | `~/Projects/BreezySLAM` | 激光里程计 |
| [sllidar_ros2](https://github.com/Slamtec/sllidar_ros2) | 本工作区 `src/`（`build.sh` 自动获取） | 雷达驱动 |

前提：机器人已按 `quad_deploy` README 完成控制栈及其依赖的安装，ONNX 模型已放置于 `~/Data/onboard_data/onnx_models/sea_nav/`（`nav_model/model.onnx` 与 `loco_model/model.onnx`，下载方式见 `quad_deploy` README）。

## 安装

```bash
# BreezySLAM（一次性，装入部署用 conda 环境）
cd ~/Projects && git clone https://github.com/simondlevy/BreezySLAM.git
cd BreezySLAM/python && python -m pip install .

# 构建本工作区
cd ~/Projects/SEA-Nav-Code/deployment
bash build.sh
```

机器路径或环境名与默认不符时，先修改 `env.sh` 顶部变量（conda 环境名、ROS2 路径、DDS 域等）。更新代码后重新执行 `bash build.sh`。

## 配置

- `env.sh`：机器级环境（conda、ROS2、CycloneDDS、`ROS_DOMAIN_ID`），每个终端 `source` 一次。
- `src/seanav_deploy/seanav_deploy/config.py`：全部运行参数（雷达串口、安装朝向 `yaw_offset_rad`、`/pose` 轴向符号、导航目标点 `goal_x / goal_y` 等）。直接修改后重启对应节点即可生效（`--symlink-install`），无需重新构建。

## 启动

一键拉起（与 `quad_deploy` 相同的 launcher，tmux 编排）：

```bash
cd ~/Projects/SEA-Nav-Code/deployment
python launch/deploy_launch.py        # PERCEPTION + RAYS_MONITOR + CONTROL_DRY（不开电机）
tmux attach -t seanav_deploy
```

按「校验」一节逐项确认后，开电机运行：

```bash
python launch/deploy_launch.py --enable CONTROL_REAL --disable CONTROL_DRY
```

调试单个组件时，也可在任意终端 `source env.sh` 后单独运行 `ros2 launch seanav_deploy perception.launch.py`、`ros2 run seanav_deploy rays_monitor`、`ros2 run seanav_deploy controller [--run]`。

注意：`lidar_odom` 启动后保持机器人静止，直到日志出现 `origin fixed`（此刻位姿被定为世界系原点与 +X 朝向）。

## 校验

开电机前逐项核对：

| 校验项 | 方法 | 期望 |
| --- | --- | --- |
| 话题频率 | `ros2 topic hz /rays`（`/scan`、`/pose` 同理） | 均约 10 Hz |
| `/rays` 维度 | `RAYS_MONITOR` 窗格 | `count=41`，值在 [0.1, 3.0] |
| `/rays` 朝向 | 正前方 1 m 处放障碍 | `idx=20` 读数 ≈ 1.0，最近点在 idx 20 附近 |
| `/rays` 左右 | 左前 / 右前放障碍 | 最近点 idx 分别落在 20–40 / 0–20 |
| `/pose` 前向 | 手推前进约 1 m | `x` 增大约 +1.0 |
| `/pose` 侧向 | 手推向左平移 | `y` 增大 |
| `/pose` 转向 | 原地逆时针转 90° | `θ` ≈ +1.57 |

不符时修改 `config.py`：`/rays` 朝向调 `LidarCfg.yaw_offset_rad`（仍左右镜像再置 `RaysCfg.invert_angle`）；`/pose` 符号调 `PoseCfg.x_sign / y_sign / yaw_sign`。

## 操作（手柄状态机）

`CONTROL_DRY` 默认 dry-run（电机不使能），确认握手通过、能收到 `rt/lowstate`、可进入 `navigation` 后，再切换 `CONTROL_REAL`。

```text
cold_start ─(站立完成 + X)─► human_teleop ─(R1 且感知 fresh)─► navigation
                                 ▲                                │
                             R2 接管                     感知 stale > 1 s
                                 └────────── safe_stop ◄──────────┘
任意状态 ─L2─► emergency ─L1─► recovery ─(站立完成 + X)─► human_teleop
```

| 手柄键 | 作用 |
| --- | --- |
| `X` | 站立完成后进入 `human_teleop` |
| `R1` | `human_teleop` → `navigation`（要求 `/rays`、`/pose` fresh） |
| `R2` | `navigation` / `safe_stop` → 人工接管 |
| `L2` | 任意状态 → `emergency`（关电机） |
| `L1` | `emergency` → `recovery` |

`navigation` 中感知持续丢失超过 1 s 自动进入 `safe_stop`（强制零指令、保持站立）；感知恢复后需再按 `R1` 恢复导航，不自动恢复。

## 常见问题

| 现象 | 排查 |
| --- | --- |
| 话题互相看不到 | 各终端是否 `source env.sh`（`ROS_DOMAIN_ID` 与 RMW 需一致） |
| 没有 `/scan` | 串口权限：`sudo chmod 666 /dev/ttyUSB0`；接线与 `LidarCfg.port` |
| `No executable found` | 未构建或未 `source env.sh`；重跑 `bash build.sh` |
| `/pose` 跳动 | `ros2 topic info /pose -v` 应只有 1 个 publisher；收紧 `PoseCfg.sigma_*` |
| 控制器停在 `[handshake] waiting` | 感知链路未启动或未发布，查看 `PERCEPTION` 窗格 |
| 控制器停在 `Robot Connection` | Go2 网络未连通：检查 `eth0` 网线，`ping` Go2 的 IP |
| `import numpy` / `onnxruntime` 失败 | ROS2 Foxy 环境为 Python 3.8，需 `numpy<1.25`、`onnxruntime<1.20` |

## 目录结构

```text
deployment/                        # colcon workspace
├── README_zh.md / README_en.md
├── env.sh                         # 机器级环境（source 使用）
├── build.sh                       # 获取雷达驱动源码 + colcon build
├── launch/
│   ├── deploy_launch.py           # ros_base BaseLauncher 入口
│   └── deploy_cfg.yaml            # tmux 节点编排（含 CONTROL_DRY / CONTROL_REAL）
└── src/
    └── seanav_deploy/             # ROS2 包（ament_python）
        ├── launch/perception.launch.py
        └── seanav_deploy/
            ├── config.py          # 全部运行参数
            ├── scan_to_rays_node.py
            ├── lidar_odom_node.py
            ├── rays_monitor_node.py
            └── controller.py      # 感知握手 + 启动 quad_deploy 控制栈
```
