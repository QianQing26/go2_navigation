# SEA-Nav 真机部署指南（Go2 + RPLIDAR A2M12 + BreezySLAM）

本指南面向第一次在 Unitree Go2 上部署 **SEA-Nav 自主导航**的读者，从零开始介绍环境配置、仓库连接、SLAM 标定与终端启动。所有命令均以 `quad_deploy` 仓库 `feat/sea-nav-agent` 分支的最新代码为准。

> **设计原则**：`quad_deploy` 内的控制栈（Nav + Loco + 安全停机 FSM）在 Sim2Sim 阶段已验证通过，真机部署**不修改** `quad_deploy`。真机适配全部放在“发布端”——把真实雷达与 SLAM 的输出整理成控制栈已经订阅的 `/rays` 和 `/pose` 两个话题。本目录提供发布端代码（`perception/`）、一键启动脚本（`scripts/`）与本说明文档。

## 0. 一分钟速览


| 终端   | 启动脚本                          | 职责                         | 产出                        |
| ---- | ----------------------------- | -------------------------- | ------------------------- |
| 终端 1 | `1_lidar.sh`                  | RPLIDAR A2M12 驱动           | `/scan`（360° 原始激光）        |
| 终端 2 | `2_rays.sh`                   | `/scan` → `/rays` 下采样适配    | `/rays`（41 束，`base_link`） |
| 终端 3 | `3_pose.sh`                   | `/scan` → BreezySLAM 激光里程计 | `/pose`（世界系 `x,y,θ`）      |
| 终端 4 | `4_monitor.sh` / `4_check.sh` | 实时监测 / 一次性体检               | 终端打印                      |
| 终端 5 | `5_controller.sh`             | SEA-Nav 控制器（Nav+Loco+FSM）  | Go2 `rt/lowcmd`           |


终端 1–4 是“感知发布端”，可用 `start_perception_tmux.sh` 一键拉起；终端 5 是控制器，确认感知正常后再单独手动启动。

---

## 1. 数据流总览

```text
RPLIDAR A2M12
   │  (终端 1: sllidar_ros2)
   ▼
 /scan  (sensor_msgs/LaserScan, 360°, ~10 Hz, frame=laser)
   ├──────────────► (终端 2: scan_to_rays_node) ──► /rays (LaserScan, 41 束, base_link, 10 Hz)
   └──────────────► (终端 3: breezy_lidar_odom_node) ──► /pose (Pose2D, 世界系, 10 Hz)

 /rays + /pose
   │  (终端 5: sea_nav_run_sdk.py，ROS2 域 1 + CycloneDDS 订阅)
   ▼
 SEA-Nav Nav 策略 (obs 550 → vx,vy,vyaw)  →  SEA-Nav Loco 策略 (obs 235 → 12 关节 + vel_pred)
   │  (Unitree SDK，DDS 域 0 / eth0)
   ▼
 Unitree Go2  (rt/lowstate 读取，rt/lowcmd 下发)
```

控制栈对外只依赖两个话题，规格固定如下（由 `SEA_Nav_NavAgentCfg` 决定，**不要改 `quad_deploy`**）：


| 话题      | 类型                          | 频率    | 内容                                                |
| ------- | --------------------------- | ----- | ------------------------------------------------- |
| `/rays` | `sensor_msgs/msg/LaserScan` | 10 Hz | 41 束，`base_link`，前向 240°（±120°），量程 `[0.1, 3.0]` m |
| `/pose` | `geometry_msgs/msg/Pose2D`  | 10 Hz | 局部世界系下机器人位姿 `(x, y, θ)`                           |


> **两条独立的 DDS 通道**：`/scan`、`/rays`、`/pose` 走 ROS2（`ROS_DOMAIN_ID=1` + CycloneDDS）；Go2 本体通信由 `sea_nav_run_sdk.py` 内部 `ChannelFactoryInitialize(0, "eth0")` 单独建立（DDS 域 0，网口 `eth0`）。两者互不影响，不要混为一谈。

---

## 2. 前提条件

**硬件**：Unitree Go2、RPLIDAR A2M12（USB 转串口，默认 `/dev/ttyUSB0`，波特率 256000）、手柄、机器人与 Go2 之间的 `eth0` 网线连通。

**软件（机器人侧需要安装）**：


| 仓库 / 包                | 建议位置                                | 用途                                      |
| --------------------- | ----------------------------------- | --------------------------------------- |
| `quad_deploy`         | `~/Projects/quad_deploy`            | 控制主程序，运行 `sea_nav_run_sdk.py`           |
| `ros_base`            | `~/Projects/ros_base`               | `quad_deploy` 依赖的 manager/node/agent 基类 |
| `unitree_sdk2_python` | `~/Projects/unitree_sdk2_python`    | 订阅 `rt/lowstate`、发布 `rt/lowcmd`         |
| `cyclonedds`          | `~/Projects/cyclonedds`             | `unitree_sdk2_python` 的通信依赖             |
| `sllidar_ros2`        | `~/seanav_ws/src/sllidar_ros2`      | RPLIDAR 驱动，发布 `/scan`（由构建脚本自动 clone）    |
| `BreezySLAM`          | `~/Projects/BreezySLAM`             | 激光里程计，供 `/pose` 使用                      |
| `seanav_perception`   | `~/seanav_ws/src/seanav_perception` | 本目录提供的发布端适配包（由构建脚本软链接进去）                |


**不需要装到机器人上**：`SEA-Nav-Code` 训练仓库（只需导出的 ONNX）、Isaac Gym / `legged_gym`、`unitree_mujoco_ros`（仅 Sim2Sim 用）。

---

## 3. 一次性环境配置

下面的步骤每台机器人只需做一次。如果代码已经从开发机同步过来（见第 4 节），可跳过对应的 `git clone`，直接安装。

### 3.1 Conda 环境与 ROS2

真机统一使用一个 Conda 环境（本指南记为 `sea`，名字可自定），该环境需具备：`rclpy`（ROS2 Foxy 的 Python 绑定）、`onnxruntime`、`numpy`、以及下文安装的 `quad_deploy` / `ros_base` / `unitree_sdk2py` / `BreezySLAM`。ROS2 侧使用 **Foxy + CycloneDDS + `ROS_DOMAIN_ID=1`**，即整套环境等价于：

```bash
conda activate sea
source /opt/ros/foxy/setup.bash
source ~/cyclonedds_ws/install/setup.bash          # CycloneDDS overlay（见 3.2）
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=1
```

> **版本约束（重要）**：ROS2 Foxy 的 `rclpy` 绑定基于 **Python 3.8**，因此 `sea` 环境必须是 Python 3.8，且 `numpy` / `onnxruntime` 不能用过新版本（否则没有 3.8 的 wheel、无法 `import`）：
>
> | 包             | 建议版本             | 原因                                            |
> | ------------- | ---------------- | --------------------------------------------- |
> | `numpy`       | `>=1.20,<1.25`   | numpy 1.25+ 要求 Python ≥3.9；1.24.x 是支持 3.8 的最后一支 |
> | `onnxruntime` | `>=1.12,<1.20`   | onnxruntime 1.20+ 要求 Python ≥3.10；1.19.2 是支持 3.8 的最后一支 |
>
> 安装示例：`pip install "numpy>=1.20,<1.25" "onnxruntime>=1.12,<1.20"`。`quad_deploy` 与 `seanav_perception` 的 `setup.py` 均未钉版本，若 pip 自动拉到过新版本，按上表降级即可。

本目录的 `scripts/env.sh` 已经把上面这一整套（激活 conda → source ROS2 Foxy → source CycloneDDS → 锁定 RMW 与域）封装好，所有终端脚本都会自动 `source` 它，因此**你不需要手动敲这些命令**。各路径/名字集中在 `scripts/config.sh`（见第 5 节），与你机器上的实际情况不符时改那里即可，例如 conda 环境不叫 `sea`、ROS2 不在 `/opt/ros/foxy`、CycloneDDS 工作区不在 `~/cyclonedds_ws`：

```bash
SEANAV_CONDA_ENV=myenv SEANAV_ROS_SETUP=/opt/ros/foxy/setup.bash bash 1_lidar.sh
```

> 若 `conda activate sea` 后 `python -c "import rclpy"` 失败，说明该环境的 Python 与系统 ROS2 Foxy 的绑定不匹配。优先修复该环境直至能稳定 `import rclpy`，不要为此新建第二套 ROS 环境。

### 3.2 安装 `quad_deploy` 依赖链

```bash
conda activate sea
# ros_base
cd ~/Projects && git clone https://github.com/11chens/ros_base.git
cd ~/Projects/ros_base && pip install -e .

# unitree_sdk2_python
cd ~/Projects && git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd ~/Projects/unitree_sdk2_python && pip install -e .
```

若安装 `unitree_sdk2_python` 报 `Could not locate cyclonedds`，先编译 CycloneDDS 再重装：

```bash
cd ~/Projects && git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd ~/Projects/cyclonedds && mkdir -p build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install -j"$(nproc)"

export CYCLONEDDS_HOME=~/Projects/cyclonedds/install
echo 'export CYCLONEDDS_HOME=~/Projects/cyclonedds/install' >> ~/.bashrc
cd ~/Projects/unitree_sdk2_python && pip install -e .
```

最后 clone 并安装 `quad_deploy`（使用 `feat/sea-nav-agent` 分支），再验证 SEA-Nav 入口可导入：

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

### 3.3 准备 ONNX 模型

统一 Loco 之后，**Nav 与 Loco 两个模型都放在同一目录** `~/Data/onboard_data/onnx_models/sea_nav/` 下（即 `sea_nav_run_sdk.py` 的 `--data` 默认值）：

```text
~/Data/onboard_data/onnx_models/sea_nav/
├── nav_model/model.onnx     # 高层导航，输入 550，输出 3 维 (vx, vy, vyaw)
└── loco_model/model.onnx    # 低层运控，输入 235，输出 12 关节 + vel_pred(3)
```

从开发机同步（路径按实际情况改）：

```bash
mkdir -p ~/Data/onboard_data/onnx_models/sea_nav
rsync -av --progress \
  <dev-pc>:~/Data/onboard_data/onnx_models/sea_nav/ \
  ~/Data/onboard_data/onnx_models/sea_nav/
```

校验维度（应为 nav 输入 550、loco 输入 235）：

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

### 3.4 雷达驱动、BreezySLAM 与感知适配包

先装 BreezySLAM（Python/C 扩展，装进 `sea` 环境）：

```bash
sudo apt update && sudo apt install -y git build-essential python3-dev tmux
cd ~/Projects && git clone https://github.com/simondlevy/BreezySLAM.git
cd ~/Projects/BreezySLAM/python && python -m pip install .
python -c "from breezyslam.algorithms import RMHC_SLAM; print('BreezySLAM OK')"
```

然后用本目录的构建脚本一键完成：clone `sllidar_ros2`、把 `seanav_perception` 软链接进 `~/seanav_ws/src`、`colcon build` 两个包：

```bash
cd ~/Projects/SEA-Nav-Code/deployment/scripts
bash build_perception.sh
ros2 pkg executables seanav_perception   # 应列出 3 个可执行节点
```

> 软链接方式意味着**本仓库是发布端代码的唯一来源**：以后改了 `perception/` 里的节点，重跑 `build_perception.sh` 即可生效，无需手动拷贝。

完成后插上雷达并确认串口权限：

```bash
ls -l /dev/ttyUSB0           # 确认设备存在
sudo chmod 666 /dev/ttyUSB0  # 若不可写
```

---

## 4. 代码与仓库连接（开发机 ↔ 机器人）

如果开发机上的代码有未提交的本地改动，推荐用 `rsync` 同步到机器人（比 `scp -r` 更适合反复同步）。常用命令：

```bash
# 同步 quad_deploy（排除缓存）
rsync -av --progress --exclude '__pycache__' --exclude '*.pyc' \
  ~/Projects/quad_deploy/  unitree@<robot_ip>:~/Projects/quad_deploy/

# 同步 ros_base（如有本地改动）
rsync -av --progress --exclude '__pycache__' \
  ~/Projects/ros_base/  unitree@<robot_ip>:~/Projects/ros_base/

# 同步本部署目录（发布端代码 + 脚本 + 文档）
rsync -av --progress \
  ~/Projects/SEA-Nav-Code/deployment/  unitree@<robot_ip>:~/Projects/SEA-Nav-Code/deployment/

# 同步 ONNX 模型
rsync -av --progress \
  ~/Data/onboard_data/onnx_models/sea_nav/  unitree@<robot_ip>:~/Data/onboard_data/onnx_models/sea_nav/
```

> 同步源码后，务必在机器人上重新执行对应的 `pip install -e .`（Python 包）或 `bash build_perception.sh`（ROS2 包），否则不一定能加载到最新代码。NoMachine 适合看图形界面、复制少量文本，代码与模型一律用 `rsync`/`scp`。

---

## 5. 参数集中配置（`scripts/config.sh`）

所有可调参数集中在 `scripts/config.sh`，每个值都支持从外层环境变量覆盖（无需改脚本本身）。常用项：


| 变量                                              | 默认                                        | 说明                                                    |
| ----------------------------------------------- | ----------------------------------------- | ----------------------------------------------------- |
| `SEANAV_CONDA_ENV`                              | `sea`                                     | 含 rclpy/onnxruntime/quad_deploy/BreezySLAM 的 conda 环境 |
| `SEANAV_WS`                                     | `~/seanav_ws`                             | 感知包的独立工作区（不污染共享的 `~/ros2_ws`）                         |
| `SEANAV_DATA`                                   | `~/Data/onboard_data/onnx_models/sea_nav` | ONNX 模型目录（含 `nav_model/` 与 `loco_model/`）             |
| `LIDAR_PORT` / `LIDAR_BAUD`                     | `/dev/ttyUSB0` / `256000`                 | 雷达串口与波特率                                              |
| `LASER_YAW_OFFSET`                              | `3.1415926`                               | 雷达安装朝向修正（当前实测倒装，需 180°）                               |
| `RAYS_INVERT_ANGLE`                             | `false`                                   | 左右镜像（加完 yaw offset 仍左右反时才置 `true`）                    |
| `POSE_X_SIGN` / `POSE_Y_SIGN` / `POSE_YAW_SIGN` | `-1.0` / `-1.0` / `1.0`                   | `/pose` 轴向符号（实测标定值）                                   |
| `POSE_USE_MOTION_PRIOR`                         | `true`                                    | 匀速运动先验，解决前进方向（x）跟不上、漂移；设 `false` 退回旧行为                |
| `POSE_MOTION_PRIOR_ALPHA`                       | `0.5`                                     | 速度估计的 EMA 平滑系数（越大越跟手、越小越平滑）                           |
| `POSE_MOTION_PRIOR_MAX_FWD_MM`                  | `80.0`                                    | 单帧前进预测限幅（mm），防发散                                      |
| `POSE_MOTION_PRIOR_MAX_DTHETA_DEG`              | `15.0`                                    | 单帧转角预测限幅（度），防发散                                       |
| `GOAL_X` / `GOAL_Y`                             | `1.0` / `0.0`                             | 目标点（与 `/pose` 同一世界系，原点为 SLAM 定原点时机器人所在处）              |


覆盖示例：

```bash
LIDAR_PORT=/dev/ttyUSB1 bash 1_lidar.sh
GOAL_X=2.0 GOAL_Y=0.5 bash 5_controller.sh --run
```

---

## 6. 启动：五个终端

每个终端脚本都会先 `source env.sh`（激活 `sea` → 初始化 ROS2 Foxy/CycloneDDS/域 1 → source `seanav_ws`），因此直接 `bash n_xxx.sh` 即可，**无需手动激活环境**。

### 6.1 推荐：一键拉起感知（tmux）

```bash
cd ~/Projects/SEA-Nav-Code/deployment/scripts
bash start_perception_tmux.sh      # 在 4 个 tmux 窗格中启动 终端 1/2/3/4
tmux attach -t seanav_perception   # 查看
# 关闭整套感知：tmux kill-session -t seanav_perception
```

控制器（终端 5）**不**包含在该 tmux 中：等感知验证通过、操作员手持手柄后，再在单独终端手动启动。

### 6.2 或：逐个手动启动

```bash
cd ~/Projects/SEA-Nav-Code/deployment/scripts
bash 1_lidar.sh      # 终端 1
bash 2_rays.sh       # 终端 2
bash 3_pose.sh       # 终端 3
bash 4_monitor.sh    # 终端 4（实时监测）  或  bash 4_check.sh（一次性体检）
```

### 6.3 各终端职责

- **终端 1 `1_lidar.sh`**：启动 `sllidar_ros2` 驱动，发布原始 `/scan`（360°，~10 Hz，`frame_id=laser`）。脚本会先检查串口存在与可写性，不可写时提示 `chmod`。
- **终端 2 `2_rays.sh`**：`scan_to_rays_node` 把 `/scan` 下采样为控制栈需要的 41 束 `/rays`（`base_link`，前向 ±120°，量程裁剪到 `[0.1, 3.0]`），并按 `LASER_YAW_OFFSET` 修正安装朝向，使 `ranges[20]` 指向正前方。
- **终端 3 `3_pose.sh`**：`breezy_lidar_odom_node` 用 `/scan` 跑 BreezySLAM 激光里程计，发布世界系 `/pose`。**启动后保持机器人静止**，直到日志出现 `origin fixed`——此刻当前朝向被定为世界系 +X，`x≈0, y≈0, θ≈0`。
- **终端 4 `4_monitor.sh` / `4_check.sh`**：前者实时打印 `/rays` ASCII 条形图（标定朝向用）；后者做一次性体检（话题数、频率、`/rays` 是否为 41 维有限值且在量程内）后自动退出。
- **终端 5 `5_controller.sh`**：SEA-Nav 控制器，详见第 8 节。

> **为什么控制器也要用脚本/`sea` 环境直接启动，而不用 `quad_launch.py`？** 控制器通过 ROS2 订阅 `/rays`、`/pose`，必须与感知端处在**同一** `ROS_DOMAIN_ID=1 + CycloneDDS` 下才能收到数据。而 `launch/sea_nav_launch_cfg.yaml` 里写死的是仿真用的 `ros_env` + `~/ros2_ws`，域/中间件不一定匹配。真机因此直接用 `5_controller.sh`（在 `sea` 环境内启动），最稳妥。

---

## 7. 方向标定与验证

在开电机前，务必确认 `/rays` 与 `/pose` 的朝向/符号正确。打开终端 4 监测，并用终端 4 体检脚本核对维度。

`/rays` 朝向（索引约定：`0` = 右前 -120°，`20` = 正前 0°，`40` = 左前 +120°）：


| 测试          | 期望                                |
| ----------- | --------------------------------- |
| 正前方 1 m 放障碍 | `ranges[20] ≈ 1.0`，最近点在 idx 20 附近 |
| 左前方放障碍      | 最近点 idx 落在 20–40                  |
| 右前方放障碍      | 最近点 idx 落在 0–20                   |


若正前方障碍不在 `ranges[20]`，优先调 `LASER_YAW_OFFSET`；加完 yaw offset 后仍左右反，再把 `RAYS_INVERT_ANGLE=true`。

`**/pose` 符号（终端 3 出现 `origin fixed` 后，机器人静止时应 `x≈0,y≈0,θ≈0`）：


| 测试          | 期望           | 对应参数（实测值）           |
| ----------- | ------------ | ------------------- |
| 手推前进约 1 m   | `x` 增大约 +1.0 | `POSE_X_SIGN=-1.0`  |
| 手推向左平移      | `y` 增大       | `POSE_Y_SIGN=-1.0`  |
| 原地逆时针转约 90° | `θ ≈ +1.57`  | `POSE_YAW_SIGN=1.0` |


`/pose` 还需满足：`ros2 topic info /pose -v` 只有 **1 个** publisher；`ros2 topic hz /pose` 约 10 Hz；终端 3 日志里 `valid_bins` 稳定大于 `min_valid_bins`，`prev=True` 不频繁出现，`dt` 明显小于 100 ms。若机器人完全静止时 `/pose` 仍跳动，可调更保守的 `POSE_SIGMA_XY_MM=10`、`POSE_SIGMA_THETA_DEG=1.5`、`POSE_MAX_SEARCH_ITER=300`。

**前进方向（x）漂移 / "跟不上节奏"**：BreezySLAM 的 `RMHC_SLAM` 在没有里程计时，`update()` 默认以 `pose_change=(0,0,0)` 在上一帧位置附近随机搜索（见官方 `algorithms.py`），无法预测运动，因此机器人主要运动方向（前进 +X）会系统性欠估计、滞后；而 y、yaw 几乎不动反而"凑合"。本部署默认开启**匀速运动先验**（`POSE_USE_MOTION_PRIOR=true`）：用上一帧 SLAM 自身估出的内部位移推算本帧 `pose_change` 喂回 `update()`，把搜索起点挪到预测位置——这样既能保留小 `sigma` 的静止稳定性，又能跟上前进运动。终端 3 日志末尾的 `vel=(.. mm/s, .. deg/s)` 即当前速度估计，前进时前者应明显非零。若先验导致异常发散，可减小 `POSE_MOTION_PRIOR_ALPHA`、收紧 `POSE_MOTION_PRIOR_MAX_FWD_MM`，或临时 `POSE_USE_MOTION_PRIOR=false` 退回旧行为对比。

> `/rays` 正常**不等于** BreezySLAM 输入正常：`/rays` 是 41 维下采样，BreezySLAM 用的是整圈 `/scan`。两者需分别验证。

---

## 8. 控制器：先 dry-run，再开电机

`5_controller.sh` 默认 **dry-run**（不开电机，安全）；加 `--run`（或 `MOTOR=1`）才开电机。底层命令等价于：

```bash
python -m quad_deploy.scripts.SEA_Nav.sea_nav_run_sdk \
    --nosimrun [--nodryrun] --data ~/Data/onboard_data/onnx_models/sea_nav \
    --goal_x <GX> --goal_y <GY>
```

**第一步：dry-run 验证链路**（确保终端 1/2/3 的 `/scan` `/rays` `/pose` 都在发）：

```bash
bash 5_controller.sh        # 默认 dry-run，电机不动
```

启动横幅应显示 `sim_run = False`、`dry_run = True`、`--> MOTOR DISABLED`、`data = .../sea_nav`、`goal = (...)`。期望：

- 能收到 Go2 `rt/lowstate`，不卡在 `Robot Connection`；
- 按手柄 `X` 进入 `human_teleop`；
- `/rays`、`/pose` fresh 时按 `R1` 进入 `navigation`，日志出现 `[nav] Entering navigation state.` 与周期性 `[Nav] ... fresh=True`。

若卡在 `Robot Connection`：

```bash
ip addr show eth0
ping <Go2_IP>
```

确认 Go2 网络接在 `eth0`（控制器内部用 DDS 域 0 / `eth0` 与 Go2 通信）。

**第二步：开电机运行**（仅在第 7 步标定与第一步 dry-run 都通过后）：

```bash
bash 5_controller.sh --run        # 横幅显示 MOTOR ENABLED
# 或指定目标：GOAL_X=2.0 GOAL_Y=0.0 bash 5_controller.sh --run
```

操作顺序：机器人置于空旷区 → 操作员握手柄随时准备 `R2`/`L2` → 等站立完成 → `X` 进 `human_teleop` → 确认可小速度遥控 → `R1` 进 `navigation` → 先只跑 1 m 短距离。

---

## 9. FSM 操作（手柄键位）

真机由手柄驱动状态机（仿真才用键盘）。状态转移：

```text
cold_start ─(站立完成 + X)─► human_teleop ─(R1 且感知 fresh)─► navigation
                                  ▲                                  │
                              R2 接管 / safe_stop 恢复                │
                                  └──────── safe_stop ◄── 感知持续 stale > 1s
任意状态 ─L2─► emergency ─L1─► recovery ─(站立完成 + X)─► human_teleop
```


| 手柄键  | 作用                                                          |
| ---- | ----------------------------------------------------------- |
| `X`  | 站立完成后进入 `human_teleop`                                      |
| `R1` | `human_teleop` → `navigation`（要求 `/rays` 与 `/pose` 都 fresh） |
| `R2` | 从 `navigation` / `safe_stop` 回到人工接管                         |
| `L2` | 任意状态 → `emergency`（关电机）                                     |
| `L1` | `emergency` → `recovery`                                    |


`navigation` 中若 `/rays` 或 `/pose` 持续 stale 超过 1 s，自动进入 `safe_stop`（每 tick 强制零命令、保持站立）；感知恢复并稳定 0.5 s 后，**按 `R1`** 才恢复导航（不自动恢复）。

**安全停机故障注入测试**：`navigation` 中停掉终端 1/3（即停 `/scan`→`/rays`/`/pose`），约 1 s 内应进入 `safe_stop`；恢复发布后按 `R1` 恢复。

---

## 10. 开跑前 checklist

- [ ] 所有终端均经 `env.sh` 初始化（`[env] ... ROS_DOMAIN_ID=1  RMW=rmw_cyclonedds_cpp`）。
- [ ] `python -c "import rclpy"`、`ros_base` / `unitree_sdk2py` / `quad_deploy` SEA-Nav 均可导入。
- [ ] ONNX 在 `~/Data/onboard_data/onnx_models/sea_nav/{nav_model,loco_model}/model.onnx`，维度 nav=550 / loco=235。
- [ ] `/scan` ~10 Hz；`/rays` ~10 Hz、41 维、`[0.1, 3.0]`、`ranges[20]` 为正前方。
- [ ] `/pose` ~10 Hz、单一 publisher、前进/平移/转向符号正确，已出现 `origin fixed`。
- [ ] Go2 `eth0` 连通，dry-run 能收 `rt/lowstate`、可进 `navigation` 且 `fresh=True`。
- [ ] 操作员熟悉 `R2` 接管与 `L2` 急停。

---

## 11. 常见问题


| 现象                                               | 先查什么                                                                                          |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `ros2` 命令缺失 / 话题互相看不到                            | 是否经 `env.sh` 初始化；`ROS_DOMAIN_ID=1`、`RMW=rmw_cyclonedds_cpp` 是否一致                              |
| `import rclpy` 失败                                | `sea` 的 Python 与 ROS2 Foxy 绑定不匹配；先修 `sea` 环境                                                  |
| `ModuleNotFoundError: ros_base / unitree_sdk2py` | 对应包未 `pip install -e .` 或不在同一 Python 环境                                                       |
| `Could not locate cyclonedds`                    | 先编译 CycloneDDS、`export CYCLONEDDS_HOME` 后重装 Unitree SDK（见 3.2）                                |
| 没有 `/scan`                                       | `/dev/ttyUSB0` 权限/接线/波特率（256000）；`bash 1_lidar.sh` 的提示                                        |
| 有 `/scan` 没 `/rays`                              | `seanav_perception` 未 build/source；`bash build_perception.sh` 后重开终端                           |
| `No executable found`                            | `setup.cfg` 缺失导致脚本没装到 `lib/seanav_perception/`；重 build                                        |
| `/rays[20]` 不是正前方                                | 先调 `LASER_YAW_OFFSET`；左右反再 `RAYS_INVERT_ANGLE=true`                                           |
| `/pose` 一直跳                                      | `ros2 topic info /pose -v` 必须只有 1 个 publisher；杀掉多余发布进程                                        |
| `malloc(): invalid size`                         | A2M12 点数超过 Breezy `Laser(360)`；本节点已压成固定 `breezy_scan_size=360` 并 `update(scan_mm)`，确认用的是本仓库节点 |
| 卡 `Robot Connection`                             | Go2 网络 / `eth0` / Unitree DDS（域 0）；`ping <Go2_IP>`                                            |
| `navigation` 进不去                                 | `/rays` 或 `/pose` stale；`ros2 topic hz /rays /pose`                                           |
| 进 `navigation` 动作异常                              | 立即 `R2` 或 `L2`；退回 dry-run 复查 `/rays`、`/pose`、ONNX 路径                                          |


---

## 附录 A：本目录结构

```text
SEA-Nav-Code/deployment/
├── README.md                              # 本文档
├── perception/
│   └── seanav_perception/                 # ROS2 发布端适配包（ament_python）
│       ├── package.xml                    # 包元数据与依赖
│       ├── setup.py                       # 注册 3 个 console_scripts 节点
│       ├── setup.cfg                      # ament_python 安装路径
│       ├── resource/
│       │   └── seanav_perception          # ament 资源索引标记文件
│       └── seanav_perception/
│           ├── __init__.py
│           ├── scan_to_rays_node.py        # /scan -> /rays
│           ├── breezy_lidar_odom_node.py   # /scan -> /pose (BreezySLAM)
│           └── rays_monitor_node.py        # /rays 实时监测
└── scripts/
    ├── config.sh                          # 所有可调参数（环境变量可覆盖）
    ├── env.sh                             # 公共环境：conda + ROS2 Foxy/CycloneDDS/域1 + seanav_ws
    ├── build_perception.sh                # 一次性构建：clone sllidar + 链接本包 + colcon build
    ├── 1_lidar.sh                         # 终端 1：/scan
    ├── 2_rays.sh                          # 终端 2：/rays
    ├── 3_pose.sh                          # 终端 3：/pose
    ├── 4_monitor.sh                       # 终端 4：实时监测
    ├── 4_check.sh                         # 终端 4：一次性体检
    ├── 5_controller.sh                    # 终端 5：SEA-Nav 控制器（默认 dry-run）
    └── start_perception_tmux.sh           # 一键拉起 终端 1–4（tmux）
```

---

## 附录 B：问题记录与解决

### 里程计漂移：前进方向（x）"跟不上" + y 反向

**现象**：`origin fixed` 后，机器人主要沿前进方向（+X）运动时，`/pose` 的 `x` 严重不准、明显滞后（"跟不上节奏"），而 `y`、`yaw` 基本正常；另外 `y` 方向整体反了。

**根因（已对照 BreezySLAM 官方 `python/breezyslam/algorithms.py` 确认）**：

`RMHC_SLAM.update()` 在不传 `pose_change` 时默认用 `(0, 0, 0)`：

```python
def update(self, scans_mm, pose_change=None, ...):
    if not pose_change:
        pose_change = (0, 0, 0)
```

而 `pose_change` 决定的是 RMHC 随机爬山搜索的**起点**（沿当前朝向前进 `dxy_mm`、再转 `dtheta`）：

```python
start_pos.x_mm += dxy_mm * cos(theta)
start_pos.y_mm += dxy_mm * sin(theta)
start_pos.theta_degrees += dtheta_degrees
new_position = self._getNewPosition(start_pos)   # 在 start_pos 附近随机搜索
```

本部署原先只调 `update(scan_mm)` → 搜索起点永远等于上一帧位置、**完全不预测运动**。机器人主要沿朝向（=前进 +X）运动，每帧需靠 `sigma_xy_mm=20mm` 的随机搜索"摸"出约 30mm 位移（0.3 m/s ÷ 10 Hz），于是前进方向被系统性**欠估计、滞后**；而 `y`、`yaw` 几乎不动，小范围随机搜索就能锁住，所以"凑合"。这也解释了"静止要小 sigma（稳）" 与 "运动要大 sigma（跟得上）" 之间的矛盾。

**解决方案：匀速运动先验（已默认开启）**

在 `breezy_lidar_odom_node.py` 中，用上一帧 SLAM 自身估出的**内部坐标位移**计算前进/转向速度（EMA 平滑），推算本帧的 `pose_change = (前进 mm, 转角 °, dt)` 喂回 `update()`，把搜索起点直接挪到预测位置：

- 既保留小 `sigma`（静止稳定），又能跟上前进运动（x 不再滞后）；
- 带单帧限幅（`max_fwd_mm` / `max_dtheta_deg`）防止反馈发散；
- 全部估计在 BreezySLAM 内部坐标系完成，再走原有的 base_link/世界系变换。

相关参数（默认值，均可在 `config.sh` 覆盖）：


| 参数                                 | 默认值    | 说明                                 |
| ---------------------------------- | ------ | ---------------------------------- |
| `POSE_USE_MOTION_PRIOR`            | `true` | 总开关；设 `false` 退回旧的 `(0,0,0)` 行为做对比 |
| `POSE_MOTION_PRIOR_ALPHA`          | `0.5`  | 速度估计 EMA 平滑系数（越大越跟手、越小越平滑）         |
| `POSE_MOTION_PRIOR_MAX_FWD_MM`     | `80.0` | 单帧前进预测限幅（mm）                       |
| `POSE_MOTION_PRIOR_MAX_DTHETA_DEG` | `15.0` | 单帧转角预测限幅（度）                        |


终端 3 的 debug 日志末尾新增 `vel=(.. mm/s, .. deg/s)`，前进时前进分量应明显非零，可据此确认先验生效。

**y 方向反向**：与运动先验无关，属轴向符号标定。实测正确值为 `POSE_Y_SIGN=-1.0`（已设为默认），即"手推向左平移 → `pose.y` 增大"。

**关键依据（官方仓库）**：

- BreezySLAM `RMHC_SLAM` / `CoreSLAM.update` 的 `pose_change` 语义与默认 `(0,0,0)`：[https://github.com/simondlevy/BreezySLAM/blob/master/python/breezyslam/algorithms.py](https://github.com/simondlevy/BreezySLAM/blob/master/python/breezyslam/algorithms.py)
- BreezySLAM 基本接口 `slam.update(scan)`：[https://github.com/simondlevy/BreezySLAM](https://github.com/simondlevy/BreezySLAM)
- Slamtec `sllidar_ros2`（A2M12 默认 `angle_compensate=true`，雷达 frame 外参需自行处理）：[https://github.com/Slamtec/sllidar_ros2](https://github.com/Slamtec/sllidar_ros2)

---

*以 `quad_deploy@feat/sea-nav-agent` 最新代码为准。真机适配代码与脚本随本目录维护，控制栈本身不改动。*