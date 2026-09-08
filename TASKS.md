# SEA-Nav 任务说明

本文档记录仓库中各个训练任务的定位、环境差异、训练关系和使用方式。
所有任务都属于 Isaac Gym 训练流程；部署流程不在本文档范围内。

## 任务总览

| 任务名 | 定位 | 环境特点 | 训练方式 |
| --- | --- | --- | --- |
| `go2_pos_rough` | 静态环境 baseline | Go2 在静态粗糙房间中进行目标导航 | 从头训练 |
| `go2_pos_dynamic` | 动态场景微调任务 | 在静态房间背景上增加动态障碍物 | 加载 `go2_pos_rough` checkpoint 后微调 |

任务名必须使用注册表中的实际名称。当前动态任务的注册名是
`go2_pos_dynamic`，不是 `go2_pos_dynamics`。

## 训练关系

```text
go2_pos_rough
    │
    │  Actor-Critic 权重迁移
    │  optimizer 重置，迭代计数重新开始
    ▼
go2_pos_dynamic
```

两个任务保持相同的导航策略输入输出接口，均使用 3 维导航指令作为高层
策略动作，并通过可配置的关节位置控制器生成 Go2 的 12 维关节位置动作。
因此，动态阶段可以使用 `--pretrained_path` 加载静态阶段策略。

低层控制器需要在两个阶段保持一致。当前 Go2 默认使用注册名
`robogauge`，其模型输入、历史帧、关节顺序和动作缩放都由控制器配置管理。
训练阶段不应在静态预训练和动态微调之间切换低层控制器。

## `go2_pos_rough`

### 任务定位

`go2_pos_rough` 是当前项目的静态环境 baseline，用于学习基本的四足运动、
目标导航、速度跟踪和静态障碍物避障能力。它也是动态场景微调的初始化策略
来源。

### 主要环境特征

- 机器人：Unitree Go2；
- 地形：`hard_room` 静态粗糙房间；
- 目标：从随机起点导航到目标位置；
- 动作：3 维导航指令，由低层关节位置控制器执行；
- 控制/本体感知频率：50 Hz；
- 外感 history 频率：10 Hz，每 5 个控制步采样一次；
- 默认控制器：`robogauge`；
- 使用域随机化、观测噪声、随机推力和现有导航奖励。

### 推荐用途

- 建立静态环境中的性能基线；
- 检查低层控制器与高层导航策略的接口是否正确；
- 生成动态环境微调使用的初始 checkpoint；
- 评估策略是否具备基本的运动和目标到达能力。

### 训练命令

```bash
CUDA_VISIBLE_DEVICES=0 python \
training/legged_gym/legged_gym/scripts/train.py \
--task go2_pos_rough \
--headless \
--num_envs 2048 \
--max_iterations 2000 \
--run_name static_pretrain
```

训练日志和模型默认保存在：

```text
training/legged_gym/logs/Go2_pos_rough/<timestamp>_static_pretrain/
```

正式作为微调起点的 checkpoint 应来自控制器配置已经确认正确之后的训练。

## `go2_pos_dynamic`

### 任务定位

`go2_pos_dynamic` 是动态场景调整任务。它继承 `go2_pos_rough` 的导航观测、
动作接口、奖励和控制器设置，在此基础上加入动态障碍物，主要用于训练四足
在动态场景中的避障导航能力。

### 主要环境特征

- 静态背景：`easy_room` 房间；
- 动态障碍物：每个环境 6 个重力关闭的盒状障碍物；
- 运动方式：二维线性运动，到达边界后反射；
- 运动方向：支持横向、纵向和斜向运动；
- 当前速度范围：`[0.5, 2.5]`；
- 感知：动态障碍物的解析射线结果与地形射线融合；
- 频率：本体感知和控制为 50 Hz，外感 ray/goal history 为 10 Hz；
- ray：41 条，角度范围 `[-120°, 120°]`，步长约 `6°`；
- 训练初始化：加载 `go2_pos_rough` 的 Actor-Critic 权重；
- 优化器：不迁移，动态阶段重新初始化。

动态障碍物是每次环境 reset 时采样的。动态阶段的 checkpoint 不应直接覆盖
静态 baseline，两个阶段应保存在不同的实验目录中。

### 推荐用途

- 在静态导航能力的基础上学习动态避障；
- 测试障碍物速度、数量和运动方向变化带来的影响；
- 比较静态策略和动态微调策略在动态环境中的性能差异；
- 后续扩展速度课程、障碍物课程和更复杂运动模式的基础任务。

### 训练前验证

训练前可以用以下脚本验证动态障碍物的三条关键链路。脚本不加载高层
导航策略，默认只使用一个环境：

```bash
CUDA_VISIBLE_DEVICES=0 python \
training/legged_gym/legged_gym/scripts/validate_dynamic_obstacles.py \
--mode all --headless --steps 10 --print-every 5
```

- `--mode rays`：打印 41 条 ray 的角度、融合距离和动态障碍物命中角度；
  去掉 `--headless` 后会在 Isaac Gym 中绘制 ray。
- `--mode gt`：打印每个动态障碍物的 simulator GT position、velocity、
  box size、水平包围半径和 actor index。
- `--mode collision`：将一个障碍物临时放到机器人基座处，检查接触力、
  collision penalty 和 termination。

环境接口 `get_dynamic_obstacle_gt()` 已提供这些特权信息，但当前为了保持
已有 checkpoint 的输入维度和兼容性，GT 尚未接入 PPO critic。

### 微调命令

```bash
STATIC_CKPT=/absolute/path/to/Go2_pos_rough/<run>/model_2000.pt

CUDA_VISIBLE_DEVICES=0 python \
training/legged_gym/legged_gym/scripts/train.py \
--task go2_pos_dynamic \
--pretrained_path "$STATIC_CKPT" \
--headless \
--num_envs 2048 \
--max_iterations 1000 \
--run_name dynamic_finetune
```

跨任务迁移使用 `--pretrained_path`，不要使用 `--resume`。前者只加载策略
权重并重置优化器，后者用于继续同一实验运行。

## 推荐实验流程

1. 使用 `test_low_level_controller.py` 测试 `robogauge` 的前进、后退、侧向、
   原地转向和停止动作。
2. 从头训练 `go2_pos_rough`，确认静态导航性能和训练曲线稳定。
3. 保存并记录静态 checkpoint 的完整路径、训练配置和实验名称。
4. 将静态 checkpoint 加载到 `go2_pos_dynamic` 中进行微调。
5. 分别在静态和动态环境中评估两个 checkpoint，检查动态微调是否导致静态
   能力明显退化。

推荐至少记录以下指标：

- 目标到达率；
- 碰撞率；
- 卡死率；
- 平均 episode reward；
- 平均 episode length；
- 平均到达时间；
- 障碍物速度和数量配置。

训练日志包括 `train.log`、`metrics.csv` 和 TensorBoard event 文件，位于每次
训练对应的 `logs/<experiment_name>/<run_name>/` 目录中。

## 新增任务记录规范

后续新增任务时，请在任务总览表中增加一行，并按下面的结构增加一个小节：

```markdown
## `<task_name>`

### 任务定位
说明这个任务解决什么问题，以及它是从头训练还是从哪个任务微调。

### 主要环境特征
- 机器人和地形；
- 障碍物或扰动；
- 观测和动作接口；
- 与父任务的关键差异。

### 推荐用途
说明该任务适合做 baseline、预训练、微调、消融实验还是最终评估。

### 训练命令
给出可直接运行的命令，并说明 checkpoint 如何加载。
```

每个新任务还应同步确认以下内容：

- 在 `legged_gym/envs/__init__.py` 中完成注册；
- 明确父任务和 checkpoint 迁移关系；
- 说明是否保持观测维度、动作维度和控制器不变；
- 为新任务使用独立的 `experiment_name`；
- 不修改其他任务的配置；
- 如增加新的环境机制，补充对应的可视化或冒烟测试方法。
