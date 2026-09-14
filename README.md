# SEA-Nav: Efficient Policy Learning for Safe and Agile Quadruped Navigation in Cluttered Environments


**Project Website**: [https://11chens.github.io/sea-nav](https://11chens.github.io/sea-nav/)

<p align="center">
  <img src="imgs/terser.jpg" width="80%">
</p>

---

## Installation

### 1. Environment Setup
Create a new Python virtual environment with Python 3.8:
```bash
conda create -n sea_nav python=3.8
conda activate sea_nav
```

### 2. Install Isaac Gym
- Download and install Isaac Gym Preview 4 from [NVIDIA Developer](https://developer.nvidia.com/isaac-gym).
- Install the python package:
```bash
cd isaacgym/python && pip install -e .
```

### 3. Install rsl_rl
- Clone this repository
- Install the package:
```bash
cd training/rsl_rl && pip install -e .
```

### 4. Install legged_gym
```bash
cd training/legged_gym && pip install -e .
```

---

## Usage

### Task documentation

The purpose and training relationship of each registered task are documented in
the [task guide](TASKS.md). In particular, `go2_pos_rough` is the static
baseline and `go2_pos_dynamic` is the dynamic-obstacle fine-tuning task.

### Training
To start training in headless mode:
```bash
python training/legged_gym/legged_gym/scripts/train.py --headless
```

### Testing
To visualize and test a trained policy:
```bash
python training/legged_gym/legged_gym/scripts/play.py
```

### Low-level controller selection
The navigation task uses a pluggable joint-position controller. Go2 uses the
registered `robogauge` controller by default; the original three-model
TorchScript controller remains available. Select a registered backend in
`LeggedRobotPosCfg.controller.name` without changing the PPO runner or
navigation policy:

```python
class controller:
    name = 'torchscript'  # or 'onnx', 'robogauge'
```

Controller-owned observation construction, model inference and history reset
are implemented under `training/legged_gym/legged_gym/controllers/`. The
`robogauge` controller loads the configured batchable `go2_moe_cts` TorchScript
policy, builds its 45-D source observation, and lets the policy maintain its
five-step history. Its source-compatible scales are linear/angular velocity
`2.0/0.25`, joint position/velocity `1.0/0.05`, command `[2.0, 2.0, 0.25]`,
and PD action scale `0.25`; its Go2 model order is kept separate from the
legacy SEA-Nav sim-to-real permutation. The ONNX backend is optional and
requires `onnxruntime`;
by default it looks for the `.onnx` counterparts of the existing controller
model files.

The first-stage dynamic-obstacle navigation task is registered separately as
`go2_pos_dynamic`, so `go2_pos_rough` is unchanged. It uses the same
navigation observation layout, adds six gravity-free linearly moving box
actors per environment, and fuses their analytic ray intersections with the
existing terrain rays. The navigation control/proprioception loop runs at
50 Hz, while the exteroceptive ray/goal history is updated at 10 Hz. Run it
with:

```bash
python training/legged_gym/legged_gym/scripts/train.py --task go2_pos_dynamic
```

Before training, the configured low-level controller can be smoke-tested
without loading a navigation PPO policy. The test resets once and then sends
forward/backward, lateral, in-place-turning, and stop commands:

```bash
CUDA_VISIBLE_DEVICES=0 python training/legged_gym/legged_gym/scripts/test_low_level_controller.py
```

The test uses a flat terrain mesh and does not alter `go2_pos_rough`. Add
`--controller torchscript` (or another registered name) to override the
controller configured by the task, `--headless` for a non-visual run, or
adjust each phase with `--phase_steps N`.

### Dynamic-obstacle validation

Before training `go2_pos_dynamic`, the three core simulator links can be
checked without loading a navigation PPO policy:

```bash
CUDA_VISIBLE_DEVICES=0 python \
training/legged_gym/legged_gym/scripts/validate_dynamic_obstacles.py \
--mode all --headless --steps 10 --print-every 5
```

Use `--mode rays` without `--headless` to open the Isaac Gym viewer and draw
all 41 rays. `--mode gt` prints dynamic obstacle world position, velocity,
box size, ray-query radius, and simulator actor indices. `--mode collision`
places one obstacle at the robot base and checks contact force, collision
penalty, and termination state. The GT interface is also available as
`env.get_dynamic_obstacle_gt(relative_to_robot=False)` for a future privileged
critic or teacher policy; it is not appended to the current PPO observation.

### Phase-3 fixed-cohort evaluation audit

The authoritative Phase-3 scenario bank is
`datasets/phase3/scenario_bank_512.pt` (512 scenarios, six obstacles,
analytic speed range 0.1--1.5 m/s). Its text metadata and hashes are recorded
in `datasets/phase3/scenario_bank_512.json`. Fixed and checkpoint-sweep
evaluations write the same `evaluation_manifest.json` schema, including policy,
estimator, bank, simulator, randomization, replay, CBF, and scenario-selection
provenance.

The pre-registered checkpoint sweep is exactly:

```text
500, 1000, 1200, 1500, 2000
```

Use `compare_evaluation_manifests.py` to audit trajectory-affecting inputs and
`compare_evaluation_runs.py` to compare per-scenario outcomes and the first
diagnostic divergence. The fixed evaluator accepts `--scenario_ids` and
`--diagnostic_scenario_ids` for small reproducibility smoke tests.

Phase-3B repeated evaluation is run serially on GPU 0 with
`run_repeated_evaluations.py`. It writes
`seed{1,2,3}/{A,B,C}/repeat{1,2,3}` and records execution timing. The
analysis command is:

```bash
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n sea_nav python \
training/legged_gym/legged_gym/scripts/aggregate_repeated_evaluations.py \
--root training/legged_gym/logs/go2_pos_dynamic/phase3_repeated_eval \
--output_dir training/legged_gym/logs/go2_pos_dynamic/phase3B_analysis
```

The analysis keeps the hierarchy scenario → evaluation repeat → training
seed. Repeats are never pooled as independent training samples; sample
standard deviations use `ddof=1`. It emits repeat variance, scenario
disagreement, per-repeat paired A/B/C effects, difficulty-stratified results,
multi-seed means, and an explicit reproducibility assessment.

---

## Deployment
For real-robot deployment on the Unitree Go2, see the [deployment README](deployment/README.md).
