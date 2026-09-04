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
five-step history. The ONNX backend is optional and requires `onnxruntime`;
by default it looks for the `.onnx` counterparts of the existing controller
model files.

---

## Deployment
For real-robot deployment on the Unitree Go2, see the [deployment README](deployment/README.md).
