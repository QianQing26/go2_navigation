# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import copy
import re
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

def class_to_dict(obj) -> dict:
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = [
            run for run in os.listdir(root)
            if run != 'exported' and os.path.isdir(os.path.join(root, run))
        ]
    except OSError:
        raise ValueError("No runs in this directory: " + root)

    if not runs:
        raise ValueError("No runs in this directory: " + root)

    if load_run==-1:
        # Evaluation scripts may create empty bookkeeping directories.  Only
        # consider runs that contain at least one checkpoint when selecting
        # the latest run automatically.
        runs_with_models = [
            run for run in runs
            if any(
                re.match(r'^model_\d+\.pt$', filename)
                for filename in os.listdir(os.path.join(root, run))
            )
        ]
        if not runs_with_models:
            raise ValueError("No checkpoints found in this directory: " + root)
        load_run = os.path.join(root, sorted(runs_with_models)[-1])
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [
            file for file in os.listdir(load_run)
            if re.match(r'^model_\d+\.pt$', file)
        ]
        if not models:
            raise ValueError("No checkpoints found in run directory: " + load_run)
        models.sort(key=lambda m: int(re.match(r'^model_(\d+)\.pt$', m).group(1)))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.seed is not None:
            env_cfg.seed = args.seed
        safety_cfg = getattr(getattr(env_cfg, 'env', None), 'predictive_safety', None)
        if safety_cfg is not None:
            if getattr(args, 'safety_mode', None) is not None:
                safety_cfg.mode = args.safety_mode
            if getattr(args, 'estimator_checkpoint', None) is not None:
                safety_cfg.estimator_checkpoint = args.estimator_checkpoint
            if getattr(args, 'calibration_delta', None) is not None:
                safety_cfg.calibration_delta = args.calibration_delta
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train

def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "go2_pos_rough", "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False,  "help": "Resume training from a checkpoint"},
        {"name": "--pretrained_path", "type": str, "default": None, "help": "Load Actor-Critic weights from a checkpoint before fine-tuning. Optimizer state and iteration are reset."},
        {"name": "--init_policy_path", "type": str, "default": None, "help": "Weights-only Phase-2 initialization. Starts with a fresh PPO optimizer at iteration 0."},
        {"name": "--safety_mode", "type": str, "default": None, "help": "Safety mode: original, synchronized_static, or predictive."},
        {"name": "--estimator_checkpoint", "type": str, "default": None, "help": "Frozen MotionEstimator checkpoint for predictive mode."},
        {"name": "--calibration_delta", "type": float, "default": None, "help": "Optional physical drift calibration; predictive pilot requires 0.0."},
        {"name": "--experiment_name", "type": str,  "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str,  "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,  "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,  "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},
        
        {"name": "--no_wandb", "action": "store_true", "default": True, "help": "Disable logging to Weights and Biases (wandb)"},
        {"name": "--test", "action": "store_true", "default": False, "help": "Run in test mode (overrides some parameters for testing)"},
        {"name": "--headless", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device=='cuda':
        args.sim_device += f":{args.sim_device_id}"
        
    # default logic for train and test
    # Training uses local TensorBoard logging only.  Keep the attribute for
    # compatibility with older scripts, but never enable WandB implicitly.
    args.wandb = False
    args.rl_device = args.sim_device
    if args.test:
        args.headless = False
        args.wandb = False
        args.num_envs = 100
        
    return args

def export_policy_as_jit(actor_critic, path, exported_policy_name):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else: 
        os.makedirs(path, exist_ok=True)
        model = 'policy.pt'
        save_path = os.path.join(path, model)
        
        # Handle DifferentiableSafeActorCritic which doesn't have a single 'actor' module
        if hasattr(actor_critic, 'backbone') and hasattr(actor_critic, 'nav_head'):
            # Create a wrapper module that mimics the forward pass of the safe actor
            class SafeActorWrapper(torch.nn.Module):
                def __init__(self, ac):
                    super().__init__()
                    self.encoder = copy.deepcopy(ac.encoder)
                    self.backbone = copy.deepcopy(ac.backbone)
                    self.nav_head = copy.deepcopy(ac.nav_head)
                    self.alpha_head = copy.deepcopy(ac.alpha_head)
                    self.cbf_layer = copy.deepcopy(ac.cbf_layer)
                    self.num_obs_one_step = ac.num_obs_one_step
                    self.num_props = ac.num_props
                    self.num_rays = ac.num_rays
                    
                def forward(self, observations):
                    obs_buf = observations[:, -self.num_obs_one_step:]
                    props = obs_buf[:, :self.num_props]
                    rays = obs_buf[:, self.num_props : self.num_props + self.num_rays]
                    
                    latent = self.encoder(observations)
                    obs_cat = torch.cat((obs_buf, latent), dim=-1)
                    
                    shared_features = self.backbone(obs_cat)
                    u_bar = self.nav_head(shared_features)
                    alpha_raw = self.alpha_head(shared_features)
                    
                    rays_real = torch.exp2(rays)
                    alpha = torch.nn.functional.softplus(alpha_raw)
                    
                    u_s = self.cbf_layer(u_bar, rays_real, alpha)
                    return u_s
                    
            model = SafeActorWrapper(actor_critic).to('cpu')
        else:
            model = copy.deepcopy(actor_critic.actor).to('cpu')
            
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(save_path)
        print(f'policy has saved in {save_path}')
    
        
class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
