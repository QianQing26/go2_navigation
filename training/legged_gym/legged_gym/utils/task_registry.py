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
import json
from datetime import datetime
from typing import Tuple, Dict, Union
import torch
import numpy as np
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
    
    def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        # copy seed
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)

        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        print(
            "[task_registry] creating environment: task={}, num_envs={}, sim_device={}, rl_device={}".format(
                name, env_cfg.env.num_envs, args.sim_device, args.rl_device
            ),
            flush=True,
        )
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            physics_engine=args.physics_engine,
                            sim_device=args.sim_device,
                            headless=args.headless)
        print("[task_registry] environment created: task={}".format(name), flush=True)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default"
                        ) -> Tuple[Union[OnPolicyRunner, OnPolicyRunner], LeggedRobotCfgPPO]:

    # def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default"
    #                     ) -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        default_log_root = os.path.join(
            LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name
        )
        if log_root == "default":
            log_root = default_log_root
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime('%m_%d_%H-%M-%S')
                + '_' + train_cfg.runner.run_name,
            )
        elif log_root is None:
            # Evaluation callers use ``log_root=None`` to avoid creating a
            # new training run.  They must still resolve resume checkpoints
            # from the task's normal experiment directory.
            log_root = default_log_root
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%m_%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        
        
        train_cfg_dict = class_to_dict(train_cfg)

        runner_class = eval(train_cfg.runner_class_name)

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            print("[task_registry] log_dir={}".format(log_dir), flush=True)
        runner = runner_class(env, train_cfg_dict, log_dir, args=args, device=args.rl_device)

        init_policy_path = getattr(args, 'init_policy_path', None)
        legacy_pretrained_path = getattr(args, 'pretrained_path', None)
        if init_policy_path and legacy_pretrained_path:
            raise ValueError(
                'Use only one of --init_policy_path and --pretrained_path'
            )
        init_policy_path = init_policy_path or legacy_pretrained_path
        loaded_resume_path = None
        if init_policy_path:
            init_policy_path = os.path.abspath(os.path.expanduser(init_policy_path))
            if not os.path.isfile(init_policy_path):
                raise FileNotFoundError(
                    "Initial policy checkpoint not found: {}".format(init_policy_path)
                )
            if getattr(args, 'resume', False) or train_cfg.runner.resume:
                raise ValueError(
                    '--init_policy_path/--pretrained_path cannot be combined '
                    'with --resume; choose weights-only initialization or resume'
                )
            print(
                "[task_registry] loading initial policy weights only (fresh optimizer): {}".format(
                    init_policy_path
                ),
                flush=True,
            )
            runner.load_weights(init_policy_path)
        elif train_cfg.runner.resume:
            # load previously trained model
            loaded_resume_path = get_load_path(
                log_root,
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )
            self.loaded_policy_path = loaded_resume_path
            print('[task_registry] loading model with optimizer state: {}'.format(
                loaded_resume_path
            ), flush=True)
            runner.load(loaded_resume_path, load_optimizer=True)

        if log_dir is not None:
            safety_cfg = getattr(getattr(env.cfg, 'env', None), 'predictive_safety', None)
            obstacle_cfg = getattr(env.cfg, 'dynamic_obstacles', None)
            curriculum_cfg = getattr(obstacle_cfg, 'curriculum', None)
            speed_range = getattr(obstacle_cfg, 'speed_range', None)
            speed_start = getattr(curriculum_cfg, 'speed_start', None)
            estimator_checkpoint = (
                getattr(safety_cfg, 'estimator_checkpoint', '')
                if safety_cfg is not None else ''
            )
            if estimator_checkpoint:
                estimator_checkpoint = os.path.expanduser(estimator_checkpoint)
                if not os.path.isabs(estimator_checkpoint):
                    estimator_checkpoint = os.path.abspath(os.path.join(
                        os.path.dirname(__file__), '../../../..', estimator_checkpoint
                    ))
            experiment_config = {
                'task': name,
                'run_name': train_cfg.runner.run_name,
                'safety_mode': getattr(safety_cfg, 'mode', 'original'),
                'initial_policy_checkpoint': init_policy_path,
                'resume_checkpoint': loaded_resume_path,
                'weights_only_initialization': bool(init_policy_path),
                'optimizer_initialization': (
                    'fresh' if init_policy_path else
                    ('loaded' if loaded_resume_path else 'fresh')
                ),
                'estimator_checkpoint': estimator_checkpoint,
                'calibration_delta': float(
                    getattr(safety_cfg, 'calibration_delta', 0.0)
                    if safety_cfg is not None else 0.0
                ),
                'seed': int(train_cfg.seed),
                'num_envs': int(env.num_envs),
                'num_steps_per_env': int(train_cfg.runner.num_steps_per_env),
                'max_iterations': int(train_cfg.runner.max_iterations),
                'curriculum': {
                    'enabled': bool(getattr(curriculum_cfg, 'enabled', False)),
                    'speed_start': list(speed_start) if speed_start is not None else None,
                    'speed_final': list(speed_range) if speed_range is not None else None,
                    'speed_steps': int(getattr(curriculum_cfg, 'speed_steps', 0)),
                },
                'algorithm': class_to_dict(train_cfg.algorithm),
                'policy': class_to_dict(train_cfg.policy),
                'runner': class_to_dict(train_cfg.runner),
                'reward': class_to_dict(env.cfg.rewards),
            }
            with open(os.path.join(log_dir, 'config.json'), 'w') as config_file:
                json.dump(experiment_config, config_file, indent=2, sort_keys=True)
            print('[task_registry] wrote experiment config: {}'.format(
                os.path.join(log_dir, 'config.json')
            ), flush=True)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
