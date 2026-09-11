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

import numpy as np
import os
import sys
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry, helpers
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from datetime import datetime
from legged_gym.envs.go2.go2_pos_config import Go2PosRoughCfg
from legged_gym.utils.helpers import class_to_dict
from rsl_rl.utils.phase2 import remaining_iterations

def print_config():
    config = class_to_dict(Go2PosRoughCfg.rewards)
    print('-'*80)
    import pprint
    pprint.pprint(config, indent=2)
    print('-'*80)

    return config

def train(args):
    if getattr(args, 'safety_mode', None) == 'predictive_calibrated':
        raise ValueError(
            'predictive_calibrated is reserved for later ablation; '
            'Phase-2 pilot supports original, synchronized_static, predictive'
        )
    print('[train] loading reward configuration', flush=True)
    config = print_config()
    print('[train] creating environment', flush=True)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    print('[train] creating PPO runner', flush=True)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    target_iteration = int(train_cfg.runner.max_iterations)
    current_iteration = int(ppo_runner.current_learning_iteration)
    iterations_to_run = remaining_iterations(target_iteration, current_iteration)
    print(
        '[train] starting learning: current_iteration={}, target_iteration={}, '
        'iterations_to_run={}'.format(
            current_iteration, target_iteration, iterations_to_run
        ),
        flush=True,
    )
    if iterations_to_run:
        ppo_runner.learn(
            num_learning_iterations=iterations_to_run,
            init_at_random_ep_len=(current_iteration == 0),
            config=config,
        )
    else:
        print('[train] target iteration already reached; no learning required', flush=True)
    
if __name__ == '__main__':
    args = get_args()
    train(args)
