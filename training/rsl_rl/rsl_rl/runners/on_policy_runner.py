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

import time
import os
import csv
from collections import deque
import statistics

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

from rsl_rl.env import VecEnv
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic
class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 args=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.args = args

        num_obs = self.env.num_obs
        num_rays = self.env.rays.shape[1]
        num_nav_actions = self.env.num_nav_actions
        actor_critic_class = eval(self.cfg["policy_class_name"])

        actor_critic: ActorCritic = actor_critic_class( 
                                        num_actions=num_nav_actions,
                                        num_props=self.env.num_props,
                                        his_len=self.env.cfg.env.his_len,
                                        num_rays=num_rays,
                                        **self.policy_cfg).to(self.device)

        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        
        self.alg.init_storage(num_envs=self.env.num_envs, num_transitions_per_env=self.num_steps_per_env, 
                obs_shape=[num_obs], action_shape=[num_nav_actions])
        
        self.log_dir = log_dir
        self.writer = None
        self.log_file = None
        self.metrics_file = None
        if self.log_dir is not None:
            if SummaryWriter is None:
                raise ImportError(
                    "TensorBoard logging requires the 'tensorboard' package. "
                    "Install it with: python -m pip install tensorboard"
                )
            # Create the run directory immediately.  The old runner waited
            # until iteration 100, which made early initialization failures
            # and long first rollouts impossible to diagnose from logs.
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.log_file = os.path.join(self.log_dir, 'train.log')
            self.metrics_file = os.path.join(self.log_dir, 'metrics.csv')
            if not os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'w', newline='') as metrics_file:
                    csv.writer(metrics_file).writerow([
                        'iteration', 'timesteps', 'collection_sec', 'learning_sec',
                        'fps', 'rollout_mean_reward', 'episodes_completed',
                        'mean_episode_reward', 'mean_episode_length',
                        'value_loss', 'surrogate_loss', 'regularization_loss',
                        'smooth_loss', 'interv_loss',
                    ])
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self._log(
            '[runner] log_dir={}'.format(self.log_dir)
        )
        self._log(
            '[runner] resetting environment: num_envs={}, steps_per_env={}'.format(
                self.env.num_envs, self.num_steps_per_env
            )
        )
        _, _ = self.env.reset()
        self._log('[runner] environment reset complete')

    def _log(self, message):
        """Print a flushed message and mirror it into the current run log."""

        print(message, flush=True)
        if self.log_file is not None:
            with open(self.log_file, 'a', encoding='utf-8') as log_file:
                log_file.write(message.rstrip() + '\n')

    def _write_metrics(self, locs):
        if self.metrics_file is None:
            return
        episodes_completed = len(locs['rewbuffer'])
        mean_episode_reward = (
            statistics.mean(locs['rewbuffer']) if episodes_completed else float('nan')
        )
        mean_episode_length = (
            statistics.mean(locs['lenbuffer']) if episodes_completed else float('nan')
        )
        iteration_time = locs['collection_time'] + locs['learn_time']
        fps = int(self.num_steps_per_env * self.env.num_envs / iteration_time) if iteration_time > 0 else 0
        with open(self.metrics_file, 'a', newline='') as metrics_file:
            csv.writer(metrics_file).writerow([
                locs['it'],
                self.tot_timesteps,
                locs['collection_time'],
                locs['learn_time'],
                fps,
                locs['rollout_mean_reward'],
                episodes_completed,
                mean_episode_reward,
                mean_episode_length,
                locs['mean_value_loss'],
                locs['mean_surrogate_loss'],
                locs['mean_regularization_loss'],
                locs['mean_smooth_loss'],
                locs['mean_interv_loss'],
            ])
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False, config=None):
        
        # initialize writer
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        infos = self.env.get_extras()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        tot_iter = self.current_learning_iteration + num_learning_iterations
        # self.num_steps_per_env = 1
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            mean_num_sim = 0
            self._log(
                '[runner] iteration {}/{} rollout start'.format(
                    it + 1, tot_iter
                )
            )
            with torch.no_grad():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, infos)
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                mean_num_sim /= (self.num_steps_per_env)

                self._log(
                    '[runner] iteration {} rollout complete ({:.3f}s), update start'.format(
                        it, collection_time
                    )
                )

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, infos)
            
            mean_value_loss, mean_surrogate_loss, mean_regularization_loss, mean_smooth_loss, mean_interv_loss = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += collection_time + learn_time
            rollout_mean_reward = float(self.alg.storage.rewards.mean().item())
            locs = locals()
            self._write_metrics(locs)

            self._log(
                '[runner] iteration {} update complete ({:.3f}s), rollout_reward={:.4f}'.format(
                    it, learn_time, rollout_mean_reward
                )
            )
            # TensorBoard is the only online logger.  Write every iteration so
            # that short runs and interrupted runs still contain a complete
            # learning curve.
            self.tensorboard_log(locals())
            if self.log_dir is not None and it % 10 == 0:
                self.print_log(locals(), extra=True)
            if self.log_dir is not None and it % self.save_interval == 0 and it > self.current_learning_iteration:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
            self.writer.flush()
            self.writer.close()
    
    def tensorboard_log(self, locs):
        """Write scalar metrics for one completed training iteration."""
        if self.writer is None:
            return

        step = int(locs['it'])
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], step)
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], step)
        self.writer.add_scalar('Loss/regularization', locs['mean_regularization_loss'], step)
        self.writer.add_scalar('Loss/smooth', locs['mean_smooth_loss'], step)
        self.writer.add_scalar('Loss/intervention', locs['mean_interv_loss'], step)
        self.writer.add_scalar('Train/rollout_reward', locs['rollout_mean_reward'], step)
        self.writer.add_scalar('Train/collection_seconds', locs['collection_time'], step)
        self.writer.add_scalar('Train/learning_seconds', locs['learn_time'], step)
        self.writer.add_scalar(
            'Train/fps',
            self.num_steps_per_env * self.env.num_envs /
            max(locs['collection_time'] + locs['learn_time'], 1e-9),
            step,
        )

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar(
                'Train/mean_episode_reward', statistics.mean(locs['rewbuffer']), step
            )
            self.writer.add_scalar(
                'Train/mean_episode_length', statistics.mean(locs['lenbuffer']), step
            )

        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                values = []
                for ep_info in locs['ep_infos']:
                    value = ep_info[key]
                    if isinstance(value, torch.Tensor):
                        values.append(value.detach().float().mean().item())
                    else:
                        values.append(float(value))
                if values:
                    self.writer.add_scalar('Rewards/{}'.format(key), statistics.mean(values), step)

        # Make the current iteration visible immediately when tailing the log
        # or monitoring TensorBoard during a long-running job.
        self.writer.flush()

    def print_log(self, locs, width=80, pad=35, extra=True):
        iteration_time = locs['collection_time'] + locs['learn_time']
        mean_reward = (
            statistics.mean(locs['rewbuffer'])
            if len(locs['rewbuffer']) > 0
            else locs.get('rollout_mean_reward', float('nan'))
        )
        mean_episode_length = (
            statistics.mean(locs['lenbuffer'])
            if len(locs['lenbuffer']) > 0
            else float('nan')
        )
        ep_string = f''
        if extra:
            if locs['ep_infos']:
                for key in locs['ep_infos'][0]:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in locs['ep_infos']:
                        # handle scalar and zero dimensional tensor infos
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat(
                            (infotensor, ep_info[key].to(self.device)))
                    value = torch.mean(infotensor)
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
            
            
        log_string = (f"""{'=' * (width)}\n\n"""
                      f"""{'Iteration:':>{pad}} {locs['it']}\n"""
                      f"""{'collection:':>{pad}} {locs['collection_time']:.3f}s\n"""
                      f"""{'Learning:':>{pad}} {locs['learn_time']:.3f}s\n"""
                      f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                      f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""""
                      f"""{'Regularization loss:':>{pad}} {locs['mean_regularization_loss']:.4f}\n"""""
                      f"""{'Smooth loss:':>{pad}} {locs['mean_smooth_loss']:.4f}\n"""""
                      f"""{'Interv loss:':>{pad}} {locs['mean_interv_loss']:.4f}\n"""""
                      f"""{'Mean reward:':>{pad}} {mean_reward:.2f}\n"""
                      f"""{'Mean episode length:':>{pad}} {mean_episode_length:.2f}\n"""
                      f"""{'Episodes completed:':>{pad}} {len(locs['rewbuffer'])}\n"""
                      )
        log_string += ep_string

        self._log(log_string.rstrip())

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
