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
import sys
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

try:
    from motion_estimator.runtime import PredictiveDriftRuntime
except ModuleNotFoundError:
    # Training entrypoints are often launched with only the two legacy
    # ``training/*`` roots on sys.path.  Add the repository root without
    # duplicating any estimator implementation in rsl_rl.
    _repository_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../../..')
    )
    if _repository_root not in sys.path:
        sys.path.insert(0, _repository_root)
    from motion_estimator.runtime import PredictiveDriftRuntime


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
                obs_shape=[num_obs], action_shape=[num_nav_actions],
                shield_rays_shape=[num_rays])

        self.safety_mode = 'original'
        self.predictive_runtime = None
        safety_cfg = getattr(self.env.cfg.env, 'predictive_safety', None)
        if safety_cfg is not None:
            self.safety_mode = str(getattr(safety_cfg, 'mode', 'original')).lower()
        valid_modes = {
            'original', 'synchronized_static', 'predictive',
            'predictive_calibrated',
        }
        if self.safety_mode not in valid_modes:
            raise ValueError(
                'Unknown predictive safety mode {!r}; choose one of {}'.format(
                    self.safety_mode, sorted(valid_modes)
                )
            )
        if self.safety_mode in {'predictive', 'predictive_calibrated'}:
            if safety_cfg is None:
                raise ValueError('Predictive mode requires env.predictive_safety config')
            checkpoint = str(getattr(safety_cfg, 'estimator_checkpoint', '')).strip()
            if not checkpoint:
                raise ValueError(
                    'Predictive mode requires env.predictive_safety.estimator_checkpoint'
                )
            if not os.path.isabs(os.path.expanduser(checkpoint)):
                checkpoint = os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), '../../../..', checkpoint
                    )
                )
            calibration_delta = float(
                getattr(safety_cfg, 'calibration_delta', 0.0)
            )
            if self.safety_mode == 'predictive':
                calibration_delta = 0.0
            self.predictive_runtime = PredictiveDriftRuntime(
                checkpoint_path=checkpoint,
                device=self.device,
                calibration_delta=calibration_delta,
                use_warmup_gate=bool(
                    getattr(safety_cfg, 'use_warmup_gate', True)
                ),
            )
        
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
                        'collision_reward', 'termination_reward',
                        'value_loss', 'surrogate_loss', 'regularization_loss',
                        'smooth_loss', 'interv_loss',
                        'intervention_frequency', 'mean_intervention_norm',
                        'max_intervention_norm', 'residual_negative_probability',
                        'mean_safety_drift', 'negative_drift_rate',
                        'drift_induced_intervention',
                        'estimator_online_mae', 'estimator_online_rmse',
                        'estimator_online_sign_accuracy',
                        'estimator_online_false_safe_rate',
                        'estimator_online_optimistic_danger_rate',
                        'estimator_online_dangerous_count',
                        'estimator_online_count', 'curriculum_progress',
                        'configured_speed_min', 'configured_speed_max',
                        'obstacle_speed_mean', 'obstacle_speed_p50',
                        'obstacle_speed_p90', 'obstacle_speed_max',
                    ])
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self._log(
            '[runner] log_dir={}'.format(self.log_dir)
        )
        self._log('[runner] safety_mode={}'.format(self.safety_mode))
        self._log(
            '[runner] resetting environment: num_envs={}, steps_per_env={}'.format(
                self.env.num_envs, self.num_steps_per_env
            )
        )
        _, _ = self.env.reset()
        self._log('[runner] environment reset complete')

    def _legacy_shield_rays(self, obs):
        """Reconstruct the exact physical rays from the actor observation."""
        one_step = int(self.env.cfg.env.num_obs_one_step)
        latest = obs[:, -one_step:]
        rays = latest[:, self.env.num_props:self.env.num_props + self.env.rays.shape[1]]
        return torch.exp2(rays)

    def build_safety_context(self, obs):
        """Return physical ``(drift, shield_rays)`` for the current step."""
        zeros = torch.zeros(
            self.env.num_envs, 1, device=obs.device, dtype=obs.dtype
        )
        if self.safety_mode == 'original':
            return zeros, self._legacy_shield_rays(obs)
        if self.safety_mode == 'synchronized_static':
            return zeros, self.env.rays_hist[:, -1, :].to(
                device=obs.device, dtype=obs.dtype
            ).clone()
        drift, rays = self.predictive_runtime.update(self.env)
        return (
            drift.to(device=obs.device, dtype=obs.dtype),
            rays.to(device=obs.device, dtype=obs.dtype),
        )

    def reset_safety_context(self, env_ids=None):
        if self.predictive_runtime is not None:
            self.predictive_runtime.reset(env_ids)

    @staticmethod
    def _new_rollout_safety_stats():
        return {
            'samples': 0,
            'interventions': 0,
            'norm_sum': 0.0,
            'norm_max': 0.0,
            'residual_negative': 0,
            'drift_sum': 0.0,
            'negative_drift': 0,
            'drift_induced_intervention': 0,
            'online_count': 0,
            'online_abs_error_sum': 0.0,
            'online_squared_error_sum': 0.0,
            'online_sign_correct': 0,
            'online_sign_count': 0,
            'online_dangerous_count': 0,
            'online_false_safe_count': 0,
            'online_optimistic_danger_count': 0,
        }

    def _record_rollout_safety_stats(self, stats, safety_drift):
        """Accumulate detached diagnostics without entering autograd."""

        layer = getattr(self.alg.actor_critic, 'cbf_layer', None)
        if layer is None or not hasattr(layer, 'last_intervention_norm'):
            return
        with torch.no_grad():
            intervention = layer.last_intervention_norm.reshape(-1).detach()
            residual = layer.last_nominal_barrier_residual.reshape(-1).detach()
            static_residual = (
                layer.last_Lgh_u.reshape(-1) + layer.last_alpha_h.reshape(-1)
            ).detach()
            drift = safety_drift.reshape(-1).detach()
            samples = int(intervention.numel())
            stats['samples'] += samples
            stats['interventions'] += int((intervention > 1.0e-6).sum())
            stats['norm_sum'] += float(intervention.sum())
            stats['norm_max'] = max(stats['norm_max'], float(intervention.max()))
            stats['residual_negative'] += int((residual < 0.0).sum())
            stats['drift_sum'] += float(drift.sum())
            stats['negative_drift'] += int((drift < 0.0).sum())
            stats['drift_induced_intervention'] += int(
                ((static_residual >= 0.0) & (static_residual + drift < 0.0)).sum()
            )

            runtime = self.predictive_runtime
            gt = getattr(self.env, 'lse_drift_gt', None)
            if self.safety_mode != 'predictive' or runtime is None or gt is None:
                return
            inference_mask = runtime.last_inference_mask
            if inference_mask is None or not bool(inference_mask.any()):
                return
            pred = drift[inference_mask.reshape(-1)]
            target = gt.reshape(-1).detach()[inference_mask.reshape(-1)]
            finite = torch.isfinite(pred) & torch.isfinite(target)
            pred, target = pred[finite], target[finite]
            if pred.numel() == 0:
                return
            error = pred - target
            stats['online_count'] += int(pred.numel())
            stats['online_abs_error_sum'] += float(error.abs().sum())
            stats['online_squared_error_sum'] += float(error.square().sum())
            sign_mask = target.abs() > 0.05
            stats['online_sign_correct'] += int(
                (torch.sign(pred[sign_mask]) == torch.sign(target[sign_mask])).sum()
            )
            stats['online_sign_count'] += int(sign_mask.sum())
            dangerous = target < -0.05
            stats['online_dangerous_count'] += int(dangerous.sum())
            stats['online_false_safe_count'] += int((pred[dangerous] >= 0.0).sum())
            stats['online_optimistic_danger_count'] += int(
                (pred[dangerous] > target[dangerous] + 0.1).sum()
            )

    def _rollout_safety_metrics(self, stats):
        samples = max(stats['samples'], 1)
        online_count = stats['online_count']
        sign_count = stats['online_sign_count']
        dangerous_count = stats['online_dangerous_count']
        return {
            'intervention_frequency': stats['interventions'] / samples,
            'mean_intervention_norm': stats['norm_sum'] / samples,
            'max_intervention_norm': stats['norm_max'],
            'residual_negative_probability': stats['residual_negative'] / samples,
            'mean_safety_drift': stats['drift_sum'] / samples,
            'negative_drift_rate': stats['negative_drift'] / samples,
            'drift_induced_intervention': stats['drift_induced_intervention'] / samples,
            'estimator_online_mae': (
                stats['online_abs_error_sum'] / online_count if online_count else 0.0
            ),
            'estimator_online_rmse': (
                (stats['online_squared_error_sum'] / online_count) ** 0.5
                if online_count else 0.0
            ),
            'estimator_online_sign_accuracy': (
                stats['online_sign_correct'] / sign_count if sign_count else 0.0
            ),
            'estimator_online_false_safe_rate': (
                stats['online_false_safe_count'] / dangerous_count
                if dangerous_count else 0.0
            ),
            'estimator_online_optimistic_danger_rate': (
                stats['online_optimistic_danger_count'] / dangerous_count
                if dangerous_count else 0.0
            ),
            'estimator_online_dangerous_count': dangerous_count,
            'estimator_online_count': online_count,
        }

    def _curriculum_metrics(self):
        curriculum = getattr(self.env, 'get_dynamic_obstacle_curriculum', None)
        speed_stats = getattr(self.env, 'get_dynamic_obstacle_speed_statistics', None)
        if curriculum is None or speed_stats is None:
            return {
                'curriculum_progress': 0.0,
                'configured_speed_min': 0.0,
                'configured_speed_max': 0.0,
                'obstacle_speed_mean': 0.0,
                'obstacle_speed_p50': 0.0,
                'obstacle_speed_p90': 0.0,
                'obstacle_speed_max': 0.0,
            }
        current = curriculum()
        actual = speed_stats()
        return {
            'curriculum_progress': float(current['progress']),
            'configured_speed_min': float(current['speed_min']),
            'configured_speed_max': float(current['speed_max']),
            'obstacle_speed_mean': float(actual['mean']),
            'obstacle_speed_p50': float(actual['p50']),
            'obstacle_speed_p90': float(actual['p90']),
            'obstacle_speed_max': float(actual['max']),
        }

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
        def mean_episode_info(key):
            values = []
            for episode_info in locs['ep_infos']:
                if key not in episode_info:
                    continue
                value = episode_info[key]
                if isinstance(value, torch.Tensor):
                    values.append(float(value.detach().float().mean()))
                else:
                    values.append(float(value))
            return statistics.mean(values) if values else float('nan')

        safety_metrics = locs['rollout_safety_metrics']
        curriculum_metrics = locs['curriculum_metrics']
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
                mean_episode_info('rew_collision'),
                mean_episode_info('rew_termination'),
                locs['mean_value_loss'],
                locs['mean_surrogate_loss'],
                locs['mean_regularization_loss'],
                locs['mean_smooth_loss'],
                locs['mean_interv_loss'],
                safety_metrics['intervention_frequency'],
                safety_metrics['mean_intervention_norm'],
                safety_metrics['max_intervention_norm'],
                safety_metrics['residual_negative_probability'],
                safety_metrics['mean_safety_drift'],
                safety_metrics['negative_drift_rate'],
                safety_metrics['drift_induced_intervention'],
                safety_metrics['estimator_online_mae'],
                safety_metrics['estimator_online_rmse'],
                safety_metrics['estimator_online_sign_accuracy'],
                safety_metrics['estimator_online_false_safe_rate'],
                safety_metrics['estimator_online_optimistic_danger_rate'],
                safety_metrics['estimator_online_dangerous_count'],
                safety_metrics['estimator_online_count'],
                curriculum_metrics['curriculum_progress'],
                curriculum_metrics['configured_speed_min'],
                curriculum_metrics['configured_speed_max'],
                curriculum_metrics['obstacle_speed_mean'],
                curriculum_metrics['obstacle_speed_p50'],
                curriculum_metrics['obstacle_speed_p90'],
                curriculum_metrics['obstacle_speed_max'],
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
        safety_drift, shield_rays = self.build_safety_context(obs)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        tot_iter = self.current_learning_iteration + num_learning_iterations
        # self.num_steps_per_env = 1
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            rollout_safety_stats = self._new_rollout_safety_stats()
            mean_num_sim = 0
            self._log(
                '[runner] iteration {}/{} rollout start'.format(
                    it + 1, tot_iter
                )
            )
            with torch.no_grad():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(
                        obs, critic_obs, safety_drift=safety_drift,
                        shield_rays=shield_rays,
                    )
                    self._record_rollout_safety_stats(
                        rollout_safety_stats, safety_drift
                    )
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.reset_safety_context(dones)
                    next_safety_drift, next_shield_rays = self.build_safety_context(obs)
                    self.alg.process_env_step(
                        obs, rewards, dones, infos,
                        next_safety_drift=next_safety_drift,
                        next_shield_rays=next_shield_rays,
                    )
                    safety_drift, shield_rays = next_safety_drift, next_shield_rays
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos and bool(dones.any()):
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
            rollout_safety_metrics = self._rollout_safety_metrics(rollout_safety_stats)
            curriculum_metrics = self._curriculum_metrics()
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
            checkpoint_iteration = it + 1
            if self.log_dir is not None and checkpoint_iteration % self.save_interval == 0:
                self.save(
                    os.path.join(self.log_dir, 'model_{}.pt'.format(checkpoint_iteration)),
                    iteration=checkpoint_iteration,
                )
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

        safety_metrics = locs['rollout_safety_metrics']
        curriculum_metrics = locs['curriculum_metrics']
        for name in (
            'intervention_frequency', 'mean_intervention_norm',
            'max_intervention_norm', 'residual_negative_probability',
            'mean_safety_drift', 'negative_drift_rate',
            'drift_induced_intervention', 'estimator_online_mae',
            'estimator_online_rmse', 'estimator_online_sign_accuracy',
            'estimator_online_false_safe_rate',
            'estimator_online_optimistic_danger_rate',
            'estimator_online_dangerous_count', 'estimator_online_count',
        ):
            self.writer.add_scalar('Safety/{}'.format(name), safety_metrics[name], step)
        for name, value in curriculum_metrics.items():
            self.writer.add_scalar('Curriculum/{}'.format(name), value, step)

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

    def save(self, path, infos=None, iteration=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': (
                self.current_learning_iteration
                if iteration is None else int(iteration)
            ),
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def load_weights(self, path):
        """Initialize policy weights while retaining a fresh PPO optimizer."""

        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        # Keep this method correct even if a caller reuses a runner object:
        # weights-only initialization must never carry Adam moments from an
        # earlier run or from the source checkpoint.
        self.alg.optimizer.state.clear()
        if hasattr(self.alg, 'penalty_optimizer'):
            self.alg.penalty_optimizer.state.clear()
        self.current_learning_iteration = 0
        return loaded_dict.get('infos')

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
