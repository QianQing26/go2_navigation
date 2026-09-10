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

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.storage import RolloutStorage
import torch.nn.functional as F

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 penalty_lr=5e-2,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # penalty params
        self.penalty_param = torch.tensor(1.0,requires_grad=True).float()
        self.penalty_optimizer = optim.Adam([self.penalty_param], lr=penalty_lr)

        # PPO-Lagragian parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape,
                     action_shape, shield_rays_shape=(41,)):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, obs_shape, action_shape,
            self.device, shield_rays_shape=shield_rays_shape
        )
        
    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()
        

    def compute_alpha_loss(self, alpha, alpha_min=0.5):
        # If alpha > alpha_min, the result is 0
        # If alpha < alpha_min, calculate the square of the difference
        penalty = F.relu(alpha_min - alpha) 
        loss_alpha = torch.mean(penalty ** 2)
        return loss_alpha

    def compute_smoothness_loss(
        self, current_states, next_states, current_safety_drift=None,
        next_safety_drift=None, current_shield_rays=None,
        next_shield_rays=None,
    ):
        if current_safety_drift is None:
            current_safety_drift = torch.zeros(
                current_states.shape[0], 1, device=current_states.device,
                dtype=current_states.dtype
            )
        if next_safety_drift is None:
            next_safety_drift = torch.zeros_like(current_safety_drift)
        if current_shield_rays is None:
            current_shield_rays = self._observation_shield_rays(current_states)
        if next_shield_rays is None:
            next_shield_rays = self._observation_shield_rays(next_states)
        batch_size = current_states.size(0)
        _u = torch.rand(batch_size, 1, device=current_states.device)
        mix_weights = ((_u - 0.5) * 2.0)
        
        # s̄ = s + (s_next - s)*u
        delta_states = next_states - current_states
        interp_states = current_states + mix_weights * delta_states
        interp_safety_drift = current_safety_drift + mix_weights * (
            next_safety_drift - current_safety_drift
        )
        interp_shield_rays = current_shield_rays + mix_weights * (
            next_shield_rays - current_shield_rays
        )
        
        # with torch.no_grad():
        self.actor_critic.act(
            current_states,
            safety_drift=current_safety_drift,
            shield_rays=current_shield_rays,
        )
        orig_actions = self.actor_critic.action_mean
        self.actor_critic.act(
            interp_states,
            safety_drift=interp_safety_drift,
            shield_rays=interp_shield_rays,
        )
        interp_actions = self.actor_critic.action_mean
        actor_smoothness = F.mse_loss(interp_actions, orig_actions)
        
        # with torch.no_grad():
        orig_values = self.actor_critic.evaluate(current_states)
        interp_values = self.actor_critic.evaluate(interp_states)
        critic_smoothness = F.mse_loss(interp_values, orig_values)
        
        total_loss = (
            1 * actor_smoothness +
            0.1 * critic_smoothness
        )
        
        return total_loss


    def _observation_shield_rays(self, observations):
        """Recover the physical rays used by the legacy actor path."""
        if not hasattr(self.actor_critic, 'extract'):
            return torch.zeros(
                observations.shape[0], self.storage.shield_rays.shape[-1],
                device=observations.device, dtype=observations.dtype
            )
        _, _, _, rays, _ = self.actor_critic.extract(observations)
        return torch.exp2(rays)

    def act(self, obs, critic_obs, safety_drift=None, shield_rays=None):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        critic_obs = obs
        supplied_context = safety_drift is not None or shield_rays is not None
        if supplied_context:
            if safety_drift is None:
                safety_drift = torch.zeros(
                    obs.shape[0], 1, device=obs.device, dtype=obs.dtype
                )
            if shield_rays is None:
                shield_rays = self._observation_shield_rays(obs)
            self.transition.actions = self.actor_critic.act(
                obs,
                safety_drift=safety_drift,
                shield_rays=shield_rays,
            ).detach()
        else:
            # Preserve the old actor call exactly for callers that do not use
            # predictive context.
            self.transition.actions = self.actor_critic.act(obs).detach()
            safety_drift = torch.zeros(
                obs.shape[0], 1, device=obs.device, dtype=obs.dtype
            )
            shield_rays = self._observation_shield_rays(obs)
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.transition.safety_drift = safety_drift.detach()
        self.transition.shield_rays = shield_rays.detach()
        return self.transition.actions
    
    def process_env_step(
        self, next_obs, rewards, dones, infos, next_safety_drift=None,
        next_shield_rays=None,
    ):
        self.transition.next_observations = next_obs
        if next_safety_drift is None:
            next_safety_drift = torch.zeros_like(self.transition.safety_drift)
        if next_shield_rays is None:
            next_shield_rays = self._observation_shield_rays(next_obs)
        self.transition.next_safety_drift = next_safety_drift.detach()
        self.transition.next_shield_rays = next_shield_rays.detach()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'bad_masks' in infos:
            self.transition.bad_masks = infos['bad_masks']
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs, infos=None):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
    
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self, **args):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_smooth_loss = 0.
        mean_regularization_loss = 0.
        mean_interv_loss = 0.0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
            
        for obs_batch, next_obs_batch, actions_batch, \
                target_values_batch, advantages_batch, returns_batch, \
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, \
                safety_drift_batch, next_safety_drift_batch, \
                shield_rays_batch, next_shield_rays_batch, \
                hid_states_batch, masks_batch, bad_masks_batch in generator:

                valid_mask = (~bad_masks_batch.bool()).flatten()
                
                self.actor_critic.act(
                    obs_batch,
                    safety_drift=safety_drift_batch,
                    shield_rays=shield_rays_batch,
                    masks=masks_batch,
                    hidden_states=hid_states_batch[0],
                )
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = (kl * valid_mask).sum() / (valid_mask.sum() + 1e-8)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate
                    
                    
                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = (torch.max(surrogate, surrogate_clipped) * valid_mask).sum() / (valid_mask.sum() + 1e-8)

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = (torch.max(value_losses, value_losses_clipped) * valid_mask.unsqueeze(1)).sum() / (valid_mask.sum() + 1e-8)
                else:
                    value_loss = ((returns_batch - value_batch).pow(2) * valid_mask.unsqueeze(1)).sum() / (valid_mask.sum() + 1e-8)
                    

                loss =  surrogate_loss \
                        + 1.0 * value_loss \
                        - self.entropy_coef * (entropy_batch * valid_mask).sum() / (valid_mask.sum() + 1e-8)
                
                clip_mins = torch.tensor([-0.5, -0.8, -1.0], device=mu_batch.device)
                clip_maxs = torch.tensor([1.7,  0.8,  1.0], device=mu_batch.device)
                range_loss = (torch.sum((mu_batch - torch.clip(mu_batch, min=clip_mins, max=clip_maxs))**2, dim=-1) * valid_mask).sum() / (valid_mask.sum() + 1e-8)
                
                smooth_loss = self.compute_smoothness_loss(
                    obs_batch, next_obs_batch, safety_drift_batch,
                    next_safety_drift_batch, shield_rays_batch,
                    next_shield_rays_batch,
                )
                regularization_loss = range_loss + 0.05 * smooth_loss
                loss += 1.0 * regularization_loss

                if hasattr(self.actor_critic, 'alpha'):
                    alpha_loss = self.compute_alpha_loss(self.actor_critic.alpha, alpha_min=1.0)
                else:
                    alpha_loss = torch.tensor(0.0)
                loss += 1.0 * alpha_loss

                # Calculate Intervention Loss: Penalize the action difference before and after the Shield
                if hasattr(self.actor_critic, 'u_bar') and hasattr(self.actor_critic, 'u_s'):
                    # Force the Nav Head to propose safer actions, and force the Alpha Head to provide a more precise alpha
                    interv_loss = torch.mean(torch.sum((self.actor_critic.u_s - self.actor_critic.u_bar)**2, dim=-1))
                    loss += 0.1 * interv_loss
                else:
                    interv_loss = torch.tensor(0.0)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_smooth_loss += smooth_loss.item()
                mean_regularization_loss += regularization_loss.item()
                mean_interv_loss += interv_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_regularization_loss /= num_updates
        mean_smooth_loss /= num_updates
        mean_interv_loss /= num_updates

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss,  mean_regularization_loss, mean_smooth_loss, mean_interv_loss
