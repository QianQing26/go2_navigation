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

from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.torch_math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, yaw_quat
from legged_gym.utils.grid2ray import *
from .legged_robot_pos_config import LeggedRobotPosCfg
from legged_gym.envs.go2.go2_pos_config import Go2PosRoughCfg
from legged_gym.utils.custom_terrain import *
from legged_gym.controllers import ControllerState, make_controller
import torch.nn.functional as F

SAVE_IMG = False
MAX_DEPTH = 10

class LeggedRobotPos(LeggedRobot):
    cfg : Go2PosRoughCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize timer_left
        self.obs_history_buf = torch.zeros(
                self.num_envs, self.cfg.env.his_len, self.cfg.env.num_obs_one_step, device=self.device, dtype=torch.float)  
        self.exteroception_step_buf = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long)
        self.exteroception_update_interval = self._get_exteroception_update_interval()
        self.actions_orig = self.actions.clone()

        # Replay and Collision History Init
        self._init_replay_buffers()

        controller_name = self.cfg.controller.name
        self.controller = make_controller(
            name=controller_name,
            controller_cfg=self.cfg.controller,
            env_cfg=self.cfg,
            num_envs=self.num_envs,
            num_actions=self.num_actions,
            device=self.device,
            joint_reindex=self.joint_reindex,
        )
        
        self.pos_hist = torch.zeros(
                self.num_envs, self.cfg.env.his_len, 2, device=self.device, dtype=torch.float)
        self.delay_goal = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)
        self.goal_hist = torch.zeros(
                self.num_envs, self.cfg.env.his_len, 2, device=self.device, dtype=torch.float)
        
        self.heading_targets = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

        self.len_y = len(self.cfg.terrain.measured_points_y)
        self.len_x = len(self.cfg.terrain.measured_points_x)
        self.c_y = int(self.len_y/2)
        
        # Dynamically find the index of the robot's center (x=0) in measured_points_x
        try:
            self.c_x = self.cfg.terrain.measured_points_x.index(0.0)
        except ValueError:
            # Fallback if exactly 0.0 is not found, find the closest
            self.c_x = np.argmin(np.abs(np.array(self.cfg.terrain.measured_points_x)))

        self.reach_goal = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool, requires_grad=False)
        self.distance = torch.zeros(self.num_envs, device=self.device, dtype=torch.float, requires_grad=False)

        self.goal_local_pos = torch.zeros(self.num_envs, 2, device=self.device, requires_grad=False)
        
        # Ray angles in radians: -2pi/3 to 2pi/3 (-120 to 120 degrees),
        # step pi/30 (6 degrees), total 41 rays.
        self.ray_angles = torch.arange(start=self.cfg.sensors.ray2d.theta_start, end=self.cfg.sensors.ray2d.theta_end, 
                                                step=self.cfg.sensors.ray2d.theta_step, device=self.device)
        self.rays = torch.ones(self.num_envs, self.ray_angles.shape[0], dtype=torch.float, device=self.device, requires_grad=False) * 5.0
        self.delay_rays = torch.ones(self.num_envs, self.ray_angles.shape[0], dtype=torch.float, device=self.device, requires_grad=False) * 5.0
        self.nav_clip_min = torch.tensor([self.cfg.commands.ranges.limit_vx[0], self.cfg.commands.ranges.limit_vy[0], self.cfg.commands.ranges.limit_vyaw[0]], dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_clip_max = torch.tensor([self.cfg.commands.ranges.limit_vx[1], self.cfg.commands.ranges.limit_vy[1], self.cfg.commands.ranges.limit_vyaw[1]], dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_actions_filtered = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        
        self.rays_hist = torch.ones(
                self.num_envs, self.cfg.env.his_len, self.ray_angles.shape[0], device=self.device, dtype=torch.float) * 5.0
        self.motion_ego_hist = torch.zeros(
                self.num_envs, self.cfg.env.his_len, 3, device=self.device, dtype=torch.float)
        self.goal_hold_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int) 
        self.stay_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.int) 
        self.goal_reached_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)  
        self.stand_still_flag = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    def _get_exteroception_update_interval(self):
        """Return the number of policy steps between exteroceptive samples."""
        frequency = float(getattr(self.cfg.env, 'exteroception_frequency', 50.0))
        if frequency <= 0.0:
            raise ValueError('env.exteroception_frequency must be positive')
        interval = int(round(1.0 / (frequency * self.dt)))
        if interval < 1:
            raise ValueError(
                'env.exteroception_frequency cannot exceed the policy frequency '
                '(dt={:.6f}s, requested={:.3f}Hz)'.format(self.dt, frequency)
            )
        actual_frequency = 1.0 / (interval * self.dt)
        if abs(actual_frequency - frequency) > 1e-4:
            raise ValueError(
                'env.exteroception_frequency={:.3f}Hz is not an integer divisor '
                'of the policy frequency (dt={:.6f}s)'.format(frequency, self.dt)
            )
        return interval

    def _init_replay_buffers(self):
        """ Initialize buffers for state replay and collision tracking. """
        # --- State Replay ---
        self.replay_len = 100 # Store ~2 seconds of history
        self.replay_root_states = torch.zeros(self.num_envs, self.replay_len, 13, device=self.device, dtype=torch.float)
        self.replay_dof_pos = torch.zeros(self.num_envs, self.replay_len, self.num_dof, device=self.device, dtype=torch.float)
        self.replay_dof_vel = torch.zeros(self.num_envs, self.replay_len, self.num_dof, device=self.device, dtype=torch.float)
        
        # --- Flags & Counters ---
        self.collision_occurred = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_collision_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.is_replay = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.replay_undo_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.fall_down = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        # --- Collision Visualization ---
        max_col_pts = getattr(self.cfg.replay, 'max_collision_points', 10)
        self.collision_pos_hist = torch.zeros(self.num_envs, max_col_pts, 3, device=self.device, dtype=torch.float)
        self.num_collisions = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
    def _update_replay_buffer(self):
        # Update replay buffer
        self.replay_root_states = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.root_states] * self.replay_len, dim=1),
            torch.cat([
                self.replay_root_states[:, 1:],
                self.root_states.unsqueeze(1)
            ], dim=1)
        )
        self.replay_dof_pos = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.dof_pos] * self.replay_len, dim=1),
            torch.cat([
                self.replay_dof_pos[:, 1:],
                self.dof_pos.unsqueeze(1)
            ], dim=1)
        )
        self.replay_dof_vel = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.dof_vel] * self.replay_len, dim=1),
            torch.cat([
                self.replay_dof_vel[:, 1:],
                self.dof_vel.unsqueeze(1)
            ], dim=1)
        )

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # Controller outputs are normalized joint-position residuals.  The
        # policy-specific residual scale belongs to the task control config,
        # matching go2_rl_gym's ``action_scale`` handling.
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        joint_pos_target = actions_scaled + self.default_dof_pos
        torques = self.p_gains * (joint_pos_target- self.dof_pos) - self.d_gains * self.dof_vel
        torques = torques 
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_actions(self, nav_actions=None):
        # Keep the filtered command available for the navigation observation
        # builder, preserving the original environment behavior.
        self.slr_commands = nav_actions
        state = ControllerState(
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            nav_command=nav_actions,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            default_dof_pos=self.default_dof_pos,
            previous_action=self.actions_orig,
            episode_length=self.episode_length_buf,
        )
        return self.controller.get_action(state)

    def post_process_actions(self):
        """ Filter and clip navigation actions to prevent sim instability. """
        alpha = self.cfg.commands.alpha
        self.nav_actions_filtered = alpha * self.nav_actions_orig + (1 - alpha) * self.nav_actions_filtered
        self.nav_actions_after_clip = torch.clip(self.nav_actions_filtered, min=self.nav_clip_min, max=self.nav_clip_max)

    def _terrain_type_for_col(self, col):
        """Return the configured terrain type that owns a terrain-map column.

        ``Terrain`` expands ``cfg.terrain.terrain_types`` into
        ``cfg.terrain.num_cols`` map columns according to
        ``terrain_proportions``.  The position task stores the expanded map
        column in ``self.terrain_types``; indexing the shorter configuration
        list directly therefore fails for the common one-type case
        (for example ``['hard_room']`` with ``num_cols=10``).
        """
        terrain_types = list(self.cfg.terrain.terrain_types)
        if not terrain_types:
            raise ValueError("cfg.terrain.terrain_types must contain at least one terrain type")
        if len(terrain_types) == 1:
            return terrain_types[0]

        proportions = np.asarray(self.cfg.terrain.terrain_proportions, dtype=np.float64)
        if proportions.size != len(terrain_types) or proportions.sum() <= 0:
            raise ValueError(
                "terrain_proportions must have one positive entry per terrain type"
            )
        boundaries = np.cumsum(proportions / proportions.sum()) * self.cfg.terrain.num_cols
        type_idx = int(np.searchsorted(boundaries, int(col), side="right"))
        type_idx = min(type_idx, len(terrain_types) - 1)
        return terrain_types[type_idx]

    def step(self, nav_actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.nav_actions_orig = torch.clip(nav_actions, -3.0, 3.0).to(self.device)
        self.post_process_actions()
        loco_actions = self._compute_actions(nav_actions=self.nav_actions_after_clip)
        self.actions_orig = torch.clip(loco_actions, -clip_actions, clip_actions).to(self.device)
        return super().step(actions=self.actions_orig)
    
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        self.custom_origins = True
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.position_targets = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # Preserve the historical two-row sampling for room tasks, while
        # allowing one-row flat test terrains without spawning off the mesh.
        num_origin_rows = max(1, min(2, int(self.cfg.terrain.num_rows)))
        self.terrain_levels = torch.randint(0, num_origin_rows, (self.num_envs,), device=self.device)
        self.goal_levels = torch.zeros(self.num_envs, device=self.device)
        self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
        self.max_terrain_level = self.cfg.terrain.num_rows
        self.ori_z = torch.zeros(self.num_envs, 1, device=self.device)
        
        for i in range(self.num_envs):
            row = int(self.terrain_levels[i])
            col = int(self.terrain_types[i])
            scaled_room = self.terrain.select_room(row, col)
            grid_size_x, grid_size_y = scaled_room.shape
            terrain_type = self._terrain_type_for_col(col)
            if terrain_type == 'flat':
                # A flat terrain has no obstacle-defined path, so the room
                # placement helper (which intentionally requires a blocked
                # path) cannot be used. Keep both poses well inside the cell.
                robot_pos = [grid_size_x // 2, grid_size_y // 2]
                goal_pos = [grid_size_x * 3 // 4, grid_size_y // 2]
            else:
                robot_pos, goal_pos = place_robot_and_goal(scaled_room)
            robot_pos = torch.tensor(robot_pos, device = self.device)
            goal_pos = torch.tensor(goal_pos, device = self.device)
            robot_origin_x = (row + (robot_pos[0])/grid_size_x) * self.terrain.env_length
            robot_origin_y = (col + (robot_pos[1])/grid_size_y) * self.terrain.env_width
            goal_origin_x = (row + (goal_pos[0])/grid_size_x) * self.terrain.env_length
            goal_origin_y = (col + (goal_pos[1])/grid_size_y) * self.terrain.env_width
            
            self.env_origins[i, 0] = robot_origin_x
            self.env_origins[i, 1] = robot_origin_y
            self.position_targets[i, 0] = goal_origin_x
            self.position_targets[i, 1] = goal_origin_y
        
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

        if self.cfg.domain_rand.randomize_yaw:
            _yaw = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(self.cfg.domain_rand.init_yaw_range[0], self.cfg.domain_rand.init_yaw_range[1])
        else:
            _yaw = torch.zeros_like(self.root_states[env_ids, 3]) 
        if self.cfg.domain_rand.randomize_roll:
            roll = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(self.cfg.domain_rand.init_roll_range[0], self.cfg.domain_rand.init_roll_range[1])
        else:
            roll = torch.zeros_like(self.root_states[env_ids, 3])
        if self.cfg.domain_rand.randomize_pitch:
            pitch = torch.zeros_like(self.root_states[env_ids, 3]).uniform_(self.cfg.domain_rand.init_pitch_range[0], self.cfg.domain_rand.init_pitch_range[1])
        else:
            pitch = torch.zeros_like(self.root_states[env_ids, 3])
        
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, _yaw)
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states_for_sim()),
                                                     gymtorch.unwrap_tensor(self._robot_actor_indices(env_ids_int32)), len(env_ids_int32))

    def _reset_collision_replay(self, env_ids):
        # Decide replay step: based on config range
        undo_range = getattr(self.cfg.replay, 'undo_steps_range', [40, 80])
        undo_steps = torch.randint(undo_range[0], undo_range[1], (len(env_ids),), device=self.device)
        
        # Cap undo_steps by current episode length
        current_len = self.episode_length_buf[env_ids]
        undo_steps = torch.min(undo_steps.long(), current_len.long())
        undo_steps = torch.min(undo_steps, torch.tensor(self.replay_len - 1, device=self.device))
        
        # Minimum history required for a valid replay
        valid_replay = undo_steps > 20
        replay_ids = env_ids[valid_replay]
        fallback_ids = env_ids[~valid_replay]
        
        if len(fallback_ids) > 0:
            if self.cfg.terrain.curriculum:
                self._update_terrain_curriculum(fallback_ids)
            self._reset_root_states(fallback_ids)
            self._reset_dofs(fallback_ids)
            self.is_replay[fallback_ids] = False
        
        if len(replay_ids) == 0:
            return replay_ids, fallback_ids

        self.is_replay[replay_ids] = True
        self.replay_undo_steps[replay_ids] = undo_steps[valid_replay]
        indices = -undo_steps[valid_replay]
        
        # Fetch from buffer
        # Buffer shape: (num_envs, replay_len, dim)
        self.root_states[replay_ids] = self.replay_root_states[replay_ids, indices]
        self.dof_pos[replay_ids] = self.replay_dof_pos[replay_ids, indices]
        self.dof_vel[replay_ids] = self.replay_dof_vel[replay_ids, indices]
        
        # Update simulation state
        env_ids_int32 = replay_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states_for_sim()),
                                                     gymtorch.unwrap_tensor(self._robot_actor_indices(env_ids_int32)), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(self._robot_actor_indices(env_ids_int32)), len(env_ids_int32))

        return replay_ids, fallback_ids

    def _reset_auxiliary_states(self, env_ids, replay_ids, normal_ids):
        """Hook for task-specific state reset alongside robot replay.

        ``replay_ids`` contains only environments for which a historical robot
        state was actually available. ``normal_ids`` also includes replay
        candidates that fell back to a normal reset because their history was
        too short.
        """

        return None

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
            
        # Separate Normal vs Replay
        # Only replay if ENABLED in config and at least one collision occurred 
        # Skip replay if it finished via success/timeout to avoid reward issues
        enable_replay = getattr(self.cfg.replay, 'enable_collision_replay', False)
        is_collision = self.collision_occurred[env_ids]
        is_success = self.goal_reached_flag[env_ids]
        is_timeout = self.time_out_buf[env_ids]
        
        prob_replay = getattr(self.cfg.replay, 'replay_prob', 0.8)
        wants_replay = enable_replay & (torch.rand(len(env_ids), device=self.device) < prob_replay) & is_collision & (~is_success) & (~is_timeout)
        
        replay_candidates = env_ids[wants_replay]
        normal_ids = env_ids[~wants_replay]
        replay_ids = env_ids.new_empty((0,), dtype=torch.long)
        replay_fallback_ids = env_ids.new_empty((0,), dtype=torch.long)
        
        # Replay Reset
        if len(replay_candidates) > 0:
            replay_ids, replay_fallback_ids = self._reset_collision_replay(replay_candidates)

        # Normal Reset
        if len(normal_ids) > 0:
            if self.cfg.terrain.curriculum:
                self._update_terrain_curriculum(normal_ids)
            self._reset_dofs(normal_ids)
            self._reset_root_states(normal_ids)
            self.is_replay[normal_ids] = False

        auxiliary_normal_ids = normal_ids
        if len(replay_fallback_ids) > 0:
            auxiliary_normal_ids = torch.cat((auxiliary_normal_ids, replay_fallback_ids))
        self._reset_auxiliary_states(env_ids, replay_ids, auxiliary_normal_ids)

        # Common Reset Logic (Buffers)
        # We do this for ALL envs
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.obs_history_buf[env_ids, :, :] = 0.
        self.exteroception_step_buf[env_ids] = 0
        self.rays_hist[env_ids, :, :] = 5.
        self.motion_ego_hist[env_ids, :, :] = 0.
        self.pos_hist[env_ids, :, :] = 0.
        self.goal_hist[env_ids, :, :] = 0.
        
        self.reset_buf[env_ids] = 1
        self.goal_reached_flag[env_ids] = 0
        self.stand_still_flag[env_ids] = 0
        self.goal_hold_timer[env_ids] = 0
        self.stay_timer[env_ids] = 0
        self.reach_goal[env_ids] = 0
        self.nav_actions_filtered[env_ids] = 0.
        self.controller.reset(env_ids)
        self.replay_undo_steps[env_ids] = 0
        
        self.contact_filt[env_ids] = False
        self.last_contacts[env_ids] = False
        self.collision_occurred[env_ids] = False # Reset collision flag
        self.last_collision_active[env_ids] = False
        self.num_collisions[env_ids] = 0 # Reset collision count for visualization
        self.collision_pos_hist[env_ids] = 0 # Clear history
        
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        
        # Extras calculation might need adjustment if mixed
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            self.extras["episode"]["goal_level"] = torch.mean(self.goal_levels.float())
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        
    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return

        move_up = self.distance[env_ids] < self.cfg.rewards.position_target_sigma_tight
        move_down = self.distance[env_ids] > self.cfg.rewards.position_target_sigma_soft
        
        self.goal_levels[env_ids] += 1 * move_up - 1 * move_down
        self.goal_levels[env_ids] = self.goal_levels[env_ids].clip(min=0, max=self.max_terrain_level)
        
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down

        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )  # (the minumum level is zero)
        
        for _, i in enumerate(env_ids):
            i = int(i)
            row = int(self.terrain_levels[i])
            col = int(self.terrain_types[i])
            scaled_room = self.terrain.select_room(row, col)
            grid_size_x, grid_size_y = scaled_room.shape
            terrain_type = self._terrain_type_for_col(col)
            if terrain_type == 'flat':
                robot_pos = [grid_size_x // 2, grid_size_y // 2]
                goal_pos = [grid_size_x * 3 // 4, grid_size_y // 2]
            else:
                robot_pos, goal_pos = place_robot_and_goal(scaled_room)
            robot_pos = torch.tensor(robot_pos, device = self.device)
            goal_pos = torch.tensor(goal_pos, device = self.device)
            robot_origin_x = (row + (robot_pos[0])/grid_size_x) * self.terrain.env_length # *10
            robot_origin_y = (col + (robot_pos[1])/grid_size_y) * self.terrain.env_width
            goal_origin_x = (row + (goal_pos[0])/grid_size_x) * self.terrain.env_length # *10
            goal_origin_y = (col + (goal_pos[1])/grid_size_y) * self.terrain.env_width
            
            self.env_origins[i, 0] = robot_origin_x
            self.env_origins[i, 1] = robot_origin_y
            self.position_targets[i, 0] = goal_origin_x
            self.position_targets[i, 1] = goal_origin_y
                         
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # step physics and render each frame
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.update_percetion()
    
    def update_percetion(self):
        self.distance = torch.norm(self.position_targets[:, :2] - self.root_states[:, :2], dim=1)
        self.far_goal = (self.distance > 0.5)
        self._get_rays()

    def _get_rays(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if not hasattr(self.terrain, 'height_points'):
            self.height_points = self._init_height_points()
            
        if env_ids is not None:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
        
        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights = torch.max(heights1, heights2)
        self.measured_heights = heights.view(self.num_envs, self.len_x, self.len_y) * self.terrain.cfg.vertical_scale
        center_height = self.measured_heights[:, self.c_x, self.c_y].unsqueeze(1).unsqueeze(2)
        raw_heights = torch.where(self.measured_heights > center_height + 0.1, 1.0, 0.0)
        self.rays = self._grid2ray(raw_heights) * self.cfg.terrain.measure_resolution

    def _grid2ray(self, grid_batch):
        base_row, base_col = self.c_x, self.c_y  
        max_radius_cells = self.cfg.sensors.ray2d.max_dist / self.cfg.terrain.measure_resolution
        final_dist_2d = batch_ray_cast_torch(grid_batch, base_row, base_col, 
                                        self.ray_angles,
                                        rad=True,
                                        max_radius=max_radius_cells, step_r=0.1)
        return final_dist_2d

    def _check_spawn_collision(self):
        """ Check if robot is spawned inside an obstacle and force reset if so. """
        # Check scan dots around the robot's initial position.
        # Check from -0.2m to +0.4m in X, and -0.2m to +0.2m in Y
        res = self.cfg.terrain.measure_resolution
        self.surr_row_start = self.c_x - int(0.2 / res)
        self.surr_row_end = self.c_x + int(0.4 / res) + 1
        self.surr_col_start = self.c_y - int(0.2 / res)
        self.surr_col_end = self.c_y + int(0.2 / res) + 1
            
        row_end = min(self.surr_row_end, self.len_x)
        col_start = max(self.surr_col_start, 0)
        col_end = min(self.surr_col_end, self.len_y)
            
        init_envs = self.initial_
        sub_heights = self.measured_heights[init_envs, self.surr_row_start:row_end, col_start:col_end]
        max_h = torch.amax(sub_heights, dim=(1, 2))
        
        bad_spawn = (max_h > 0.1)
        bad_spawn_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        bad_spawn_mask[init_envs] = bad_spawn
        self.reset_buf |= bad_spawn_mask

    def _update_collision_hist(self, new_collisions):
        """ Record collision positions for debug visualization. """
        max_pts = getattr(self.cfg.replay, 'max_collision_points', 10)
        # We store a point every few steps when in collision to avoid buffer overflow/clutter
        store_mask = new_collisions & (self.episode_length_buf % 5 == 0)
        if store_mask.any():
            self.collision_pos_hist[store_mask, self.num_collisions[store_mask] % max_pts] = self.root_states[store_mask, :3]
            self.num_collisions[store_mask] += 1

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # 0. Initialization check (first step after reset/spawn)
        # Masking initial steps to avoid penalizing spawn-in-obstacle errors
        self.initial_ = self.episode_length_buf <= 1
        self.reach_goal = self.distance < 0.5

        # Terminal Contacts (Strict termination with high penalty)
        # Based on cfg.asset.terminate_after_contacts_on
        self.terminate_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :2], dim=-1) > 1.0, dim=1)
        self.terminate_buf &= (~self.initial_) # Mask death penalty during spawn
        
        self.reset_buf = self.terminate_buf.clone() # Death always causes reset
        # Hard Force Reset (e.g., getting squashed/glitched)
        # Auxiliary actors (for example dynamic obstacles) are appended to
        # the rigid-body contact tensor.  Their contacts must not terminate
        # the robot environment; only the robot's own bodies can indicate
        # that the robot was squashed or physically corrupted.
        robot_contact_forces = self.contact_forces[:, :self.num_bodies, :2]
        self.reset_buf |= torch.any(torch.norm(robot_contact_forces, dim=-1) > 50.0, dim=1)

        # Spawn-in-obstacle detection
        if self.initial_.any():
            self._check_spawn_collision()
        
        # Add initial flag to extras as a bad_mask to discard transitions in PPO
        self.extras["bad_masks"] = self.initial_

        # Collision Tracking & Early Replay Reset (NON-terminal, no -500 penalty)
        new_collisions = torch.any((torch.norm(self.contact_forces[:, self.penalised_contact_indices, :2], dim=-1) > 1.0), dim=1)
        new_collisions &= (~self.initial_) # Mask collision tracking for replay during initial steps
        
        # Detect the onset of collision to avoid repetitive triggering over consecutive frames
        is_new_collision = new_collisions & (~self.last_collision_active)
        
        # Curriculum for early_reset_prob: pull range from config
        prob_range = getattr(self.cfg.replay, 'early_reset_prob_range', [0.1, 0.67])
        early_prob_min, early_prob_max = prob_range[0], prob_range[1]
        
        # goal_levels is updated during curriculum, providing a per-env difficulty measure
        # Fixed scaling: 0.1 at level 0, max prob at level 1.5
        early_prob = early_prob_min + (early_prob_max - early_prob_min) * (self.goal_levels / 1.5).clip(max=1.0)
        
        trigger_replay_mask = is_new_collision & (torch.rand(self.num_envs, device=self.device) < early_prob)
        
        # IMPORTANT: Updated to include termination penalty for early resets to discourage collision-seeking behavior.
        self.reset_buf |= trigger_replay_mask
        self.terminate_buf |= trigger_replay_mask # Add penalty
        
        # Record collision positions for history
        if new_collisions.any():
            self._update_collision_hist(new_collisions)

        self.collision_occurred |= new_collisions
        self.last_collision_active = new_collisions # Record current state for next frame
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.fall_down = self.projected_gravity[:, 2] > -0.8
        
        # Stricter static (stuck) condition:
        # Either very low velocity OR very small displacement in the last ~100 steps
        v_low = (torch.norm(self.base_lin_vel[:, :2], dim=-1) < 0.1) & (torch.abs(self.base_ang_vel[:, 2]) < 0.1)
        d_low = torch.norm(self.root_states[:, :2] - self.pos_hist[:, 0, :2], dim=-1) < 0.2
        
        self.not_just_reset = (self.episode_length_buf/self.max_episode_length) > 0.1
        self.static = (v_low | d_low) & self.not_just_reset
        
        self.goal_hold_timer = self.goal_hold_timer + (self.reach_goal).int()
        self.stay_timer = self.stay_timer + (self.static).int()
        
        goal_reached_time = self.cfg.env.goal_reached_time 
        stay_time = self.cfg.env.stay_time 
        self.goal_reached_flag = (self.goal_hold_timer >= goal_reached_time)
        self.stand_still_flag = (self.stay_timer >= (stay_time))

        self.reset_buf |= self.goal_reached_flag
        self.reset_buf |= self.stand_still_flag
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self.fall_down

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            if torch.isnan(rew).nonzero().any():
                raise ValueError(f"NaN detected in reward term '{name}'")
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    def _get_perception(self):
        """Update asynchronous exteroception and build the delayed sample.

        The policy still runs at the control frequency (50 Hz), but the
        exteroceptive sensor/history is sampled at ``env.exteroception_frequency``
        (10 Hz for Go2).  The history therefore advances only every five policy
        steps.  Between sensor updates, the current proprioception is refreshed
        in the newest fused history frame while the ray/goal history remains
        unchanged.
        """
        pos_diff = self.position_targets - self.root_states[:, 0:3]
        self.goal_local_pos = quat_rotate_inverse(yaw_quat(self.base_quat), pos_diff)[:, :2]

        is_new_episode = self.episode_length_buf <= 1
        self.exteroception_step_buf[is_new_episode] = 0
        self.exteroception_step_buf += 1
        update_ids = (
            is_new_episode
            | (self.exteroception_step_buf >= self.exteroception_update_interval)
        ).nonzero(as_tuple=False).flatten()
        if len(update_ids) == 0:
            return update_ids

        self.exteroception_step_buf[update_ids] = 0
        self.rays_rand = self.rays.clone() + torch.rand_like(self.rays) * 0.0
        motion_ego = torch.cat((
            self.base_lin_vel[:, :2], self.base_ang_vel[:, 2:3]
        ), dim=-1)
        reset_ids = update_ids[is_new_episode[update_ids]]
        rolling_ids = update_ids[~is_new_episode[update_ids]]

        if len(reset_ids) > 0:
            self.rays_hist[reset_ids] = self.rays_rand[reset_ids].unsqueeze(1).expand(
                -1, self.cfg.env.his_len, -1
            )
            self.motion_ego_hist[reset_ids] = motion_ego[reset_ids].unsqueeze(1).expand(
                -1, self.cfg.env.his_len, -1
            )
            self.goal_hist[reset_ids] = self.goal_local_pos[reset_ids].unsqueeze(1).expand(
                -1, self.cfg.env.his_len, -1
            )

        if len(rolling_ids) > 0:
            self.rays_hist[rolling_ids] = torch.cat((
                self.rays_hist[rolling_ids, 1:],
                self.rays_rand[rolling_ids].unsqueeze(1),
            ), dim=1)
            self.motion_ego_hist[rolling_ids] = torch.cat((
                self.motion_ego_hist[rolling_ids, 1:],
                motion_ego[rolling_ids].unsqueeze(1),
            ), dim=1)
            self.goal_hist[rolling_ids] = torch.cat((
                self.goal_hist[rolling_ids, 1:],
                self.goal_local_pos[rolling_ids].unsqueeze(1),
            ), dim=1)

        # commands.delay_time is expressed in seconds.  Since the history is
        # now sampled at the sensor rate, select the corresponding old sensor
        # frame instead of using a 50 Hz frame index.
        delay_time = float(getattr(self.cfg.commands, 'delay_time', 0.0))
        delay_frames = max(0, int(round(delay_time * self.cfg.env.exteroception_frequency)))
        delay_index = -min(self.cfg.env.his_len, delay_frames + 1)
        self.delay_rays[update_ids] = self.rays_hist[update_ids, delay_index, :]
        self.delay_goal[update_ids] = self.goal_hist[update_ids, delay_index, :]
        return update_ids

    def compute_observations(self):
        """ Computes observations
        """
        self._update_replay_buffer()
        self.prop_buf = torch.cat((
                                self.projected_gravity, # 0:3
                                self.slr_commands[:, :3] * self.commands_scale[:3], # 3:6
                                # self.nav_actions_orig * self.commands_scale, # 3:6
                                self.base_lin_vel * 1.0, # 6:9 
                                self.base_ang_vel * 1.0 # 9:12
                                ), dim=-1)
            
        noise_scales = self.cfg.noise.noise_scales
        noise_vec = torch.cat((
                               torch.ones(3) * noise_scales.gravity,
                               torch.zeros(3),
                               torch.ones(3) * noise_scales.lin_vel * 1.0,
                               torch.ones(3) * noise_scales.ang_vel * 1.0,
                               ), dim=0)
        
        if self.cfg.noise.add_noise:
            self.prop_buf += (2 * torch.rand_like(self.prop_buf) - 1) * noise_vec.to(self.device)

        exteroception_update_ids = self._get_perception()
        
        env_ids = (self.episode_length_buf % 10 == 0).nonzero(as_tuple=False).flatten()
        self.pos_hist[env_ids] = torch.where(
            (self.episode_length_buf[env_ids] <= 1)[:, None, None],
            torch.stack([self.root_states[env_ids,:2]] * self.cfg.env.his_len, dim=1),
            torch.cat([
                self.pos_hist[env_ids, 1:],
                self.root_states[env_ids,:2].unsqueeze(1)
            ], dim=1)
        )

        obs_buf = torch.cat((
                            self.prop_buf, # 12
                            torch.log2(self.delay_rays.clip(min=0.1, max=5.0)), # num_rays 41
                            self.delay_goal, # 2
                            ), dim=-1)

        is_new_episode = self.episode_length_buf <= 1
        reset_ids = is_new_episode.nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            self.obs_history_buf[reset_ids] = obs_buf[reset_ids].unsqueeze(1).expand(
                -1, self.cfg.env.his_len, -1
            )

        rolling_ids = exteroception_update_ids[~is_new_episode[exteroception_update_ids]]
        if len(rolling_ids) > 0:
            self.obs_history_buf[rolling_ids] = torch.cat((
                self.obs_history_buf[rolling_ids, 1:],
                obs_buf[rolling_ids].unsqueeze(1),
            ), dim=1)

        # Proprioception is available at 50 Hz.  Keep the newest fused frame
        # current even when the exteroceptive history does not advance.
        non_update_ids = (~is_new_episode).nonzero(as_tuple=False).flatten()
        if len(non_update_ids) > 0:
            self.obs_history_buf[non_update_ids, -1, :self.cfg.env.num_props] = \
                self.prop_buf[non_update_ids]

        self.obs_buf = self.obs_history_buf.view(self.num_envs, -1)
        
    def _draw_ray_vis(self):
        """ Draws visualizations for rays with configurable groups/FOVs. 
            Refactored to separate selection and rendering logic.
        """
        vis_cfg = self.cfg.visualization
        if not vis_cfg.draw_rays:
            return

        for name, params in vis_cfg.ray_groups.items():
            fov_deg, style_key = params[0], params[1]
            if style_key not in vis_cfg.points:
                continue
            
            # Select Rays based on configuration logic
            ray_mask = self._select_rays_by_config(fov_deg)
            
            # Render the selected rays
            if ray_mask.any():
                self._render_rays(ray_mask, style_key)

    def _select_rays_by_config(self, fov_config):
        """ Helper: Returns a boolean mask of rays to be visualized based on fov configuration. """
        if fov_config is None:
            return torch.ones_like(self.ray_angles, dtype=torch.bool)
            
        # Handle 'guide' semantic key
        if str(fov_config).lower() == 'guide':
            mask = torch.zeros_like(self.ray_angles, dtype=torch.bool)
            if hasattr(self, 'guide_ray_idx') and self.guide_ray_idx is not None:
                # Visualize for the first environment (debugging view)
                mask[self.guide_ray_idx[0]] = True
            return mask
            
        # Handle numerical FOV degrees
        try:
            fov_rad = np.deg2rad(float(fov_config))
            return torch.abs(self.ray_angles) <= fov_rad / 2.0
        except ValueError:
             return torch.zeros_like(self.ray_angles, dtype=torch.bool)

    def _render_rays(self, ray_mask, style_key):
        """ Helper: Transforms and draws the selected rays for the primary environment (Env 0). """
        env_idx = 0 # Focus debug on first env
        
        # Retrieve Style Geometry
        vis_cfg = self.cfg.visualization
        p_geom = vis_cfg.points[style_key]
        sphere_geom = gymutil.WireframeSphereGeometry(p_geom[0], p_geom[1], p_geom[1], None, color=p_geom[2])

        # Get Active Data
        active_angles = self.ray_angles[ray_mask]
        active_lengths = self.rays[env_idx, ray_mask]
        
        # Coordinate Transformation: Polar (Local) -> Cartesian (Local)
        local_x = active_lengths * torch.cos(active_angles)
        local_y = active_lengths * torch.sin(active_angles)
        local_points = torch.stack([local_x, local_y, torch.zeros_like(local_x)], dim=-1)
        
        # Local -> World Transform
        robot_pos = self.root_states[env_idx, :3]
        robot_quat = self.base_quat[env_idx]
        
        # Apply rotation (broadcast quaternion to match number of rays)
        rays_vec_world = quat_apply_yaw(robot_quat.unsqueeze(0).repeat(local_points.shape[0], 1), local_points)
        rays_end_world = robot_pos + rays_vec_world
        
        # Drawing Loop (Draw dots along each ray)
        start_pos = robot_pos
        for i in range(len(active_angles)):
            end_pos = rays_end_world[i]
            # Interpolate 10 dots along the ray
            for step in range(1, 11):
                p = start_pos + (end_pos - start_pos) * (step / 10.0)
                pose = gymapi.Transform(gymapi.Vec3(p[0], p[1], p[2]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_idx], pose)

    def _draw_scan_dots_vis(self):
        """ Draws visualizations for debugging (slows down simulation a lot).
            Visualizes the obstacle detection Region of Interest (ROI) from the height map.
        """
        # Can use any ROI range for debugging, e.g., from proximity reward config
        rew_cfg = self.cfg.rewards.proximity_config
        _, is_obstacle_mask, flat_indices_grid = self._get_height_roi_info(rew_cfg.rear_x_range, rew_cfg.rear_y_range, rew_cfg.obstacle_height_th)
        flat_indices = flat_indices_grid.flatten()

        for i in range(self.num_envs):
            if i > 0: break # Only draw for first env

            # Points in base frame (from height_points)
            subset_points_base = self.height_points[i][flat_indices]
            
            # Move to world frame
            subset_points_world = quat_apply_yaw(self.base_quat[i].repeat(subset_points_base.shape[0], 1), subset_points_base) + (self.root_states[i, :3])
            
            # Update heights
            heights_flat = self.measured_heights[i].flatten()
            subset_heights = heights_flat[flat_indices]
            subset_points_world[:, 2] = subset_heights
            
            # Draw using _draw_point with style keys
            is_obs = is_obstacle_mask[i].flatten().bool()
            if is_obs.any():
                self._draw_point(subset_points_world[is_obs], style="scan_dot_obs", env_ids=[i])
            if (~is_obs).any():
                self._draw_point(subset_points_world[~is_obs], style="scan_dot_safe", env_ids=[i])

    def _draw_point(self, pos, style="goal_marker", env_ids=None):
        """ Draws spheres at given positions with configurable styles. 
            Args:
                pos (torch.Tensor): Positions to draw (num_envs, N, 3) or (N, 3) 
                style (str): Style key in cfg.visualization.points
                env_ids (list): Subset of environments to draw for. Defaults to all.
        """
        vis_cfg = self.cfg.visualization
        if style not in vis_cfg.points:
            return
            
        params = vis_cfg.points[style]
        geom = gymutil.WireframeSphereGeometry(params[0], params[1], params[1], None, color=params[2])
        
        # Pre-cache "red" geom for replay if necessary (Special case for collision tracking)
        geom_replay = None
        if style == "goal_marker" and "collision_marker" in vis_cfg.points:
            p_red = vis_cfg.points["collision_marker"]
            geom_replay = gymutil.WireframeSphereGeometry(p_red[0], p_red[1], p_red[1], None, color=p_red[2])

        env_ids = env_ids if env_ids is not None else range(self.num_envs)
        
        for i in env_ids:
            # Handle both (num_envs, N, 3) and (N, 3)
            p_batch = pos[i] if pos.dim() == 3 else pos
            points_np = p_batch.cpu().numpy()
            
            # Use red if it's a replay situation and we are drawing the main target
            current_geom = geom
            if style == "goal_marker" and self.is_replay[i] and geom_replay is not None:
                current_geom = geom_replay
                
            for p in points_np:
                sphere_pose = gymapi.Transform(gymapi.Vec3(p[0], p[1], p[2]), r=None)
                gymutil.draw_lines(current_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _draw_debug_vis(self):
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        if getattr(self.cfg.visualization, 'draw_position_target', True):
            self._draw_point(self.position_targets, style="goal_marker")
        if getattr(self.cfg.visualization, 'draw_collision_points', False):
            self._draw_collision_history_vis()
        if getattr(self.cfg.visualization, 'draw_scan_dots', False):
            self._draw_scan_dots_vis()
        if getattr(self.cfg.visualization, 'draw_rays', False):
            self._draw_ray_vis()
            
    def _draw_collision_history_vis(self):
        """ Draws past collision points for selected envs. """
        env_ids = [0] # Usually just for debugging first env
        for i in env_ids:
            num = self.num_collisions[i]
            if num > 0:
                limit = min(num, self.collision_pos_hist.shape[1])
                self._draw_point(self.collision_pos_hist[i, :limit], style="collision_marker", env_ids=[i])

    def _get_clearance(self, fov_deg=None):
        """ Returns min and max clearance within a specified field of view. """
        if fov_deg is not None:
            angle_threshold = (fov_deg / 2.0) * np.pi / 180.0
            indices_fov = (torch.abs(self.ray_angles) <= angle_threshold).nonzero(as_tuple=True)[0]
        else:
            indices_fov = torch.arange(len(self.ray_angles), device=self.device)
            
        subset_rays = self.rays[:, indices_fov]
        min_clearance = torch.min(subset_rays, dim=-1)[0]
        max_clearance = torch.max(subset_rays, dim=-1)[0]
        return min_clearance, max_clearance
        
    def _get_guidance_nav_alignment(self, fov_deg=None):
        """ Calculates the alignment of the optimal navigation direction based on smoothed continuous openings. """
        # Clip rays to remove magnitude bias from very distant empty spaces.
        rays_clipped = torch.clamp(self.rays, max=self.cfg.rewards.close_obst_vel_config.max_rays_clip)
        
        # Identify the center of the guidance broad opening via smoothing.
        kernel_size = self.cfg.rewards.close_obst_vel_config.kernel_size # 30 degrees smoothing window (5 * 6deg)
        # Pad to keep the same number of output dimensions
        rays_padded = torch.nn.functional.pad(rays_clipped.unsqueeze(1), (kernel_size//2, kernel_size//2), mode='replicate')
        smoothed_rays = torch.nn.functional.avg_pool1d(rays_padded, kernel_size, stride=1).squeeze(1)
        
        # Apply FOV masking (Mask out rays outside the FOV with a strong negative value)
        if fov_deg is not None:
            angle_threshold = (fov_deg / 2.0) * np.pi / 180.0
            mask_fov = torch.abs(self.ray_angles) <= angle_threshold
            smoothed_rays_masked = torch.where(mask_fov, smoothed_rays, torch.tensor(-1.0, device=self.device))
        else:
            smoothed_rays_masked = smoothed_rays

        center_bias = -torch.abs(self.ray_angles) * 0.001
        scores = smoothed_rays_masked + center_bias
        
        _, guide_opening_idx_global = torch.max(scores, dim=-1)
        
        self.guide_ray_idx = guide_opening_idx_global # stored for visualization
        
        dir_alignment = torch.cos(self.ray_angles[guide_opening_idx_global]).clip(min=0.0)
        return dir_alignment

    def _get_height_roi_info(self, x_range, y_range, height_th):
        """ Returns processing information for a specific height map Region of Interest (ROI).
            Args:
                x_range (list): [min_x, max_x] relative to robot center in meters.
                y_range (list): [min_y, max_y] relative to robot center in meters.
                height_th (float): Height difference threshold to consider a point an obstacle.
            Returns:
                rear_heights (torch.Tensor): Height samples in the ROI (num_envs, rows, cols)
                is_obstacle (torch.Tensor): Boolean mask of obstacles (num_envs, rows, cols)
                flat_indices (torch.Tensor): Flat indices of the ROI points for height_points lookup
        """
        res = getattr(self.cfg.terrain, 'measure_resolution', 0.1)
        idx_x_start = int(x_range[0] / res)
        idx_x_end = int(x_range[1] / res)
        idx_y_start = int(y_range[0] / res)
        idx_y_end = int(y_range[1] / res)
        
        r_start = max(0, self.c_x + idx_x_start)
        r_end = min(self.measured_heights.shape[1], self.c_x + idx_x_end)
        c_start = max(0, self.c_y + idx_y_start)
        c_end = min(self.measured_heights.shape[2], self.c_y + idx_y_end)
        
        rear_heights = self.measured_heights[:, r_start:r_end, c_start:c_end]
        center_h_val = self.measured_heights[:, self.c_x, self.c_y].unsqueeze(1).unsqueeze(2)
        is_obstacle = (rear_heights > (center_h_val + height_th)).float()
        
        # Calculate indices for coordinate lookup (used by visualization)
        rows = torch.arange(r_start, r_end, device=self.device)
        cols = torch.arange(c_start, c_end, device=self.device)
        r_grid, c_grid = torch.meshgrid(rows, cols, indexing='ij')
        flat_indices = r_grid * self.len_y + c_grid
        
        return rear_heights, is_obstacle, flat_indices

    #### rewards
    def _reward_reach_pos_target_tight(self):
        rew_cfg = self.cfg.rewards.reach_pos_target_tight_config
        reach_reaward =  (rew_cfg.reach_bonus_weight /(1.0 + 2 * torch.square(self.distance))) * (self.distance < rew_cfg.distance_threshold)
        return reach_reaward

    def _reward_velo_dir(self):
        rew_cfg = self.cfg.rewards.velo_dir_config
        goal_dir_norm = self.goal_local_pos / (self.distance.unsqueeze(1) + 1e-4)
        alignment = goal_dir_norm[:, 0].clip(min=0.0)
        target_speed = (self.distance * rew_cfg.target_speed_scale).clip(max=rew_cfg.target_speed_max)
        forward_vel = torch.clamp(self.base_lin_vel[:, 0], min=0.0)
        vel_reward = alignment * forward_vel
        vel_reward = torch.clamp(vel_reward, max=target_speed)
        reach_bonus = rew_cfg.reach_bonus_weight / (1.0 + 2 * torch.square(self.distance))
        return vel_reward + reach_bonus
   
    def _reward_termination(self):
        return (self.terminate_buf).float() 
    
    def _reward_close_obst_vel(self): 
        rew_cfg = self.cfg.rewards.close_obst_vel_config
        
        front_clearance, _ = self._get_clearance(fov_deg=None)
        dir_alignment = self._get_guidance_nav_alignment(fov_deg=rew_cfg.fov_deg)
        # Limit velocity based on nearest obstacle in FOV
        x_vel = self.base_lin_vel[:, 0].clip(min=0.0)
        safe_vel_limit = (front_clearance * rew_cfg.safe_vel_scale).clip(max=rew_cfg.safe_vel_max)
        # Positive reward only for safe velocities
        reward_vel_clamped = torch.min(x_vel, safe_vel_limit)
        # Base positive reward constrained by alignment and path viability
        reward_base = dir_alignment * reward_vel_clamped
        # Overspeed penalty: Explicitly penalize velocities exceeding safe limits
        overspeed = (x_vel - safe_vel_limit).clip(min=0.0)
        overspeed_penalty = overspeed * rew_cfg.overspeed_penalty_weight
        reward = (reward_base - overspeed_penalty).clip(min=0.0)
        reach_bonus = rew_cfg.reach_bonus_weight / (1.0 + 2 * torch.square(self.distance))
        return reward * self.far_goal + (~self.far_goal) * reach_bonus

    def _reward_stuck(self):
        rew_cfg = self.cfg.rewards.stuck_config
        distances_hist = torch.norm(self.pos_hist - self.root_states[:, None, :2], dim=-1)
        move_dist_max = torch.max(distances_hist, dim=-1)[0]
        stuck = move_dist_max < rew_cfg.move_dist_threshold
        stand_velo = (torch.abs(self.base_ang_vel[:, 2]) + torch.abs(self.base_lin_vel[:, 1].clip(max=0.5))) + torch.abs(self.base_lin_vel[:, 0].clip(max=0.5))
        _, max_front_space = self._get_clearance(fov_deg=rew_cfg.fov_deg)
        is_dead_end = max_front_space < rew_cfg.dead_end_threshold
        no_backward = self.base_lin_vel[:, 0] > rew_cfg.backward_vel_threshold
        no_turn_back = torch.abs(self.base_ang_vel[:, 2]) < rew_cfg.turn_vel_threshold
        no_escape = is_dead_end & (no_backward | no_turn_back)
        return  self.not_just_reset * self.far_goal * (no_escape + is_dead_end) * stuck + stand_velo * (~self.far_goal)
    
    def _reward_collision(self):
        rew_cfg = self.cfg.rewards.collision_config
        mask = (~self.initial_).float()
        th = rew_cfg.contact_force_th
        rew_coll = torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices, :2], dim=-1) > th), dim=1) * mask
        rew_coll_head_base = torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices[8:11], :2], dim=-1) > th), dim=1) * mask
        rew_coll_legs = torch.sum((torch.norm(self.contact_forces[:, self.penalised_contact_indices[0:8], :2], dim=-1) > th), dim=1) * mask
        vel_square = torch.square(self.base_lin_vel[:, :2]).sum(dim=-1) + torch.square(self.base_ang_vel[:, 2])
        total_coll = rew_coll * rew_cfg.generic_weight + rew_coll_head_base * rew_cfg.head_weight + rew_cfg.leg_weight * rew_coll_legs
        return (1.0 + rew_cfg.vel_square_scale * vel_square) * total_coll
