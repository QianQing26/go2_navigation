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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfgPPO
from legged_gym.envs.base.legged_robot_pos_config import LeggedRobotPosCfg
import numpy as np


class Go2PosRoughCfg( LeggedRobotPosCfg ):
    class controller(LeggedRobotPosCfg.controller):
        # RoboGauge's exported go2_moe_cts policy. It is a stateful
        # batchable TorchScript module with a five-step internal history.
        name = 'robogauge'
        model_path = '{LEGGED_GYM_ROOT_DIR}/legged_gym/ctrl_model/go2_moe_cts_137000_0.6365_batchable.pt'
        obs_dim = 45
        history_length = 5
        obs_scale_ang_vel = 0.25
        obs_scale_dof_pos = 1.0
        obs_scale_dof_vel = 0.05
        command_scale = [2.0, 2.0, 0.25]
        # Match go2_rl_gym's training observation noise.
        add_noise = True
        noise_level = 1.0
        noise_scales = [0.2, 0.05, 0.01, 1.5]

    class loco:
        num_obs_buf = 45
        his_len = 10
        class normalization:
            class obs_scales:
                lin_vel = 2.0
                ang_vel = 0.25
                dof_pos = 1.0
                dof_vel = 0.05

    class env(LeggedRobotPosCfg.env):
        goal_reached_time = 150
        stay_time = 150
        his_len = 10
        num_nav_actions = 3
        num_props = 12
        num_rays = 41
        num_goal_obs = 2 # target x,y
        num_obs_one_step = num_props + num_rays + num_goal_obs
        num_observations = num_obs_one_step * his_len 
        # The policy/control loop runs at 50 Hz while the exteroceptive sensor
        # history is sampled at 10 Hz (one new frame every five policy steps).
        proprioception_frequency = 50.0
        exteroception_frequency = 10.0
        num_envs = 2048
        episode_length_s = 60 # episode length in seconds
        debug_viz = True

    class replay:
        enable_collision_replay = True
        replay_prob = 0.8
        # Prob to trigger reset on non-fatal collision: [min, max]
        # Increases linearly based on goal_level
        early_reset_prob_range = [0.1, 0.5] 
        undo_steps_range = [100, 150]
        max_collision_points = 10

    class visualization:
        draw_scan_dots = False
        draw_rays = True
        draw_collision_points = True
        draw_position_target = True

        # Configuration for ray visualization groups
        # format: {name: [fov_deg, style_key]}
        # If fov_deg is None, draws all rays. Use 'guide' to draw the calculated guidance ray.
        ray_groups = {
            # "all": [None, "ray_pink"],
            "guidance_navigation": ["guide", "guide_ray_marker"],
        }

        # Configuration for point visualization Styles
        # format: {name: [radius, resolution, color(r,g,b)]}
        # Decoupled from color names to allow different sizes for same colors
        points = {
            "ray_pink":   [0.015, 4, (1.0, 0.2, 1.0)],
            "ray_yellow": [0.015, 4, (1.0, 1.0, 0.0)],
            "ray_red":    [0.015, 4, (1.0, 0.0, 0.0)],
            "ray_green":  [0.015, 4, (0.0, 1.0, 0.0)],
            "goal_marker": [0.2, 8, (0.0, 0.0, 1.0)],
            "collision_marker": [0.10, 4, (1.0, 0.0, 0.0)],
            "scan_dot_obs": [0.05, 4, (1.0, 0.0, 0.0)],
            "scan_dot_safe": [0.05, 4, (0.0, 1.0, 0.0)],
            "guide_ray_marker": [0.06, 6, (0.0, 1.0, 1.0)],
        }

    class init_state( LeggedRobotPosCfg.init_state ):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }
        

    class commands(LeggedRobotPosCfg.commands):
        curriculum = False
        max_curriculum = 1.
        num_commands = 3
        delay_time = 0.1 # [s] time delay for the command
        alpha = 0.5 # Filter coefficient: alpha*new + (1-alpha)*old
        class ranges(LeggedRobotPosCfg.commands.ranges):
            limit_vx = [-0.5, 2.0]
            limit_vy = [-1.0, 1.0]
            limit_vyaw = [-1.0, 1.0]

    class control( LeggedRobotPosCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        # Match the gains used to train the RoboGauge Go2 policy.
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad
            
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotPosCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/go2_description/urdf/go2_description_v8.urdf'
        flip_visual_attachments = True
        fix_base_link = False
        name = "Go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "Head_upper", "Head_lower", "base"] # collision reward
        terminate_after_contacts_on = ["base", "Head_upper", "Head_lower"] # termination rewrad
        # terminate_after_contacts_on = [] # no termination
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class terrain( LeggedRobotPosCfg.terrain ):
        terrain_types = ['hard_room'] # easy_room, middle_room, hard_room
        terrain_proportions = [1.0]
        num_rows = 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        measure_heights = True

    class domain_rand:
        randomize_friction = True
        friction_range = [-0.2, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1.5, 1.5]
        push_robots = True
        push_interval_s = 2.5
        max_push_vel_xy = 0.0  # not used
        
        randomize_yaw = True
        randomize_roll = False
        randomize_pitch = False
        init_yaw_range = [-3.14, 3.14]
        init_roll_range = [-0.1, 0.1]
        init_pitch_range = [-0.1, 0.1]
        randomize_xy = True
        init_x_range = [-0.5, 0.5]
        init_y_range = [-0.5, 0.5]
        randomize_velo = False
        init_vlinx_range = [-0.5,0.5]
        init_vliny_range = [-0.5,0.5]
        init_vlinz_range = [-0.5,0.5]
        init_vang_range = [-0.5,0.5]

    class sensors:
        class ray2d:
            enable = False
            log2 = True
            min_dist = 0.1
            max_dist = 3.0
            theta_start = -2*np.pi/3
            theta_end = 2*np.pi/3 + 0.0001
            theta_step = np.pi/30

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.2
            height_measurements = 2.0
            ray2d = 1.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.0
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            height_measurements = 0.1    

    class rewards():
        class scales():
            termination = -100.0
            collision = -4.0
            close_obst_vel = 5.0
            stuck = -5.0
            velo_dir = 4.0
            reach_pos_target_tight = 10.0
            ang_vel_xy = -0.05
            proximity = 0.0

        class proximity_config:
            # Ray distance penalty
            slope = 15.0
            min_dist = 0.1
            
            # Speed penalty in narrow space
            narrow_slope = 10.0
            speed_limit_scale = 1.0
            speed_limit_min = 0.1
            speed_limit_max = 1.0
            
            # Rear obstacle detection (height map)
            rear_x_range = [-0.4, -0.1] # Rear ROI in meters relative to center
            rear_y_range = [-0.3, 0.3]  # Width ROI in meters relative to center
            obstacle_height_th = 0.15   # Height difference to consider an obstacle
            rear_penalty_weight = 1.0
            speed_penalty_weight = 2.0

        class velo_dir_config:
            target_speed_max = 0.5
            target_speed_scale = 1.0
            orbit_penalty_weight = 1.0 # Scale for y_vel^2 + yaw_vel^2
            reach_bonus_weight = 1.0   # Scale for 1/(1+2*d^2)

        class reach_pos_target_tight_config:
            distance_threshold = 0.5
            reach_bonus_weight = 1.0 # Scale for 1/(1+2*d^2)

        class close_obst_vel_config:
            max_rays_clip = 2.0
            kernel_size = 5 # smoothing window
            fov_deg = 150.0
            safe_vel_scale = 1.0
            safe_vel_max = 0.5
            reach_bonus_weight = 1.0   # Scale for 1/(1+2*d^2)
            overspeed_penalty_weight = 0.2 # Penalty for exceeding safe velocity

        class stuck_config:
            fov_deg = 120.0            # Total FOV in degrees
            move_dist_threshold = 0.1  # Distance moved in hist to be "not stuck"
            dead_end_threshold = 1.0   # Max space in front to be "dead end"
            backward_vel_threshold = 0.0 # v_x > 0 means not moving backward
            turn_vel_threshold = 1.0     # |yaw_vel| < 1.0 means not turning back

        class collision_config:
            contact_force_th = 0.1
            vel_square_scale = 4.0  # (1.0 + 4.0*vel_square)
            # Weights for different body parts
            base_weight = 6.0
            head_weight = 10.0
            leg_weight = 10.0
            generic_weight = 1.0

        pos_level = 'normal'
        soft_dof_pos_limit = 0.95
        base_height_target = 0.25
        only_positive_rewards = False
        position_target_sigma_soft = 2.0
        position_target_sigma_tight = 0.5
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.85
        max_contact_force = 100.

class Go2PosRoughCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunner'
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.003
        
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'Go2_pos_rough'
        
        policy_class_name = 'DifferentiableSafeActorCritic'
        # policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        max_iterations = 2000 # number of policy updates
