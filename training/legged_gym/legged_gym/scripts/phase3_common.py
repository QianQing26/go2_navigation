"""Shared simulator setup and state restoration for Phase-3 evaluation."""

import os
import subprocess

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from rsl_rl.utils.phase2 import PHASE2_SPEED_FINAL


def configure_evaluation_cfg(
    env_cfg, train_cfg, seed, num_envs, safety_mode='original',
    estimator_checkpoint='', stratified_speed=False,
):
    """Apply only evaluation controls; reward and policy configs are untouched."""

    env_cfg.seed = int(seed)
    env_cfg.env.num_envs = int(num_envs)
    env_cfg.env.debug_viz = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.replay.enable_collision_replay = False
    if hasattr(env_cfg.replay, 'enable_dynamic_obstacle_replay'):
        env_cfg.replay.enable_dynamic_obstacle_replay = False
    env_cfg.visualization.draw_rays = False
    env_cfg.visualization.draw_position_target = False

    safety_cfg = env_cfg.env.predictive_safety
    safety_cfg.mode = safety_mode
    safety_cfg.estimator_checkpoint = estimator_checkpoint
    safety_cfg.calibration_delta = 0.0
    safety_cfg.use_warmup_gate = True
    train_cfg.runner.resume = False

    if stratified_speed:
        # This is a bank-generation-only sampling policy.  It makes the
        # cohort cover the full difficulty range without changing training.
        env_cfg.dynamic_obstacles.dataset_speed_sampling = 'stratified'
        env_cfg.dynamic_obstacles.dataset_speed_sampling_range = list(
            PHASE2_SPEED_FINAL
        )


def repository_commit():
    """Return the current repository commit when the script is run from git."""

    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../../../..'
    ))
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_root,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ''
    return result.stdout.strip()


def _clear_episode_buffers(env, episode_length=0):
    """Clear reset-owned state without invoking a stochastic environment reset."""

    # ``BaseTask.reset`` normally creates these navigation tensors as a side
    # effect of its zero-action step.  Fixed-bank generation deliberately
    # stops before that step, so initialize the same state explicitly.
    if not hasattr(env, 'slr_commands'):
        env.slr_commands = torch.zeros(
            env.num_envs, env.num_nav_actions,
            device=env.device, dtype=torch.float,
        )
    if not hasattr(env, 'nav_actions_orig'):
        env.nav_actions_orig = torch.zeros_like(env.slr_commands)
    if not hasattr(env, 'nav_actions_after_clip'):
        env.nav_actions_after_clip = torch.zeros_like(env.slr_commands)

    env.episode_length_buf[:] = int(episode_length)
    env.reset_buf.zero_()
    env.time_out_buf.zero_()
    for name in (
        'goal_reached_flag', 'stand_still_flag', 'reach_goal',
        'collision_occurred', 'last_collision_active', 'is_replay',
        'fall_down', 'last_contacts', 'contact_filt',
    ):
        value = getattr(env, name, None)
        if value is not None:
            value.zero_()
    for name in ('goal_hold_timer', 'stay_timer', 'replay_undo_steps', 'num_collisions'):
        value = getattr(env, name, None)
        if value is not None:
            value.zero_()
    for name in (
        'last_actions', 'last_dof_vel', 'last_root_vel', 'nav_actions_orig',
        'nav_actions_filtered', 'nav_actions_after_clip', 'actions_orig',
        'actions', 'slr_commands', 'rays_hist', 'motion_ego_hist',
        'goal_hist', 'obs_history_buf', 'pos_hist',
    ):
        value = getattr(env, name, None)
        if value is not None:
            value.zero_()
    if hasattr(env, 'controller'):
        env.controller.reset()
    env.common_step_counter = 0


def prepare_fresh_scenario_state(env):
    """Sample exactly one real reset, then leave the env before its first action."""

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(env_ids)
    _clear_episode_buffers(env, episode_length=0)
    _refresh_robot_derived_state(env)
    env.update_percetion()
    env.compute_observations()
    env.reset_buf.zero_()
    env.time_out_buf.zero_()
    return env_ids


def _refresh_robot_derived_state(env):
    env.base_quat[:] = env.root_states[:, 3:7]
    env.base_lin_vel[:] = quat_rotate_inverse(
        env.base_quat, env.root_states[:, 7:10]
    )
    env.base_ang_vel[:] = quat_rotate_inverse(
        env.base_quat, env.root_states[:, 10:13]
    )
    env.projected_gravity[:] = quat_rotate_inverse(
        env.base_quat, env.gravity_vec
    )


def capture_scenario_tensors(env):
    """Capture every reset-time state used by the current dynamic task."""

    return {
        'scenario_id': torch.arange(
            env.num_envs, device=env.device, dtype=torch.long
        ),
        'robot_root_state': env.root_states.detach().clone(),
        'robot_dof_state': env.dof_state.view(env.num_envs, env.num_dof, 2).detach().clone(),
        'position_target': env.position_targets.detach().clone(),
        'env_origin': env.env_origins.detach().clone(),
        'terrain_level': env.terrain_levels.detach().clone(),
        'terrain_type': env.terrain_types.detach().clone(),
        'goal_level': env.goal_levels.detach().clone(),
        'dynamic_obstacle_start': env.dynamic_obstacle_start.detach().clone(),
        'dynamic_obstacle_velocity': env.dynamic_obstacle_velocity.detach().clone(),
        'dynamic_obstacle_time': env.dynamic_obstacle_time.detach().clone(),
        'dynamic_obstacle_effective_low': env.dynamic_obstacle_effective_low.detach().clone(),
        'dynamic_obstacle_effective_high': env.dynamic_obstacle_effective_high.detach().clone(),
        'dynamic_obstacle_current_speed_range': env.dynamic_obstacle_current_speed_range.detach().clone(),
        'episode_length': env.episode_length_buf.detach().clone(),
        'initial_commands': env.slr_commands.detach().clone(),
        'initial_nav_actions_filtered': env.nav_actions_filtered.detach().clone(),
        'initial_actions_orig': env.actions_orig.detach().clone(),
        'scenario_vmax_speed_mps': torch.linalg.vector_norm(
            env.dynamic_obstacle_velocity, dim=-1
        ).amax(dim=-1),
    }


def capture_static_terrain(env):
    return {
        'terrain_height_field_raw': torch.as_tensor(
            env.terrain.height_field_raw.copy()
        ),
        'terrain_env_origins': torch.as_tensor(
            env.terrain.env_origins.copy(), dtype=torch.float32
        ),
    }


def restore_scenario_batch(env, bank, scenario_indices):
    """Restore a batch without calling reset_idx or sampling a second episode."""

    scenario_indices = torch.as_tensor(
        scenario_indices, device=env.device, dtype=torch.long
    )
    if scenario_indices.numel() != env.num_envs:
        raise ValueError(
            'fixed-cohort batch must contain exactly env.num_envs scenarios '
            '(got {}, expected {})'.format(
                scenario_indices.numel(), env.num_envs
            )
            )
    data = bank['scenarios']
    # The serialized bank is intentionally CPU-resident.  Keep indexing on
    # CPU, then transfer each selected state tensor to the simulator device.
    scenario_indices_cpu = scenario_indices.detach().cpu()
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    for key in (
        'position_target', 'env_origin', 'terrain_level', 'terrain_type',
        'goal_level', 'dynamic_obstacle_start', 'dynamic_obstacle_velocity',
        'dynamic_obstacle_time', 'dynamic_obstacle_effective_low',
        'dynamic_obstacle_effective_high', 'dynamic_obstacle_current_speed_range',
    ):
        getattr(env, {
            'position_target': 'position_targets',
            'env_origin': 'env_origins',
            'terrain_level': 'terrain_levels',
            'terrain_type': 'terrain_types',
            'goal_level': 'goal_levels',
            'dynamic_obstacle_start': 'dynamic_obstacle_start',
            'dynamic_obstacle_velocity': 'dynamic_obstacle_velocity',
            'dynamic_obstacle_time': 'dynamic_obstacle_time',
            'dynamic_obstacle_effective_low': 'dynamic_obstacle_effective_low',
            'dynamic_obstacle_effective_high': 'dynamic_obstacle_effective_high',
            'dynamic_obstacle_current_speed_range': 'dynamic_obstacle_current_speed_range',
        }[key])[ids] = data[key][scenario_indices_cpu].to(env.device)

    env.root_states[:] = data['robot_root_state'][scenario_indices_cpu].to(env.device)
    env.dof_state.view(env.num_envs, env.num_dof, 2)[:] = (
        data['robot_dof_state'][scenario_indices_cpu].to(env.device)
    )
    _clear_episode_buffers(env, episode_length=0)
    env.episode_length_buf[:] = data['episode_length'][scenario_indices_cpu].to(env.device)
    env.slr_commands[:] = data['initial_commands'][scenario_indices_cpu].to(env.device)
    env.nav_actions_filtered[:] = data['initial_nav_actions_filtered'][scenario_indices_cpu].to(env.device)
    env.actions_orig[:] = data['initial_actions_orig'][scenario_indices_cpu].to(env.device)
    _refresh_robot_derived_state(env)
    env._write_dynamic_obstacle_states()
    env._set_dynamic_obstacle_states_in_sim()
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env._root_states_for_sim()),
        gymtorch.unwrap_tensor(env._robot_actor_indices(ids.to(dtype=torch.int32))),
        env.num_envs,
    )
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env._robot_actor_indices(ids.to(dtype=torch.int32))),
        env.num_envs,
    )
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_dof_state_tensor(env.sim)
    _refresh_robot_derived_state(env)
    env.update_percetion()
    env.compute_observations()
    env.reset_buf.zero_()
    env.time_out_buf.zero_()
