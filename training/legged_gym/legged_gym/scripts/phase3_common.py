"""Shared simulator setup and state restoration for Phase-3 evaluation."""

import hashlib
import json
import os
import random
import subprocess

import numpy as np
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from rsl_rl.utils.phase2 import PHASE2_SPEED_FINAL
from rsl_rl.utils.phase3 import scenario_ids_digest, sha256_file


EVALUATOR_VERSION = 'phase3.fixed_cohort.evaluator.v2'


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


def configure_deterministic_evaluation(seed):
    """Configure deterministic host/inference settings for one evaluator.

    Isaac Gym's GPU PhysX solver is not guaranteed to be bitwise
    deterministic.  These settings remove the software-side random sources
    first and are recorded in the manifest so a remaining simulator-level
    divergence can be diagnosed rather than hidden.
    """

    seed = int(seed)
    # This must be set before CUDA kernels are initialized.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    deterministic_algorithms = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        deterministic_algorithms = False
    try:
        torch.set_float32_matmul_precision('highest')
    except AttributeError:
        pass
    return {
        'python_random_seed': seed,
        'numpy_random_seed': seed,
        'torch_random_seed': seed,
        'cuda_random_seed': seed if cuda_available else None,
        'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'torch_deterministic_algorithms': deterministic_algorithms,
        'cuda_tf32_disabled': cuda_available,
        'cudnn_deterministic': cuda_available,
        'cudnn_benchmark': False,
    }


def _json_safe(value, depth=0):
    """Convert config/SWIG values to bounded JSON-compatible data."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if depth > 4:
        return repr(value)
    if isinstance(value, torch.Tensor):
        if value.numel() <= 64:
            return value.detach().cpu().tolist()
        return {
            'dtype': str(value.dtype), 'shape': list(value.shape),
        }
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), depth + 1)
    if hasattr(value, '__fspath__'):
        return os.fspath(value)
    return repr(value)


def snapshot_public_config(value):
    """Snapshot public scalar/list attributes, including inherited cfg attrs."""

    if value is None:
        return None
    result = {}
    for name in sorted(set(dir(value))):
        if name.startswith('_'):
            continue
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if callable(item) or isinstance(item, (staticmethod, classmethod)):
            continue
        if isinstance(item, type):
            result[name] = snapshot_public_config(item)
        elif isinstance(item, (bool, int, float, str, list, tuple, dict)):
            result[name] = _json_safe(item)
        elif isinstance(item, (np.ndarray, np.generic, torch.Tensor)):
            result[name] = _json_safe(item)
    return result


def repository_dirty():
    """Return whether tracked repository files differ from HEAD."""

    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../../../..'
    ))
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'], cwd=repo_root,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def repository_diff_sha256():
    """Hash the tracked working-tree diff for dirty-manifest provenance."""

    repo_root = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '../../../..'
    ))
    try:
        result = subprocess.run(
            ['git', 'diff', 'HEAD', '--binary'], cwd=repo_root,
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return ''
    return hashlib.sha256(result.stdout).hexdigest()


def build_evaluation_manifest(
    *, args, env, env_cfg, sim_params, policy_path, estimator_checkpoint,
    bank_path, bank_metadata, scenario_ids, num_envs, max_steps, control_dt,
    safety_mode, deterministic_settings, headless, cbf_layer=None,
    task_name='go2_pos_dynamic', evaluation_kind='fixed_cohort',
    evaluator_script=None,
):
    """Build the shared manifest used by fixed and checkpoint-sweep paths."""

    scenario_ids = [int(value) for value in scenario_ids]
    layer = cbf_layer
    safety_cfg = getattr(getattr(env_cfg, 'env', None), 'predictive_safety', None)
    if layer is not None:
        d_safe = float(layer.d_safe)
        kappa = float(layer.kappa)
        damping = float(layer.damping_factor)
    else:
        d_safe = float(getattr(safety_cfg, 'd_safe', 0.0))
        kappa = float(getattr(safety_cfg, 'kappa', 0.0))
        damping = float(getattr(safety_cfg, 'damping_factor', 0.0))

    def cfg(name):
        return snapshot_public_config(getattr(env_cfg, name, None))

    env_args = snapshot_public_config(args)
    sim_snapshot = snapshot_public_config(sim_params)
    physx = getattr(sim_params, 'physx', None)
    if physx is not None:
        sim_snapshot['physx'] = snapshot_public_config(physx)
    sim_dt = float(getattr(sim_params, 'dt', control_dt))
    scenario_range = [min(scenario_ids), max(scenario_ids)] if scenario_ids else []
    estimator_sha = (
        sha256_file(estimator_checkpoint)
        if estimator_checkpoint else ''
    )
    evaluator_path = os.path.abspath(evaluator_script or __file__)
    manifest = {
        'manifest_version': 1,
        'git_commit': repository_commit(),
        'git_dirty': repository_dirty(),
        'git_diff_sha256': repository_diff_sha256(),
        'evaluator_script': os.path.relpath(
            evaluator_path,
            os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')),
        ),
        'evaluator_script_sha256': sha256_file(evaluator_path),
        'evaluator_version': EVALUATOR_VERSION,
        'policy_path': os.path.abspath(policy_path),
        'policy_sha256': sha256_file(policy_path),
        'safety_mode': safety_mode,
        'estimator_checkpoint': os.path.abspath(estimator_checkpoint) if estimator_checkpoint else '',
        'estimator_sha256': estimator_sha,
        'scenario_bank_path': os.path.abspath(bank_path),
        'scenario_bank_file_sha256': sha256_file(bank_path),
        'scenario_bank_hash': bank_metadata['bank_hash'],
        'scenario_count': len(scenario_ids),
        'scenario_id_range': scenario_range,
        'scenario_ids_digest': scenario_ids_digest(scenario_ids),
        'scenario_ids': scenario_ids,
        'simulation_seed': int(env_cfg.seed),
        'num_envs': int(num_envs),
        'control_dt_s': float(control_dt),
        'sim_dt_s': sim_dt,
        'max_episode_steps': int(max_steps),
        'timeout_s': float(max_steps) * float(control_dt),
        'task': task_name,
        'terrain_config': cfg('terrain'),
        'obstacle_count': int(env.num_dynamic_obstacles),
        'obstacle_config': cfg('dynamic_obstacles'),
        'noise_enabled': bool(getattr(env_cfg.noise, 'add_noise', False)),
        'noise_config': cfg('noise'),
        'domain_randomization': cfg('domain_rand'),
        'replay': cfg('replay'),
        'headless': bool(headless),
        'deterministic_inference': True,
        'deterministic_settings': deterministic_settings,
        'calibration_delta': float(getattr(safety_cfg, 'calibration_delta', 0.0)),
        'd_safe': d_safe,
        'kappa': kappa,
        'damping': damping,
        'simulation_parameters': sim_snapshot,
        'environment_config': {
            name: cfg(name) for name in (
                'env', 'control', 'normalization', 'asset', 'init_state',
                'commands', 'sensors', 'visualization', 'motion_estimation',
            )
        },
        'isaac_gym_arguments': env_args,
        'provenance': {
            'evaluation_kind': evaluation_kind,
        },
    }
    # Ensure this function itself never emits a non-JSON value from a SWIG
    # parameter or a custom config object.
    return _json_safe(manifest)


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
