"""Evaluate frozen Go2 policies with the Phase-1 predictive CBF modes.

The script runs the same frozen policy under four shield contexts:
``original``, ``synchronized_static``, ``predictive`` and
``predictive_calibrated``.  It writes episode metrics, a small-env diagnostic
trace, collision traces, a summary and (when matplotlib is available) figures.

Example::

    CUDA_VISIBLE_DEVICES=0 python \
      training/legged_gym/legged_gym/scripts/evaluate_predictive_cbf.py \
      --policy_path training/legged_gym/logs/Go2_pos_rough/<run>/model_2000.pt \
      --estimator_checkpoint motion_estimator/artifacts/<run>/best.pt \
      --safety_mode predictive --num_envs 64 --num_episodes 100 --headless \
      --rl_device cuda:0
"""

import argparse
import csv
import datetime
import json
import os
import sys
import math

from isaacgym import gymapi  # noqa: F401 - initialize Isaac Gym before torch
from legged_gym.envs import *  # noqa: F401,F403 - register tasks
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import get_args, task_registry

import torch


TASK_NAME = 'go2_pos_dynamic'
MODES = {
    'original', 'synchronized_static', 'predictive',
    'predictive_calibrated',
}


def _parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--policy_path', required=True)
    parser.add_argument('--estimator_checkpoint', default='')
    parser.add_argument('--safety_mode', choices=sorted(MODES), default='original')
    parser.add_argument('--calibration_delta', type=float, default=0.0)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--num_episodes', type=int, default=100)
    parser.add_argument('--max_steps_per_episode', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--monitor_envs', type=int, default=8)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument(
        '--scenario', choices=['default', 'controlled_crossing'],
        default='default',
        help='Use a repeatable approaching/crossing obstacle arrangement.',
    )
    script_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return script_args


def _resolve_path(path):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError('Checkpoint not found: {}'.format(path))
    return path


def _configure(env_cfg, train_cfg, script_args):
    env_cfg.seed = script_args.seed
    env_cfg.env.num_envs = script_args.num_envs
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
    safety_cfg.mode = script_args.safety_mode
    safety_cfg.calibration_delta = script_args.calibration_delta
    safety_cfg.estimator_checkpoint = script_args.estimator_checkpoint
    safety_cfg.use_warmup_gate = True
    train_cfg.runner.resume = False


def _set_controlled_crossing(env):
    """Place one obstacle on a deterministic transverse crossing trajectory."""
    num_obstacles = env.num_dynamic_obstacles
    starts = [[1.4, -2.0], [-2.4, -2.4], [-2.4, 2.4], [2.4, 2.4],
              [2.4, -2.4], [-2.0, 2.0]][:num_obstacles]
    velocities = [[0.0, 1.2]] + [[0.0, 0.0]] * (num_obstacles - 1)
    positions = torch.as_tensor(starts, device=env.device, dtype=torch.float)
    speeds = torch.as_tensor(velocities, device=env.device, dtype=torch.float)
    env.dynamic_obstacle_start[:] = positions.unsqueeze(0)
    env.dynamic_obstacle_velocity[:] = speeds.unsqueeze(0)
    env.dynamic_obstacle_time.zero_()
    env._write_dynamic_obstacle_states()
    env._set_dynamic_obstacle_states_in_sim()
    env._get_rays()
    env.compute_observations()


def _bool_list(tensor):
    return tensor.detach().bool().reshape(-1).cpu().tolist()


def _write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_diagnostics(output_dir, timeseries):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    if not timeseries:
        return None
    figure_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figure_dir, exist_ok=True)
    steps = [row['global_step'] for row in timeseries]
    residual = [row['residual'] for row in timeseries]
    eta = [row['eta'] for row in timeseries]
    intervention = [row['intervention_norm'] for row in timeseries]
    drift = [row['safety_drift'] for row in timeseries]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(steps, residual, linewidth=0.8, label='barrier residual')
    axes[0].axhline(0.0, color='black', linewidth=0.7)
    axes[0].legend()
    axes[1].plot(steps, eta, linewidth=0.8, label='eta')
    axes[1].plot(steps, intervention, linewidth=0.8, label='intervention norm')
    axes[1].legend()
    axes[2].plot(steps, drift, linewidth=0.8, label='physical drift')
    axes[2].legend()
    axes[2].set_xlabel('control step')
    fig.tight_layout()
    figure_path = os.path.join(figure_dir, 'predictive_cbf_diagnostics.png')
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)
    return figure_path


def evaluate(args, script_args):
    policy_path = _resolve_path(script_args.policy_path)
    if script_args.safety_mode in {'predictive', 'predictive_calibrated'}:
        if not script_args.estimator_checkpoint:
            raise ValueError('--estimator_checkpoint is required in predictive modes')
        script_args.estimator_checkpoint = _resolve_path(
            script_args.estimator_checkpoint
        )

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    _configure(env_cfg, train_cfg, script_args)
    args.task = TASK_NAME
    args.num_envs = script_args.num_envs
    args.seed = script_args.seed
    args.wandb = False

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env, name=TASK_NAME, args=args, train_cfg=train_cfg, log_root=None
    )
    runner.load(policy_path, load_optimizer=False)
    runner.alg.actor_critic.eval()
    # Keep terminal flags available until this script has recorded their cause;
    # reset completed envs explicitly below.
    env.do_reset = False

    if script_args.scenario == 'controlled_crossing':
        _set_controlled_crossing(env)

    output_dir = script_args.output_dir
    if output_dir is None:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR, 'logs', TASK_NAME,
            'predictive_cbf_eval', stamp + '_' + script_args.safety_mode,
        )
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    num_envs = env.num_envs
    monitor_envs = min(max(1, script_args.monitor_envs), num_envs)
    dt = float(env.cfg.control.decimation * env.cfg.sim.dt)
    obs = env.get_observations().to(env.device)
    runner.reset_safety_context()

    episode_ids = torch.arange(num_envs, device=env.device, dtype=torch.long)
    episode_steps = torch.zeros(num_envs, device=env.device, dtype=torch.long)
    episode_rewards = torch.zeros(num_envs, device=env.device)
    episode_distance = torch.zeros(num_envs, device=env.device)
    previous_positions = env.root_states[:, :2].detach().clone()
    episodes = []
    timeseries = []
    collision_traces = []
    total_steps = 0
    total_intervention = 0.0
    total_active = 0
    total_residual_negative = 0
    total_samples = 0
    max_intervention = 0.0
    first_trigger_step = None

    layer = getattr(runner.alg.actor_critic, 'cbf_layer', None)
    max_rollout_steps = script_args.max_steps_per_episode * max(
        1, int(math.ceil(float(script_args.num_episodes) / num_envs))
    )
    with torch.inference_mode():
        for global_step in range(max_rollout_steps):
            safety_drift, shield_rays = runner.build_safety_context(obs)
            action = runner.alg.actor_critic.act_inference(
                obs,
                safety_drift=safety_drift,
                shield_rays=shield_rays,
            )

            diag = {
                'h_comp': layer.last_h_comp.reshape(-1),
                'safety_drift': layer.last_safety_drift.reshape(-1),
                'Lgh_u': layer.last_Lgh_u.reshape(-1),
                'alpha_h': layer.last_alpha_h.reshape(-1),
                'residual': layer.last_nominal_barrier_residual.reshape(-1),
                'eta': layer.last_eta.reshape(-1),
                'intervention_norm': layer.last_intervention_norm.reshape(-1),
            }
            intervention = diag['intervention_norm']
            residual = diag['residual']
            active = intervention > 1.0e-6
            total_intervention += float(intervention.sum())
            total_active += int(active.sum())
            total_residual_negative += int((residual < 0.0).sum())
            total_samples += num_envs
            max_intervention = max(max_intervention, float(intervention.max()))
            if first_trigger_step is None and bool(active.any()):
                first_trigger_step = global_step

            old_positions = previous_positions
            obs, _, rewards, dones, infos = env.step(action)
            obs = obs.to(env.device)
            episode_steps += 1
            episode_rewards += rewards
            new_positions = env.root_states[:, :2].detach().clone()
            episode_distance += torch.linalg.vector_norm(
                new_positions - old_positions, dim=-1
            )
            previous_positions = new_positions

            updated = _bool_list(env.exteroception_updated_mask)
            counts = env.exteroception_history_count.detach().cpu().tolist()
            collision_flags = _bool_list(env.collision_occurred)
            success_flags = _bool_list(env.goal_reached_flag)
            timeout_flags = _bool_list(env.time_out_buf)
            stuck_flags = _bool_list(env.stand_still_flag)
            terminate_flags = _bool_list(env.terminate_buf)
            natural_done = dones.bool().reshape(-1)
            forced_mask = (episode_steps >= script_args.max_steps_per_episode) & (
                ~natural_done
            )
            terminal_mask = natural_done | forced_mask
            forced_flags = forced_mask.detach().cpu().tolist()

            for env_id in range(monitor_envs):
                row = {
                    'global_step': global_step,
                    'env_id': env_id,
                    'episode_id': int(episode_ids[env_id]),
                    'episode_step': int(episode_steps[env_id]),
                    'exteroception_updated': int(updated[env_id]),
                    'history_count': int(counts[env_id]),
                    'h_comp': float(diag['h_comp'][env_id]),
                    'safety_drift': float(diag['safety_drift'][env_id]),
                    'Lgh_u': float(diag['Lgh_u'][env_id]),
                    'alpha_h': float(diag['alpha_h'][env_id]),
                    'residual': float(diag['residual'][env_id]),
                    'eta': float(diag['eta'][env_id]),
                    'intervention_norm': float(diag['intervention_norm'][env_id]),
                    'done': int(terminal_mask[env_id]),
                    'collision': int(collision_flags[env_id]),
                    'success': int(success_flags[env_id]),
                    'timeout': int(timeout_flags[env_id] or forced_flags[env_id]),
                    'stuck': int(stuck_flags[env_id]),
                    'terminate': int(terminate_flags[env_id]),
                    'forced_limit': int(forced_flags[env_id]),
                }
                timeseries.append(row)
                if collision_flags[env_id]:
                    collision_traces.append(row.copy())

            terminal_ids = terminal_mask.nonzero(as_tuple=False).flatten()
            for env_id in terminal_ids.detach().cpu().tolist():
                length = int(episode_steps[env_id])
                record = {
                    'env_id': env_id,
                    'episode_id': int(episode_ids[env_id]),
                    'length': length,
                    'duration_s': length * dt,
                    'reward': float(episode_rewards[env_id]),
                    'distance_m': float(episode_distance[env_id]),
                    'mean_speed_mps': float(episode_distance[env_id]) / max(length * dt, 1.0e-6),
                    'success': int(success_flags[env_id]),
                    'collision': int(collision_flags[env_id]),
                    'timeout': int(timeout_flags[env_id] or forced_flags[env_id]),
                    'stuck': int(stuck_flags[env_id]),
                    'terminate': int(terminate_flags[env_id]),
                    'forced_limit': int(forced_flags[env_id]),
                }
                episodes.append(record)
                episode_ids[env_id] += num_envs
                episode_steps[env_id] = 0
                episode_rewards[env_id] = 0.0
                episode_distance[env_id] = 0.0

            if terminal_ids.numel() > 0:
                # The environment is configured not to auto-reset so terminal
                # flags remain observable. Reset only after recording them.
                env.reset_idx(terminal_ids)
                env.compute_observations()
                previous_positions[terminal_ids] = env.root_states[terminal_ids, :2]
                runner.reset_safety_context(terminal_ids)

            total_steps += 1
            if len(episodes) >= script_args.num_episodes:
                break

        # Make the result explicit if an episode did not terminate naturally.
        if len(episodes) < script_args.num_episodes:
            for env_id in range(num_envs):
                if len(episodes) >= script_args.num_episodes:
                    break
                if int(episode_steps[env_id]) == 0:
                    continue
                length = int(episode_steps[env_id])
                episodes.append({
                    'env_id': env_id,
                    'episode_id': int(episode_ids[env_id]),
                    'length': length,
                    'duration_s': length * dt,
                    'reward': float(episode_rewards[env_id]),
                    'distance_m': float(episode_distance[env_id]),
                    'mean_speed_mps': float(episode_distance[env_id]) / max(length * dt, 1.0e-6),
                    'success': 0,
                    'collision': 0,
                    'timeout': 1,
                    'stuck': 0,
                    'terminate': 0,
                    'forced_limit': 1,
                })

    episodes = episodes[:script_args.num_episodes]
    episode_fields = list(episodes[0].keys()) if episodes else [
        'env_id', 'episode_id', 'length', 'duration_s', 'reward',
        'distance_m', 'mean_speed_mps', 'success', 'collision', 'timeout',
        'stuck', 'terminate', 'forced_limit',
    ]
    timeseries_fields = [
        'global_step', 'env_id', 'episode_id', 'episode_step',
        'exteroception_updated', 'history_count', 'h_comp', 'safety_drift',
        'Lgh_u', 'alpha_h', 'residual', 'eta', 'intervention_norm', 'done',
        'collision', 'success', 'timeout', 'stuck', 'terminate', 'forced_limit',
    ]
    _write_csv(os.path.join(output_dir, 'episodes.csv'), episodes, episode_fields)
    _write_csv(os.path.join(output_dir, 'timeseries.csv'), timeseries, timeseries_fields)
    _write_csv(os.path.join(output_dir, 'collision_traces.csv'), collision_traces, timeseries_fields)
    figure_path = _plot_diagnostics(output_dir, timeseries)

    def mean_field(name):
        return sum(float(row[name]) for row in episodes) / max(len(episodes), 1)

    summary = {
        'task': TASK_NAME,
        'policy_path': policy_path,
        'estimator_checkpoint': script_args.estimator_checkpoint or None,
        'safety_mode': script_args.safety_mode,
        'calibration_delta_mps': (
            script_args.calibration_delta
            if script_args.safety_mode == 'predictive_calibrated' else 0.0
        ),
        'scenario': script_args.scenario,
        'seed': script_args.seed,
        'num_envs': num_envs,
        'episodes_requested': script_args.num_episodes,
        'episodes_completed_or_forced': len(episodes),
        'control_dt_s': dt,
        'success_rate': mean_field('success'),
        'collision_rate': mean_field('collision'),
        'timeout_rate': mean_field('timeout'),
        'stuck_rate': mean_field('stuck'),
        'mean_episode_length': mean_field('length'),
        'mean_speed_mps': mean_field('mean_speed_mps'),
        'intervention_frequency': total_active / max(total_samples, 1),
        'mean_intervention_norm': total_intervention / max(total_samples, 1),
        'max_intervention_norm': max_intervention,
        'residual_negative_probability': total_residual_negative / max(total_samples, 1),
        'first_intervention_step': first_trigger_step,
        'first_intervention_time_s': None if first_trigger_step is None else first_trigger_step * dt,
        'controlled_trigger_delta_s': None,
        'controlled_trigger_delta_note': (
            'Run the same controlled_crossing seed for static and predictive '
            'outputs to compute the trigger-time delta.'
            if script_args.scenario == 'controlled_crossing' else None
        ),
        'diagnostic_envs': monitor_envs,
        'artifacts': {
            'episodes_csv': 'episodes.csv',
            'timeseries_csv': 'timeseries.csv',
            'collision_traces_csv': 'collision_traces.csv',
            'figure': os.path.relpath(figure_path, output_dir) if figure_path else None,
        },
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    script_args = _parse_script_args()
    isaac_args = get_args()
    evaluate(isaac_args, script_args)
