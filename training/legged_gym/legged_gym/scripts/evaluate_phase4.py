"""Phase-4 fixed-cohort failure analysis and privileged oracle evaluator.

This evaluator is deliberately separate from the Phase-3 evaluator.  It
keeps the learned policy, actor observation, reward, current shield rays,
10-Hz estimator refresh, 50-Hz controller, and locomotion dynamics fixed.
Oracle modes replace only the drift supplied to the same CBF projection.
"""

import argparse
import csv
import datetime
import json
import math
import os
import sys
from collections import deque

from isaacgym import gymapi  # noqa: F401 - initialize Isaac Gym first
import torch

from legged_gym.envs import *  # noqa: F401,F403 - register tasks
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import get_args, task_registry
from rsl_rl.modules.cbf_lse_layer import DEFAULT_D_SAFE, DEFAULT_KAPPA
from rsl_rl.utils.phase2 import classify_terminal_outcome
from rsl_rl.utils.phase3 import (
    DIFFICULTY_BINS,
    difficulty_bin,
    fixed_cohort_batches_for_indices,
    load_scenario_bank,
    outcome_counts,
    parse_scenario_ids,
)
from rsl_rl.utils.phase4 import (
    ORACLE_HORIZONS,
    PHASE4_MODES,
    classify_failure,
    extract_pre_collision_window,
    mode_spec,
    primary_failure_label,
    select_multi_horizon_drift,
    warning_time,
)

from phase3_common import (
    build_evaluation_manifest,
    configure_deterministic_evaluation,
    configure_evaluation_cfg,
    restore_scenario_batch,
)


TASK_NAME = 'go2_pos_dynamic'
AUTHORITATIVE_BANK_HASH = (
    '466b093b1b7698d322047c29d3bc81e61fb84e427d2d755a8e95f56d00fa8bee'
)
FAILURE_WINDOW_SAMPLES = 100


def _parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--policy_path', required=True)
    parser.add_argument('--scenario_bank', required=True)
    parser.add_argument('--estimator_checkpoint', required=True)
    parser.add_argument('--training_seed', type=int, required=True)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--mode', choices=PHASE4_MODES, required=True)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--max_steps_per_episode', type=int, default=3000)
    parser.add_argument('--scenario_ids', default=None)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--progress_interval_steps', type=int, default=250)
    parser.add_argument('--failure_analysis', action='store_true')
    script_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return script_args


def _resolve_file(path, label):
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(path):
        raise FileNotFoundError('{} not found: {}'.format(label, path))
    return path


def _write_csv(path, rows, fields):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _json_tensor(value, local_id):
    return json.dumps(
        value[local_id].detach().cpu().tolist(), separators=(',', ':')
    )


def _scalar(value, local_id=0):
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[local_id].detach().cpu())
    return float(value)


def _yaw_from_quaternion(quaternion, local_id):
    quat = quaternion[local_id]
    sin_yaw = 2.0 * (quat[3] * quat[2] + quat[0] * quat[1])
    cos_yaw = 1.0 - 2.0 * (quat[1].square() + quat[2].square())
    return float(torch.atan2(sin_yaw, cos_yaw).detach().cpu())


def _difficulty_report(rows):
    result = []
    for name, lower, upper in DIFFICULTY_BINS:
        subset = [row for row in rows if row['difficulty_bin'] == name]

        def mean(field):
            values = [float(row[field]) for row in subset]
            return sum(values) / len(values) if values else float('nan')

        result.append({
            'difficulty_bin': name,
            'lower_mps': lower,
            'upper_mps': upper,
            'num_scenarios': len(subset),
            'safe_success_rate': mean('safe_success'),
            'collision_rate': mean('collision'),
            'stuck_rate': mean('stuck'),
            'timeout_rate': mean('timeout'),
            'mean_robot_speed_mps': mean('mean_speed_mps'),
        })
    return result


WINDOW_FIELDS = [
    'seed', 'scenario_id', 'repeat', 'control_step', 'time_s',
    'time_to_collision_s', 'is_extero_update',
    'h_lse', 'alpha', 'alpha_h', 'u_bar_x', 'u_bar_y', 'u_bar_yaw',
    'Lgh_x', 'Lgh_y', 'Lgh_norm_sq', 'Lgh_u', 'q_static',
    'd_hat', 'r_hat',
    'd_gt_0p1', 'd_gt_0p2', 'd_gt_0p3', 'd_gt_0p5',
    'r_gt_0p1', 'r_gt_0p2', 'r_gt_0p3', 'r_gt_0p5',
    'static_unsafe', 'learned_predictive_unsafe',
    'gt_0p1_unsafe', 'gt_0p2_unsafe', 'gt_0p3_unsafe', 'gt_0p5_unsafe',
    'estimator_false_safe_0p1',
    'u_safe_hat_x', 'u_safe_hat_y', 'u_safe_hat_yaw',
    'intervention_hat_norm',
    'u_safe_gt_0p1', 'u_safe_gt_0p2', 'u_safe_gt_0p3', 'u_safe_gt_0p5',
    'intervention_gt_0p1_norm', 'intervention_gt_0p2_norm',
    'intervention_gt_0p3_norm', 'intervention_gt_0p5_norm',
    'pre_clip_action', 'post_clip_action', 'clip_delta', 'clip_event',
    'v_actual_x', 'v_actual_y', 'yaw_rate_actual', 'tracking_error_xy',
    'minimum_shield_ray', 'minimum_dynamic_obstacle_center_distance',
    'closest_dynamic_obstacle_speed', 'closest_dynamic_relative_speed',
    'selected_horizon',
]


EPISODE_FIELDS = [
    'seed', 'scenario_id', 'repeat', 'difficulty_bin', 'vmax_speed_mps',
    'terminal_outcome', 'goal_reached', 'safe_success', 'collision',
    'stuck', 'timeout', 'other_failure', 'episode_length', 'duration_s',
    'mean_speed_mps', 'total_reward', 'intervention_frequency',
    'mean_intervention_norm', 'max_intervention_norm',
    'mean_gt_drift', 'negative_gt_drift_rate', 'selected_horizon_mode',
]


def _mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else float('nan')


def _stats_record():
    return {
        'samples': 0, 'interventions': 0, 'intervention_sum': 0.0,
        'intervention_max': 0.0, 'gt_drift_sum': 0.0,
        'gt_negative': 0, 'gt_updates': 0, 'speed_sum': 0.0,
        'reward_sum': 0.0, 'selected_horizons': [],
    }


def _episode_record(seed, repeat, scenario_id, vmax, steps, dt, stats, flags):
    outcome = classify_terminal_outcome(
        flags['collision'], flags['goal_reached'], flags['timeout'], flags['stuck']
    )
    samples = max(int(stats['samples']), 1)
    selected = stats['selected_horizons']
    selected_mode = ''
    if selected:
        counts = {h: selected.count(h) for h in sorted(set(selected))}
        selected_mode = json.dumps(counts, sort_keys=True)
    return {
        'seed': int(seed), 'scenario_id': int(scenario_id), 'repeat': int(repeat),
        'difficulty_bin': difficulty_bin(vmax), 'vmax_speed_mps': float(vmax),
        'terminal_outcome': outcome,
        'goal_reached': int(flags['goal_reached']),
        'safe_success': int(outcome == 'safe_success'),
        'collision': int(outcome == 'collision_failure'),
        'stuck': int(outcome == 'stuck_failure'),
        'timeout': int(outcome == 'timeout_failure'),
        'other_failure': int(outcome == 'other_failure'),
        'episode_length': int(steps), 'duration_s': float(steps) * dt,
        'mean_speed_mps': stats['speed_sum'] / max(samples, 1),
        'total_reward': float(stats['reward_sum']),
        'intervention_frequency': stats['interventions'] / samples,
        'mean_intervention_norm': stats['intervention_sum'] / samples,
        'max_intervention_norm': stats['intervention_max'],
        'mean_gt_drift': stats['gt_drift_sum'] / max(stats['gt_updates'], 1),
        'negative_gt_drift_rate': stats['gt_negative'] / max(stats['gt_updates'], 1),
        'selected_horizon_mode': selected_mode,
    }


def _failure_summary(rows, collision_time_s, seed, scenario_id, repeat):
    update_rows = [row for row in rows if int(row['is_extero_update'])]
    if not update_rows:
        update_rows = list(rows)
    errors = [abs(float(row['d_hat']) - float(row['d_gt_0p1'])) for row in update_rows]
    dangerous = [
        abs(float(row['d_hat']) - float(row['d_gt_0p1']))
        for row in update_rows if float(row['r_gt_0p1']) < 0.0
    ]
    optimistic = [
        max(0.0, float(row['d_hat']) - float(row['d_gt_0p1']))
        for row in update_rows if float(row['r_gt_0p1']) < 0.0
    ]
    false_safe = [row for row in update_rows if int(row['estimator_false_safe_0p1'])]
    first_gt = min(
        [float(row['time_s']) for row in update_rows if float(row['r_gt_0p1']) < 0.0],
        default=float('nan'),
    )
    first_hat = min(
        [float(row['time_s']) for row in update_rows if float(row['r_hat']) < 0.0],
        default=float('nan'),
    )
    delay = first_hat - first_gt if not (math.isnan(first_gt) or math.isnan(first_hat)) else float('nan')

    def mean_field(field):
        return _mean([float(row[field]) for row in rows])

    warning = {
        'warning_time_0p1': warning_time(collision_time_s, update_rows, 'r_gt_0p1'),
        'warning_time_0p2': warning_time(collision_time_s, update_rows, 'r_gt_0p2'),
        'warning_time_0p3': warning_time(collision_time_s, update_rows, 'r_gt_0p3'),
        'warning_time_0p5': warning_time(collision_time_s, update_rows, 'r_gt_0p5'),
    }
    metrics = {
        'seed': int(seed), 'scenario_id': int(scenario_id), 'repeat': int(repeat),
        'collision_time_s': float(collision_time_s),
        'num_pre_collision_samples': len(rows),
        'num_gt_danger_updates': sum(float(row['r_gt_0p1']) < 0.0 for row in update_rows),
        'num_false_safe_updates': len(false_safe),
        'false_safe_fraction': len(false_safe) / max(len(update_rows), 1),
        'mean_abs_drift_error': _mean(errors),
        'max_abs_drift_error': max(errors) if errors else float('nan'),
        'dangerous_drift_mae': _mean(dangerous),
        'optimistic_danger_mean': _mean(optimistic),
        'optimistic_danger_max': max(optimistic) if optimistic else float('nan'),
        'first_gt_0p1_danger_time': first_gt,
        'first_hat_danger_time': first_hat,
        'learned_detection_delay_s': delay,
        'mean_oracle_extra_intervention_0p1': _mean([
            float(row['intervention_gt_0p1_norm']) - float(row['intervention_hat_norm'])
            for row in rows
        ]),
        'max_oracle_extra_intervention_0p1': max([
            float(row['intervention_gt_0p1_norm']) - float(row['intervention_hat_norm'])
            for row in rows
        ], default=float('nan')),
        'clip_fraction': _mean([float(row['clip_event']) for row in rows]),
        'max_clip_delta': max([float(row['clip_delta']) for row in rows], default=float('nan')),
        'mean_tracking_error': mean_field('tracking_error_xy'),
        'max_tracking_error': max([float(row['tracking_error_xy']) for row in rows], default=float('nan')),
        'mean_robot_speed_pre_collision': _mean([
            math.hypot(float(row['v_actual_x']), float(row['v_actual_y']))
            for row in rows
        ]),
        'min_obstacle_distance': min([
            float(row['minimum_dynamic_obstacle_center_distance']) for row in rows
        ], default=float('nan')),
    }
    metrics.update(warning)
    tags = classify_failure(metrics)
    metrics['failure_tags'] = json.dumps(tags, separators=(',', ':'))
    metrics['primary_failure_label'] = primary_failure_label(tags)
    return metrics


def _build_step_row(
    env, actor, layer, learned_drift, gt_cache, shield_rays,
    oracle_mode, selected_horizon, seed, scenario_id, repeat, control_step, dt,
    active_ids,
):
    """Capture all pre-action safety values for the active environments."""

    u_bar = actor.u_bar.detach()
    u_safe = actor.u_s.detach()
    alpha = actor.alpha.detach()
    h_lse = layer.last_h_comp.detach().reshape(-1)
    alpha_h = layer.last_alpha_h.detach().reshape(-1)
    lgh = layer.last_Lgh.detach()
    lgh_u = layer.last_Lgh_u.detach().reshape(-1)
    lgh_norm_sq = lgh.square().sum(dim=-1)
    q_static = lgh_u + alpha_h
    d_hat = learned_drift.detach().reshape(-1)
    r_hat = d_hat + q_static

    shadow = {}
    for horizon in ORACLE_HORIZONS:
        key = '{:.1f}'.format(horizon)
        result = layer.project(
            u_bar, shield_rays, alpha, safety_drift=gt_cache[horizon]
        )
        shadow[horizon] = result

    obstacle_gt = env.get_dynamic_obstacle_gt()
    robot_xy = env.root_states[:, :2]
    obstacle_xy = obstacle_gt['trajectory_position'][:, :, :2]
    obstacle_delta = obstacle_xy - robot_xy[:, None, :]
    obstacle_distance = torch.linalg.vector_norm(obstacle_delta, dim=-1)
    closest_index = obstacle_distance.argmin(dim=-1)
    closest_distance = obstacle_distance.min(dim=-1).values
    obstacle_speed = torch.linalg.vector_norm(
        obstacle_gt['trajectory_velocity'][:, :, :2], dim=-1
    )
    closest_speed = torch.gather(obstacle_speed, 1, closest_index[:, None]).reshape(-1)
    rel_velocity = obstacle_gt['trajectory_velocity'][:, :, :2] - env.root_states[:, None, 7:9]
    closest_relative_speed = torch.gather(
        torch.linalg.vector_norm(rel_velocity, dim=-1), 1, closest_index[:, None]
    ).reshape(-1)

    rows = []
    for local_id in active_ids:
        row = {
            '_local_id': int(local_id),
            'seed': int(seed), 'scenario_id': int(scenario_id[local_id]),
            'repeat': int(repeat), 'control_step': int(control_step),
            'time_s': float(control_step) * dt, 'time_to_collision_s': '',
            'is_extero_update': int(env.exteroception_updated_mask[local_id]),
            'h_lse': float(h_lse[local_id]), 'alpha': float(alpha[local_id].reshape(-1)[0]),
            'alpha_h': float(alpha_h[local_id]),
            'u_bar_x': float(u_bar[local_id, 0]), 'u_bar_y': float(u_bar[local_id, 1]),
            'u_bar_yaw': float(u_bar[local_id, 2]),
            'Lgh_x': float(lgh[local_id, 0]), 'Lgh_y': float(lgh[local_id, 1]),
            'Lgh_norm_sq': float(lgh_norm_sq[local_id]), 'Lgh_u': float(lgh_u[local_id]),
            'q_static': float(q_static[local_id]), 'd_hat': float(d_hat[local_id]),
            'r_hat': float(r_hat[local_id]),
            'pre_clip_action': _json_tensor(u_safe, local_id),
            'u_safe_hat_x': float(u_safe[local_id, 0]),
            'u_safe_hat_y': float(u_safe[local_id, 1]),
            'u_safe_hat_yaw': float(u_safe[local_id, 2]),
            'intervention_hat_norm': float(layer.last_intervention_norm[local_id]),
            'static_unsafe': int(float(q_static[local_id]) < 0.0),
            'learned_predictive_unsafe': int(float(r_hat[local_id]) < 0.0),
            'minimum_shield_ray': float(shield_rays[local_id].min()),
            'minimum_dynamic_obstacle_center_distance': float(closest_distance[local_id]),
            'closest_dynamic_obstacle_speed': float(closest_speed[local_id]),
            'closest_dynamic_relative_speed': float(closest_relative_speed[local_id]),
            'selected_horizon': float(selected_horizon[local_id]),
        }
        for horizon in ORACLE_HORIZONS:
            suffix = '{:.1f}'.format(horizon).replace('.', 'p')
            gt = gt_cache[horizon]
            result = shadow[horizon]
            drift = float(gt[local_id])
            residual = float(result['nominal_barrier_residual'][local_id])
            safe = result['safe_action']
            row['d_gt_{}'.format(suffix)] = drift
            row['r_gt_{}'.format(suffix)] = residual
            row['gt_{}_unsafe'.format(suffix)] = int(residual < 0.0)
            row['u_safe_gt_{}'.format(suffix)] = _json_tensor(safe, local_id)
            row['intervention_gt_{}_norm'.format(suffix)] = float(
                result['intervention_norm'][local_id]
            )
        row['estimator_false_safe_0p1'] = int(
            float(r_hat[local_id]) >= 0.0
            and float(row['r_gt_0p1']) < 0.0
        )
        rows.append(row)
    return rows


def evaluate(args, script_args):
    policy_path = _resolve_file(script_args.policy_path, 'policy checkpoint')
    bank_path = _resolve_file(script_args.scenario_bank, 'scenario bank')
    estimator_path = _resolve_file(script_args.estimator_checkpoint, 'estimator checkpoint')
    bank = load_scenario_bank(bank_path)
    metadata = bank['metadata']
    if metadata.get('bank_hash') != AUTHORITATIVE_BANK_HASH:
        raise ValueError(
            'scenario bank hash mismatch: expected {}, got {}'.format(
                AUTHORITATIVE_BANK_HASH, metadata.get('bank_hash')
            )
        )
    scenarios = bank['scenarios']
    num_scenarios = int(metadata['num_scenarios'])
    selected_ids = parse_scenario_ids(script_args.scenario_ids, num_scenarios)
    num_envs = int(script_args.num_envs)
    if num_envs < 1:
        raise ValueError('--num_envs must be positive')
    batches = fixed_cohort_batches_for_indices(selected_ids, num_envs)
    if script_args.max_steps_per_episode < 1:
        raise ValueError('--max_steps_per_episode must be positive')
    if script_args.repeat < 1:
        raise ValueError('--repeat must be positive')

    spec = mode_spec(script_args.mode)
    generation_seed = int(metadata['generation_seed'])
    deterministic_settings = configure_deterministic_evaluation(generation_seed)
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    configure_evaluation_cfg(
        env_cfg, train_cfg, generation_seed, num_envs,
        safety_mode='predictive', estimator_checkpoint=estimator_path,
    )
    args.task = TASK_NAME
    args.num_envs = num_envs
    args.seed = generation_seed
    args.wandb = False
    args.headless = True

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    try:
        train_cfg.runner.resume = False
        torch.use_deterministic_algorithms(False, warn_only=True)
        runner, _ = task_registry.make_alg_runner(
            env=env, name=TASK_NAME, args=args, train_cfg=train_cfg, log_root=None
        )
        torch.use_deterministic_algorithms(True, warn_only=True)
        runner.load(policy_path, load_optimizer=False)
        runner.alg.actor_critic.eval()
        env.do_reset = False
        layer = runner.alg.actor_critic.cbf_layer
        dt = float(env.cfg.control.decimation * env.cfg.sim.dt)
        output_dir = os.path.abspath(os.path.expanduser(script_args.output_dir))
        os.makedirs(output_dir, exist_ok=True)

        episodes = []
        collision_windows = []
        collision_summaries = []
        finished_count = 0
        full_diagnostics = bool(script_args.failure_analysis)
        if full_diagnostics or spec['oracle_multi_horizon']:
            required_horizons = list(ORACLE_HORIZONS)
        elif spec['safety_drift_source'] == 'gt':
            required_horizons = [spec['oracle_horizon_s']]
        else:
            required_horizons = []
        selected_scenario_ids = scenarios['scenario_id'][torch.as_tensor(selected_ids)].tolist()

        with torch.inference_mode():
            for batch_number, batch_indices in enumerate(batches, start=1):
                print(
                    '[phase4] mode={} batch {}/{} scenarios={}-{}'.format(
                        script_args.mode, batch_number, len(batches),
                        batch_indices[0], batch_indices[-1]
                    ), flush=True,
                )
                restore_scenario_batch(env, bank, batch_indices)
                runner.reset_safety_context()
                obs = env.get_observations().to(env.device)
                batch_scenario_ids = scenarios['scenario_id'][
                    torch.as_tensor(batch_indices)
                ].tolist()
                finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
                episode_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
                episode_rewards = torch.zeros(num_envs, dtype=torch.float, device=env.device)
                stats = [_stats_record() for _ in range(num_envs)]
                history = [deque(maxlen=FAILURE_WINDOW_SAMPLES) for _ in range(num_envs)]
                collision_recorded = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
                gt_cache = {
                    horizon: torch.zeros(num_envs, 1, device=env.device)
                    for horizon in required_horizons
                }

                for control_step in range(script_args.max_steps_per_episode):
                    active_before = ~finished
                    if not bool(active_before.any()):
                        break
                    learned_drift, shield_rays = runner.build_safety_context(obs)
                    extero_update = env.exteroception_updated_mask.bool()
                    if bool(extero_update.any()):
                        for horizon in required_horizons:
                            gt_cache[horizon] = env.compute_lse_drift_gt(horizon)['drift'].detach().clone()
                    horizon_tensor = torch.zeros(num_envs, device=env.device)
                    if script_args.mode == 'predictive_learned':
                        selected_drift = learned_drift
                    else:
                        if spec['oracle_multi_horizon']:
                            stacked = torch.cat([gt_cache[h] for h in ORACLE_HORIZONS], dim=1)
                            selected_drift, selected_idx = stacked.min(dim=1, keepdim=True)
                            horizon_values = torch.as_tensor(
                                ORACLE_HORIZONS, device=env.device, dtype=stacked.dtype
                            )
                            horizon_tensor = horizon_values[selected_idx.reshape(-1)]
                        else:
                            horizon = spec['oracle_horizon_s']
                            selected_drift = gt_cache[horizon]
                            horizon_tensor[:] = float(horizon)
                    if script_args.mode == 'predictive_learned':
                        horizon_tensor[:] = float('nan')

                    actor = runner.alg.actor_critic
                    action = actor.act_inference(
                        obs, safety_drift=selected_drift, shield_rays=shield_rays
                    )
                    pre_clip_action = action.detach().clone()
                    active_ids = active_before.nonzero(as_tuple=False).flatten().tolist()
                    rows = []
                    if full_diagnostics:
                        rows = _build_step_row(
                            env, actor, layer, learned_drift, gt_cache, shield_rays,
                            script_args.mode, horizon_tensor, script_args.training_seed,
                            torch.as_tensor(batch_scenario_ids, device=env.device),
                            script_args.repeat, control_step, dt, active_ids,
                        )

                    action[finished] = 0.0
                    obs, _, rewards, dones, _ = env.step(action)
                    obs = obs.to(env.device)
                    episode_steps[active_before] += 1
                    episode_rewards[active_before] += rewards[active_before]

                    # Complete execution-path diagnostics after the actual
                    # navigation command has passed filtering and clipping.
                    post_clip = env.nav_actions_after_clip.detach()
                    actual_v = env.base_lin_vel.detach()
                    actual_yaw = env.base_ang_vel[:, 2].detach()
                    if full_diagnostics:
                        for row in rows:
                            local_id = int(row['_local_id'])
                            row['post_clip_action'] = _json_tensor(post_clip, local_id)
                            delta = torch.linalg.vector_norm(
                                pre_clip_action[local_id] - post_clip[local_id]
                            )
                            row['clip_delta'] = float(delta)
                            row['clip_event'] = int(float(delta) > 1.0e-6)
                            row['v_actual_x'] = float(actual_v[local_id, 0])
                            row['v_actual_y'] = float(actual_v[local_id, 1])
                            row['yaw_rate_actual'] = float(actual_yaw[local_id])
                            row['tracking_error_xy'] = float(torch.linalg.vector_norm(
                                actual_v[local_id, :2] - post_clip[local_id, :2]
                            ))
                            history[local_id].append(row)
                            stats[local_id]['samples'] += 1
                            stats[local_id]['interventions'] += int(
                                float(row['intervention_hat_norm']) > 1.0e-6
                            )
                            stats[local_id]['intervention_sum'] += float(row['intervention_hat_norm'])
                            stats[local_id]['intervention_max'] = max(
                                stats[local_id]['intervention_max'],
                                float(row['intervention_hat_norm']),
                            )
                            stats[local_id]['speed_sum'] += math.hypot(
                                float(row['v_actual_x']), float(row['v_actual_y'])
                            )
                            if int(row['is_extero_update']):
                                stats[local_id]['gt_updates'] += 1
                                stats[local_id]['gt_drift_sum'] += float(row['d_gt_0p1'])
                                stats[local_id]['gt_negative'] += int(float(row['d_gt_0p1']) < 0.0)
                                if not math.isnan(float(row['selected_horizon'])):
                                    stats[local_id]['selected_horizons'].append(
                                        float(row['selected_horizon'])
                                    )
                    else:
                        live_intervention = layer.last_intervention_norm.detach().reshape(-1)
                        for local_id in active_ids:
                            stats[local_id]['samples'] += 1
                            intervention = float(live_intervention[local_id])
                            stats[local_id]['interventions'] += int(intervention > 1.0e-6)
                            stats[local_id]['intervention_sum'] += intervention
                            stats[local_id]['intervention_max'] = max(
                                stats[local_id]['intervention_max'], intervention
                            )
                            stats[local_id]['speed_sum'] += math.hypot(
                                float(actual_v[local_id, 0]), float(actual_v[local_id, 1])
                            )
                            if bool(extero_update[local_id]) and required_horizons:
                                stats[local_id]['gt_updates'] += 1
                                if spec['oracle_multi_horizon']:
                                    stats[local_id]['gt_drift_sum'] += float(selected_drift[local_id])
                                    stats[local_id]['selected_horizons'].append(
                                        float(horizon_tensor[local_id])
                                    )
                                elif spec['safety_drift_source'] == 'gt':
                                    stats[local_id]['gt_drift_sum'] += float(selected_drift[local_id])
                                    stats[local_id]['selected_horizons'].append(
                                        float(horizon_tensor[local_id])
                                    )

                    collision_flags = env.collision_occurred.detach().bool().reshape(-1)
                    goal_flags = env.goal_reached_flag.detach().bool().reshape(-1)
                    timeout_flags = env.time_out_buf.detach().bool().reshape(-1)
                    stuck_flags = env.stand_still_flag.detach().bool().reshape(-1)
                    natural_done = dones.detach().bool().reshape(-1)
                    forced = active_before & (
                        episode_steps >= script_args.max_steps_per_episode
                    ) & ~natural_done
                    terminal = active_before & (natural_done | forced)

                    for local_id in active_ids:
                        if (
                            full_diagnostics
                            and bool(collision_flags[local_id])
                            and not bool(collision_recorded[local_id])
                        ):
                            collision_time = float(control_step + 1) * dt
                            window = extract_pre_collision_window(
                                history[local_id], collision_time, FAILURE_WINDOW_SAMPLES
                            )
                            collision_windows.extend(window)
                            collision_recorded[local_id] = True
                            collision_summaries.append(_failure_summary(
                                window, collision_time, script_args.training_seed,
                                batch_scenario_ids[local_id], script_args.repeat,
                            ))

                    for local_id in terminal.nonzero(as_tuple=False).flatten().tolist():
                        flags = {
                            'collision': bool(collision_flags[local_id]),
                            'goal_reached': bool(goal_flags[local_id]),
                            'timeout': bool(timeout_flags[local_id] | forced[local_id]),
                            'stuck': bool(stuck_flags[local_id]),
                        }
                        stats[local_id]['reward_sum'] = float(episode_rewards[local_id])
                        episodes.append(_episode_record(
                            script_args.training_seed, script_args.repeat,
                            batch_scenario_ids[local_id],
                            float(scenarios['scenario_vmax_speed_mps'][batch_indices[local_id]]),
                            int(episode_steps[local_id]), dt, stats[local_id], flags,
                        ))
                        finished[local_id] = True
                        finished_count += 1

                    if control_step == 0 or (control_step + 1) % script_args.progress_interval_steps == 0:
                        print(
                            '[phase4] mode={} batch {}/{} step {}/{} active={}'.format(
                                script_args.mode, batch_number, len(batches),
                                control_step + 1, script_args.max_steps_per_episode,
                                int((~finished).sum()),
                            ), flush=True,
                        )

                if not bool(finished.all()):
                    raise RuntimeError(
                        'Phase-4 batch did not produce one terminal outcome per scenario'
                    )

        episodes.sort(key=lambda row: int(row['scenario_id']))
        os.makedirs(output_dir, exist_ok=True)
        _write_csv(os.path.join(output_dir, 'episodes.csv'), episodes, EPISODE_FIELDS)
        if script_args.failure_analysis:
            _write_csv(os.path.join(output_dir, 'collision_windows.csv'), collision_windows, WINDOW_FIELDS)
            _write_csv(
                os.path.join(output_dir, 'collision_episode_summary.csv'),
                collision_summaries,
                list(collision_summaries[0].keys()) if collision_summaries else [
                    'seed', 'scenario_id', 'repeat'
                ],
            )

        outcomes = outcome_counts(episodes)
        summary = {
            'task': TASK_NAME,
            'training_seed': int(script_args.training_seed),
            'repeat': int(script_args.repeat),
            'mode': script_args.mode,
            'safety_drift_source': spec['safety_drift_source'],
            'oracle_horizon_s': spec['oracle_horizon_s'],
            'oracle_multi_horizon': spec['oracle_multi_horizon'],
            'oracle_horizon_set': spec['oracle_horizon_set'],
            'drift_refresh_hz': 10.0,
            'control_frequency_hz': 50.0,
            'control_dt_s': dt,
            'zoh_control_steps': int(round(0.1 / dt)),
            'policy_path': policy_path,
            'estimator_checkpoint': estimator_path,
            'scenario_bank': bank_path,
            'scenario_bank_hash': metadata['bank_hash'],
            'scenario_count': len(episodes),
            'one_episode_per_scenario': True,
            'safe_success_rate': outcomes['rates']['safe_success'],
            'collision_rate': outcomes['rates']['collision_failure'],
            'stuck_rate': outcomes['rates']['stuck_failure'],
            'timeout_rate': outcomes['rates']['timeout_failure'],
            'other_failure_rate': outcomes['rates']['other_failure'],
            'terminal_outcome_counts': outcomes['counts'],
            'terminal_outcome_rates': outcomes['rates'],
            'mean_robot_speed_mps': _mean([row['mean_speed_mps'] for row in episodes]),
            'intervention_frequency': _mean([row['intervention_frequency'] for row in episodes]),
            'mean_intervention_norm': _mean([row['mean_intervention_norm'] for row in episodes]),
            'max_intervention_norm': max([row['max_intervention_norm'] for row in episodes], default=float('nan')),
            'difficulty_stratified': _difficulty_report(episodes),
            'collision_episode_count': len(collision_summaries),
            'artifacts': {
                'episodes_csv': 'episodes.csv',
                'collision_windows_csv': 'collision_windows.csv' if script_args.failure_analysis else None,
                'collision_episode_summary_csv': 'collision_episode_summary.csv' if script_args.failure_analysis else None,
                'evaluation_manifest': 'evaluation_manifest.json',
            },
        }
        with open(os.path.join(output_dir, 'summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2, allow_nan=True, sort_keys=True)

        manifest = build_evaluation_manifest(
            args=args, env=env, env_cfg=env_cfg, sim_params=env.sim_params,
            policy_path=policy_path, estimator_checkpoint=estimator_path,
            bank_path=bank_path, bank_metadata=metadata, scenario_ids=selected_scenario_ids,
            num_envs=num_envs, max_steps=script_args.max_steps_per_episode,
            control_dt=dt, safety_mode='predictive',
            deterministic_settings=deterministic_settings,
            headless=True, cbf_layer=layer, task_name=TASK_NAME,
            evaluation_kind='phase4_failure_analysis' if script_args.failure_analysis else 'phase4_oracle',
            evaluator_script=__file__,
        )
        manifest.update({
            'training_seed': int(script_args.training_seed),
            'repeat': int(script_args.repeat),
            'safety_drift_source': spec['safety_drift_source'],
            'oracle_horizon_s': spec['oracle_horizon_s'],
            'oracle_multi_horizon': spec['oracle_multi_horizon'],
            'oracle_horizon_set': spec['oracle_horizon_set'],
            'drift_refresh_hz': 10.0,
            'control_frequency_hz': 50.0,
            'zoh_control_steps': int(round(0.1 / dt)),
            'actor_observation_unchanged': True,
            'policy_weights_unchanged': True,
            'shadow_projection_does_not_control_trajectory': True,
        })
        with open(os.path.join(output_dir, 'evaluation_manifest.json'), 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        if script_args.failure_analysis:
            taxonomy_counts = {}
            for row in collision_summaries:
                for tag in json.loads(row['failure_tags']):
                    taxonomy_counts[tag] = taxonomy_counts.get(tag, 0) + 1
            _write_csv(
                os.path.join(output_dir, 'failure_taxonomy_counts.csv'),
                [
                    {'label': label, 'count': count,
                     'fraction_of_collisions': count / max(len(collision_summaries), 1)}
                    for label, count in sorted(taxonomy_counts.items())
                ],
                ['label', 'count', 'fraction_of_collisions'],
            )
            with open(os.path.join(output_dir, 'failure_analysis_summary.json'), 'w') as handle:
                json.dump({
                    'training_seed': int(script_args.training_seed),
                    'repeat': int(script_args.repeat),
                    'collision_count': len(collision_summaries),
                    'scenario_count': len(episodes),
                    'validated_training_seeds': [1, 2],
                    'taxonomy_counts': taxonomy_counts,
                    'continuous_metrics_file': 'collision_episode_summary.csv',
                }, handle, indent=2, sort_keys=True, allow_nan=True)
        print(json.dumps(summary, indent=2, allow_nan=True, sort_keys=True), flush=True)
        return summary
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    script_args = _parse_script_args()
    isaac_args = get_args()
    evaluate(isaac_args, script_args)
