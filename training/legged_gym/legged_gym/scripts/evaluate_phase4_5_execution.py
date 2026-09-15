"""Phase-4.5 execution-path causal audit.

This evaluator keeps the learned policy, estimator, CBF geometry, scenario bank,
and low-level controller fixed while recording the four navigation command
stages and the realized one-step response:

    u_CBF -> clip[-3, 3] -> low-pass(beta) -> navigation bounds.

The evaluator is nominal/noise-free by default and is intentionally separate
from the formal Phase-4 evaluator.  It never changes training defaults and the
shadow re-projection is diagnostic only.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict, deque

from isaacgym import gymapi  # noqa: F401 - initialize Isaac Gym first
import torch

from legged_gym.envs import *  # noqa: F401,F403 - register tasks
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import get_args, task_registry
from rsl_rl.utils.phase2 import classify_terminal_outcome
from rsl_rl.utils.phase3 import (
    fixed_cohort_batches_for_indices,
    load_scenario_bank,
    outcome_counts,
    parse_scenario_ids,
)
from rsl_rl.utils.phase4_5 import (
    barrier_residual,
    distribution,
    reconstruct_execution_stages,
    shadow_reprojection,
    stage_deltas,
    stage_residuals,
    tracking_violation,
    violation_events,
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
WINDOW_SAMPLES = 100
EVENT_TOLERANCE = 1.0e-6

EPISODE_FIELDS = [
    'training_seed', 'scenario_id', 'terminal_outcome', 'goal_reached',
    'safe_success', 'collision', 'stuck', 'timeout', 'episode_length',
    'duration_s', 'mean_speed_mps', 'intervention_frequency',
    'mean_intervention_norm', 'max_intervention_norm',
]

TIMESERIES_FIELDS = [
    'global_step', 'scenario_id', 'episode_step', 'time_s',
    'terminal_outcome', 'robot_xy', 'robot_yaw', 'goal_xy',
    'u_cbf', 'u_rawclip', 'u_filter', 'u_cmd',
    'raw_clamp_delta', 'filter_delta', 'bound_clip_delta',
    'filter_delta_yaw_abs', 'bound_clip_delta_yaw_abs',
    'raw_clamp_event', 'filter_change_event', 'bound_clip_event',
    'r_learned_cbf', 'r_learned_rawclip', 'r_learned_filter',
    'r_learned_cmd', 'r_gt_cbf', 'r_gt_rawclip', 'r_gt_filter', 'r_gt_cmd',
    'delta_r_rawclip_learned', 'delta_r_filter_learned',
    'delta_r_bound_learned', 'delta_r_rawclip_gt', 'delta_r_filter_gt',
    'delta_r_bound_gt',
    'V_raw_learned', 'V_filter_learned', 'V_bound_learned',
    'V_raw_gt', 'V_filter_gt', 'V_bound_gt',
    'shadow_reproject_action', 'shadow_reproject_norm',
    'shadow_r_learned', 'shadow_r_gt', 'shadow_repair_learned',
    'shadow_repair_gt', 'shadow_reproject_out_of_bounds',
    'v_actual_pre', 'v_actual_post', 'tracking_error_pre',
    'tracking_error_post', 'r_actual_post_learned', 'r_actual_post_gt',
    'V_track_learned', 'V_track_gt', 'command_norm', 'command_yaw_abs',
    'robot_speed_pre', 'robot_speed_post', 'intervention_norm',
    'actual_velocity_change',
    'goal_reached', 'collision', 'stuck', 'timeout', 'done',
]

WINDOW_FIELDS = ['window_type', 'window_index'] + TIMESERIES_FIELDS


def _parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--policy_path', required=True)
    parser.add_argument('--scenario_bank', required=True)
    parser.add_argument('--estimator_checkpoint', required=True)
    parser.add_argument('--training_seed', type=int, required=True)
    parser.add_argument('--scenario_ids', default=None)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--max_steps_per_episode', type=int, default=3000)
    parser.add_argument('--progress_interval_steps', type=int, default=250)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--filter_beta', type=float, default=None)
    parser.add_argument(
        '--controller_noise', choices=('off', 'on'), default='off',
        help='Nominal Phase-4.5 defaults to off; on is robustness-only.',
    )
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


def _json_value(value, local_id):
    return json.dumps(
        value[local_id].detach().cpu().tolist(), separators=(',', ':')
    )


def _json_vector(value):
    return json.dumps(value.detach().cpu().tolist(), separators=(',', ':'))


def _yaw_from_quaternion(quaternion, local_id):
    quat = quaternion[local_id]
    sin_yaw = 2.0 * (quat[3] * quat[2] + quat[0] * quat[1])
    cos_yaw = 1.0 - 2.0 * (quat[1].square() + quat[2].square())
    return float(torch.atan2(sin_yaw, cos_yaw).detach().cpu())


def _float(value):
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def _episode_record(seed, scenario_id, steps, dt, stats, flags):
    outcome = classify_terminal_outcome(
        flags['collision'], flags['goal_reached'], flags['timeout'], flags['stuck']
    )
    samples = max(stats['samples'], 1)
    return {
        'training_seed': int(seed),
        'scenario_id': int(scenario_id),
        'terminal_outcome': outcome,
        'goal_reached': int(flags['goal_reached']),
        'safe_success': int(outcome == 'safe_success'),
        'collision': int(outcome == 'collision_failure'),
        'stuck': int(outcome == 'stuck_failure'),
        'timeout': int(outcome == 'timeout_failure'),
        'episode_length': int(steps),
        'duration_s': float(steps) * dt,
        'mean_speed_mps': stats['speed_sum'] / samples,
        'intervention_frequency': stats['interventions'] / samples,
        'mean_intervention_norm': stats['intervention_sum'] / samples,
        'max_intervention_norm': stats['intervention_max'],
    }


def _row_value(row, field):
    value = row.get(field, '')
    if value in ('', None):
        return float('nan')
    return float(value)


def _mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else float('nan')


def _row_metrics_summary(rows):
    fields = (
        'raw_clamp_delta', 'filter_delta', 'bound_clip_delta',
        'filter_delta_yaw_abs', 'bound_clip_delta_yaw_abs',
        'delta_r_rawclip_learned', 'delta_r_filter_learned',
        'delta_r_bound_learned', 'delta_r_rawclip_gt',
        'delta_r_filter_gt', 'delta_r_bound_gt', 'tracking_error_pre',
        'tracking_error_post', 'command_norm', 'robot_speed_post',
        'actual_velocity_change',
    )
    result = {'sample_count': len(rows)}
    for field in fields:
        result[field] = distribution([_row_value(row, field) for row in rows])
    event_fields = (
        'raw_clamp_event', 'filter_change_event', 'bound_clip_event',
        'V_raw_learned', 'V_filter_learned', 'V_bound_learned',
        'V_raw_gt', 'V_filter_gt', 'V_bound_gt',
        'V_track_learned', 'V_track_gt', 'shadow_repair_learned',
        'shadow_repair_gt', 'shadow_reproject_out_of_bounds',
    )
    for field in event_fields:
        values = [_row_value(row, field) for row in rows]
        result[field + '_rate'] = _mean(values)
        result[field + '_count'] = int(sum(value == 1.0 for value in values))
    return result


def _build_outcome_table(timeseries_path, episodes):
    by_id = {int(row['scenario_id']): row['terminal_outcome'] for row in episodes}
    grouped = defaultdict(list)
    with open(timeseries_path, newline='') as handle:
        for row in csv.DictReader(handle):
            outcome = by_id[int(row['scenario_id'])]
            grouped[outcome].append(row)
            grouped['all'].append(row)
    output = []
    for outcome in ('safe_success', 'collision_failure', 'stuck_failure', 'timeout_failure', 'other_failure', 'all'):
        if outcome not in grouped:
            continue
        metrics = _row_metrics_summary(grouped[outcome])
        for metric, value in sorted(metrics.items()):
            if isinstance(value, dict):
                for statistic, statistic_value in sorted(value.items()):
                    output.append({
                        'outcome': outcome, 'metric': metric,
                        'statistic': statistic, 'value': statistic_value,
                    })
            else:
                output.append({
                    'outcome': outcome, 'metric': metric,
                    'statistic': 'value', 'value': value,
                })
    return output


def evaluate(args, script_args):
    policy_path = _resolve_file(script_args.policy_path, 'policy checkpoint')
    bank_path = _resolve_file(script_args.scenario_bank, 'scenario bank')
    estimator_path = _resolve_file(
        script_args.estimator_checkpoint, 'estimator checkpoint'
    )
    bank = load_scenario_bank(bank_path)
    metadata = bank['metadata']
    if metadata.get('bank_hash') != AUTHORITATIVE_BANK_HASH:
        raise ValueError(
            'scenario bank hash mismatch: expected {}, got {}'.format(
                AUTHORITATIVE_BANK_HASH, metadata.get('bank_hash')
            )
        )
    scenarios = bank['scenarios']
    selected_ids = parse_scenario_ids(
        script_args.scenario_ids, int(metadata['num_scenarios'])
    )
    num_envs = int(script_args.num_envs)
    if num_envs < 1 or len(selected_ids) % num_envs != 0:
        raise ValueError(
            'Phase-4.5 requires complete fixed batches: num_envs={} selected={}'
            .format(num_envs, len(selected_ids))
        )
    if script_args.max_steps_per_episode < 1:
        raise ValueError('--max_steps_per_episode must be positive')
    if script_args.filter_beta is not None and not 0.0 < script_args.filter_beta <= 1.0:
        raise ValueError('--filter_beta must be in (0, 1]')

    generation_seed = int(metadata['generation_seed'])
    deterministic_settings = configure_deterministic_evaluation(generation_seed)
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    configure_evaluation_cfg(
        env_cfg, train_cfg, generation_seed, num_envs,
        safety_mode='predictive', estimator_checkpoint=estimator_path,
    )
    # Phase-4.5 nominal is explicitly noise-free.  The default config remains
    # unchanged; this is an evaluator-local protocol setting.
    env_cfg.controller.add_noise = script_args.controller_noise == 'on'
    if script_args.controller_noise == 'off':
        env_cfg.controller.noise_level = 0.0
    if script_args.filter_beta is not None:
        env_cfg.commands.alpha = float(script_args.filter_beta)
    beta = float(env_cfg.commands.alpha)

    args.task = TASK_NAME
    args.num_envs = num_envs
    args.seed = generation_seed
    args.wandb = False
    args.headless = True

    output_dir = os.path.abspath(os.path.expanduser(script_args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    timeseries_path = os.path.join(output_dir, 'execution_timeseries.csv')
    try:
        if bool(getattr(env.controller, 'noise_enabled', False)) != (
            script_args.controller_noise == 'on'
        ):
            raise RuntimeError('controller noise override did not reach live controller')
        if bool(getattr(env_cfg.noise, 'add_noise', False)):
            raise RuntimeError('environment observation noise unexpectedly enabled')
        if any(bool(getattr(env_cfg.domain_rand, name, False)) for name in (
            'randomize_friction', 'randomize_base_mass', 'push_robots',
        )):
            raise RuntimeError('domain randomization unexpectedly enabled')

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
        nav_min = env.nav_clip_min.detach().clone()
        nav_max = env.nav_clip_max.detach().clone()
        batches = fixed_cohort_batches_for_indices(selected_ids, num_envs)
        scenario_ids = scenarios['scenario_id'][torch.as_tensor(selected_ids)].tolist()

        episodes = []
        window_rows = []
        all_values = defaultdict(list)
        event_counts = defaultdict(int)
        sample_count = 0
        with open(timeseries_path, 'w', newline='') as timeseries_handle:
            timeseries_writer = csv.DictWriter(
                timeseries_handle, fieldnames=TIMESERIES_FIELDS,
                extrasaction='ignore',
            )
            timeseries_writer.writeheader()

            with torch.inference_mode():
                for batch_number, batch_indices in enumerate(batches, start=1):
                    batch_scenario_ids = scenarios['scenario_id'][
                        torch.as_tensor(batch_indices)
                    ].tolist()
                    print(
                        '[phase4.5] beta={} seed={} batch {}/{} scenarios={}-{}'.format(
                            beta, script_args.training_seed,
                            batch_number, len(batches),
                            batch_indices[0], batch_indices[-1],
                        ), flush=True,
                    )
                    restore_scenario_batch(env, bank, batch_indices)
                    runner.reset_safety_context()
                    obs = env.get_observations().to(env.device)
                    finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
                    episode_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
                    previous_positions = env.root_states[:, :2].detach().clone()
                    stats = [{
                        'samples': 0, 'interventions': 0,
                        'intervention_sum': 0.0, 'intervention_max': 0.0,
                        'speed_sum': 0.0,
                    } for _ in range(num_envs)]
                    histories = [deque(maxlen=WINDOW_SAMPLES) for _ in range(num_envs)]

                    for control_step in range(script_args.max_steps_per_episode):
                        active_before = ~finished
                        if not bool(active_before.any()):
                            break

                        learned_drift, shield_rays = runner.build_safety_context(obs)
                        gt_drift = env.lse_drift_gt.detach().clone()
                        actual_pre = env.base_lin_vel.detach().clone()
                        previous_filter = env.nav_actions_filtered.detach().clone()
                        action = runner.alg.actor_critic.act_inference(
                            obs, safety_drift=learned_drift, shield_rays=shield_rays
                        )
                        u_cbf = action.detach().clone()
                        if not torch.allclose(u_cbf, runner.alg.actor_critic.u_s.detach()):
                            raise RuntimeError('u_CBF is not the live actor CBF output')
                        alpha = runner.alg.actor_critic.alpha.detach().clone()
                        lgh = layer.last_Lgh.detach().clone()
                        alpha_h = layer.last_alpha_h.detach().clone()
                        stages = reconstruct_execution_stages(
                            u_cbf, previous_filter, beta, nav_min, nav_max
                        )
                        deltas = stage_deltas(stages)
                        residual_learned = stage_residuals(
                            learned_drift, lgh, alpha_h, stages
                        )
                        residual_gt = stage_residuals(
                            gt_drift, lgh, alpha_h, stages
                        )
                        events_learned = violation_events(residual_learned)
                        events_gt = violation_events(residual_gt)
                        shadow_learned = shadow_reprojection(
                            layer, stages['filter'], shield_rays, alpha,
                            learned_drift, nav_min, nav_max,
                        )
                        shadow_gt = shadow_reprojection(
                            layer, stages['filter'], shield_rays, alpha,
                            gt_drift, nav_min, nav_max,
                        )
                        r_actual_post_learned = None
                        r_actual_post_gt = None

                        action[finished] = 0.0
                        obs, _, rewards, dones, _ = env.step(action)
                        obs = obs.to(env.device)
                        actual_post = env.base_lin_vel.detach().clone()
                        actual_yaw_post = env.base_ang_vel[:, 2].detach().clone()
                        new_positions = env.root_states[:, :2].detach().clone()
                        episode_steps[active_before] += 1
                        previous_positions = new_positions

                        # Verify the live env implementation against the pure
                        # reconstruction before using the values in diagnostics.
                        active_ids = active_before.nonzero(as_tuple=False).flatten()
                        if not torch.allclose(
                            env.nav_actions_orig[active_ids], stages['rawclip'][active_ids],
                            atol=2e-6, rtol=2e-6,
                        ):
                            raise RuntimeError('live raw clip does not match reconstruction')
                        if not torch.allclose(
                            env.nav_actions_filtered[active_ids], stages['filter'][active_ids],
                            atol=2e-6, rtol=2e-6,
                        ):
                            raise RuntimeError('live filter does not match reconstruction')
                        if not torch.allclose(
                            env.nav_actions_after_clip[active_ids], stages['cmd'][active_ids],
                            atol=2e-6, rtol=2e-6,
                        ):
                            raise RuntimeError('live command bounds do not match reconstruction')

                        collision_flags = env.collision_occurred.detach().bool().reshape(-1)
                        goal_flags = env.goal_reached_flag.detach().bool().reshape(-1)
                        timeout_flags = env.time_out_buf.detach().bool().reshape(-1)
                        stuck_flags = env.stand_still_flag.detach().bool().reshape(-1)
                        natural_done = dones.detach().bool().reshape(-1)
                        forced = active_before & (
                            episode_steps >= script_args.max_steps_per_episode
                        ) & ~natural_done
                        terminal = active_before & (natural_done | forced)

                        r_actual_post_learned = barrier_residual(
                            learned_drift, lgh, alpha_h,
                            torch.cat((actual_post[:, :2], u_cbf[:, 2:3]), dim=-1),
                        )
                        r_actual_post_gt = barrier_residual(
                            gt_drift, lgh, alpha_h,
                            torch.cat((actual_post[:, :2], u_cbf[:, 2:3]), dim=-1),
                        )
                        track_events_learned = tracking_violation(
                            residual_learned['cmd'], r_actual_post_learned
                        )
                        track_events_gt = tracking_violation(
                            residual_gt['cmd'], r_actual_post_gt
                        )

                        for local_id in active_ids.tolist():
                            row = {
                                'global_step': int(control_step),
                                'scenario_id': int(batch_scenario_ids[local_id]),
                                'episode_step': int(episode_steps[local_id]),
                                'time_s': float(episode_steps[local_id]) * dt,
                                'terminal_outcome': '',
                                'robot_xy': _json_value(env.root_states[:, :2], local_id),
                                'robot_yaw': _yaw_from_quaternion(env.root_states[:, 3:7], local_id),
                                'goal_xy': _json_value(env.position_targets[:, :2], local_id),
                                'u_cbf': _json_value(stages['cbf'], local_id),
                                'u_rawclip': _json_value(stages['rawclip'], local_id),
                                'u_filter': _json_value(stages['filter'], local_id),
                                'u_cmd': _json_value(stages['cmd'], local_id),
                                'raw_clamp_delta': _float(torch.linalg.vector_norm(deltas['raw_clamp'][local_id])),
                                'filter_delta': _float(torch.linalg.vector_norm(deltas['filter'][local_id])),
                                'bound_clip_delta': _float(torch.linalg.vector_norm(deltas['bound'][local_id])),
                                'filter_delta_yaw_abs': abs(_float(deltas['filter'][local_id, 2])),
                                'bound_clip_delta_yaw_abs': abs(_float(deltas['bound'][local_id, 2])),
                                'raw_clamp_event': int(_float(torch.linalg.vector_norm(deltas['raw_clamp'][local_id])) > EVENT_TOLERANCE),
                                'filter_change_event': int(_float(torch.linalg.vector_norm(deltas['filter'][local_id])) > EVENT_TOLERANCE),
                                'bound_clip_event': int(_float(torch.linalg.vector_norm(deltas['bound'][local_id])) > EVENT_TOLERANCE),
                                'r_learned_cbf': _float(residual_learned['cbf'][local_id]),
                                'r_learned_rawclip': _float(residual_learned['rawclip'][local_id]),
                                'r_learned_filter': _float(residual_learned['filter'][local_id]),
                                'r_learned_cmd': _float(residual_learned['cmd'][local_id]),
                                'r_gt_cbf': _float(residual_gt['cbf'][local_id]),
                                'r_gt_rawclip': _float(residual_gt['rawclip'][local_id]),
                                'r_gt_filter': _float(residual_gt['filter'][local_id]),
                                'r_gt_cmd': _float(residual_gt['cmd'][local_id]),
                                'delta_r_rawclip_learned': _float((residual_learned['rawclip'] - residual_learned['cbf'])[local_id]),
                                'delta_r_filter_learned': _float((residual_learned['filter'] - residual_learned['rawclip'])[local_id]),
                                'delta_r_bound_learned': _float((residual_learned['cmd'] - residual_learned['filter'])[local_id]),
                                'delta_r_rawclip_gt': _float((residual_gt['rawclip'] - residual_gt['cbf'])[local_id]),
                                'delta_r_filter_gt': _float((residual_gt['filter'] - residual_gt['rawclip'])[local_id]),
                                'delta_r_bound_gt': _float((residual_gt['cmd'] - residual_gt['filter'])[local_id]),
                                'V_raw_learned': int(events_learned['raw_clamp'][local_id]),
                                'V_filter_learned': int(events_learned['filter'][local_id]),
                                'V_bound_learned': int(events_learned['bound'][local_id]),
                                'V_raw_gt': int(events_gt['raw_clamp'][local_id]),
                                'V_filter_gt': int(events_gt['filter'][local_id]),
                                'V_bound_gt': int(events_gt['bound'][local_id]),
                                'shadow_reproject_action': _json_value(shadow_learned['action'], local_id),
                                'shadow_reproject_norm': _float(shadow_learned['intervention_norm'][local_id]),
                                'shadow_r_learned': _float(shadow_learned['residual'][local_id]),
                                'shadow_r_gt': _float(shadow_gt['residual'][local_id]),
                                'shadow_repair_learned': int(
                                    residual_learned['filter'][local_id] < 0.0
                                    and shadow_learned['residual'][local_id] >= 0.0
                                ),
                                'shadow_repair_gt': int(
                                    residual_gt['filter'][local_id] < 0.0
                                    and shadow_gt['residual'][local_id] >= 0.0
                                ),
                                'shadow_reproject_out_of_bounds': int(shadow_learned['out_of_bounds'][local_id]),
                                'v_actual_pre': _json_value(actual_pre, local_id),
                                'v_actual_post': _json_value(actual_post, local_id),
                                'tracking_error_pre': _float(torch.linalg.vector_norm(actual_pre[local_id, :2] - stages['cmd'][local_id, :2])),
                                'tracking_error_post': _float(torch.linalg.vector_norm(actual_post[local_id, :2] - stages['cmd'][local_id, :2])),
                                'r_actual_post_learned': _float(r_actual_post_learned[local_id]),
                                'r_actual_post_gt': _float(r_actual_post_gt[local_id]),
                                'V_track_learned': int(track_events_learned[local_id]),
                                'V_track_gt': int(track_events_gt[local_id]),
                                'command_norm': _float(torch.linalg.vector_norm(stages['cmd'][local_id])),
                                'command_yaw_abs': abs(_float(stages['cmd'][local_id, 2])),
                                'robot_speed_pre': _float(torch.linalg.vector_norm(actual_pre[local_id, :2])),
                                'robot_speed_post': _float(torch.linalg.vector_norm(actual_post[local_id, :2])),
                                'intervention_norm': _float(layer.last_intervention_norm[local_id]),
                                'actual_velocity_change': _float(torch.linalg.vector_norm(
                                    actual_post[local_id, :2] - actual_pre[local_id, :2]
                                )),
                                'goal_reached': int(goal_flags[local_id]),
                                'collision': int(collision_flags[local_id]),
                                'stuck': int(stuck_flags[local_id]),
                                'timeout': int(timeout_flags[local_id] | forced[local_id]),
                                'done': int(terminal[local_id]),
                            }
                            histories[local_id].append(row)
                            timeseries_writer.writerow(row)
                            timeseries_handle.flush()
                            sample_count += 1

                            current_stats = stats[local_id]
                            current_stats['samples'] += 1
                            intervention = row['intervention_norm']
                            current_stats['interventions'] += int(intervention > EVENT_TOLERANCE)
                            current_stats['intervention_sum'] += intervention
                            current_stats['intervention_max'] = max(
                                current_stats['intervention_max'], intervention
                            )
                            current_stats['speed_sum'] += row['robot_speed_post']
                            for name, value in (
                                ('delta_r_rawclip_learned', row['delta_r_rawclip_learned']),
                                ('delta_r_filter_learned', row['delta_r_filter_learned']),
                                ('delta_r_bound_learned', row['delta_r_bound_learned']),
                                ('delta_r_rawclip_gt', row['delta_r_rawclip_gt']),
                                ('delta_r_filter_gt', row['delta_r_filter_gt']),
                                ('delta_r_bound_gt', row['delta_r_bound_gt']),
                                ('tracking_error_pre', row['tracking_error_pre']),
                                ('tracking_error_post', row['tracking_error_post']),
                                ('raw_clamp_delta', row['raw_clamp_delta']),
                                ('filter_delta', row['filter_delta']),
                                ('bound_clip_delta', row['bound_clip_delta']),
                                ('filter_delta_yaw_abs', row['filter_delta_yaw_abs']),
                                ('bound_clip_delta_yaw_abs', row['bound_clip_delta_yaw_abs']),
                                ('actual_velocity_change', row['actual_velocity_change']),
                            ):
                                all_values[name].append(value)
                            for name in (
                                'raw_clamp_event', 'filter_change_event', 'bound_clip_event',
                                'V_raw_learned', 'V_filter_learned', 'V_bound_learned',
                                'V_raw_gt', 'V_filter_gt', 'V_bound_gt',
                                'V_track_learned', 'V_track_gt',
                                'shadow_repair_learned', 'shadow_repair_gt',
                                'shadow_reproject_out_of_bounds',
                            ):
                                event_counts[name] += int(row[name])

                        for local_id in active_ids.tolist():
                            if not bool(terminal[local_id]):
                                continue
                            flags = {
                                'collision': bool(collision_flags[local_id]),
                                'goal_reached': bool(goal_flags[local_id]),
                                'timeout': bool(timeout_flags[local_id] | forced[local_id]),
                                'stuck': bool(stuck_flags[local_id]),
                            }
                            outcome = classify_terminal_outcome(
                                flags['collision'], flags['goal_reached'],
                                flags['timeout'], flags['stuck'],
                            )
                            episodes.append(_episode_record(
                                script_args.training_seed,
                                batch_scenario_ids[local_id],
                                int(episode_steps[local_id]), dt,
                                stats[local_id], flags,
                            ))
                            if outcome in ('collision_failure', 'safe_success'):
                                window_type = (
                                    'collision_last_2s'
                                    if outcome == 'collision_failure'
                                    else 'safe_success_pre_goal_last_2s'
                                )
                                for window_index, history_row in enumerate(histories[local_id]):
                                    item = dict(history_row)
                                    item['window_type'] = window_type
                                    item['window_index'] = int(window_index)
                                    item['terminal_outcome'] = outcome
                                    window_rows.append(item)
                            finished[local_id] = True

                        if control_step == 0 or (control_step + 1) % script_args.progress_interval_steps == 0:
                            print(
                                '[phase4.5] step {}/{} active={}'.format(
                                    control_step + 1,
                                    script_args.max_steps_per_episode,
                                    int((~finished).sum()),
                                ), flush=True,
                            )

                    if not bool(finished.all()):
                        raise RuntimeError(
                            'Phase-4.5 batch did not produce one terminal outcome per scenario'
                        )

        episodes.sort(key=lambda row: int(row['scenario_id']))
        _write_csv(os.path.join(output_dir, 'episodes.csv'), episodes, EPISODE_FIELDS)
        _write_csv(os.path.join(output_dir, 'terminal_windows.csv'), window_rows, WINDOW_FIELDS)
        outcome_table = _build_outcome_table(timeseries_path, episodes)
        _write_csv(
            os.path.join(output_dir, 'execution_stage_by_outcome.csv'),
            outcome_table,
            ['outcome', 'metric', 'statistic', 'value'],
        )
        outcomes = outcome_counts(episodes)
        mean_robot_speed = _mean([row['mean_speed_mps'] for row in episodes])
        mean_intervention_frequency = _mean(
            [row['intervention_frequency'] for row in episodes]
        )
        mean_intervention_norm = _mean(
            [row['mean_intervention_norm'] for row in episodes]
        )
        summary = {
            'task': TASK_NAME,
            'training_seed': int(script_args.training_seed),
            'policy_path': policy_path,
            'estimator_checkpoint': estimator_path,
            'scenario_bank': bank_path,
            'scenario_bank_hash': metadata['bank_hash'],
            'scenario_ids': selected_ids,
            'num_envs': num_envs,
            'control_dt_s': dt,
            'filter_beta': beta,
            'controller_noise_enabled': bool(getattr(env_cfg.controller, 'add_noise', False)),
            'controller_noise_level': float(getattr(env_cfg.controller, 'noise_level', 0.0)),
            'safe_success_rate': outcomes['rates']['safe_success'],
            'collision_rate': outcomes['rates']['collision_failure'],
            'stuck_rate': outcomes['rates']['stuck_failure'],
            'timeout_rate': outcomes['rates']['timeout_failure'],
            'mean_robot_speed_mps': mean_robot_speed,
            'intervention_frequency': mean_intervention_frequency,
            'mean_intervention_norm': mean_intervention_norm,
            'terminal_outcome_counts': outcomes['counts'],
            'terminal_outcome_rates': outcomes['rates'],
            'execution_chain': {
                'u_cbf': 'actor.act_inference output after ExactLSECBFLayer',
                'u_rawclip': 'clip(u_cbf, -3, 3)',
                'u_filter': 'beta*u_rawclip + (1-beta)*previous_filter',
                'u_cmd': 'clip(u_filter, nav_clip_min, nav_clip_max)',
                'beta': beta,
                'nav_clip_min': nav_min.cpu().tolist(),
                'nav_clip_max': nav_max.cpu().tolist(),
            },
            'residual_definition': 'r(u,d)=d + Lgh*u_xy + alpha*h',
            'tracking_semantics': {
                'base_lin_vel_frame': 'body frame via quat_rotate_inverse',
                'tracking_error_post': 'one control-step response, not steady-state',
            },
            'stage_degradation_distributions': {
                name: distribution(values) for name, values in all_values.items()
            },
            'event_counts': dict(sorted(event_counts.items())),
            'event_rates': {
                name: count / max(sample_count, 1)
                for name, count in sorted(event_counts.items())
            },
            'matched_terminal_windows': {
                'collision_last_2s_count': sum(
                    row['window_type'] == 'collision_last_2s' for row in window_rows
                ),
                'safe_success_pre_goal_last_2s_count': sum(
                    row['window_type'] == 'safe_success_pre_goal_last_2s'
                    for row in window_rows
                ),
                'by_window': {
                    name: _row_metrics_summary([
                        row for row in window_rows if row['window_type'] == name
                    ])
                    for name in ('collision_last_2s', 'safe_success_pre_goal_last_2s')
                },
            },
            'shadow_reprojection': {
                'is_diagnostic_only': True,
                'executes_u_filter': True,
                'repair_rate_gt': event_counts['shadow_repair_gt'] / max(sample_count, 1),
            },
            'artifacts': {
                'episodes_csv': 'episodes.csv',
                'execution_timeseries_csv': 'execution_timeseries.csv',
                'terminal_windows_csv': 'terminal_windows.csv',
                'execution_stage_by_outcome_csv': 'execution_stage_by_outcome.csv',
                'execution_stage_summary_json': 'execution_stage_summary.json',
                'evaluation_manifest': 'evaluation_manifest.json',
            },
        }
        with open(os.path.join(output_dir, 'summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=True)
        with open(os.path.join(output_dir, 'execution_stage_summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=True)

        manifest = build_evaluation_manifest(
            args=args, env=env, env_cfg=env_cfg, sim_params=env.sim_params,
            policy_path=policy_path, estimator_checkpoint=estimator_path,
            bank_path=bank_path, bank_metadata=metadata, scenario_ids=scenario_ids,
            num_envs=num_envs, max_steps=script_args.max_steps_per_episode,
            control_dt=dt, safety_mode='predictive',
            deterministic_settings=deterministic_settings, headless=True,
            cbf_layer=layer, task_name=TASK_NAME,
            evaluation_kind='phase4_5_execution_audit', evaluator_script=__file__,
        )
        manifest.update({
            'phase4_5_protocol': 'noise_free_nominal_repeated_fixed_cohort',
            'filter_beta': beta,
            'controller_noise_enabled': bool(getattr(env_cfg.controller, 'add_noise', False)),
            'controller_noise_level': float(getattr(env_cfg.controller, 'noise_level', 0.0)),
            'observation_noise_enabled': bool(getattr(env_cfg.noise, 'add_noise', False)),
            'friction_randomization_enabled': bool(getattr(env_cfg.domain_rand, 'randomize_friction', False)),
            'mass_randomization_enabled': bool(getattr(env_cfg.domain_rand, 'randomize_base_mass', False)),
            'push_enabled': bool(getattr(env_cfg.domain_rand, 'push_robots', False)),
            'domain_randomization_enabled': any(bool(getattr(env_cfg.domain_rand, name, False)) for name in (
                'randomize_friction', 'randomize_base_mass', 'push_robots',
            )),
            'replay_enabled': bool(getattr(env_cfg.replay, 'enable_collision_replay', False)),
            'execution_chain': summary['execution_chain'],
            'residual_definition': summary['residual_definition'],
            'tracking_semantics': summary['tracking_semantics'],
            'shadow_projection_does_not_control_trajectory': True,
        })
        with open(os.path.join(output_dir, 'evaluation_manifest.json'), 'w') as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=True)
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True), flush=True)
        return summary
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    script_args = _parse_script_args()
    isaac_args = get_args()
    evaluate(isaac_args, script_args)
