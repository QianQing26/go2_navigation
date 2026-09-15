"""Evaluate exactly one frozen episode for every scenario in a fixed bank."""

import argparse
import csv
import datetime
import json
import math
import os
import sys

from isaacgym import gymapi  # noqa: F401 - initialize Isaac Gym first
import torch
from legged_gym.envs import *  # noqa: F401,F403 - register tasks
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import get_args, task_registry
from rsl_rl.utils.phase2 import classify_terminal_outcome, snapshot_control_context
from rsl_rl.utils.phase3 import (
    DIFFICULTY_BINS,
    difficulty_bin,
    fixed_cohort_batches_for_indices,
    load_scenario_bank,
    outcome_counts,
    parse_scenario_ids,
    validate_fixed_cohort_rows,
)

from phase3_common import (
    build_evaluation_manifest,
    configure_deterministic_evaluation,
    configure_evaluation_cfg,
    restore_scenario_batch,
)


TASK_NAME = 'go2_pos_dynamic'
MODES = {'original', 'synchronized_static', 'predictive'}


def _parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--policy_path', required=True)
    parser.add_argument('--scenario_bank', required=True)
    parser.add_argument('--safety_mode', choices=sorted(MODES), required=True)
    parser.add_argument('--estimator_checkpoint', default='')
    parser.add_argument(
        '--controller_noise', choices=('on', 'off'), default='on',
        help=(
            'Evaluation-only RoboGauge observation-noise override. The default '
            '"on" preserves the historical fixed-cohort protocol; "off" sets '
            'controller.add_noise=False and controller.noise_level=0.'
        ),
    )
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--max_steps_per_episode', type=int, default=3000)
    parser.add_argument('--monitor_envs', type=int, default=8)
    parser.add_argument(
        '--scenario_ids', default=None,
        help='Comma-separated scenario IDs; omitted means the complete bank.',
    )
    parser.add_argument(
        '--diagnostic_scenario_ids', default=None,
        help='Comma-separated IDs to retain detailed trajectory diagnostics for.',
    )
    parser.add_argument('--evaluation_kind', default='fixed_cohort')
    parser.add_argument(
        '--progress_interval_steps', type=int, default=250,
        help='Print active-env progress every N control steps.',
    )
    parser.add_argument('--output_dir', default=None)
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
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


EPISODE_FIELDS = [
    'scenario_id', 'scenario_bank_hash', 'difficulty_bin', 'vmax_speed_mps',
    'terminal_outcome', 'goal_reached', 'safe_success', 'collision', 'stuck', 'timeout',
    'other_failure', 'episode_length', 'duration_s', 'mean_speed_mps',
    'total_reward', 'intervention_frequency', 'mean_intervention_norm',
    'max_intervention_norm', 'residual_negative_probability',
    'mean_safety_drift', 'negative_drift_rate',
    'drift_induced_intervention_rate',
]

TIMESERIES_FIELDS = [
    'global_step', 'env_id', 'scenario_id', 'episode_step',
    'exteroception_updated', 'history_count', 'shield_rays_min',
    'shield_rays_max', 'h_comp', 'safety_drift', 'Lgh_u', 'alpha_h',
    'residual', 'eta', 'intervention_norm', 'done', 'collision',
    'safe_success', 'timeout', 'stuck', 'other_failure', 'forced_limit',
    'robot_xy', 'robot_yaw', 'goal_xy', 'obstacle_xy', 'obstacle_velocity',
    'shield_rays', 'safety_drift_value', 'u_bar', 'u_s', 'alpha',
    'Lgh_norm_sq', 'denominator', 'pre_clip_action', 'post_clip_action',
]


def _json_tensor_value(value, local_id):
    """Serialize one diagnostic vector without losing its shape."""

    return json.dumps(
        value[local_id].detach().cpu().tolist(),
        separators=(',', ':'),
    )


def _yaw_from_quaternion(quaternion, local_id):
    quat = quaternion[local_id]
    sin_yaw = 2.0 * (quat[3] * quat[2] + quat[0] * quat[1])
    cos_yaw = 1.0 - 2.0 * (quat[1].square() + quat[2].square())
    return float(torch.atan2(sin_yaw, cos_yaw))


def _snapshot_diagnostic_context(
    env, actor_critic, layer, context, shield_rays, pre_clip_action, local_id
):
    """Capture the pre-step state needed to explain a trajectory divergence."""

    fallback = torch.zeros(1, device=env.device, dtype=shield_rays.dtype)
    lgh = getattr(layer, 'last_Lgh', None)
    lgh_norm_sq = (
        lgh.square().sum(dim=-1, keepdim=True)
        if lgh is not None else fallback
    )
    damping = float(getattr(layer, 'damping_factor', 0.0))
    obstacle_states = env.dynamic_obstacle_states
    return {
        'robot_xy': _json_tensor_value(env.root_states[:, :2], local_id),
        'robot_yaw': _yaw_from_quaternion(env.root_states[:, 3:7], local_id),
        'goal_xy': _json_tensor_value(env.position_targets[:, :2], local_id),
        'obstacle_xy': _json_tensor_value(obstacle_states[:, :, :2], local_id),
        'obstacle_velocity': _json_tensor_value(obstacle_states[:, :, 7:9], local_id),
        'shield_rays': _json_tensor_value(shield_rays, local_id),
        'safety_drift_value': float(context['safety_drift'][local_id]),
        'u_bar': _json_tensor_value(actor_critic.u_bar, local_id),
        'u_s': _json_tensor_value(actor_critic.u_s, local_id),
        'alpha': _json_tensor_value(actor_critic.alpha, local_id),
        'Lgh_norm_sq': float(lgh_norm_sq.reshape(-1)[local_id]),
        'denominator': float(lgh_norm_sq.reshape(-1)[local_id]) + damping,
        'pre_clip_action': _json_tensor_value(pre_clip_action, local_id),
    }


def _make_episode_record(
    row_index, scenario_id, bank_hash, vmax_speed, steps, dt, distance,
    reward, stats, flags,
):
    outcome = classify_terminal_outcome(
        flags['collision'], flags['goal_reached'], flags['timeout'], flags['stuck']
    )
    length = int(steps)
    samples = max(int(stats['samples']), 1)
    return {
        'scenario_id': int(scenario_id),
        'scenario_bank_hash': bank_hash,
        'difficulty_bin': difficulty_bin(vmax_speed),
        'vmax_speed_mps': float(vmax_speed),
        'terminal_outcome': outcome,
        'goal_reached': int(flags['goal_reached']),
        'safe_success': int(outcome == 'safe_success'),
        'collision': int(outcome == 'collision_failure'),
        'stuck': int(outcome == 'stuck_failure'),
        'timeout': int(outcome == 'timeout_failure'),
        'other_failure': int(outcome == 'other_failure'),
        'episode_length': length,
        'duration_s': length * dt,
        'mean_speed_mps': float(distance) / max(length * dt, 1.0e-6),
        'total_reward': float(reward),
        'intervention_frequency': stats['interventions'] / samples,
        'mean_intervention_norm': stats['intervention_sum'] / samples,
        'max_intervention_norm': stats['intervention_max'],
        'residual_negative_probability': stats['residual_negative'] / samples,
        'mean_safety_drift': stats['drift_sum'] / samples,
        'negative_drift_rate': stats['negative_drift'] / samples,
        'drift_induced_intervention_rate': (
            stats['drift_induced_intervention'] / samples
        ),
    }


def _difficulty_report(rows):
    report = []
    for name, lower, upper in DIFFICULTY_BINS:
        subset = [row for row in rows if row['difficulty_bin'] == name]
        report.append({
            'difficulty_bin': name,
            'lower_mps': lower,
            'upper_mps': upper,
            'num_scenarios': len(subset),
            'safe_success_rate': _mean(subset, 'safe_success'),
            'collision_rate': _mean(subset, 'collision'),
            'stuck_rate': _mean(subset, 'stuck'),
            'timeout_rate': _mean(subset, 'timeout'),
            # Difficulty is defined by analytic obstacle speed, not the
            # robot's realized displacement speed in this episode.
            'mean_speed_mps': _mean(subset, 'vmax_speed_mps'),
            'mean_robot_speed_mps': _mean(subset, 'mean_speed_mps'),
        })
    return report


def evaluate(args, script_args):
    policy_path = _resolve_file(script_args.policy_path, 'policy checkpoint')
    bank = load_scenario_bank(script_args.scenario_bank)
    metadata = bank['metadata']
    scenarios = bank['scenarios']
    num_scenarios = int(metadata['num_scenarios'])
    selected_ids = parse_scenario_ids(script_args.scenario_ids, num_scenarios)
    selected_indices = list(selected_ids)
    num_envs = int(script_args.num_envs)
    batches = fixed_cohort_batches_for_indices(selected_indices, num_envs)
    diagnostic_ids = (
        parse_scenario_ids(script_args.diagnostic_scenario_ids, num_scenarios)
        if script_args.diagnostic_scenario_ids is not None else
        selected_ids[:max(0, int(script_args.monitor_envs))]
    )
    diagnostic_id_set = set(diagnostic_ids)
    if not diagnostic_id_set.issubset(set(selected_ids)):
        raise ValueError('diagnostic scenarios must be in the evaluated cohort')
    if int(metadata.get('obstacle_count', -1)) < 1:
        raise ValueError('scenario bank has no valid obstacle count')
    if script_args.max_steps_per_episode < 1:
        raise ValueError('--max_steps_per_episode must be positive')
    if script_args.progress_interval_steps < 1:
        raise ValueError('--progress_interval_steps must be positive')
    if script_args.safety_mode == 'predictive':
        if not script_args.estimator_checkpoint:
            raise ValueError('--estimator_checkpoint is required for predictive mode')
        script_args.estimator_checkpoint = _resolve_file(
            script_args.estimator_checkpoint, 'estimator checkpoint'
        )

    generation_seed = int(metadata['generation_seed'])
    deterministic_settings = configure_deterministic_evaluation(generation_seed)
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    configure_evaluation_cfg(
        env_cfg, train_cfg, generation_seed, num_envs,
        safety_mode=script_args.safety_mode,
        estimator_checkpoint=script_args.estimator_checkpoint,
    )
    # This is deliberately an evaluator-local override.  The training config
    # default remains unchanged so the test can compare the historical
    # controller-noise protocol against a noise-free fixed-cohort protocol.
    if not hasattr(env_cfg, 'controller'):
        raise AttributeError('evaluation config has no controller section')
    if script_args.controller_noise == 'off':
        env_cfg.controller.add_noise = False
        env_cfg.controller.noise_level = 0.0
    else:
        env_cfg.controller.add_noise = True
    args.task = TASK_NAME
    args.num_envs = num_envs
    args.seed = generation_seed
    args.wandb = False

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    try:
        print(
            '[fixed-cohort] environment ready: scenarios={}, num_envs={}, '
            'bank_hash={}'.format(
                len(selected_ids), num_envs, metadata['bank_hash']
            ),
            flush=True,
        )
        if int(env.num_dynamic_obstacles) != int(metadata['obstacle_count']):
            raise ValueError('scenario bank obstacle count does not match evaluator')
        terrain_raw = bank.get('extras', {}).get('terrain_height_field_raw')
        terrain_origins = bank.get('extras', {}).get('terrain_env_origins')
        if terrain_raw is None or terrain_origins is None:
            raise ValueError('scenario bank does not contain static terrain metadata')
        actual_terrain = torch.as_tensor(env.terrain.height_field_raw)
        if not torch.equal(actual_terrain.cpu(), terrain_raw.cpu()):
            raise ValueError(
                'terrain generated from bank generation_seed does not match '
                'the stored terrain height field'
            )
        actual_origins = torch.as_tensor(env.terrain.env_origins)
        actual_origins_f = actual_origins.cpu().float()
        expected_origins_f = terrain_origins.cpu().float()
        if not torch.equal(actual_origins_f, expected_origins_f):
            max_origin_diff = float(
                (actual_origins_f - expected_origins_f).abs().max()
            )
            raise ValueError(
                'terrain origins do not match scenario bank: actual_shape={} '
                'expected_shape={} max_abs_diff={}'.format(
                    tuple(actual_origins.shape), tuple(terrain_origins.shape),
                    max_origin_diff,
                )
            )

        train_cfg.runner.resume = False
        # PyTorch 2.4.1 has a CUDA advanced-indexing assertion in the
        # framework's stochastic reset broadcast when deterministic algorithms
        # are enabled before runner construction.  The fixed-bank evaluator
        # never uses that reset for measured episodes, so initialize the
        # runner first and restore the deterministic setting immediately after.
        torch.use_deterministic_algorithms(False, warn_only=True)
        runner, _ = task_registry.make_alg_runner(
            env=env, name=TASK_NAME, args=args, train_cfg=train_cfg, log_root=None
        )
        torch.use_deterministic_algorithms(True, warn_only=True)
        runner.load(policy_path, load_optimizer=False)
        runner.alg.actor_critic.eval()
        env.do_reset = False
        print(
            '[fixed-cohort] policy loaded; starting {} complete batches '
            '(max_steps_per_episode={})'.format(
                len(batches), script_args.max_steps_per_episode
            ),
            flush=True,
        )
        layer = getattr(runner.alg.actor_critic, 'cbf_layer', None)
        dt = float(env.cfg.control.decimation * env.cfg.sim.dt)
        monitor_envs = max(0, int(script_args.monitor_envs))
        episodes = []
        timeseries = []
        collision_traces = []

        with torch.inference_mode():
            for batch_number, batch_indices in enumerate(batches, start=1):
                print(
                    '[fixed-cohort] batch {}/{} start: scenario_ids={}-{}'.format(
                        batch_number, len(batches), batch_indices[0],
                        batch_indices[-1],
                    ),
                    flush=True,
                )
                restore_scenario_batch(env, bank, batch_indices)
                runner.reset_safety_context()
                obs = env.get_observations().to(env.device)
                finished = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
                episode_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
                episode_rewards = torch.zeros(num_envs, dtype=torch.float, device=env.device)
                episode_distance = torch.zeros(num_envs, dtype=torch.float, device=env.device)
                previous_positions = env.root_states[:, :2].detach().clone()
                stats = [{
                    'samples': 0, 'interventions': 0, 'intervention_sum': 0.0,
                    'intervention_max': 0.0, 'residual_negative': 0,
                    'drift_sum': 0.0, 'negative_drift': 0,
                    'drift_induced_intervention': 0,
                } for _ in range(num_envs)]

                for global_step in range(script_args.max_steps_per_episode):
                    active_before = ~finished
                    if not bool(active_before.any()):
                        break
                    safety_drift, shield_rays = runner.build_safety_context(obs)
                    action = runner.alg.actor_critic.act_inference(
                        obs, safety_drift=safety_drift, shield_rays=shield_rays
                    )
                    pre_clip_action = action.detach().clone()
                    context = snapshot_control_context(
                        env, safety_drift, shield_rays, layer
                    )
                    diagnostics = {}
                    for local_id in range(num_envs):
                        if batch_indices[local_id] in diagnostic_id_set:
                            diagnostics[local_id] = _snapshot_diagnostic_context(
                                env, runner.alg.actor_critic, layer, context,
                                shield_rays, pre_clip_action, local_id,
                            )
                    action[finished] = 0.0
                    for local_id in active_before.nonzero(as_tuple=False).flatten().tolist():
                        current = stats[local_id]
                        intervention = float(context['intervention_norm'][local_id])
                        residual = float(context['residual'][local_id])
                        drift = float(context['safety_drift'][local_id])
                        current['samples'] += 1
                        current['interventions'] += int(intervention > 1.0e-6)
                        current['intervention_sum'] += intervention
                        current['intervention_max'] = max(
                            current['intervention_max'], intervention
                        )
                        current['residual_negative'] += int(residual < 0.0)
                        current['drift_sum'] += drift
                        current['negative_drift'] += int(drift < 0.0)
                        static_residual = (
                            float(context['Lgh_u'][local_id])
                            + float(context['alpha_h'][local_id])
                        )
                        current['drift_induced_intervention'] += int(
                            static_residual >= 0.0 and static_residual + drift < 0.0
                        )

                    obs, _, rewards, dones, _ = env.step(action)
                    obs = obs.to(env.device)
                    episode_steps[active_before] += 1
                    episode_rewards[active_before] += rewards[active_before]
                    new_positions = env.root_states[:, :2].detach().clone()
                    episode_distance[active_before] += torch.linalg.vector_norm(
                        new_positions[active_before] - previous_positions[active_before],
                        dim=-1,
                    )
                    previous_positions = new_positions

                    collision_flags = env.collision_occurred.detach().bool().reshape(-1)
                    goal_flags = env.goal_reached_flag.detach().bool().reshape(-1)
                    timeout_flags = env.time_out_buf.detach().bool().reshape(-1)
                    stuck_flags = env.stand_still_flag.detach().bool().reshape(-1)
                    natural_done = dones.detach().bool().reshape(-1)
                    forced = active_before & (
                        episode_steps >= script_args.max_steps_per_episode
                    ) & ~natural_done
                    terminal = active_before & (natural_done | forced)

                    for local_id in range(num_envs):
                        if batch_indices[local_id] not in diagnostic_id_set:
                            continue
                        if not bool(active_before[local_id]):
                            continue
                        outcome = classify_terminal_outcome(
                            collision_flags[local_id], goal_flags[local_id],
                            timeout_flags[local_id] | forced[local_id],
                            stuck_flags[local_id],
                        )
                        row = {
                            'global_step': global_step,
                            'env_id': local_id,
                            'scenario_id': batch_indices[local_id],
                            'episode_step': int(episode_steps[local_id]),
                            'exteroception_updated': int(context['exteroception_updated'][local_id]),
                            'history_count': int(context['history_count'][local_id]),
                            'shield_rays_min': float(context['shield_rays_min'][local_id]),
                            'shield_rays_max': float(context['shield_rays_max'][local_id]),
                            'h_comp': float(context['h_comp'][local_id]),
                            'safety_drift': float(context['safety_drift'][local_id]),
                            'Lgh_u': float(context['Lgh_u'][local_id]),
                            'alpha_h': float(context['alpha_h'][local_id]),
                            'residual': float(context['residual'][local_id]),
                            'eta': float(context['eta'][local_id]),
                            'intervention_norm': float(context['intervention_norm'][local_id]),
                            'done': int(terminal[local_id]),
                            'collision': int(outcome == 'collision_failure'),
                            'safe_success': int(outcome == 'safe_success'),
                            'timeout': int(timeout_flags[local_id] | forced[local_id]),
                            'stuck': int(outcome == 'stuck_failure'),
                            'other_failure': int(outcome == 'other_failure'),
                            'forced_limit': int(forced[local_id]),
                        }
                        row.update(diagnostics[local_id])
                        row['post_clip_action'] = _json_tensor_value(
                            env.nav_actions_after_clip, local_id
                        )
                        timeseries.append(row)
                        if row['collision']:
                            collision_traces.append(dict(row))

                    for local_id in terminal.nonzero(as_tuple=False).flatten().tolist():
                        scenario_index = batch_indices[local_id]
                        scenario_id = int(scenarios['scenario_id'][scenario_index])
                        flags = {
                            'collision': bool(collision_flags[local_id]),
                            'goal_reached': bool(goal_flags[local_id]),
                            'timeout': bool(timeout_flags[local_id] | forced[local_id]),
                            'stuck': bool(stuck_flags[local_id]),
                        }
                        episodes.append(_make_episode_record(
                            scenario_index, scenario_id, metadata['bank_hash'],
                            float(scenarios['scenario_vmax_speed_mps'][scenario_index]),
                            int(episode_steps[local_id]), dt,
                            float(episode_distance[local_id]),
                            float(episode_rewards[local_id]), stats[local_id], flags,
                        ))
                        finished[local_id] = True

                    if (
                        global_step == 0
                        or (global_step + 1) % script_args.progress_interval_steps == 0
                    ):
                        print(
                            '[fixed-cohort] batch {}/{} step {}/{}: '
                            'active_envs={}'.format(
                                batch_number, len(batches), global_step + 1,
                                script_args.max_steps_per_episode,
                                int((~finished).sum()),
                            ),
                            flush=True,
                        )

                if not bool(finished.all()):
                    raise RuntimeError(
                        'fixed-cohort batch did not produce one terminal outcome '
                        'per scenario within max_steps_per_episode'
                    )
                print(
                    '[fixed-cohort] batch {}/{} complete: total_episodes={}'.format(
                        batch_number, len(batches), len(episodes)
                    ),
                    flush=True,
                )

        episodes.sort(key=lambda row: int(row['scenario_id']))
        by_id, bank_hash = validate_fixed_cohort_rows(episodes)
        if set(by_id) != set(selected_ids):
            raise RuntimeError(
                'fixed-cohort evaluation did not cover the selected scenarios'
            )
        output_dir = script_args.output_dir
        if output_dir is None:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = os.path.join(
                LEGGED_GYM_ROOT_DIR, 'logs', TASK_NAME,
                'fixed_cohort', stamp + '_' + script_args.safety_mode,
            )
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        _write_csv(os.path.join(output_dir, 'episodes.csv'), episodes, EPISODE_FIELDS)
        _write_csv(
            os.path.join(output_dir, 'timeseries.csv'),
            timeseries, TIMESERIES_FIELDS,
        )
        _write_csv(
            os.path.join(output_dir, 'collision_traces.csv'),
            collision_traces, TIMESERIES_FIELDS,
        )
        outcomes = outcome_counts(episodes)
        total_samples = sum(int(row['episode_length']) for row in episodes)
        summary = {
            'task': TASK_NAME,
            'policy_path': policy_path,
            'safety_mode': script_args.safety_mode,
            'controller_noise_enabled': bool(
                getattr(env_cfg.controller, 'add_noise', False)
            ),
            'controller_noise_level': float(
                getattr(env_cfg.controller, 'noise_level', 0.0)
            ),
            'estimator_checkpoint': script_args.estimator_checkpoint,
            'calibration_delta': 0.0,
            'scenario_bank': os.path.abspath(os.path.expanduser(script_args.scenario_bank)),
            'scenario_bank_hash': bank_hash,
            'scenario_bank_generation_seed': generation_seed,
            'num_scenarios': len(selected_ids),
            'scenario_ids': selected_ids,
            'scenario_id_range': [min(selected_ids), max(selected_ids)],
            'num_envs': num_envs,
            'one_episode_per_scenario': True,
            'first_completion_bias': False,
            'control_dt_s': dt,
            'safe_success_rate': outcomes['rates']['safe_success'],
            'success_rate': outcomes['rates']['safe_success'],
            'goal_reached_rate': _mean(episodes, 'goal_reached'),
            'collision_rate': outcomes['rates']['collision_failure'],
            'stuck_rate': outcomes['rates']['stuck_failure'],
            'timeout_rate': outcomes['rates']['timeout_failure'],
            'other_failure_rate': outcomes['rates']['other_failure'],
            'terminal_outcome_counts': outcomes['counts'],
            'terminal_outcome_rates': outcomes['rates'],
            'terminal_outcome_rate_sum': sum(outcomes['rates'].values()),
            'mean_episode_length': _mean(episodes, 'episode_length'),
            'mean_episode_duration_s': _mean(episodes, 'duration_s'),
            'mean_speed_mps': _mean(episodes, 'mean_speed_mps'),
            'intervention_frequency': sum(
                float(row['intervention_frequency']) * int(row['episode_length'])
                for row in episodes
            ) / max(total_samples, 1),
            'mean_intervention_norm': sum(
                float(row['mean_intervention_norm']) * int(row['episode_length'])
                for row in episodes
            ) / max(total_samples, 1),
            'max_intervention_norm': max(
                float(row['max_intervention_norm']) for row in episodes
            ),
            'residual_negative_probability': sum(
                float(row['residual_negative_probability']) * int(row['episode_length'])
                for row in episodes
            ) / max(total_samples, 1),
            'mean_safety_drift': sum(
                float(row['mean_safety_drift']) * int(row['episode_length'])
                for row in episodes
            ) / max(total_samples, 1),
            'negative_drift_rate': sum(
                float(row['negative_drift_rate']) * int(row['episode_length'])
                for row in episodes
            ) / max(total_samples, 1),
            'drift_induced_intervention_rate': sum(
                float(row['drift_induced_intervention_rate']) * int(row['episode_length'])
                for row in episodes
            ) / max(total_samples, 1),
            'difficulty_stratified': _difficulty_report(episodes),
            'diagnostic_scenarios': diagnostic_ids,
            'artifacts': {
                'episodes_csv': 'episodes.csv',
                'timeseries_csv': 'timeseries.csv',
                'collision_traces_csv': 'collision_traces.csv',
                'evaluation_manifest': 'evaluation_manifest.json',
            },
        }
        with open(os.path.join(output_dir, 'summary.json'), 'w') as handle:
            json.dump(summary, handle, indent=2)
        manifest = build_evaluation_manifest(
            args=args, env=env, env_cfg=env_cfg, sim_params=env.sim_params,
            policy_path=policy_path,
            estimator_checkpoint=script_args.estimator_checkpoint,
            bank_path=script_args.scenario_bank, bank_metadata=metadata,
            scenario_ids=selected_ids, num_envs=num_envs,
            max_steps=script_args.max_steps_per_episode, control_dt=dt,
            safety_mode=script_args.safety_mode,
            deterministic_settings=deterministic_settings,
            headless=getattr(args, 'headless', True), cbf_layer=layer,
            task_name=TASK_NAME, evaluation_kind=script_args.evaluation_kind,
            evaluator_script=__file__,
        )
        manifest.update({
            'controller_noise_enabled': bool(
                getattr(env_cfg.controller, 'add_noise', False)
            ),
            'controller_noise_level': float(
                getattr(env_cfg.controller, 'noise_level', 0.0)
            ),
            'controller_noise_protocol': script_args.controller_noise,
            'observation_noise_enabled': bool(
                getattr(env_cfg.noise, 'add_noise', False)
            ),
            'friction_randomization_enabled': bool(
                getattr(env_cfg.domain_rand, 'randomize_friction', False)
            ),
            'mass_randomization_enabled': bool(
                getattr(env_cfg.domain_rand, 'randomize_base_mass', False)
            ),
            'push_enabled': bool(
                getattr(env_cfg.domain_rand, 'push_robots', False)
            ),
            'domain_randomization_enabled': any((
                bool(getattr(env_cfg.domain_rand, 'randomize_friction', False)),
                bool(getattr(env_cfg.domain_rand, 'randomize_base_mass', False)),
                bool(getattr(env_cfg.domain_rand, 'push_robots', False)),
            )),
        })
        with open(
            os.path.join(output_dir, 'evaluation_manifest.json'), 'w'
        ) as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    script_args = _parse_script_args()
    isaac_args = get_args()
    evaluate(isaac_args, script_args)
