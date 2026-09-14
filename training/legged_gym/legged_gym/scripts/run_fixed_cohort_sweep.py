"""Run the fixed-cohort evaluator at selected policy checkpoints."""

import argparse
import csv
import json
import os
import subprocess
import sys

from rsl_rl.utils.phase3 import CANONICAL_CHECKPOINTS

DEFAULT_CHECKPOINTS = CANONICAL_CHECKPOINTS


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario_bank', required=True)
    parser.add_argument('--safety_mode', required=True)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--estimator_checkpoint', default='')
    parser.add_argument('--checkpoints', nargs='+', type=int, default=list(DEFAULT_CHECKPOINTS))
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--max_steps_per_episode', type=int, default=3000)
    parser.add_argument('--monitor_envs', type=int, default=8)
    parser.add_argument('--scenario_ids', default=None)
    parser.add_argument('--diagnostic_scenario_ids', default=None)
    return parser.parse_args()


def _write_csv(path, rows):
    fields = [
        'mode', 'checkpoint', 'policy_path', 'safe_success_rate', 'collision_rate',
        'stuck_rate', 'timeout_rate', 'other_failure_rate',
        'mean_episode_length', 'mean_episode_duration_s', 'mean_speed_mps',
        'intervention_frequency', 'mean_intervention_norm',
        'max_intervention_norm', 'residual_negative_probability',
        'mean_safety_drift', 'negative_drift_rate',
        'drift_induced_intervention_rate',
    ]
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(script_args):
    if script_args.safety_mode == 'predictive' and not script_args.estimator_checkpoint:
        raise ValueError('--estimator_checkpoint is required for predictive mode')
    output_dir = os.path.abspath(os.path.expanduser(script_args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    evaluator = os.path.join(os.path.dirname(__file__), 'evaluate_fixed_cohort.py')
    rows = []
    for checkpoint in script_args.checkpoints:
        checkpoint = int(checkpoint)
        policy_path = os.path.abspath(os.path.join(
            os.path.expanduser(script_args.run_dir),
            'model_{}.pt'.format(checkpoint),
        ))
        if not os.path.isfile(policy_path):
            raise FileNotFoundError('policy checkpoint not found: {}'.format(policy_path))
        checkpoint_dir = os.path.join(output_dir, 'checkpoint_{}'.format(checkpoint))
        command = [
            sys.executable, evaluator,
            '--policy_path', policy_path,
            '--scenario_bank', os.path.abspath(os.path.expanduser(script_args.scenario_bank)),
            '--safety_mode', script_args.safety_mode,
            '--num_envs', str(int(script_args.num_envs)),
            '--max_steps_per_episode', str(int(script_args.max_steps_per_episode)),
            '--monitor_envs', str(int(script_args.monitor_envs)),
            '--evaluation_kind', 'checkpoint_sweep',
            '--output_dir', checkpoint_dir,
            '--headless',
            '--sim_device', 'cuda:0',
            '--rl_device', 'cuda:0',
        ]
        if script_args.scenario_ids is not None:
            command.extend(['--scenario_ids', script_args.scenario_ids])
        if script_args.diagnostic_scenario_ids is not None:
            command.extend([
                '--diagnostic_scenario_ids', script_args.diagnostic_scenario_ids
            ])
        if script_args.estimator_checkpoint:
            command.extend(['--estimator_checkpoint', os.path.abspath(os.path.expanduser(script_args.estimator_checkpoint))])
        print('[fixed-cohort-sweep] running checkpoint {}'.format(checkpoint), flush=True)
        subprocess.check_call(command)
        with open(os.path.join(checkpoint_dir, 'summary.json')) as handle:
            summary = json.load(handle)
        rows.append({
            'mode': script_args.safety_mode,
            'checkpoint': checkpoint,
            'policy_path': policy_path,
            **{
                key: summary.get(key, 0.0) for key in (
                    'safe_success_rate', 'collision_rate', 'stuck_rate',
                    'timeout_rate', 'other_failure_rate',
                    'mean_episode_length', 'mean_episode_duration_s',
                    'mean_speed_mps', 'intervention_frequency',
                    'mean_intervention_norm', 'max_intervention_norm',
                    'residual_negative_probability', 'mean_safety_drift',
                    'negative_drift_rate', 'drift_induced_intervention_rate',
                )
            },
        })
    _write_csv(os.path.join(output_dir, 'checkpoint_sweep.csv'), rows)
    with open(os.path.join(output_dir, 'checkpoint_sweep.json'), 'w') as handle:
        json.dump({
            'scenario_bank': os.path.abspath(os.path.expanduser(script_args.scenario_bank)),
            'safety_mode': script_args.safety_mode,
            'checkpoints': [int(value) for value in script_args.checkpoints],
            'rows': rows,
        }, handle, indent=2)
    print(json.dumps(rows, indent=2), flush=True)
    return rows


if __name__ == '__main__':
    run_sweep(_parse_args())
