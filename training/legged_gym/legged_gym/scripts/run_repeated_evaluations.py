"""Run the Phase-3 repeated fixed-cohort protocol serially on CUDA:0."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time


METHOD_NAMES = {
    'A': 'A_original',
    'B': 'B_synchronized_static',
    'C': 'C_predictive',
}
REPEATS = (1, 2, 3)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario_bank', required=True)
    parser.add_argument('--estimator_checkpoint', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument(
        '--model', action='append', nargs=3, metavar=('SEED', 'METHOD', 'PATH'),
        required=True, help='Repeat once per seed/method: --model 1 A checkpoint.pt',
    )
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--max_steps_per_episode', type=int, default=3000)
    parser.add_argument('--monitor_envs', type=int, default=0)
    parser.add_argument('--progress_interval_steps', type=int, default=250)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def _complete(path):
    return all(os.path.isfile(os.path.join(path, name)) for name in (
        'episodes.csv', 'summary.json', 'evaluation_manifest.json',
    ))


def run_repeated_evaluations(args):
    output_root = os.path.abspath(os.path.expanduser(args.output_root))
    scenario_bank = os.path.abspath(os.path.expanduser(args.scenario_bank))
    estimator = os.path.abspath(os.path.expanduser(args.estimator_checkpoint))
    evaluator = os.path.join(os.path.dirname(__file__), 'evaluate_fixed_cohort.py')
    models = {}
    for seed_text, method, path in args.model:
        seed = int(seed_text)
        if method not in METHOD_NAMES:
            raise ValueError('method must be A, B, or C: {}'.format(method))
        key = (seed, method)
        if key in models:
            raise ValueError('duplicate model specification: {}'.format(key))
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            raise FileNotFoundError('model checkpoint not found: {}'.format(path))
        models[key] = path
    expected = {(seed, method) for seed in (1, 2, 3) for method in METHOD_NAMES}
    if set(models) != expected:
        raise ValueError('must specify exactly seed1/2/3 for methods A/B/C')

    os.makedirs(output_root, exist_ok=True)
    started = time.time()
    started_at = datetime.datetime.now().isoformat()
    records = []
    # This runner is deliberately serial: concurrent Isaac Gym/PhysX jobs can
    # add an avoidable source of GPU contention to the repeated measurement.
    for seed in (1, 2, 3):
        for method in ('A', 'B', 'C'):
            for repeat in REPEATS:
                output_dir = os.path.join(
                    output_root, 'seed{}'.format(seed), method,
                    'repeat{}'.format(repeat),
                )
                if _complete(output_dir) and not args.force:
                    records.append({
                        'seed': seed, 'method': method, 'repeat': repeat,
                        'output_dir': output_dir, 'status': 'reused',
                    })
                    print('[repeated-eval] reusing {}'.format(output_dir), flush=True)
                    continue
                command = [
                    sys.executable, evaluator,
                    '--policy_path', models[(seed, method)],
                    '--scenario_bank', scenario_bank,
                    '--safety_mode', METHOD_NAMES[method].split('_', 1)[1],
                    '--num_envs', str(int(args.num_envs)),
                    '--max_steps_per_episode', str(int(args.max_steps_per_episode)),
                    '--monitor_envs', str(int(args.monitor_envs)),
                    '--progress_interval_steps', str(int(args.progress_interval_steps)),
                    '--evaluation_kind', 'phase3B_repeated_fixed_cohort',
                    '--output_dir', output_dir,
                    '--headless', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
                    '--estimator_checkpoint', estimator,
                ]
                # The execution environment is constrained to GPU 0 even if a
                # caller has a broader CUDA_VISIBLE_DEVICES in its shell.
                run_env = os.environ.copy()
                run_env['CUDA_VISIBLE_DEVICES'] = '0'
                begin = time.time()
                print(
                    '[repeated-eval] start seed{} method{} repeat{} -> {}'.format(
                        seed, method, repeat, output_dir
                    ), flush=True,
                )
                subprocess.check_call(command, env=run_env)
                elapsed = time.time() - begin
                records.append({
                    'seed': seed, 'method': method, 'repeat': repeat,
                    'output_dir': output_dir, 'status': 'completed',
                    'elapsed_seconds': elapsed,
                })
                print(
                    '[repeated-eval] complete seed{} method{} repeat{} elapsed={:.1f}s'.format(
                        seed, method, repeat, elapsed
                    ), flush=True,
                )
    execution = {
        'started_at': started_at,
        'finished_at': datetime.datetime.now().isoformat(),
        'elapsed_seconds': time.time() - started,
        'cuda_visible_devices': '0',
        'sim_device': 'cuda:0', 'rl_device': 'cuda:0',
        'serial': True, 'records': records,
        'scenario_bank': scenario_bank,
        'estimator_checkpoint': estimator,
        'num_envs': int(args.num_envs),
        'max_steps_per_episode': int(args.max_steps_per_episode),
        'monitor_envs': int(args.monitor_envs),
    }
    with open(os.path.join(output_root, 'execution_log.json'), 'w') as handle:
        json.dump(execution, handle, indent=2, sort_keys=True)
    print(json.dumps(execution, indent=2, sort_keys=True), flush=True)
    return execution


if __name__ == '__main__':
    run_repeated_evaluations(_parse_args())
