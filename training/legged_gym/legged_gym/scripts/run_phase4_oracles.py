"""Run the Phase-4 failure analysis and oracle matrix serially on CUDA:0."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time


MODES = (
    'predictive_learned', 'oracle_gt_0p1', 'oracle_gt_0p2',
    'oracle_gt_0p3', 'oracle_gt_0p5', 'oracle_gt_multi',
)
SEEDS = (1, 2)
REPEATS = (1, 2, 3)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario_bank', required=True)
    parser.add_argument('--estimator_checkpoint', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument(
        '--model', action='append', nargs=2, metavar=('SEED', 'PATH'),
        required=True,
    )
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--max_steps_per_episode', type=int, default=3000)
    parser.add_argument('--progress_interval_steps', type=int, default=250)
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def _complete(path, failure=False):
    files = ['episodes.csv', 'summary.json', 'evaluation_manifest.json']
    if failure:
        files += ['collision_windows.csv', 'collision_episode_summary.csv',
                  'failure_taxonomy_counts.csv', 'failure_analysis_summary.json']
    return all(os.path.isfile(os.path.join(path, name)) for name in files)


def _run(command, output_dir, records, label):
    started = time.time()
    print('[phase4-runner] start {} -> {}'.format(label, output_dir), flush=True)
    run_env = os.environ.copy()
    run_env['CUDA_VISIBLE_DEVICES'] = '0'
    run_env.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/sea_nav_torch_extensions')
    subprocess.check_call(command, env=run_env)
    elapsed = time.time() - started
    records.append({
        'label': label, 'output_dir': output_dir,
        'status': 'completed', 'elapsed_seconds': elapsed,
    })
    print('[phase4-runner] complete {} elapsed={:.1f}s'.format(label, elapsed), flush=True)


def main(args):
    output_root = os.path.abspath(os.path.expanduser(args.output_root))
    scenario_bank = os.path.abspath(os.path.expanduser(args.scenario_bank))
    estimator = os.path.abspath(os.path.expanduser(args.estimator_checkpoint))
    evaluator = os.path.join(os.path.dirname(__file__), 'evaluate_phase4.py')
    models = {int(seed): os.path.abspath(os.path.expanduser(path)) for seed, path in args.model}
    if set(models) != set(SEEDS):
        raise ValueError('must provide exactly --model 1 PATH and --model 2 PATH')
    for path in list(models.values()) + [scenario_bank, estimator]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    os.makedirs(output_root, exist_ok=True)
    records = []
    started = time.time()
    started_at = datetime.datetime.now().isoformat()

    for seed in SEEDS:
        failure_dir = os.path.join(output_root, 'failure_analysis', 'seed{}'.format(seed))
        command = [
            sys.executable, evaluator,
            '--policy_path', models[seed], '--scenario_bank', scenario_bank,
            '--estimator_checkpoint', estimator, '--training_seed', str(seed),
            '--repeat', '1', '--mode', 'predictive_learned',
            '--num_envs', str(args.num_envs),
            '--max_steps_per_episode', str(args.max_steps_per_episode),
            '--progress_interval_steps', str(args.progress_interval_steps),
            '--output_dir', failure_dir, '--failure_analysis', '--headless',
            '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
        ]
        if _complete(failure_dir, failure=True) and not args.force:
            records.append({'label': 'failure_seed{}'.format(seed), 'output_dir': failure_dir, 'status': 'reused'})
        else:
            _run(command, failure_dir, records, 'failure_seed{}'.format(seed))

        for mode in MODES:
            for repeat in REPEATS:
                output_dir = os.path.join(
                    output_root, 'oracle_eval', 'seed{}'.format(seed),
                    mode, 'repeat{}'.format(repeat),
                )
                command = [
                    sys.executable, evaluator,
                    '--policy_path', models[seed], '--scenario_bank', scenario_bank,
                    '--estimator_checkpoint', estimator,
                    '--training_seed', str(seed), '--repeat', str(repeat),
                    '--mode', mode, '--num_envs', str(args.num_envs),
                    '--max_steps_per_episode', str(args.max_steps_per_episode),
                    '--progress_interval_steps', str(args.progress_interval_steps),
                    '--output_dir', output_dir, '--headless',
                    '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
                ]
                label = 'seed{}_{}_repeat{}'.format(seed, mode, repeat)
                if _complete(output_dir) and not args.force:
                    records.append({'label': label, 'output_dir': output_dir, 'status': 'reused'})
                else:
                    _run(command, output_dir, records, label)

    execution = {
        'started_at': started_at,
        'finished_at': datetime.datetime.now().isoformat(),
        'elapsed_seconds': time.time() - started,
        'validated_training_seeds': list(SEEDS),
        'modes': list(MODES), 'repeats': list(REPEATS),
        'cuda_visible_devices': '0', 'sim_device': 'cuda:0', 'rl_device': 'cuda:0',
        'serial': True, 'num_envs': int(args.num_envs),
        'max_steps_per_episode': int(args.max_steps_per_episode),
        'scenario_bank': scenario_bank, 'estimator_checkpoint': estimator,
        'records': records,
    }
    with open(os.path.join(output_root, 'execution_log.json'), 'w') as handle:
        json.dump(execution, handle, indent=2, sort_keys=True)
    print(json.dumps(execution, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    main(_parse_args())
