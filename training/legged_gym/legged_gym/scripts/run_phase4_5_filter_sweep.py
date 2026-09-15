"""Run the Phase-4.5 filter-only ablation serially on CUDA:0."""

import argparse
import os
import subprocess
import sys


BETAS = ('0.5', '0.75', '1.0')


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed1_model', required=True)
    parser.add_argument('--seed2_model', required=True)
    parser.add_argument('--scenario_bank', required=True)
    parser.add_argument('--estimator_checkpoint', required=True)
    parser.add_argument('--output_root', required=True)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--max_steps_per_episode', type=int, default=3000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--progress_interval_steps', type=int, default=500)
    return parser.parse_args()


def _beta_name(beta):
    return beta.replace('.', 'p')


def main(args=None):
    args = _parse_args() if args is None else args
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    evaluator = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'evaluate_phase4_5_execution.py'
    )
    models = ((1, args.seed1_model), (2, args.seed2_model))
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env.setdefault('TORCH_EXTENSIONS_DIR', '/tmp/sea_nav_torch_extensions')
    env.setdefault('PYTHONPATH', os.pathsep.join((
        os.path.join(repo_root, 'training/legged_gym'),
        os.path.join(repo_root, 'training/rsl_rl'),
        env.get('PYTHONPATH', ''),
    )))
    for seed, model in models:
        for beta in BETAS:
            for repeat in range(1, int(args.repeats) + 1):
                output_dir = os.path.join(
                    os.path.abspath(os.path.expanduser(args.output_root)),
                    'seed{}'.format(seed), 'beta_{}'.format(_beta_name(beta)),
                    'repeat_{}'.format(repeat),
                )
                summary_path = os.path.join(output_dir, 'summary.json')
                if os.path.isfile(summary_path):
                    print('[phase4.5-sweep] skip existing {}'.format(output_dir), flush=True)
                    continue
                command = [
                    sys.executable, evaluator,
                    '--policy_path', os.path.abspath(os.path.expanduser(model)),
                    '--scenario_bank', os.path.abspath(os.path.expanduser(args.scenario_bank)),
                    '--estimator_checkpoint', os.path.abspath(os.path.expanduser(args.estimator_checkpoint)),
                    '--training_seed', str(seed), '--num_envs', str(args.num_envs),
                    '--max_steps_per_episode', str(args.max_steps_per_episode),
                    '--filter_beta', beta, '--controller_noise', 'off',
                    '--output_dir', output_dir,
                    '--progress_interval_steps', str(args.progress_interval_steps),
                    '--headless', '--sim_device', 'cuda:0', '--rl_device', 'cuda:0',
                ]
                print(
                    '[phase4.5-sweep] seed={} beta={} repeat={}/{}'.format(
                        seed, beta, repeat, args.repeats
                    ), flush=True,
                )
                subprocess.run(command, cwd=repo_root, env=env, check=True)


if __name__ == '__main__':
    main()
