"""Generate one reproducible fixed scenario bank for Phase-3 evaluation."""

import argparse
import os
import sys

from isaacgym import gymapi  # noqa: F401 - initialize Isaac Gym first
from legged_gym.envs import *  # noqa: F401,F403 - register tasks
from legged_gym.utils import get_args, task_registry
from rsl_rl.utils.phase2 import PHASE2_SPEED_FINAL
from rsl_rl.utils.phase3 import SCENARIO_BANK_VERSION, save_scenario_bank

from phase3_common import (
    capture_scenario_tensors,
    capture_static_terrain,
    configure_evaluation_cfg,
    prepare_fresh_scenario_state,
    repository_commit,
)


TASK_NAME = 'go2_pos_dynamic'


def _parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num_scenarios', type=int, default=512)
    parser.add_argument('--generation_seed', type=int, default=20260911)
    script_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return script_args


def generate(args, script_args):
    num_scenarios = int(script_args.num_scenarios)
    if num_scenarios < 1:
        raise ValueError('--num_scenarios must be positive')

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    configure_evaluation_cfg(
        env_cfg, train_cfg,
        seed=script_args.generation_seed,
        num_envs=num_scenarios,
        safety_mode='original',
        stratified_speed=True,
    )
    args.task = TASK_NAME
    args.num_envs = num_scenarios
    args.seed = int(script_args.generation_seed)
    args.wandb = False

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    try:
        prepare_fresh_scenario_state(env)
        scenarios = capture_scenario_tensors(env)
        extras = capture_static_terrain(env)
        metadata = {
            'version': SCENARIO_BANK_VERSION,
            'generation_seed': int(script_args.generation_seed),
            'num_scenarios': num_scenarios,
            'generation_num_envs': num_scenarios,
            'task': TASK_NAME,
            'obstacle_count': int(env.num_dynamic_obstacles),
            'obstacle_speed_range': list(PHASE2_SPEED_FINAL),
            'sampling': 'stratified_analytic_vmax',
            'terrain_type': list(env.cfg.terrain.terrain_types),
            'terrain_num_rows': int(env.cfg.terrain.num_rows),
            'terrain_num_cols': int(env.cfg.terrain.num_cols),
            'terrain_mesh_type': str(env.cfg.terrain.mesh_type),
            'git_commit': repository_commit(),
        }
        bank_hash = save_scenario_bank(
            script_args.output, scenarios, metadata, extras=extras
        )
        print(
            '[scenario-bank] wrote {} scenarios to {} hash={}'.format(
                num_scenarios, os.path.abspath(script_args.output), bank_hash
            ),
            flush=True,
        )
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    script_args = _parse_script_args()
    isaac_args = get_args()
    generate(isaac_args, script_args)
