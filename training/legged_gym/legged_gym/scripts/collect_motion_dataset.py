"""Collect synchronized motion-estimation supervision from a frozen policy.

The collector stores raw synchronized ray history and ego-motion history on
10 Hz exteroception frames.  It never modifies the policy observation layout
and it disables collision replay only in its in-memory environment config.

Example::

    CUDA_VISIBLE_DEVICES=0 python collect_motion_dataset.py \
        --policy_path logs/Go2_pos_rough/<run>/model_2000.pt \
        --num_envs 512 --max_samples 200000 --output_dir /tmp/motion_dataset_v2 \
        --headless --sim_device cuda:0 --rl_device cuda:0
"""

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

from isaacgym import gymapi  # noqa: F401 - initialize Isaac Gym before torch
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - register tasks
from legged_gym.utils import get_args, get_load_path, task_registry


TASK_NAME = 'go2_pos_dynamic'


def _parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--policy_path', type=str, default=None)
    parser.add_argument('--num_envs', type=int, default=512)
    parser.add_argument('--max_samples', type=int, default=200000)
    parser.add_argument('--max_policy_steps', type=int, default=1000000)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--shard_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument(
        '--include_warmup', action='store_true',
        help='Include frames before history_len samples are available.',
    )
    script_args, remaining_args = parser.parse_known_args()
    # Isaac Gym parses simulator/device options after the collector-specific
    # arguments have been removed.
    sys.argv = [sys.argv[0]] + remaining_args
    return script_args


def _latest_rough_policy():
    policy_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'Go2_pos_rough')
    return get_load_path(policy_root, load_run=-1, checkpoint=-1)


def _git_sha():
    repo_root = os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, '..', '..'))
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_root,
            check=False, capture_output=True, text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _configure_env(env_cfg, script_args):
    if script_args.num_envs < 1:
        raise ValueError('--num_envs must be positive')
    if script_args.max_samples < 1:
        raise ValueError('--max_samples must be positive')
    if script_args.max_policy_steps < 1:
        raise ValueError('--max_policy_steps must be positive')
    if script_args.shard_size < 1:
        raise ValueError('--shard_size must be positive')

    env_cfg.env.num_envs = script_args.num_envs
    env_cfg.seed = script_args.seed
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


def _yaw_from_quat(quat):
    sin_yaw = 2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1])
    cos_yaw = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    return torch.atan2(sin_yaw, cos_yaw)


def _source_switch_type(current_source_id, future_source_id):
    """Encode source transitions: 0 none, 1 S->D, 2 D->S, 3 D->D."""
    current_dynamic = current_source_id >= 0
    future_dynamic = future_source_id >= 0
    switch_type = torch.zeros_like(current_source_id, dtype=torch.long)
    switch_type[(~current_dynamic) & future_dynamic] = 1
    switch_type[current_dynamic & (~future_dynamic)] = 2
    switch_type[current_dynamic & future_dynamic & (current_source_id != future_source_id)] = 3
    return switch_type


def _validate_batch(env, indices):
    history_len = int(env.cfg.env.his_len)
    num_rays = int(env.ray_angles.numel())
    rays_hist = env.rays_hist[indices]
    motion_hist = env.motion_ego_hist[indices]
    closing = env.closing_rate_gt[indices]
    expected = (len(indices), history_len, num_rays)
    if tuple(rays_hist.shape) != expected:
        raise RuntimeError('rays_hist shape mismatch: {} != {}'.format(
            tuple(rays_hist.shape), expected,
        ))
    expected_motion = (len(indices), history_len, 3)
    if tuple(motion_hist.shape) != expected_motion:
        raise RuntimeError('motion_ego_hist shape mismatch: {} != {}'.format(
            tuple(motion_hist.shape), expected_motion,
        ))
    expected_closing = (len(indices), num_rays)
    if tuple(closing.shape) != expected_closing:
        raise RuntimeError('closing_rate_gt shape mismatch: {} != {}'.format(
            tuple(closing.shape), expected_closing,
        ))

    tensors = {
        'rays_hist': rays_hist,
        'motion_ego_hist': motion_hist,
        'closing_rate_gt': closing,
        'current_fused_rays': env.rays[indices],
        'static_rays': env.static_rays[indices],
        'dynamic_rays': env.dynamic_rays[indices],
        'future_fused_rays': env.closing_rate_gt_future_fused_rays[indices],
        'future_dynamic_rays': env.future_dynamic_rays[indices],
        'active_obstacle_radial_velocity_gt': (
            env.active_obstacle_radial_velocity_gt[indices]
        ),
    }
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise RuntimeError('non-finite tensor encountered while collecting')

    # ``ray2d.min_dist`` is enforced by the analytic dynamic-ray query, but
    # the legacy static grid-ray path samples the terrain grid directly and
    # can return a valid sub-min-distance value (for example 0.05 m). Keep
    # that raw value in the dataset instead of clipping/changing observations.
    ray_min = -1e-4
    ray_max = float(env.cfg.sensors.ray2d.max_dist) + 1e-4
    invalid_rays = (rays_hist < ray_min) | (rays_hist > ray_max)
    if invalid_rays.any():
        invalid_rows = invalid_rays.any(dim=(1, 2))
        bad_values = rays_hist[invalid_rows]
        raise RuntimeError(
            'rays_hist contains values outside sensor range: '
            'env_ids={} history_count={} min={:.6f} max={:.6f} range=[{:.6f}, {:.6f}]'.format(
                indices[invalid_rows].detach().cpu().tolist(),
                env.exteroception_history_count[indices[invalid_rows]].detach().cpu().tolist(),
                float(bad_values.min()), float(bad_values.max()),
                0.0, ray_max,
            )
        )
    if not torch.allclose(
        rays_hist[:, -1], env.rays[indices], atol=1e-5, rtol=1e-5
    ):
        raise RuntimeError('rays_hist latest frame is not the current fused ray')
    expected_ids = (len(indices), num_rays)
    for name in [
        'dynamic_active_obstacle_id',
        'future_dynamic_active_obstacle_id',
        'current_fused_source_id',
        'future_fused_source_id',
    ]:
        if tuple(getattr(env, name)[indices].shape) != expected_ids:
            raise RuntimeError('{} shape mismatch'.format(name))
    for name in [
        'source_switch_mask',
        'same_dynamic_source_mask',
        'active_obstacle_radial_velocity_valid_mask',
    ]:
        if tuple(getattr(env, name)[indices].shape) != expected_ids:
            raise RuntimeError('{} shape mismatch'.format(name))


def _cpu_payload(env, indices, episode_ids, episode_steps, global_step):
    gt = env.get_dynamic_obstacle_gt(relative_to_robot=False)
    yaw = _yaw_from_quat(env.base_quat[indices])
    current_source_id = env.current_fused_source_id[indices]
    future_source_id = env.future_fused_source_id[indices]
    source_switch_mask = env.source_switch_mask[indices]
    return {
        'rays_hist': env.rays_hist[indices].detach().cpu().float(),
        'motion_ego_hist': env.motion_ego_hist[indices].detach().cpu().float(),
        'closing_rate_gt': env.closing_rate_gt[indices].detach().cpu().float(),
        'current_fused_rays': env.rays[indices].detach().cpu().float(),
        'static_rays': env.static_rays[indices].detach().cpu().float(),
        'dynamic_rays': env.dynamic_rays[indices].detach().cpu().float(),
        'future_fused_rays': env.closing_rate_gt_future_fused_rays[indices].detach().cpu().float(),
        'future_dynamic_rays': env.future_dynamic_rays[indices].detach().cpu().float(),
        'dynamic_ray_hit_mask': env.dynamic_ray_hit_mask[indices].detach().cpu(),
        'obstacle_position_world': gt['position'][indices, :, :2].detach().cpu().float(),
        'obstacle_trajectory_position_world': (
            gt['trajectory_position'][indices, :, :2].detach().cpu().float()
        ),
        # Keep the old key as an explicitly deprecated physical diagnostic;
        # all new speed-conditioned analysis uses the trajectory field below.
        'obstacle_velocity_world': (
            gt['physical_root_velocity'][indices, :, :2].detach().cpu().float()
        ),
        'obstacle_physics_velocity_world': (
            gt['physical_root_velocity'][indices, :, :2].detach().cpu().float()
        ),
        'obstacle_trajectory_velocity_world': (
            gt['trajectory_velocity'][indices, :, :2].detach().cpu().float()
        ),
        'active_dynamic_obstacle_id': (
            env.dynamic_active_obstacle_id[indices].detach().cpu().long()
        ),
        'future_active_dynamic_obstacle_id': (
            env.future_dynamic_active_obstacle_id[indices].detach().cpu().long()
        ),
        'current_fused_source_id': current_source_id.detach().cpu().long(),
        'future_fused_source_id': future_source_id.detach().cpu().long(),
        'source_switch_mask': source_switch_mask.detach().cpu(),
        'same_dynamic_source_mask': (
            env.same_dynamic_source_mask[indices].detach().cpu()
        ),
        'source_switch_type': _source_switch_type(
            current_source_id, future_source_id
        ).detach().cpu().long(),
        'continuous_closing_rate_gt': torch.where(
            ~source_switch_mask,
            env.closing_rate_gt[indices],
            torch.zeros_like(env.closing_rate_gt[indices]),
        ).detach().cpu().float(),
        'active_obstacle_radial_velocity_gt': (
            env.active_obstacle_radial_velocity_gt[indices].detach().cpu().float()
        ),
        'active_obstacle_radial_velocity_valid_mask': (
            env.active_obstacle_radial_velocity_valid_mask[indices].detach().cpu()
        ),
        'robot_xy_world': env.root_states[indices, :2].detach().cpu().float(),
        'robot_yaw': yaw.detach().cpu().float(),
        'env_id': indices.detach().cpu().long(),
        'episode_id': episode_ids[indices].detach().cpu().long(),
        'episode_step': episode_steps[indices].detach().cpu().long(),
        'global_policy_step': torch.full(
            (len(indices),), global_step, dtype=torch.long
        ),
        'gt_horizon': torch.full(
            (len(indices),), float(env.gt_horizon), dtype=torch.float32
        ),
    }


def _append_payload(buffer, payload):
    for key, value in payload.items():
        buffer[key].append(value)


def _flush_shard(buffer, buffered_count, output_dir, shard_index):
    if buffered_count == 0:
        return None
    shard = {
        key: torch.cat(values, dim=0) for key, values in buffer.items()
    }
    path = os.path.join(output_dir, 'shard_{:05d}.pt'.format(shard_index))
    torch.save(shard, path)
    for values in buffer.values():
        values.clear()
    return {
        'path': os.path.basename(path),
        'num_samples': int(buffered_count),
        'file_size_bytes': int(os.path.getsize(path)),
    }


def _make_splits(output_dir, shard_infos, seed):
    episode_ids = set()
    for info in shard_infos:
        shard = torch.load(os.path.join(output_dir, info['path']), map_location='cpu')
        episode_ids.update(int(value) for value in torch.unique(shard['episode_id']))
    episode_ids = sorted(episode_ids)
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(len(episode_ids), generator=generator).tolist()
    shuffled = [episode_ids[index] for index in permutation]
    train_count = int(0.8 * len(shuffled))
    val_count = int(0.1 * len(shuffled))
    if len(shuffled) >= 3:
        train_count = max(1, train_count)
        val_count = max(1, val_count)
        if train_count + val_count >= len(shuffled):
            val_count = max(0, len(shuffled) - train_count - 1)
    splits = {
        'train': shuffled[:train_count],
        'val': shuffled[train_count:train_count + val_count],
        'test': shuffled[train_count + val_count:],
    }
    split_lookup = {
        episode_id: split for split, ids in splits.items() for episode_id in ids
    }
    sample_counts = {split: 0 for split in splits}
    for info in shard_infos:
        shard = torch.load(os.path.join(output_dir, info['path']), map_location='cpu')
        for episode_id in shard['episode_id'].tolist():
            sample_counts[split_lookup[int(episode_id)]] += 1
    result = {
        'seed': int(seed),
        'episode_counts': {split: len(ids) for split, ids in splits.items()},
        'sample_counts': sample_counts,
        'episode_ids': splits,
    }
    with open(os.path.join(output_dir, 'splits.json'), 'w') as file:
        json.dump(result, file, indent=2)
    return result


def collect(args, script_args):
    policy_path = script_args.policy_path or _latest_rough_policy()
    policy_path = os.path.abspath(os.path.expanduser(policy_path))
    if not os.path.isfile(policy_path):
        raise FileNotFoundError('Policy checkpoint not found: {}'.format(policy_path))

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    _configure_env(env_cfg, script_args)
    if script_args.output_dir is None:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR, 'logs', TASK_NAME, 'motion_dataset', stamp
        )
    else:
        output_dir = os.path.abspath(os.path.expanduser(script_args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    if os.path.isfile(os.path.join(output_dir, 'manifest.json')) or glob.glob(
        os.path.join(output_dir, 'shard_*.pt')
    ):
        raise FileExistsError(
            'Output directory already contains dataset files: {}'.format(output_dir)
        )

    args.task = TASK_NAME
    args.num_envs = script_args.num_envs
    args.seed = script_args.seed
    args.headless = True
    args.wandb = False

    env = None
    shard_infos = []
    total_samples = 0
    below_configured_min_count = 0
    total_ray_value_count = 0
    global_step = 0
    episode_ids = None
    episode_steps = None
    buffer = {}
    buffered_count = 0
    shard_index = 0

    try:
        env, _ = task_registry.make_env(
            name=TASK_NAME, args=args, env_cfg=env_cfg
        )
        episode_ids = torch.arange(
            script_args.num_envs, device=env.device, dtype=torch.long
        )
        episode_steps = torch.zeros(
            script_args.num_envs, device=env.device, dtype=torch.long
        )
        train_cfg.runner.resume = False
        runner, _ = task_registry.make_alg_runner(
            env=env, name=TASK_NAME, args=args,
            train_cfg=train_cfg, log_root=None,
        )
        runner.load(policy_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        obs, _ = env.reset()

        print('[collector] task={} policy={} device={} envs={}'.format(
            TASK_NAME, policy_path, env.device, script_args.num_envs,
        ), flush=True)
        print('[collector] output_dir={} max_samples={} shard_size={}'.format(
            output_dir, script_args.max_samples, script_args.shard_size,
        ), flush=True)

        with torch.no_grad():
            while (
                total_samples < script_args.max_samples
                and global_step < script_args.max_policy_steps
            ):
                actions = policy(obs.detach())
                obs, _, _, dones, _ = env.step(actions.detach())
                global_step += 1

                done_mask = dones.to(dtype=torch.bool)
                episode_ids = episode_ids + done_mask.to(torch.long) * script_args.num_envs
                episode_steps = torch.where(
                    done_mask, torch.zeros_like(episode_steps), episode_steps + 1
                )

                updated = env.exteroception_updated_mask
                history_ready = env.exteroception_history_count >= env.cfg.env.his_len
                valid = updated & (history_ready | script_args.include_warmup)
                indices = torch.nonzero(valid, as_tuple=False).flatten()
                if len(indices) == 0:
                    continue

                remaining = script_args.max_samples - total_samples
                indices = indices[:remaining]
                _validate_batch(env, indices)
                payload = _cpu_payload(
                    env, indices, episode_ids, episode_steps, global_step
                )
                configured_min = float(env.cfg.sensors.ray2d.min_dist)
                below_configured_min_count += int(
                    (env.rays_hist[indices] < configured_min).sum().item()
                )
                total_ray_value_count += int(env.rays_hist[indices].numel())
                if not buffer:
                    buffer = {key: [] for key in payload}
                start = 0
                while start < len(indices):
                    room = script_args.shard_size - buffered_count
                    take = min(room, len(indices) - start)
                    chunk = {
                        key: value[start:start + take]
                        for key, value in payload.items()
                    }
                    _append_payload(buffer, chunk)
                    buffered_count += take
                    total_samples += take
                    start += take
                    if buffered_count == script_args.shard_size:
                        info = _flush_shard(
                            buffer, buffered_count, output_dir, shard_index
                        )
                        shard_infos.append(info)
                        print('[collector] wrote {} (total={})'.format(
                            info['path'], total_samples,
                        ), flush=True)
                        shard_index += 1
                        buffered_count = 0
                if total_samples >= script_args.max_samples:
                    break

        if buffered_count:
            info = _flush_shard(
                buffer, buffered_count, output_dir, shard_index
            )
            shard_infos.append(info)
            buffered_count = 0
            print('[collector] wrote {} (total={})'.format(
                info['path'], total_samples,
            ), flush=True)
    finally:
        if env is not None:
            if env.viewer is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)

    if total_samples == 0:
        raise RuntimeError('No valid 10 Hz samples were collected')
    if total_samples < script_args.max_samples:
        raise RuntimeError(
            'Reached max_policy_steps={} with only {} of {} samples'.format(
                script_args.max_policy_steps, total_samples, script_args.max_samples
            )
        )

    history_len = int(env_cfg.env.his_len)
    num_rays = int(env_cfg.env.num_rays)
    ray_angles_deg = torch.rad2deg(env.ray_angles).detach().cpu().tolist()
    manifest = {
        'task': TASK_NAME,
        'total_samples': total_samples,
        'num_shards': len(shard_infos),
        'shards': shard_infos,
        'per_sample_shapes': {
            'rays_hist': [history_len, num_rays],
            'motion_ego_hist': [history_len, 3],
            'closing_rate_gt': [num_rays],
            'current_fused_rays': [num_rays],
            'static_rays': [num_rays],
            'dynamic_rays': [num_rays],
            'future_fused_rays': [num_rays],
            'future_dynamic_rays': [num_rays],
            'obstacle_trajectory_position_world': [env.num_dynamic_obstacles, 2],
            'obstacle_trajectory_velocity_world': [env.num_dynamic_obstacles, 2],
            'obstacle_physics_velocity_world': [env.num_dynamic_obstacles, 2],
            'active_dynamic_obstacle_id': [num_rays],
            'future_active_dynamic_obstacle_id': [num_rays],
            'current_fused_source_id': [num_rays],
            'future_fused_source_id': [num_rays],
            'source_switch_mask': [num_rays],
            'same_dynamic_source_mask': [num_rays],
            'source_switch_type': [num_rays],
            'continuous_closing_rate_gt': [num_rays],
            'active_obstacle_radial_velocity_gt': [num_rays],
            'active_obstacle_radial_velocity_valid_mask': [num_rays],
        },
        'dataset_version': 'v2',
        'trajectory_velocity_semantics': (
            'Reflected analytic velocity returned by '
            '_compute_dynamic_obstacle_states_at_time; do not use raw '
            'PhysX root-state velocity for speed-conditioned analysis.'
        ),
        'active_source_encoding': {
            'static_environment': -1,
            'dynamic_obstacle_ids': '0..M-1',
        },
        'policy_path': policy_path,
        'git_commit': _git_sha(),
        'seed': script_args.seed,
        'gt_horizon': float(env_cfg.motion_estimation.gt_horizon),
        'exteroception_frequency': float(env_cfg.env.exteroception_frequency),
        'policy_dt': float(env_cfg.control.decimation * env_cfg.sim.dt),
        'exteroception_update_interval': int(
            round(1.0 / (env_cfg.env.exteroception_frequency * (
                env_cfg.control.decimation * env_cfg.sim.dt
            )))
        ),
        'history_length': history_len,
        'ray_angles_deg': ray_angles_deg,
        'obstacle_speed_range': list(env_cfg.dynamic_obstacles.speed_range),
        'obstacle_speed_curriculum': {
            'enabled': bool(env_cfg.dynamic_obstacles.curriculum.enabled),
            'speed_start': list(env_cfg.dynamic_obstacles.curriculum.speed_start),
            'speed_steps': int(env_cfg.dynamic_obstacles.curriculum.speed_steps),
        },
        'include_warmup': bool(script_args.include_warmup),
        'ray_range_semantics': {
            'raw_dataset_range_checked': [0.0, float(env_cfg.sensors.ray2d.max_dist)],
            'configured_dynamic_ray_min_dist': float(env_cfg.sensors.ray2d.min_dist),
            'below_configured_min_count': below_configured_min_count,
            'total_ray_value_count': total_ray_value_count,
            'below_configured_min_ratio': (
                float(below_configured_min_count) / max(total_ray_value_count, 1)
            ),
            'note': (
                'Static grid rays are not clamped to min_dist; raw sub-min '
                'values are preserved.'
            ),
        },
    }
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as file:
        json.dump(manifest, file, indent=2)
    with open(os.path.join(output_dir, 'collection_config.json'), 'w') as file:
        json.dump({
            'dataset_version': 'v2',
            'policy_path': policy_path,
            'num_envs': script_args.num_envs,
            'max_samples': script_args.max_samples,
            'max_policy_steps': script_args.max_policy_steps,
            'shard_size': script_args.shard_size,
            'seed': script_args.seed,
            'headless': True,
            'include_warmup': script_args.include_warmup,
            'trajectory_velocity_semantics': (
                'analytic reflected trajectory velocity'
            ),
            'active_source_encoding': {
                'static_environment': -1,
                'dynamic_obstacle_ids': '0..M-1',
            },
        }, file, indent=2)
    splits = _make_splits(output_dir, shard_infos, script_args.seed)
    print('[collector] complete samples={} shards={} splits={}'.format(
        total_samples, len(shard_infos), splits['sample_counts'],
    ), flush=True)
    return output_dir


if __name__ == '__main__':
    script_args = _parse_script_args()
    args = get_args()
    collect(args, script_args)
