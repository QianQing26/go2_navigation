"""Record a policy in a small dynamic-obstacle diagnostic scene.

The script intentionally overrides only the in-memory configuration.  The
training configuration of ``go2_pos_dynamic`` is not modified.  It is useful
for separating a policy/environment bug from an overly difficult six-obstacle
scene.

Example (off-screen recording on GPU 0)::

    CUDA_VISIBLE_DEVICES=0 python \
        training/legged_gym/legged_gym/scripts/record_dynamic_policy.py \
        --policy_path training/legged_gym/logs/Go2_pos_rough/<run>/model_2000.pt \
        --num_obstacles 2 --speed_range 0.2 0.5 --episodes 5 --headless

The output directory contains ``policy.mp4``, per-step ``steps.csv`` and a
``summary.json`` file.  Isaac Gym camera sensors are used instead of the
viewer, so VNC is not required.  A valid graphics device is still required;
use ``--graphics_device_id 0`` when running headlessly.
"""

import argparse
import csv
import datetime
import json
import os
import sys

from isaacgym import gymapi
import cv2
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - register tasks
from legged_gym.utils import get_args, get_load_path, task_registry


TASK_NAME = 'go2_pos_dynamic'


def _parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--policy_path', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--max_steps_per_episode', type=int, default=500)
    parser.add_argument('--num_obstacles', type=int, default=2)
    parser.add_argument(
        '--speed_range', type=float, nargs=2, default=[0.2, 0.5],
        metavar=('MIN', 'MAX'),
    )
    parser.add_argument(
        '--terrain', choices=['easy_room', 'flat'], default='easy_room'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory for the MP4 and diagnostics. A timestamped directory is used by default.',
    )
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--fps', type=int, default=50)
    script_args, remaining_args = parser.parse_known_args()
    # Let Isaac Gym parse its own arguments, including --headless and device
    # identifiers, without seeing options owned by this script.
    sys.argv = [sys.argv[0]] + remaining_args
    return script_args


def _latest_rough_policy():
    policy_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'Go2_pos_rough')
    return get_load_path(policy_root, load_run=-1, checkpoint=-1)


def _configure_scene(env_cfg, script_args):
    if script_args.num_obstacles < 1:
        raise ValueError('--num_obstacles must be positive')
    if script_args.episodes < 1:
        raise ValueError('--episodes must be positive')
    if script_args.max_steps_per_episode < 1:
        raise ValueError('--max_steps_per_episode must be positive')
    speed_min, speed_max = script_args.speed_range
    if speed_min < 0.0 or speed_max < speed_min:
        raise ValueError('--speed_range must satisfy 0 <= min <= max')

    env_cfg.env.num_envs = 1
    env_cfg.env.debug_viz = False
    env_cfg.env.enable_headless_rendering = True
    env_cfg.env.headless_camera_width = script_args.width
    env_cfg.env.headless_camera_height = script_args.height
    env_cfg.env.episode_length_s = max(
        env_cfg.env.episode_length_s,
        script_args.max_steps_per_episode * env_cfg.control.decimation * env_cfg.sim.dt,
    )
    env_cfg.dynamic_obstacles.num_obstacles = script_args.num_obstacles
    env_cfg.dynamic_obstacles.speed_range = [speed_min, speed_max]

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.replay.enable_collision_replay = False
    if hasattr(env_cfg.replay, 'enable_dynamic_obstacle_replay'):
        env_cfg.replay.enable_dynamic_obstacle_replay = False

    env_cfg.terrain.num_rows = 2 if script_args.terrain != 'flat' else 1
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrain_types = [script_args.terrain]
    env_cfg.terrain.terrain_proportions = [1.0]
    if script_args.terrain == 'flat':
        # Flat placement is handled explicitly by LeggedRobotPos.  Disable
        # room curriculum, which otherwise searches for a blocked path.
        env_cfg.terrain.curriculum = False

    env_cfg.visualization.draw_rays = False
    env_cfg.visualization.draw_position_target = False


def _camera_location(env):
    robot = env.root_states[0, :3].detach().cpu().numpy()
    camera_position = gymapi.Vec3(
        float(robot[0] + 5.0),
        float(robot[1] - 5.0),
        float(max(robot[2] + 4.0, 4.0)),
    )
    target = gymapi.Vec3(
        float(robot[0]), float(robot[1]), float(robot[2] + 0.3)
    )
    return camera_position, target


def _capture_frame(env, camera_handle, camera_props, writer, overlay):
    # BaseTask.render() renders sensors, but headless mode does not call
    # step_graphics().  Explicitly do both here for an off-screen camera.
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    image = env.gym.get_camera_image(
        env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR
    )
    if image is None:
        raise RuntimeError('Isaac Gym returned no camera image')
    image = np.asarray(image).reshape(
        (camera_props.height, camera_props.width, -1)
    )
    image = image[:, :, :3].copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.putText(
        image, overlay, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    writer.write(image)


def _diagnostic_values(env):
    obstacle_delta = (
        env.dynamic_obstacle_states[0, :, :2] - env.root_states[0, :2]
    )
    min_obstacle_distance = torch.linalg.vector_norm(
        obstacle_delta, dim=-1
    ).min()
    robot_force = torch.linalg.vector_norm(
        env.contact_forces[:, :env.num_bodies, :2], dim=-1
    ).max()
    obstacle_force = torch.linalg.vector_norm(
        env.contact_forces[:, env.num_bodies:, :2], dim=-1
    ).max()
    hard_force = robot_force > 50.0
    return {
        'min_obstacle_distance': float(min_obstacle_distance.detach().cpu()),
        'max_robot_contact_force': float(robot_force.detach().cpu()),
        'max_obstacle_contact_force': float(obstacle_force.detach().cpu()),
        'terminate': bool(env.terminate_buf[0].detach().cpu()),
        'hard_force_reset': bool(hard_force.detach().cpu()),
        'timeout': bool(env.time_out_buf[0].detach().cpu()),
        'fall_down': bool(env.fall_down[0].detach().cpu()),
        'goal_reached': bool(env.goal_reached_flag[0].detach().cpu()),
        'stand_still': bool(env.stand_still_flag[0].detach().cpu()),
        'collision_occurred': bool(env.collision_occurred[0].detach().cpu()),
    }


def record(args, script_args):
    policy_path = script_args.policy_path or _latest_rough_policy()
    policy_path = os.path.abspath(os.path.expanduser(policy_path))
    if not os.path.isfile(policy_path):
        raise FileNotFoundError('Policy checkpoint not found: {}'.format(policy_path))

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    _configure_scene(env_cfg, script_args)

    # The camera sensor requires a graphics device even though no viewer is
    # created.  Use the compute device by default if Isaac Gym parsed -1.
    args.task = TASK_NAME
    args.num_envs = 1
    args.headless = True
    args.wandb = False
    if getattr(args, 'graphics_device_id', -1) < 0:
        args.graphics_device_id = args.compute_device_id
        print(
            '[record] graphics_device_id was -1; using compute device {}'.format(
                args.graphics_device_id
            ),
            flush=True,
        )

    env, _ = task_registry.make_env(
        name=TASK_NAME, args=args, env_cfg=env_cfg
    )
    env.do_reset = False

    train_cfg.runner.resume = False
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=TASK_NAME,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )
    runner.load(policy_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    if script_args.output_dir is None:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR, 'logs', TASK_NAME, 'diagnostics', stamp
        )
    else:
        output_dir = os.path.abspath(os.path.expanduser(script_args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, 'policy.mp4')
    csv_path = os.path.join(output_dir, 'steps.csv')
    summary_path = os.path.join(output_dir, 'summary.json')

    camera_props = env.camera_properties
    camera_handle = env.camera_handles[0]
    camera_position, camera_target = _camera_location(env)
    env.gym.set_camera_location(
        camera_handle, env.envs[0], camera_position, camera_target
    )

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        script_args.fps,
        (camera_props.width, camera_props.height),
    )
    if not writer.isOpened():
        raise RuntimeError('Unable to open video writer: {}'.format(video_path))

    fieldnames = [
        'global_step', 'episode', 'episode_step', 'reward', 'episode_reward',
        'done', 'terminate', 'timeout', 'fall_down',
        'hard_force_reset', 'goal_reached', 'stand_still',
        'collision_occurred',
        'min_obstacle_distance', 'max_robot_contact_force',
        'max_obstacle_contact_force',
    ]
    completed_episodes = []
    global_step = 0
    episode = 1
    episode_step = 0
    episode_reward = 0.0
    obs, _ = env.reset()

    print('[record] task={} terrain={} obstacles={} speed_range={}'.format(
        TASK_NAME, script_args.terrain, script_args.num_obstacles,
        script_args.speed_range,
    ), flush=True)
    print('[record] policy={}'.format(policy_path), flush=True)
    print('[record] video={}'.format(video_path), flush=True)

    try:
        with open(csv_path, 'w', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()
            with torch.no_grad():
                while len(completed_episodes) < script_args.episodes:
                    actions = policy(obs.detach())
                    obs, _, rewards, dones, _ = env.step(actions.detach())
                    global_step += 1
                    episode_step += 1
                    reward = float(rewards[0].detach().cpu())
                    episode_reward += reward
                    diagnostics = _diagnostic_values(env)
                    done = bool(dones[0].detach().cpu())
                    forced_timeout = episode_step >= script_args.max_steps_per_episode
                    if forced_timeout and not done:
                        diagnostics['timeout'] = True
                    done = done or forced_timeout

                    camera_position, camera_target = _camera_location(env)
                    env.gym.set_camera_location(
                        camera_handle, env.envs[0],
                        camera_position, camera_target,
                    )
                    overlay = (
                        'episode {}/{} step {} reward {:.3f} min_dist {:.2f} '
                        'term {} fall {}'.format(
                            episode, script_args.episodes, episode_step,
                            reward, diagnostics['min_obstacle_distance'],
                            int(diagnostics['terminate']),
                            int(diagnostics['fall_down']),
                        )
                    )
                    _capture_frame(
                        env, camera_handle, camera_props, writer, overlay
                    )

                    row = {
                        'global_step': global_step,
                        'episode': episode,
                        'episode_step': episode_step,
                        'reward': reward,
                        'episode_reward': episode_reward,
                        'done': int(done),
                        **diagnostics,
                    }
                    csv_writer.writerow(row)
                    csv_file.flush()

                    if done:
                        cause = []
                        if diagnostics['terminate']:
                            cause.append('contact')
                        if diagnostics['hard_force_reset']:
                            cause.append('hard_force')
                        if diagnostics['fall_down']:
                            cause.append('fall_down')
                        if diagnostics['timeout']:
                            cause.append('timeout')
                        if diagnostics['goal_reached']:
                            cause.append('goal_reached')
                        if diagnostics['stand_still']:
                            cause.append('stand_still')
                        if not cause:
                            cause.append('reset')
                        completed_episodes.append({
                            'episode': episode,
                            'length': episode_step,
                            'reward': episode_reward,
                            'cause': cause,
                        })
                        print(
                            '[record] episode={} length={} reward={:.4f} cause={}'.format(
                                episode, episode_step, episode_reward, ','.join(cause)
                            ),
                            flush=True,
                        )
                        if len(completed_episodes) >= script_args.episodes:
                            break
                        episode += 1
                        episode_step = 0
                        episode_reward = 0.0
                        obs, _ = env.reset()
    finally:
        writer.release()
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)

    summary = {
        'task': TASK_NAME,
        'policy_path': policy_path,
        'terrain': script_args.terrain,
        'num_obstacles': script_args.num_obstacles,
        'speed_range': list(script_args.speed_range),
        'episodes_requested': script_args.episodes,
        'episodes_completed': completed_episodes,
        'video_path': video_path,
        'steps_csv': csv_path,
    }
    with open(summary_path, 'w') as summary_file:
        json.dump(summary, summary_file, indent=2)
    print('[record] summary={}'.format(summary_path), flush=True)


if __name__ == '__main__':
    script_args = _parse_script_args()
    args = get_args()
    record(args, script_args)
