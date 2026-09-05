"""Visualize the first-stage dynamic-obstacle Go2 environment.

This script is intentionally separate from ``play.py``.  It keeps the
dynamic-obstacle task configuration, creates one visible environment, and
optionally drives the robot with a policy trained on ``go2_pos_rough``.
"""

import argparse
import os
import sys

from isaacgym import gymapi, gymtorch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - registers all tasks
from legged_gym.envs.go2.go2_pos_dynamic_config import (
    Go2PosDynamicCfg,
    Go2PosDynamicCfgPPO,
)
from legged_gym.utils import get_args, get_load_path, task_registry
from legged_gym.utils.helpers import class_to_dict, parse_sim_params, set_seed
from legged_gym.utils.terrain import Terrain
import torch


def _parse_visual_arguments():
    """Parse script-only arguments and leave Isaac Gym arguments untouched."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--policy_path',
        type=str,
        default=None,
        help='Explicit PPO checkpoint. Defaults to the latest go2_pos_rough checkpoint.',
    )
    parser.add_argument(
        '--no_policy',
        action='store_true',
        help='Show only terrain and dynamic obstacles; do not create a Go2 actor.',
    )
    parser.add_argument(
        '--visual_steps',
        type=int,
        default=100000,
        help='Maximum number of viewer steps. Close the viewer to stop earlier.',
    )
    parser.add_argument(
        '--no_follow_camera',
        action='store_true',
        help='Keep the initial camera position instead of following the robot.',
    )
    visual_args, remaining_args = parser.parse_known_args()
    # gymutil.parse_arguments() will parse the Isaac Gym arguments below.  It
    # must not see the script-only arguments, otherwise it reports them as
    # unknown options.
    sys.argv = [sys.argv[0]] + remaining_args
    return visual_args


def _latest_rough_policy():
    policy_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, 'logs', 'Go2_pos_rough'
    )
    return get_load_path(policy_root, load_run=-1, checkpoint=-1)


def _set_follow_camera(env):
    """Place an isometric camera behind the first robot."""

    robot_pos = env.root_states[0, :3].detach().cpu().tolist()
    camera_pos = gymapi.Vec3(
        robot_pos[0] + 5.0,
        robot_pos[1] - 5.0,
        max(robot_pos[2] + 3.5, 3.5),
    )
    camera_target = gymapi.Vec3(
        robot_pos[0], robot_pos[1], robot_pos[2] + 0.3
    )
    env.gym.viewer_camera_look_at(
        env.viewer, None, camera_pos, camera_target
    )


def _prepare_visual_config(env_cfg):
    """Make the scene small and deterministic enough for interactive viewing."""

    env_cfg.env.num_envs = 1
    env_cfg.env.debug_viz = True
    env_cfg.env.episode_length_s = 60
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.replay.enable_collision_replay = False

    # The dynamic task samples terrain levels 0/1 during origin creation, so
    # retain two terrain rows while reducing the scene to one terrain type.
    env_cfg.terrain.num_rows = 2
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.terrain_types = ['easy_room']
    env_cfg.terrain.terrain_proportions = [1.0]

    env_cfg.visualization.draw_rays = True
    env_cfg.visualization.draw_position_target = True
    env_cfg.visualization.draw_collision_points = False


def _visualize_scene_only(args, visual_steps):
    """Visualize terrain and moving boxes without creating a Go2 actor."""

    # Do not instantiate DynamicObstacleGo2Pos here.  This path intentionally
    # avoids loading the Go2 asset, low-level controller, and PPO policy.
    args.headless = False
    args.wandb = False
    cfg = Go2PosDynamicCfg()
    cfg.terrain.num_rows = 1
    cfg.terrain.num_cols = 1
    cfg.terrain.terrain_types = ['easy_room']
    cfg.terrain.terrain_proportions = [1.0]
    cfg.terrain.border_size = 1.0
    set_seed(Go2PosDynamicCfgPPO.seed)

    sim_params = parse_sim_params(args, {'sim': class_to_dict(cfg.sim)})
    gym = gymapi.acquire_gym()
    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        args.physics_engine,
        sim_params,
    )
    if sim is None:
        raise RuntimeError('Failed to create Isaac Gym simulation')

    viewer = None
    try:
        terrain = Terrain(cfg.terrain, 1)
        mesh_params = gymapi.TriangleMeshParams()
        mesh_params.nb_vertices = terrain.vertices.shape[0]
        mesh_params.nb_triangles = terrain.triangles.shape[0]
        mesh_params.transform.p.x = -cfg.terrain.border_size
        mesh_params.transform.p.y = -cfg.terrain.border_size
        mesh_params.transform.p.z = 0.0
        mesh_params.static_friction = cfg.terrain.static_friction
        mesh_params.dynamic_friction = cfg.terrain.dynamic_friction
        mesh_params.restitution = cfg.terrain.restitution
        gym.add_triangle_mesh(
            sim,
            terrain.vertices.flatten(order='C'),
            terrain.triangles.flatten(order='C'),
            mesh_params,
        )

        env = gym.create_env(
            sim,
            gymapi.Vec3(0.0, 0.0, 0.0),
            gymapi.Vec3(0.0, 0.0, 0.0),
            1,
        )
        obstacle_cfg = cfg.dynamic_obstacles
        size = [float(value) for value in obstacle_cfg.size]
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.density = float(obstacle_cfg.density)
        obstacle_asset = gym.create_box(
            sim, size[0], size[1], size[2], asset_options
        )

        origin_x, origin_y, origin_z = [
            float(value) for value in terrain.env_origins[0, 0]
        ]
        starts_relative = [
            [-2.0, -1.2],
            [2.0, 1.2],
            [-1.2, 1.4],
            [1.2, -1.4],
            [-2.0, 1.4],
            [2.0, -1.4],
        ]
        # Deliberately use diagonal trajectories as well as the horizontal and
        # vertical ones, so the visualizer exercises the same 2-D motion model
        # used by the dynamic task.
        velocities = [
            [1.60, 0.90],
            [-1.20, 1.20],
            [1.00, -1.40],
            [-1.60, -0.80],
            [0.70, 1.60],
            [-1.40, 0.90],
        ]
        starts = [
            [origin_x + position[0], origin_y + position[1]]
            for position in starts_relative
        ]
        colors = [
            gymapi.Vec3(0.95, 0.20, 0.05),
            gymapi.Vec3(0.05, 0.50, 0.95),
            gymapi.Vec3(0.70, 0.10, 0.95),
            gymapi.Vec3(0.10, 0.85, 0.30),
        ]
        obstacle_handles = []
        actor_indices = []
        for obstacle_id in range(int(obstacle_cfg.num_obstacles)):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(
                starts[obstacle_id % len(starts)][0],
                starts[obstacle_id % len(starts)][1],
                origin_z + size[2] * 0.5,
            )
            handle = gym.create_actor(
                env,
                obstacle_asset,
                pose,
                'dynamic_obstacle_{}'.format(obstacle_id),
                0,
                0,
                0,
            )
            body_props = gym.get_actor_rigid_body_properties(env, handle)
            for body_prop in body_props:
                body_prop.mass = float(obstacle_cfg.mass)
            gym.set_actor_rigid_body_properties(
                env, handle, body_props, recomputeInertia=True
            )
            gym.set_rigid_body_color(
                env, handle, 0, gymapi.MESH_VISUAL,
                colors[obstacle_id % len(colors)]
            )
            obstacle_handles.append(handle)
            actor_indices.append(gym.get_actor_index(env, handle, gymapi.DOMAIN_SIM))

        gym.prepare_sim(sim)
        root_states = gymtorch.wrap_tensor(
            gym.acquire_actor_root_state_tensor(sim)
        ).view(-1, 13)
        actor_indices = torch.as_tensor(
            actor_indices, dtype=torch.int32, device=root_states.device
        )
        origin = torch.tensor(
            [origin_x, origin_y], dtype=torch.float, device=root_states.device
        )
        low = torch.tensor(
            [obstacle_cfg.bounds[0][0], obstacle_cfg.bounds[1][0]],
            dtype=torch.float, device=root_states.device
        )
        high = torch.tensor(
            [obstacle_cfg.bounds[0][1], obstacle_cfg.bounds[1][1]],
            dtype=torch.float, device=root_states.device
        )
        starts_tensor = torch.tensor(
            starts[:len(obstacle_handles)], dtype=torch.float, device=root_states.device
        ) - origin
        velocity_tensor = torch.tensor(
            velocities[:len(obstacle_handles)], dtype=torch.float, device=root_states.device
        )
        span = high - low
        orientation = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], dtype=torch.float, device=root_states.device
        )
        time = 0.0

        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        gym.viewer_camera_look_at(
            viewer,
            None,
            gymapi.Vec3(origin_x + 7.0, origin_y - 7.0, origin_z + 8.0),
            gymapi.Vec3(origin_x, origin_y, origin_z),
        )
        print('[visualize] scene-only mode: Go2 actor and policy are not loaded', flush=True)
        print('[visualize] close the Isaac Gym window to stop', flush=True)

        for _ in range(visual_steps):
            if gym.query_viewer_has_closed(viewer):
                break
            time += float(cfg.sim.dt)
            travel = starts_tensor + velocity_tensor * time
            phase = torch.remainder(travel - low, 2.0 * span)
            reflected = torch.where(phase <= span, phase, 2.0 * span - phase) + low
            direction = torch.where(phase <= span, 1.0, -1.0)
            velocity = velocity_tensor * direction

            root_states[actor_indices, 0:2] = reflected + origin
            root_states[actor_indices, 2] = origin_z + size[2] * 0.5
            root_states[actor_indices, 3:7] = orientation
            root_states[actor_indices, 7:9] = velocity
            root_states[actor_indices, 9:13] = 0.0
            gym.set_actor_root_state_tensor_indexed(
                sim,
                gymtorch.unwrap_tensor(root_states),
                gymtorch.unwrap_tensor(actor_indices),
                len(actor_indices),
            )
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        gym.destroy_sim(sim)


def visualize(args, visual_args):
    if visual_args.no_policy:
        return _visualize_scene_only(args, visual_args.visual_steps)

    task_name = 'go2_pos_dynamic'
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    _prepare_visual_config(env_cfg)

    # Viewer rendering is the purpose of this script; override the training
    # default even if --headless was supplied accidentally.
    args.headless = False
    args.test = False
    args.wandb = False

    print('[visualize] creating task={} with one environment'.format(task_name), flush=True)
    env, _ = task_registry.make_env(
        name=task_name, args=args, env_cfg=env_cfg
    )

    # Construct the same actor-critic class as the navigation task, but do not
    # create a training log directory or start a training loop.
    train_cfg.runner.resume = False
    ppo_runner, _ = task_registry.make_alg_runner(
        env=env,
        name=task_name,
        args=args,
        train_cfg=train_cfg,
        log_root=None,
    )

    policy_path = visual_args.policy_path or _latest_rough_policy()
    if not os.path.isfile(policy_path):
        raise FileNotFoundError('Policy checkpoint not found: {}'.format(policy_path))
    ppo_runner.load(policy_path, load_optimizer=False)
    policy = ppo_runner.get_inference_policy(device=env.device)
    print('[visualize] loaded policy: {}'.format(policy_path), flush=True)

    obs, _ = env.reset()
    _set_follow_camera(env)
    print('[visualize] viewer started; close the Isaac Gym window to stop', flush=True)

    try:
        with torch.no_grad():
            for step in range(visual_args.visual_steps):
                if policy is None:
                    actions = torch.zeros(
                        env.num_envs, env.num_nav_actions, device=env.device
                    )
                else:
                    actions = policy(obs.detach())

                obs, _, rewards, dones, _ = env.step(actions.detach())
                if not visual_args.no_follow_camera and step % 5 == 0:
                    _set_follow_camera(env)

                if bool(dones.any()):
                    print('[visualize] episode reset at step {}'.format(step), flush=True)
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    visual_args = _parse_visual_arguments()
    args = get_args()
    visualize(args, visual_args)
