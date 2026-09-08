"""Validate dynamic-obstacle rays, simulator GT, and collision handling.

Examples:
    python .../validate_dynamic_obstacles.py --mode rays
    python .../validate_dynamic_obstacles.py --mode gt --headless
    python .../validate_dynamic_obstacles.py --mode collision --headless
    python .../validate_dynamic_obstacles.py --mode closing --headless
"""

import argparse
import sys

from isaacgym import gymapi  # noqa: F401 - initialize Isaac Gym before torch

from legged_gym.envs import *  # noqa: F401,F403 - registers all tasks
from legged_gym.envs.go2.go2_pos_dynamic_config import Go2PosDynamicCfg
from legged_gym.utils import get_args, task_registry
import numpy as np
import torch


def _parse_script_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--mode', choices=['rays', 'gt', 'collision', 'spawn', 'closing', 'all'], default='all'
    )
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--print-every', type=int, default=5)
    parser.add_argument('--num-envs', type=int, default=1)
    parser.add_argument(
        '--headless', action='store_true',
        help='Disable the viewer; useful for GT and collision validation.'
    )
    script_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return script_args


def _prepare_config(num_envs=1):
    cfg = Go2PosDynamicCfg()
    cfg.seed = 1
    cfg.env.num_envs = int(num_envs)
    cfg.env.debug_viz = True
    cfg.env.episode_length_s = 60
    cfg.noise.add_noise = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_base_mass = False
    cfg.domain_rand.push_robots = False
    cfg.replay.enable_collision_replay = False
    cfg.replay.early_reset_prob_range = [0.0, 0.0]
    cfg.terrain.num_rows = 2
    cfg.terrain.num_cols = 1
    cfg.terrain.terrain_types = ['easy_room']
    cfg.terrain.terrain_proportions = [1.0]
    cfg.visualization.draw_rays = True
    cfg.visualization.draw_position_target = False
    cfg.visualization.draw_collision_points = False
    cfg.visualization.ray_groups = {
        'all_rays': [None, 'ray_green'],
    }
    return cfg


def _print_gt(env, step):
    gt = env.get_dynamic_obstacle_gt(relative_to_robot=False)
    position = gt['position'][0].detach().cpu().numpy()
    velocity = gt['velocity'][0].detach().cpu().numpy()
    size = gt['size'][0].detach().cpu().numpy()
    radius = gt['radius'][0, :, 0].detach().cpu().numpy()
    print('[gt] step={}'.format(step), flush=True)
    print('  position_world=\n{}'.format(np.array2string(position, precision=3)), flush=True)
    print('  velocity_world=\n{}'.format(np.array2string(velocity, precision=3)), flush=True)
    print('  size_xyz={} radius={}'.format(
        np.array2string(size[0], precision=3),
        np.array2string(radius, precision=3),
    ), flush=True)
    print('  actor_indices={}'.format(gt['actor_indices'][0].detach().cpu().tolist()), flush=True)


def _print_rays(env, step):
    angles = torch.rad2deg(env.ray_angles).detach().cpu().numpy()
    rays = env.rays[0].detach().cpu().numpy()
    dynamic_rays = env.dynamic_rays[0].detach().cpu().numpy()
    hit_mask = env.dynamic_ray_hit_mask[0].detach().cpu().numpy()
    dynamic_angles = np.rad2deg(np.arctan2(
        (env.dynamic_obstacle_states[0, :, 1] - env.root_states[0, 1]).detach().cpu().numpy(),
        (env.dynamic_obstacle_states[0, :, 0] - env.root_states[0, 0]).detach().cpu().numpy(),
    ))
    print('[rays] step={} angles_deg={}'.format(
        step, np.array2string(angles, precision=1)
    ), flush=True)
    print('  fused_41ray={}'.format(np.array2string(rays, precision=3)), flush=True)
    print('  dynamic_41ray={}'.format(np.array2string(dynamic_rays, precision=3)), flush=True)
    print('  dynamic_hit_angles_deg={}'.format(
        np.array2string(angles[hit_mask], precision=1)
    ), flush=True)
    print('  obstacle_center_angles_deg={}'.format(
        np.array2string(dynamic_angles, precision=1)
    ), flush=True)


def _set_static_obstacles(env, relative_xy, velocities=None):
    positions = torch.as_tensor(
        relative_xy, device=env.device, dtype=torch.float
    )
    if velocities is None:
        velocities = torch.zeros_like(positions)
    else:
        velocities = torch.as_tensor(
            velocities, device=env.device, dtype=torch.float
        )
    env.dynamic_obstacle_start[0] = positions
    env.dynamic_obstacle_velocity[0] = velocities
    env.dynamic_obstacle_time[0] = 0.0
    env._write_dynamic_obstacle_states()
    env._set_dynamic_obstacle_states_in_sim()
    env._get_rays()
    print('[setup] robot_world={} env_origin={} obstacle_world_after_write=\n{}'.format(
        np.array2string(env.root_states[0, :3].detach().cpu().numpy(), precision=3),
        np.array2string(env.env_origins[0].detach().cpu().numpy(), precision=3),
        np.array2string(
            env.dynamic_obstacle_states[0, :, :3].detach().cpu().numpy(),
            precision=3,
        ),
    ), flush=True)


def _run_rays(env, steps, print_every):
    # Obstacle 0 moves diagonally across the 41-ray sector.  The remaining
    # obstacles stay outside the close ray range to make the result readable.
    starts = [
        [1.4, -2.0], [-2.4, -2.4], [-2.4, 2.4], [2.4, 2.4],
        [2.4, -2.4], [-2.0, 2.0],
    ]
    velocities = [[0.0, 1.2]] + [[0.0, 0.0]] * (env.num_dynamic_obstacles - 1)
    _set_static_obstacles(env, starts, velocities)
    env.do_reset = False
    print('[rays] expected: obstacle 0 should move through the 41-ray sector', flush=True)
    for step in range(steps):
        if step % print_every == 0:
            _print_rays(env, step)
        env.step(torch.zeros(1, env.num_nav_actions, device=env.device))


def _run_gt(env, steps, print_every):
    env.do_reset = False
    for step in range(steps):
        if step % print_every == 0:
            _print_gt(env, step)
        env.step(torch.zeros(1, env.num_nav_actions, device=env.device))


def _run_collision(env, steps, print_every):
    # Put one box through the robot base.  The box is held there with a zero
    # velocity trajectory so the contact is deterministic.
    robot_relative = (
        env.root_states[0, :2] - env.env_origins[0, :2]
    ).detach().cpu().tolist()
    starts = [robot_relative, [-2.4, -2.4], [-2.4, 2.4], [2.4, 2.4],
              [2.4, -2.4], [-2.0, 2.0]]
    starts = starts[:env.num_dynamic_obstacles]
    _set_static_obstacles(env, starts)
    env.do_reset = False
    print('[collision] obstacle 0 forced into the robot base', flush=True)
    for step in range(steps):
        _, _, rewards, dones, _ = env.step(
            torch.zeros(1, env.num_nav_actions, device=env.device)
        )
        if step % print_every == 0:
            all_force = torch.linalg.vector_norm(env.contact_forces[0], dim=-1).max()
            penalized_force = torch.linalg.vector_norm(
                env.contact_forces[0, env.penalised_contact_indices], dim=-1
            ).max()
            termination_force = torch.linalg.vector_norm(
                env.contact_forces[0, env.termination_contact_indices], dim=-1
            ).max()
            collision_reward = (
                env._reward_collision()[0] * env.reward_scales['collision']
            )
            print(
                '[collision] step={} all_force={:.3f} penalized_force={:.3f} '
                'termination_force={:.3f} collision_reward={:.3f} '
                'terminate={} done={} total_reward={:.3f}'.format(
                    step,
                    float(all_force),
                    float(penalized_force),
                    float(termination_force),
                    float(collision_reward),
                    bool(env.terminate_buf[0]),
                    bool(dones[0]),
                    float(rewards[0]),
                ),
                flush=True,
            )


def _run_spawn(env, steps, print_every):
    """Check initial static/dynamic clearance and reset causes."""

    def room_oob():
        local_xy = (
            env.dynamic_obstacle_states[..., :2]
            - env.env_origins[:, None, :2]
        )
        below = local_xy < env.dynamic_obstacle_effective_low[:, None, :]
        above = local_xy > env.dynamic_obstacle_effective_high[:, None, :]
        return (below | above).any(dim=-1).any(dim=-1)

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(env_ids)
    env.do_reset = False
    actions = torch.zeros(env.num_envs, env.num_nav_actions, device=env.device)
    min_robot_distance = float(env.cfg.dynamic_obstacles.min_robot_distance)

    print('[spawn] checking {} freshly reset environments'.format(env.num_envs), flush=True)
    initial_gt = env.get_dynamic_obstacle_gt(relative_to_robot=True)
    initial_xy = initial_gt['position'][..., :2]
    initial_dist = torch.linalg.vector_norm(initial_xy, dim=-1).amin(dim=1)
    initial_delta = initial_xy[:, :, None, :] - initial_xy[:, None, :, :]
    initial_overlap = (
        (initial_delta[..., 0].abs() < float(initial_gt['size'][0, 0, 0]))
        & (initial_delta[..., 1].abs() < float(initial_gt['size'][0, 0, 1]))
    )
    initial_overlap &= ~torch.eye(
        env.num_dynamic_obstacles, dtype=torch.bool, device=env.device
    ).unsqueeze(0)
    print(
        '[spawn] before_physics dynamic_too_close={} obstacle_overlap={} '
        'room_oob={} min_dynamic_dist={:.3f}'.format(
            int((initial_dist < min_robot_distance).sum()),
            int(initial_overlap.any(dim=-1).any(dim=-1).sum()),
            int(room_oob().sum()),
            float(initial_dist.min()),
        ),
        flush=True,
    )
    for step in range(steps):
        env.step(actions)
        gt = env.get_dynamic_obstacle_gt(relative_to_robot=True)
        dynamic_xy = gt['position'][..., :2]
        dynamic_dist = torch.linalg.vector_norm(dynamic_xy, dim=-1).amin(dim=1)
        pair_delta = dynamic_xy[:, :, None, :] - dynamic_xy[:, None, :, :]
        pairwise_xy_overlap = (
            (pair_delta[..., 0].abs() < float(gt['size'][0, 0, 0]))
            & (pair_delta[..., 1].abs() < float(gt['size'][0, 0, 1]))
        )
        pairwise_xy_overlap &= ~torch.eye(
            env.num_dynamic_obstacles, dtype=torch.bool, device=env.device
        ).unsqueeze(0)
        obstacle_overlap = pairwise_xy_overlap.any(dim=-1).any(dim=-1)

        static_bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if hasattr(env, 'surr_row_start'):
            row_end = min(env.surr_row_end, env.len_x)
            col_start = max(env.surr_col_start, 0)
            col_end = min(env.surr_col_end, env.len_y)
            static_heights = env.measured_heights[:, env.surr_row_start:row_end, col_start:col_end]
            static_bad = torch.amax(static_heights, dim=(1, 2)) > 0.1

        all_force = torch.linalg.vector_norm(env.contact_forces[:, :, :2], dim=-1).amax(dim=1)
        robot_force = torch.linalg.vector_norm(
            env.contact_forces[:, :env.num_bodies, :2], dim=-1
        ).amax(dim=1)
        obstacle_force = torch.linalg.vector_norm(
            env.contact_forces[:, env.num_bodies:, :2], dim=-1
        ).amax(dim=1)
        dynamic_bad = dynamic_dist < min_robot_distance
        room_bad = room_oob()
        if step % print_every == 0:
            print(
                '[spawn] step={} static_bad={} dynamic_too_close={} '
                'obstacle_overlap={} room_oob={} '
                'contact_force_gt50={} robot_force_gt50={} obstacle_force_gt50={} '
                'terminate={} fall_down={} min_dynamic_dist={:.3f} '
                'max_robot_force={:.3f} max_obstacle_force={:.3f}'.format(
                    step,
                    int(static_bad.sum()),
                    int(dynamic_bad.sum()),
                    int(obstacle_overlap.sum()),
                    int(room_bad.sum()),
                    int((all_force > 50.0).sum()),
                    int((robot_force > 50.0).sum()),
                    int((obstacle_force > 50.0).sum()),
                    int(env.terminate_buf.sum()),
                    int(env.fall_down.sum()),
                    float(dynamic_dist.min()),
                    float(robot_force.max()),
                    float(obstacle_force.max()),
                ),
                flush=True,
            )


def _robot_frame_to_world(env, vector_xy):
    """Convert one robot-yaw-frame XY vector to the world frame."""
    quat = env.base_quat[0]
    sin_yaw = 2.0 * (quat[3] * quat[2] + quat[0] * quat[1])
    cos_yaw = 1.0 - 2.0 * (quat[1].square() + quat[2].square())
    vector_xy = torch.as_tensor(vector_xy, device=env.device, dtype=torch.float)
    return torch.stack((
        vector_xy[0] * cos_yaw - vector_xy[1] * sin_yaw,
        vector_xy[0] * sin_yaw + vector_xy[1] * cos_yaw,
    ))


def _world_to_robot_frame(env, vector_xy):
    """Convert one world-frame XY vector to the robot yaw frame."""
    quat = env.base_quat[0]
    sin_yaw = 2.0 * (quat[3] * quat[2] + quat[0] * quat[1])
    cos_yaw = 1.0 - 2.0 * (quat[1].square() + quat[2].square())
    vector_xy = torch.as_tensor(vector_xy, device=env.device, dtype=torch.float)
    return torch.stack((
        vector_xy[0] * cos_yaw + vector_xy[1] * sin_yaw,
        -vector_xy[0] * sin_yaw + vector_xy[1] * cos_yaw,
    ))


def _set_closing_case(env, robot_frame_offset, robot_frame_velocity):
    """Set a deterministic primary obstacle without advancing the simulator."""
    low = env.dynamic_obstacle_effective_low[0]
    high = env.dynamic_obstacle_effective_high[0]
    span = high - low
    fractions = torch.tensor(
        [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85],
         [0.85, 0.85], [0.5, 0.15], [0.5, 0.85]],
        device=env.device, dtype=torch.float,
    )
    starts = low + fractions[:env.num_dynamic_obstacles] * span
    robot_local = env.root_states[0, :2] - env.env_origins[0, :2]
    primary = robot_local + _robot_frame_to_world(env, robot_frame_offset)
    starts[0] = torch.minimum(torch.maximum(primary, low + 0.01), high - 0.01)
    velocities = torch.zeros_like(starts)
    velocities[0] = _robot_frame_to_world(env, robot_frame_velocity)
    env.dynamic_obstacle_start[:] = starts.unsqueeze(0)
    env.dynamic_obstacle_velocity[:] = velocities.unsqueeze(0)
    env.dynamic_obstacle_time[:] = 0.0
    env._write_dynamic_obstacle_states()
    env._set_dynamic_obstacle_states_in_sim()
    env._get_rays()


def _print_closing(env, name):
    angles = torch.rad2deg(env.ray_angles).detach().cpu().numpy()
    current = env.rays[0].detach().cpu().numpy()
    future = env.closing_rate_gt_future_fused_rays[0].detach().cpu().numpy()
    closing = env.closing_rate_gt[0].detach().cpu().numpy()
    print('[closing] case={} horizon={:.3f}s'.format(name, env.gt_horizon), flush=True)
    print('  angles_deg={}'.format(np.array2string(angles, precision=1)), flush=True)
    print('  current_fused_ray={}'.format(np.array2string(current, precision=3)), flush=True)
    print('  future_fused_ray={}'.format(np.array2string(future, precision=3)), flush=True)
    print('  closing_rate_gt={}'.format(np.array2string(closing, precision=3)), flush=True)


def _run_closing(env, steps, print_every):
    """Run five deterministic checks for the counterfactual closing label."""
    del steps, print_every
    env.do_reset = False
    tolerance = 1e-5

    # A: zero obstacle velocity leaves both dynamic and fused boundaries fixed.
    _set_closing_case(env, [2.0, 0.0], [0.0, 0.0])
    _print_closing(env, 'A-static')
    assert torch.allclose(env.closing_rate_gt, torch.zeros_like(env.closing_rate_gt), atol=tolerance)

    # B/C: use the same visible center ray and select rays where the dynamic
    # obstacle is the current fused boundary, avoiding static-wall masking.
    _set_closing_case(env, [2.0, 0.0], [-1.0, 0.0])
    _print_closing(env, 'B-approaching')
    visible = env.dynamic_ray_hit_mask & (env.static_rays > env.dynamic_rays + tolerance)
    assert visible.any() and env.closing_rate_gt[0, visible[0]].max() > tolerance

    _set_closing_case(env, [2.0, 0.0], [1.0, 0.0])
    _print_closing(env, 'C-receding')
    visible = env.dynamic_ray_hit_mask & (env.static_rays > env.dynamic_rays + tolerance)
    assert visible.any() and env.closing_rate_gt[0, visible[0]].min() < -tolerance

    # D: lateral motion changes the angular sector and contracts at least one
    # safety boundary ray; print every ray for manual sector inspection.
    _set_closing_case(env, [2.0, -0.8], [0.0, 4.0])
    _print_closing(env, 'D-crossing')
    assert torch.isfinite(env.closing_rate_gt).all()
    assert torch.max(torch.abs(env.closing_rate_gt)) > tolerance

    # E: place the obstacle just below the effective x-bound and query beyond
    # it. The pure trajectory query must reflect, unlike constant velocity.
    high_x = env.dynamic_obstacle_effective_high[0, 0]
    robot_local = env.root_states[0, :2] - env.env_origins[0, :2]
    reflection_world_offset = torch.stack((high_x - robot_local[0] - 0.02, robot_local[1] * 0.0))
    reflection_offset = _world_to_robot_frame(env, reflection_world_offset)
    reflection_velocity = _world_to_robot_frame(env, [1.0, 0.0])
    _set_closing_case(env, reflection_offset, reflection_velocity)
    _print_closing(env, 'E-reflection')
    future_position, _ = env._compute_dynamic_obstacle_states_at_time(
        query_time=env.dynamic_obstacle_time + env.gt_horizon
    )
    current_position = env.dynamic_obstacle_states[..., :2]
    current_velocity = env.dynamic_obstacle_states[..., 7:9]
    constant_velocity_position = current_position + current_velocity * env.gt_horizon
    reflected_step = future_position[:, 0] - current_position[:, 0]
    constant_step = constant_velocity_position[:, 0] - current_position[:, 0]
    print('  reflection_future_xy={} constant_velocity_xy={}'.format(
        np.array2string(future_position[0, 0].detach().cpu().numpy(), precision=3),
        np.array2string(constant_velocity_position[0, 0].detach().cpu().numpy(), precision=3),
    ), flush=True)
    assert torch.linalg.vector_norm(future_position - constant_velocity_position)[:, 0].max() > tolerance
    assert (reflected_step[:, 0] * constant_step[:, 0] < 0.0).any()
    print('[closing] all five sanity cases passed', flush=True)


def main():
    script_args = _parse_script_args()
    args = get_args()
    args.headless = script_args.headless
    args.wandb = False
    if script_args.num_envs < 1:
        raise ValueError('--num-envs must be positive')
    args.num_envs = script_args.num_envs

    cfg = _prepare_config(script_args.num_envs)
    env, _ = task_registry.make_env(
        name='go2_pos_dynamic', args=args, env_cfg=cfg
    )
    print(
        '[validate] task=go2_pos_dynamic mode={} headless={} device={}'.format(
            script_args.mode, script_args.headless, env.device
        ),
        flush=True,
    )
    print(
        '[validate] ray_count={} ray_range_deg=({:.1f}, {:.1f}) '
        'ray_step_deg={:.1f}'.format(
            env.ray_angles.numel(),
            float(torch.rad2deg(env.ray_angles[0])),
            float(torch.rad2deg(env.ray_angles[-1])),
            float(torch.rad2deg(env.ray_angles[1] - env.ray_angles[0])),
        ),
        flush=True,
    )

    try:
        env.reset()
        modes = (
            ['rays', 'gt', 'collision', 'closing']
            if script_args.mode == 'all'
            else [script_args.mode]
        )
        for mode_index, mode in enumerate(modes):
            # Each check owns a fresh simulator state.  In particular, the
            # ray check deliberately places boxes near room boundaries, where
            # PhysX may otherwise leave residual contact velocity for the GT
            # check that follows it.
            if mode_index > 0:
                env.reset()
            if mode == 'rays':
                _run_rays(env, script_args.steps, script_args.print_every)
            elif mode == 'gt':
                _run_gt(env, script_args.steps, script_args.print_every)
            elif mode == 'collision':
                _run_collision(env, min(script_args.steps, 5), script_args.print_every)
            elif mode == 'spawn':
                _run_spawn(env, script_args.steps, script_args.print_every)
            elif mode == 'closing':
                _run_closing(env, script_args.steps, script_args.print_every)
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
