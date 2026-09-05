"""Go2 navigation environment with linearly moving box obstacles."""

import math

import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.base.legged_robot_pos import LeggedRobotPos


class DynamicObstacleGo2Pos(LeggedRobotPos):
    """Go2 position-navigation task with batched kinematic obstacles.

    The robot remains the first actor in every Isaac Gym environment. Dynamic
    obstacles are additional gravity-free rigid boxes whose root states are
    advanced along reflected linear trajectories before every physics step.
    """

    def _create_envs(self):
        obstacle_cfg = self.cfg.dynamic_obstacles
        self.num_dynamic_obstacles = int(obstacle_cfg.num_obstacles)
        if self.num_dynamic_obstacles < 1:
            raise ValueError("num_obstacles must be at least 1")
        self.actors_per_env = 1 + self.num_dynamic_obstacles
        self._robot_actor_indices_list = []
        self._dynamic_obstacle_actor_indices_list = []
        self.dynamic_obstacle_handles = []
        super()._create_envs()

    def _create_auxiliary_assets(self):
        obstacle_cfg = self.cfg.dynamic_obstacles

        size = [float(value) for value in obstacle_cfg.size]
        if len(size) != 3 or min(size) <= 0.0:
            raise ValueError("dynamic_obstacles.size must contain three positive values")

        obstacle_options = gymapi.AssetOptions()
        obstacle_options.fix_base_link = False
        obstacle_options.disable_gravity = True
        obstacle_options.density = float(obstacle_cfg.density)
        self.dynamic_obstacle_asset = self.gym.create_box(
            self.sim, size[0], size[1], size[2], obstacle_options
        )

    def _create_additional_actors(self, env_handle, env_id):
        obstacle_cfg = self.cfg.dynamic_obstacles
        size = [float(value) for value in obstacle_cfg.size]
        robot_handle = self.actor_handles[env_id]
        self._robot_actor_indices_list.append(
            self.gym.get_actor_index(env_handle, robot_handle, gymapi.DOMAIN_SIM)
        )

        env_obstacle_handles = []
        env_obstacle_indices = []
        for obstacle_id in range(self.num_dynamic_obstacles):
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(0.0, 0.0, size[2] * 0.5)
            handle = self.gym.create_actor(
                env_handle,
                self.dynamic_obstacle_asset,
                pose,
                f"dynamic_obstacle_{obstacle_id}",
                env_id,
                0,
                0,
            )
            body_props = self.gym.get_actor_rigid_body_properties(
                env_handle, handle
            )
            for body_prop in body_props:
                body_prop.mass = float(obstacle_cfg.mass)
            self.gym.set_actor_rigid_body_properties(
                env_handle, handle, body_props, recomputeInertia=True
            )
            env_obstacle_handles.append(handle)
            env_obstacle_indices.append(
                self.gym.get_actor_index(env_handle, handle, gymapi.DOMAIN_SIM)
            )
        self.dynamic_obstacle_handles.append(env_obstacle_handles)
        self._dynamic_obstacle_actor_indices_list.append(env_obstacle_indices)

    def _init_buffers(self):
        super()._init_buffers()

        # Isaac Gym stores all actor roots in one flat tensor. Make a view for
        # the robot and retain the full tensor for indexed obstacle updates.
        actors_per_env = 1 + self.num_dynamic_obstacles
        self.actor_root_states = self.all_root_states.view(
            self.num_envs, actors_per_env, 13
        )
        # LeggedRobot._init_buffers already selected the robot view.
        self.dynamic_obstacle_states = self.actor_root_states[:, 1:, :]
        self.dynamic_obstacle_actor_indices = torch.as_tensor(
            self._dynamic_obstacle_actor_indices_list,
            dtype=torch.long,
            device=self.device,
        )

        # The parent initialized these buffers while root_states still
        # referred to the flat actor tensor; rebuild them for robot-only state.
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.last_base_twist = torch.zeros_like(self.root_states[:, 7:13])
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        obstacle_cfg = self.cfg.dynamic_obstacles
        size = [float(value) for value in obstacle_cfg.size]
        self.dynamic_obstacle_time = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self.dynamic_obstacle_start = torch.zeros(
            self.num_envs,
            self.num_dynamic_obstacles,
            2,
            device=self.device,
            dtype=torch.float,
        )
        self.dynamic_obstacle_velocity = torch.zeros_like(self.dynamic_obstacle_start)
        self.dynamic_obstacle_bounds = torch.as_tensor(
            obstacle_cfg.bounds, device=self.device, dtype=torch.float
        )
        if self.dynamic_obstacle_bounds.shape != (2, 2):
            raise ValueError("dynamic_obstacles.bounds must be [[xmin, xmax], [ymin, ymax]]")
        if torch.any(self.dynamic_obstacle_bounds[:, 1] <= self.dynamic_obstacle_bounds[:, 0]):
            raise ValueError("dynamic_obstacles.bounds must have positive spans")
        self.dynamic_obstacle_radius = math.sqrt(size[0] ** 2 + size[1] ** 2) * 0.5
        self.dynamic_obstacle_height = size[2]

        self._reset_dynamic_obstacles(
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        )

    def _sample_dynamic_obstacles(self, env_ids):
        obstacle_cfg = self.cfg.dynamic_obstacles
        count = len(env_ids)
        low = self.dynamic_obstacle_bounds[:, 0].view(1, 1, 2)
        high = self.dynamic_obstacle_bounds[:, 1].view(1, 1, 2)
        span = high - low

        robot_local = (
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2]
        ).unsqueeze(1)
        goal_local = (
            self.position_targets[env_ids, :2] - self.env_origins[env_ids, :2]
        ).unsqueeze(1)
        starts = low + torch.rand(
            count,
            self.num_dynamic_obstacles,
            2,
            device=self.device,
        ) * span
        min_robot_distance = float(obstacle_cfg.min_robot_distance)
        min_goal_distance = float(obstacle_cfg.min_goal_distance)
        for _ in range(8):
            valid = (torch.linalg.vector_norm(starts - robot_local, dim=-1) > min_robot_distance).all(dim=1)
            valid &= (torch.linalg.vector_norm(starts - goal_local, dim=-1) > min_goal_distance).all(dim=1)
            invalid = ~valid
            if not invalid.any():
                break
            starts[invalid] = low + torch.rand(
                int(invalid.sum()) * self.num_dynamic_obstacles,
                2,
                device=self.device,
            ).view(int(invalid.sum()), self.num_dynamic_obstacles, 2) * span

        speed_min, speed_max = [float(value) for value in obstacle_cfg.speed_range]
        if speed_min < 0.0 or speed_max < speed_min:
            raise ValueError("dynamic_obstacles.speed_range is invalid")
        speed = speed_min + torch.rand(
            count, self.num_dynamic_obstacles, 1, device=self.device
        ) * (speed_max - speed_min)
        angle = torch.rand(
            count, self.num_dynamic_obstacles, 1, device=self.device
        ) * (2.0 * math.pi)
        velocity = speed * torch.cat((torch.cos(angle), torch.sin(angle)), dim=-1)
        return starts, velocity

    def _write_dynamic_obstacle_states(self, env_ids=None):
        self._write_dynamic_obstacle_states_at_time(env_ids=env_ids)

    def _write_dynamic_obstacle_states_at_time(self, env_ids=None, time_override=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        low = self.dynamic_obstacle_bounds[:, 0].view(1, 1, 2)
        high = self.dynamic_obstacle_bounds[:, 1].view(1, 1, 2)
        span = high - low
        if time_override is None:
            time = self.dynamic_obstacle_time[env_ids].view(-1, 1, 1)
        else:
            time = time_override.view(-1, 1, 1)
        travel = self.dynamic_obstacle_start[env_ids] + self.dynamic_obstacle_velocity[env_ids] * time
        phase = torch.remainder(travel - low, 2.0 * span)
        reflected = torch.where(phase <= span, phase, 2.0 * span - phase) + low
        direction = torch.where(phase <= span, 1.0, -1.0)
        velocity = self.dynamic_obstacle_velocity[env_ids] * direction

        states = self.dynamic_obstacle_states[env_ids]
        states[..., 0:2] = self.env_origins[env_ids, None, 0:2] + reflected
        states[..., 2] = self.dynamic_obstacle_height * 0.5
        states[..., 3:7] = 0.0
        states[..., 6] = 1.0
        states[..., 7:9] = velocity
        states[..., 9:13] = 0.0

    def _set_dynamic_obstacle_states_in_sim(self, env_ids=None):
        if env_ids is None:
            actor_indices = self.dynamic_obstacle_actor_indices.reshape(-1)
        else:
            actor_indices = self.dynamic_obstacle_actor_indices[env_ids].reshape(-1)
        actor_indices = actor_indices.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states_for_sim()),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices),
        )

    def _reset_dynamic_obstacles(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        starts, velocity = self._sample_dynamic_obstacles(env_ids)
        self.dynamic_obstacle_time[env_ids] = 0.0
        self.dynamic_obstacle_start[env_ids] = starts
        self.dynamic_obstacle_velocity[env_ids] = velocity
        self._write_dynamic_obstacle_states(env_ids)
        self._set_dynamic_obstacle_states_in_sim(env_ids)

    def _advance_dynamic_obstacles(self):
        self.dynamic_obstacle_time += float(self.sim_params.dt)
        self._write_dynamic_obstacle_states()
        # Update only obstacle actor ids so the robot state advanced by the
        # previous physics substep is never overwritten.
        self._set_dynamic_obstacle_states_in_sim()

    def _pre_physics_step_callback(self):
        self._advance_dynamic_obstacles()

    def _reset_auxiliary_states(self, env_ids, replay_ids, normal_ids):
        """Reset or rewind obstacle trajectories with the robot replay."""

        if len(normal_ids) > 0:
            # This includes ordinary resets and replay candidates whose robot
            # history was too short for a valid replay.
            self._reset_dynamic_obstacles(normal_ids)

        if len(replay_ids) == 0:
            return

        if not getattr(self.cfg.replay, 'enable_dynamic_obstacle_replay', False):
            # Preserve the old optional behavior: replay the robot but sample
            # a fresh obstacle trajectory.
            self._reset_dynamic_obstacles(replay_ids)
            return

        # One replay step corresponds to one policy/control step, while the
        # obstacle clock advances once per simulator substep.
        rewind_dt = (
            self.replay_undo_steps[replay_ids].to(dtype=torch.float)
            * float(self.cfg.control.decimation)
            * float(self.sim_params.dt)
        )
        replay_time = torch.clamp(
            self.dynamic_obstacle_time[replay_ids] - rewind_dt, min=0.0
        )
        self.dynamic_obstacle_time[replay_ids] = replay_time
        self._write_dynamic_obstacle_states_at_time(
            env_ids=replay_ids, time_override=replay_time
        )
        self._set_dynamic_obstacle_states_in_sim(replay_ids)

    def _get_rays(self, env_ids=None):
        """Fuse terrain rays with analytic ray-box approximations."""

        super()._get_rays(env_ids)
        obstacle_xy = self.dynamic_obstacle_states[..., :2]
        delta = obstacle_xy - self.root_states[:, None, :2]

        # Transform obstacle centers from world coordinates into the robot's
        # yaw-aligned frame, matching the terrain ray convention.
        quat = self.base_quat
        sin_yaw = 2.0 * (quat[:, 3] * quat[:, 2] + quat[:, 0] * quat[:, 1])
        cos_yaw = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
        x = delta[..., 0] * cos_yaw[:, None] + delta[..., 1] * sin_yaw[:, None]
        y = -delta[..., 0] * sin_yaw[:, None] + delta[..., 1] * cos_yaw[:, None]

        angles = self.ray_angles.view(1, 1, -1)
        ray_x = torch.cos(angles)
        ray_y = torch.sin(angles)
        along = x.unsqueeze(-1) * ray_x + y.unsqueeze(-1) * ray_y
        perpendicular = -x.unsqueeze(-1) * ray_y + y.unsqueeze(-1) * ray_x
        discriminant = self.dynamic_obstacle_radius**2 - perpendicular.square()
        near = along - torch.sqrt(torch.clamp(discriminant, min=0.0))
        valid = (
            (discriminant >= 0.0)
            & (near >= float(self.cfg.sensors.ray2d.min_dist))
            & (near <= float(self.cfg.sensors.ray2d.max_dist))
        )
        dynamic_rays = torch.where(
            valid,
            near,
            torch.full_like(near, float(self.cfg.sensors.ray2d.max_dist)),
        ).amin(dim=1)
        self.rays = torch.minimum(self.rays, dynamic_rays)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
