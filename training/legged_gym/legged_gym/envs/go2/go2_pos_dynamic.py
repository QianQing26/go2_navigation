"""Go2 navigation environment with linearly moving box obstacles."""

import math

import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse

from legged_gym.envs.base.legged_robot_pos import LeggedRobotPos
from rsl_rl.modules.cbf_lse_layer import DEFAULT_D_SAFE, DEFAULT_KAPPA
from rsl_rl.utils.phase2 import curriculum_speed_range
from rsl_rl.utils.phase4 import lse_drift_from_fused_rays


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
            # These are staging poses only.  ``_init_buffers`` writes the
            # sampled trajectory states before the first simulation step, but
            # Isaac Gym still builds the initial PhysX scene from the poses
            # supplied here.  Keeping all boxes at the same position creates
            # an overlapping contact cluster for every environment and can
            # overflow the GPU contact/solver buffers at larger num_envs.
            # Separate them vertically until the trajectory state is written.
            staging_z = size[2] * 0.5 + (obstacle_id + 1) * (size[2] + 1.0)
            pose.p = gymapi.Vec3(0.0, 0.0, staging_z)
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
        self.dynamic_obstacle_current_speed_range = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.float
        )
        self.dynamic_obstacle_bounds = torch.as_tensor(
            obstacle_cfg.bounds, device=self.device, dtype=torch.float
        )
        if self.dynamic_obstacle_bounds.shape != (2, 2):
            raise ValueError("dynamic_obstacles.bounds must be [[xmin, xmax], [ymin, ymax]]")
        if torch.any(self.dynamic_obstacle_bounds[:, 1] <= self.dynamic_obstacle_bounds[:, 0]):
            raise ValueError("dynamic_obstacles.bounds must have positive spans")
        self.dynamic_obstacle_radius = math.sqrt(size[0] ** 2 + size[1] ** 2) * 0.5
        self.dynamic_obstacle_size = torch.as_tensor(
            size, device=self.device, dtype=torch.float
        )
        self.dynamic_obstacle_height = size[2]
        self.dynamic_obstacle_effective_low = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.float
        )
        self.dynamic_obstacle_effective_high = torch.zeros_like(
            self.dynamic_obstacle_effective_low
        )
        if hasattr(self, 'terrain') and hasattr(self.terrain, 'env_origins'):
            self.terrain_cell_origins = torch.as_tensor(
                self.terrain.env_origins, device=self.device, dtype=torch.float
            )
        else:
            self.terrain_cell_origins = None
        self.dynamic_rays = torch.full_like(
            self.rays, float(self.cfg.sensors.ray2d.max_dist)
        )
        self.dynamic_ray_hit_mask = torch.zeros_like(
            self.rays, dtype=torch.bool
        )
        self.static_rays = torch.full_like(
            self.rays, float(self.cfg.sensors.ray2d.max_dist)
        )
        self.closing_rate_gt = torch.zeros_like(self.rays)
        self.lse_drift_gt = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.float
        )
        self.closing_rate_gt_future_fused_rays = torch.full_like(
            self.rays, float(self.cfg.sensors.ray2d.max_dist)
        )
        self.future_dynamic_rays = torch.full_like(
            self.rays, float(self.cfg.sensors.ray2d.max_dist)
        )
        source_shape = self.rays.shape
        self.dynamic_active_obstacle_id = torch.full(
            source_shape, -1, device=self.device, dtype=torch.long
        )
        self.future_dynamic_active_obstacle_id = torch.full(
            source_shape, -1, device=self.device, dtype=torch.long
        )
        self.current_fused_source_id = torch.full(
            source_shape, -1, device=self.device, dtype=torch.long
        )
        self.future_fused_source_id = torch.full(
            source_shape, -1, device=self.device, dtype=torch.long
        )
        self.source_switch_mask = torch.zeros(
            source_shape, device=self.device, dtype=torch.bool
        )
        self.same_dynamic_source_mask = torch.zeros(
            source_shape, device=self.device, dtype=torch.bool
        )
        self.active_obstacle_radial_velocity_gt = torch.zeros_like(self.rays)
        self.active_obstacle_radial_velocity_valid_mask = torch.zeros(
            source_shape, device=self.device, dtype=torch.bool
        )
        self.active_obstacle_geometric_closing_rate_gt = torch.zeros_like(
            self.rays
        )
        self.active_obstacle_geometric_closing_valid_mask = torch.zeros(
            source_shape, device=self.device, dtype=torch.bool
        )
        self.active_obstacle_ray_discriminant_sqrt = torch.zeros_like(self.rays)
        motion_estimation_cfg = getattr(self.cfg, 'motion_estimation', None)
        self.gt_horizon = float(
            getattr(motion_estimation_cfg, 'gt_horizon', 0.1)
        )
        if self.gt_horizon <= 0.0:
            raise ValueError('motion_estimation.gt_horizon must be positive')
        configured_d_safe = float(
            getattr(motion_estimation_cfg, 'gt_d_safe', DEFAULT_D_SAFE)
        )
        configured_kappa = float(
            getattr(motion_estimation_cfg, 'gt_kappa', DEFAULT_KAPPA)
        )
        if abs(configured_d_safe - DEFAULT_D_SAFE) > 1.0e-6:
            raise ValueError(
                'motion_estimation.gt_d_safe must match the live CBF d_safe '
                '({:.6f}); got {:.6f}'.format(DEFAULT_D_SAFE, configured_d_safe)
            )
        if abs(configured_kappa - DEFAULT_KAPPA) > 1.0e-6:
            raise ValueError(
                'motion_estimation.gt_kappa must match the live CBF kappa '
                '({:.6f}); got {:.6f}'.format(DEFAULT_KAPPA, configured_kappa)
            )
        self.gt_d_safe = DEFAULT_D_SAFE
        self.gt_kappa = DEFAULT_KAPPA

        self._reset_dynamic_obstacles(
            torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        )

    def _get_dynamic_obstacle_speed_range(self):
        """Return the current curriculum speed range in m/s.

        The curriculum advances with policy steps, not simulator substeps,
        because obstacle velocities are sampled once per environment reset.
        Existing trajectories therefore remain deterministic until their
        next reset/replay.
        """

        obstacle_cfg = self.cfg.dynamic_obstacles
        final_min, final_max = [
            float(value) for value in obstacle_cfg.speed_range
        ]
        curriculum_cfg = getattr(obstacle_cfg, 'curriculum', None)
        if curriculum_cfg is None or not getattr(curriculum_cfg, 'enabled', False):
            return final_min, final_max

        start_min, start_max = [
            float(value) for value in curriculum_cfg.speed_start
        ]
        curriculum_steps = int(curriculum_cfg.speed_steps)
        if curriculum_steps < 1:
            raise ValueError(
                'dynamic_obstacles.curriculum.speed_steps must be positive'
            )
        if not (0.0 <= start_min <= start_max):
            raise ValueError(
                'dynamic_obstacles.curriculum.speed_start must satisfy '
                '0 <= min <= max'
            )
        if not (0.0 <= final_min <= final_max):
            raise ValueError(
                'dynamic_obstacles.speed_range must satisfy 0 <= min <= max'
            )

        progress = min(float(self.common_step_counter) / curriculum_steps, 1.0)
        return curriculum_speed_range(
            progress, (start_min, start_max), (final_min, final_max)
        )

    def get_dynamic_obstacle_curriculum(self):
        """Return the current shared obstacle-speed curriculum state."""

        obstacle_cfg = self.cfg.dynamic_obstacles
        curriculum_cfg = getattr(obstacle_cfg, 'curriculum', None)
        if curriculum_cfg is None or not getattr(curriculum_cfg, 'enabled', False):
            progress = 1.0
            speed_min, speed_max = self._get_dynamic_obstacle_speed_range()
            speed_steps = 0
        else:
            speed_steps = int(curriculum_cfg.speed_steps)
            progress = min(float(self.common_step_counter) / speed_steps, 1.0)
            speed_min, speed_max = self._get_dynamic_obstacle_speed_range()
        return {
            'progress': progress,
            'speed_min': speed_min,
            'speed_max': speed_max,
            'speed_steps': speed_steps,
        }

    def get_dynamic_obstacle_speed_statistics(self):
        """Return statistics of the active analytic trajectory velocities.

        The values intentionally come from the reflected analytic trajectory,
        not from PhysX root-state velocities.  This is the difficulty signal
        shared by all three Phase-2 safety modes.
        """

        _, trajectory_velocity = self._compute_dynamic_obstacle_states_at_time()
        speeds = torch.linalg.vector_norm(trajectory_velocity, dim=-1).reshape(-1)
        return {
            'mean': float(speeds.mean()),
            'p50': float(torch.quantile(speeds, 0.50)),
            'p90': float(torch.quantile(speeds, 0.90)),
            'max': float(speeds.max()),
        }

    def _get_dynamic_obstacle_speed_ranges(self, env_ids):
        """Return one sampling interval per environment.

        The optional ``dataset_speed_sampling`` attributes are intentionally
        read from the in-memory config only.  Training configs do not define
        them, so the normal curriculum path remains unchanged.
        """
        obstacle_cfg = self.cfg.dynamic_obstacles
        mode = getattr(obstacle_cfg, 'dataset_speed_sampling', 'curriculum')
        if mode == 'curriculum':
            low, high = self._get_dynamic_obstacle_speed_range()
            return torch.full(
                (len(env_ids),), low, device=self.device, dtype=torch.float
            ), torch.full(
                (len(env_ids),), high, device=self.device, dtype=torch.float
            )

        sampling_range = getattr(
            obstacle_cfg, 'dataset_speed_sampling_range', obstacle_cfg.speed_range
        )
        full_min, full_max = [float(value) for value in sampling_range]
        if not (0.0 <= full_min < full_max):
            raise ValueError(
                'dataset_speed_sampling_range must satisfy 0 <= min < max'
            )
        if mode == 'uniform':
            low = torch.full(
                (len(env_ids),), full_min, device=self.device, dtype=torch.float
            )
            high = torch.full(
                (len(env_ids),), full_max, device=self.device, dtype=torch.float
            )
            return low, high
        if mode != 'stratified':
            raise ValueError(
                'dataset_speed_sampling must be curriculum, uniform, or stratified'
            )

        # Group global environment ids into three approximately equal blocks.
        # This gives every reset a stable low/medium/high speed assignment and
        # avoids rejection sampling on the GPU.
        group = torch.div(
            env_ids * 3, max(self.num_envs, 1), rounding_mode='floor'
        ).clamp(max=2)
        bin_low = torch.tensor(
            [full_min, 0.5, 1.0], device=self.device, dtype=torch.float
        )
        bin_high = torch.tensor(
            [0.5, 1.0, full_max], device=self.device, dtype=torch.float
        )
        return bin_low[group], bin_high[group]

    def _get_room_constrained_bounds(self, env_ids):
        """Return obstacle-center bounds that stay inside each room cell.

        The configured bounds are expressed relative to ``env_origins`` (the
        robot spawn frame).  ``env_origins`` is not generally the center of a
        room, so using the configured range directly can cross a wall.  Clamp
        it against the terrain cell bounds and reserve half the box size plus
        the configured static clearance for the obstacle footprint.
        """

        if self.terrain_cell_origins is None:
            # Plane/none terrains do not have room cells.  Keep the configured
            # bounds unchanged in that case.
            low = self.dynamic_obstacle_bounds[:, 0].view(1, 2).expand(
                len(env_ids), -1
            )
            high = self.dynamic_obstacle_bounds[:, 1].view(1, 2).expand(
                len(env_ids), -1
            )
            return low, high

        terrain_levels = self.terrain_levels[env_ids].to(dtype=torch.long)
        terrain_types = self.terrain_types[env_ids].to(dtype=torch.long)
        room_centers = self.terrain_cell_origins[
            terrain_levels, terrain_types, :2
        ]
        room_half_extent = torch.tensor(
            [self.terrain.env_length * 0.5, self.terrain.env_width * 0.5],
            device=self.device,
            dtype=torch.float,
        )
        static_clearance = float(
            getattr(self.cfg.dynamic_obstacles, 'static_clearance', 0.0)
        )
        room_margin = self.dynamic_obstacle_size[:2] * 0.5 + static_clearance
        room_min_local = (
            room_centers - room_half_extent - self.env_origins[env_ids, :2]
            + room_margin
        )
        room_max_local = (
            room_centers + room_half_extent - self.env_origins[env_ids, :2]
            - room_margin
        )
        configured_low = self.dynamic_obstacle_bounds[:, 0].view(1, 2)
        configured_high = self.dynamic_obstacle_bounds[:, 1].view(1, 2)
        low = torch.maximum(configured_low, room_min_local)
        high = torch.minimum(configured_high, room_max_local)
        if torch.any(high <= low):
            bad_ids = env_ids[(high <= low).any(dim=1)]
            raise RuntimeError(
                'Dynamic obstacle bounds have no room-safe area for envs {}'.format(
                    bad_ids.detach().cpu().tolist()
                )
            )
        return low, high

    def _sample_dynamic_obstacles(self, env_ids):
        obstacle_cfg = self.cfg.dynamic_obstacles
        count = len(env_ids)
        effective_low, effective_high = self._get_room_constrained_bounds(env_ids)
        self.dynamic_obstacle_effective_low[env_ids] = effective_low
        self.dynamic_obstacle_effective_high[env_ids] = effective_high
        low = effective_low.view(count, 1, 2)
        high = effective_high.view(count, 1, 2)
        span = high - low

        robot_local = (
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2]
        ).unsqueeze(1)
        goal_local = (
            self.position_targets[env_ids, :2] - self.env_origins[env_ids, :2]
        ).unsqueeze(1)
        min_robot_distance = float(obstacle_cfg.min_robot_distance)
        min_goal_distance = float(obstacle_cfg.min_goal_distance)
        obstacle_clearance = float(getattr(obstacle_cfg, 'obstacle_clearance', 0.0))
        static_clearance = float(getattr(obstacle_cfg, 'static_clearance', 0.0))
        max_attempts = int(getattr(obstacle_cfg, 'max_spawn_attempts', 128))
        if max_attempts < 1:
            raise ValueError('dynamic_obstacles.max_spawn_attempts must be positive')

        obstacle_size = self.dynamic_obstacle_size[:2].view(1, 1, 2)

        def sample(count_to_sample):
            return low + torch.rand(
                count_to_sample,
                self.num_dynamic_obstacles,
                2,
                device=self.device,
            ) * span

        def static_free(candidate_starts, candidate_env_ids):
            """Check the terrain height map under each box footprint."""

            if self.height_samples is None:
                return torch.ones(
                    candidate_starts.shape[:2], dtype=torch.bool, device=self.device
                )

            resolution = float(self.cfg.terrain.horizontal_scale)
            half_extent = torch.ceil(
                (obstacle_size + static_clearance) / (2.0 * resolution)
            ).to(dtype=torch.long)
            offsets_x = torch.arange(
                -int(half_extent[0, 0, 0]),
                int(half_extent[0, 0, 0]) + 1,
                device=self.device,
                dtype=torch.long,
            )
            offsets_y = torch.arange(
                -int(half_extent[0, 0, 1]),
                int(half_extent[0, 0, 1]) + 1,
                device=self.device,
                dtype=torch.long,
            )
            offset_x, offset_y = torch.meshgrid(offsets_x, offsets_y, indexing='ij')
            offsets = torch.stack((offset_x.flatten(), offset_y.flatten()), dim=-1)

            world_xy = self.env_origins[candidate_env_ids, None, :2] + candidate_starts
            grid_xy = torch.floor(
                (world_xy + float(self.cfg.terrain.border_size)) / resolution
            ).to(dtype=torch.long)
            grid_xy = grid_xy.unsqueeze(2) + offsets.view(1, 1, -1, 2)
            in_bounds = (
                (grid_xy[..., 0] >= 0)
                & (grid_xy[..., 0] < self.height_samples.shape[0])
                & (grid_xy[..., 1] >= 0)
                & (grid_xy[..., 1] < self.height_samples.shape[1])
            )
            safe_grid_xy = torch.stack(
                (
                    grid_xy[..., 0].clamp(0, self.height_samples.shape[0] - 1),
                    grid_xy[..., 1].clamp(0, self.height_samples.shape[1] - 1),
                ),
                dim=-1,
            )
            heights = self.height_samples[
                safe_grid_xy[..., 0], safe_grid_xy[..., 1]
            ]
            # Keep one validity flag per environment and obstacle.  The
            # caller then requires every obstacle in that environment to be
            # valid; reducing both dimensions here would lose that axis.
            return (in_bounds & (heights <= 0.1)).all(dim=-1)

        # Place obstacles one at a time.  Rejecting an entire six-obstacle
        # layout at once has a very low acceptance rate in a room of this
        # size, even when a valid layout exists.  Sequential placement keeps
        # already accepted obstacles and only resamples the current one.
        starts = torch.empty(
            count, self.num_dynamic_obstacles, 2, device=self.device
        )
        obstacle_x_clearance = float(self.dynamic_obstacle_size[0]) + obstacle_clearance
        obstacle_y_clearance = float(self.dynamic_obstacle_size[1]) + obstacle_clearance
        for obstacle_id in range(self.num_dynamic_obstacles):
            placed_current = torch.zeros(count, dtype=torch.bool, device=self.device)
            for _ in range(max_attempts):
                unresolved = ~placed_current
                if not unresolved.any():
                    break
                candidate = sample(count)[:, 0, :]
                valid = (
                    torch.linalg.vector_norm(candidate - robot_local[:, 0, :], dim=-1)
                    > min_robot_distance
                )
                valid &= (
                    torch.linalg.vector_norm(candidate - goal_local[:, 0, :], dim=-1)
                    > min_goal_distance
                )
                candidate_for_static = candidate.unsqueeze(1)
                valid &= static_free(candidate_for_static, env_ids)[:, 0]
                if obstacle_id > 0:
                    delta = candidate[:, None, :] - starts[:, :obstacle_id, :]
                    valid &= ~(
                        (delta[..., 0].abs() < obstacle_x_clearance)
                        & (delta[..., 1].abs() < obstacle_y_clearance)
                    ).any(dim=1)
                accepted = unresolved & valid
                starts[accepted, obstacle_id] = candidate[accepted]
                placed_current |= accepted
            if not placed_current.all():
                failed = int((~placed_current).sum())
                raise RuntimeError(
                    'Unable to place dynamic obstacle {} for {} environments '
                    'after {} attempts'.format(
                        obstacle_id, failed, max_attempts
                    )
                )

        speed_min, speed_max = self._get_dynamic_obstacle_speed_ranges(env_ids)
        if torch.any(speed_min < 0.0) or torch.any(speed_max < speed_min):
            raise ValueError("dynamic_obstacles.speed_range is invalid")
        self.dynamic_obstacle_current_speed_range[env_ids, 0] = speed_min
        self.dynamic_obstacle_current_speed_range[env_ids, 1] = speed_max
        speed_min = speed_min.view(count, 1, 1)
        speed_max = speed_max.view(count, 1, 1)
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

    def _compute_dynamic_obstacle_states_at_time(
        self, env_ids=None, query_time=None
    ):
        """Purely evaluate reflected obstacle trajectories at an arbitrary time."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )

        if query_time is None:
            time = self.dynamic_obstacle_time[env_ids]
        else:
            time = torch.as_tensor(
                query_time, device=self.device,
                dtype=self.dynamic_obstacle_time.dtype,
            ).reshape(-1)
            if time.numel() == 1:
                time = time.expand(len(env_ids))
            elif time.numel() != len(env_ids):
                raise ValueError(
                    'query_time must be a scalar or have one value per env_id'
                )

        low = self.dynamic_obstacle_effective_low[env_ids].view(-1, 1, 2)
        high = self.dynamic_obstacle_effective_high[env_ids].view(-1, 1, 2)
        span = high - low
        travel = self.dynamic_obstacle_start[env_ids] + self.dynamic_obstacle_velocity[env_ids] * time.view(-1, 1, 1)
        phase = torch.remainder(travel - low, 2.0 * span)
        reflected = torch.where(phase <= span, phase, 2.0 * span - phase) + low
        direction = torch.where(phase <= span, 1.0, -1.0)
        velocity = self.dynamic_obstacle_velocity[env_ids] * direction
        position = self.env_origins[env_ids, None, 0:2] + reflected
        return position, velocity

    def _write_dynamic_obstacle_states_at_time(self, env_ids=None, time_override=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )
        position, velocity = self._compute_dynamic_obstacle_states_at_time(
            env_ids=env_ids, query_time=time_override
        )

        # ``tensor[env_ids]`` is an advanced-indexing copy in PyTorch.  Assign
        # directly to the original tensor so the values are written back to
        # the simulator root-state buffer.
        self.dynamic_obstacle_states[env_ids, :, 0:2] = (
            position
        )
        self.dynamic_obstacle_states[env_ids, :, 2] = self.dynamic_obstacle_height * 0.5
        self.dynamic_obstacle_states[env_ids, :, 3:7] = 0.0
        self.dynamic_obstacle_states[env_ids, :, 6] = 1.0
        self.dynamic_obstacle_states[env_ids, :, 7:9] = velocity
        self.dynamic_obstacle_states[env_ids, :, 9:13] = 0.0

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

    def _compute_dynamic_rays_from_positions(
        self, obstacle_xy, robot_xy, robot_quat
    ):
        """Purely compute analytic dynamic rays for arbitrary poses.

        This preserves the existing ray convention, range limits, and
        bounding-circle approximation while allowing counterfactual queries.
        """
        delta = obstacle_xy - robot_xy.unsqueeze(1)

        # Transform obstacle centers from world coordinates into the robot's
        # yaw-aligned frame, matching the terrain ray convention.
        sin_yaw = 2.0 * (robot_quat[:, 3] * robot_quat[:, 2] + robot_quat[:, 0] * robot_quat[:, 1])
        cos_yaw = 1.0 - 2.0 * (robot_quat[:, 1].square() + robot_quat[:, 2].square())
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
        ray_distances = torch.where(
            valid,
            near,
            torch.full_like(near, float(self.cfg.sensors.ray2d.max_dist)),
        )
        dynamic_rays, active_obstacle_id = ray_distances.min(dim=1)
        hit_mask = valid.any(dim=1)
        active_obstacle_id = torch.where(
            hit_mask,
            active_obstacle_id,
            torch.full_like(active_obstacle_id, -1),
        )
        return dynamic_rays, hit_mask, active_obstacle_id

    def compute_lse_drift_gt(self, horizon):
        """Query the frozen-robot LSE barrier drift at an arbitrary horizon.

        The current static ray field is held fixed while only the analytic
        dynamic obstacles are advanced.  This is the same counterfactual
        semantics used by the existing ``lse_drift_gt`` label and is kept
        stateless so evaluation-only oracle variants cannot affect the live
        trajectory.
        """

        horizon = float(horizon)
        if horizon <= 0.0:
            raise ValueError('GT horizon must be positive')
        static_rays = self.static_rays
        current_dynamic_rays = self.dynamic_rays
        current_fused = torch.minimum(static_rays, current_dynamic_rays)
        future_position, _ = self._compute_dynamic_obstacle_states_at_time(
            query_time=self.dynamic_obstacle_time + horizon
        )
        future_dynamic_rays, _, future_active_obstacle_id = (
            self._compute_dynamic_rays_from_positions(
                future_position, self.root_states[:, :2], self.base_quat
            )
        )
        future_fused = torch.minimum(static_rays, future_dynamic_rays)
        current_h = current_fused - self.gt_d_safe
        future_h = future_fused - self.gt_d_safe
        current_lse = -torch.logsumexp(
            -self.gt_kappa * current_h, dim=-1, keepdim=True
        ) / self.gt_kappa
        future_lse = -torch.logsumexp(
            -self.gt_kappa * future_h, dim=-1, keepdim=True
        ) / self.gt_kappa
        return {
            'drift': lse_drift_from_fused_rays(
                current_fused, future_fused, horizon,
                self.gt_d_safe, self.gt_kappa,
            ),
            'current_fused_rays': current_fused,
            'future_fused_rays': future_fused,
            'future_dynamic_rays': future_dynamic_rays,
            'future_active_obstacle_id': future_active_obstacle_id,
            'current_lse': current_lse,
            'future_lse': future_lse,
        }

    def _update_closing_rate_gt(
        self, static_rays, current_dynamic_rays, current_active_obstacle_id
    ):
        """Build the existing configured-horizon GT labels."""
        gt = self.compute_lse_drift_gt(self.gt_horizon)
        current_fused = gt['current_fused_rays']
        future_dynamic_rays = gt['future_dynamic_rays']
        future_fused = gt['future_fused_rays']
        future_active_obstacle_id = gt['future_active_obstacle_id']
        _, current_trajectory_velocity = self._compute_dynamic_obstacle_states_at_time()
        self.future_dynamic_rays = future_dynamic_rays
        self.closing_rate_gt = (current_fused - future_fused) / self.gt_horizon
        self.closing_rate_gt_future_fused_rays = future_fused
        self.lse_drift_gt = gt['drift']

        current_source_id = torch.where(
            static_rays <= current_dynamic_rays,
            torch.full_like(current_active_obstacle_id, -1),
            current_active_obstacle_id,
        )
        future_source_id = torch.where(
            static_rays <= future_dynamic_rays,
            torch.full_like(future_active_obstacle_id, -1),
            future_active_obstacle_id,
        )
        source_switch = current_source_id != future_source_id
        same_dynamic = (
            (current_source_id >= 0)
            & (future_source_id == current_source_id)
        )

        # The environmental derivative freezes the robot.  Rotate the
        # reflected analytic obstacle velocity into the current robot yaw
        # frame, but do not subtract the robot velocity.
        selected_ids = current_source_id.clamp(min=0)
        selected_velocity = torch.gather(
            current_trajectory_velocity,
            1,
            selected_ids.unsqueeze(-1).expand(-1, -1, 2),
        )
        sin_yaw = 2.0 * (
            self.base_quat[:, 3] * self.base_quat[:, 2]
            + self.base_quat[:, 0] * self.base_quat[:, 1]
        )
        cos_yaw = 1.0 - 2.0 * (
            self.base_quat[:, 1].square() + self.base_quat[:, 2].square()
        )
        velocity_body_x = (
            selected_velocity[..., 0] * cos_yaw[:, None]
            + selected_velocity[..., 1] * sin_yaw[:, None]
        )
        velocity_body_y = (
            -selected_velocity[..., 0] * sin_yaw[:, None]
            + selected_velocity[..., 1] * cos_yaw[:, None]
        )
        ray_angles = self.ray_angles.view(1, -1)
        radial_velocity = -(
            velocity_body_x * torch.cos(ray_angles)
            + velocity_body_y * torch.sin(ray_angles)
        )

        self.dynamic_active_obstacle_id = current_active_obstacle_id
        self.future_dynamic_active_obstacle_id = future_active_obstacle_id
        self.current_fused_source_id = current_source_id
        self.future_fused_source_id = future_source_id
        self.source_switch_mask = source_switch
        self.same_dynamic_source_mask = same_dynamic
        self.active_obstacle_radial_velocity_gt = torch.where(
            same_dynamic, radial_velocity, torch.zeros_like(radial_velocity)
        )
        self.active_obstacle_radial_velocity_valid_mask = same_dynamic

        # Exact environmental derivative of the analytic ray-circle boundary.
        # The robot pose is frozen; only the active obstacle trajectory moves.
        current_position, _ = self._compute_dynamic_obstacle_states_at_time()
        selected_position = torch.gather(
            current_position,
            1,
            selected_ids.unsqueeze(-1).expand(-1, -1, 2),
        )
        delta_x = selected_position[..., 0] - self.root_states[:, None, 0]
        delta_y = selected_position[..., 1] - self.root_states[:, None, 1]
        position_body_x = (
            delta_x * cos_yaw[:, None] + delta_y * sin_yaw[:, None]
        )
        position_body_y = (
            -delta_x * sin_yaw[:, None] + delta_y * cos_yaw[:, None]
        )
        ray_cos = torch.cos(ray_angles)
        ray_sin = torch.sin(ray_angles)
        perpendicular = -position_body_x * ray_sin + position_body_y * ray_cos
        discriminant = self.dynamic_obstacle_radius ** 2 - perpendicular.square()
        discriminant_sqrt = torch.sqrt(torch.clamp(discriminant, min=0.0))
        velocity_body_x = (
            selected_velocity[..., 0] * cos_yaw[:, None]
            + selected_velocity[..., 1] * sin_yaw[:, None]
        )
        velocity_body_y = (
            -selected_velocity[..., 0] * sin_yaw[:, None]
            + selected_velocity[..., 1] * cos_yaw[:, None]
        )
        normal_velocity = velocity_body_x * ray_cos + velocity_body_y * ray_sin
        tangent_velocity = -velocity_body_x * ray_sin + velocity_body_y * ray_cos
        denominator = discriminant_sqrt.clamp_min(1e-6)
        geometric_closing = -normal_velocity - (
            perpendicular * tangent_velocity / denominator
        )
        geometric_valid = same_dynamic & (discriminant >= 0.0)
        self.active_obstacle_geometric_closing_rate_gt = torch.where(
            geometric_valid, geometric_closing, torch.zeros_like(geometric_closing)
        )
        self.active_obstacle_geometric_closing_valid_mask = geometric_valid
        self.active_obstacle_ray_discriminant_sqrt = torch.where(
            geometric_valid, discriminant_sqrt, torch.zeros_like(discriminant_sqrt)
        )

    def _get_rays(self, env_ids=None):
        """Fuse terrain rays with analytic ray-box approximations."""

        super()._get_rays(env_ids)
        static_rays = self.rays.clone()
        dynamic_rays, hit_mask, active_obstacle_id = (
            self._compute_dynamic_rays_from_positions(
                self.dynamic_obstacle_states[..., :2],
                self.root_states[:, :2],
                self.base_quat,
            )
        )
        self.static_rays = static_rays
        self.dynamic_rays = dynamic_rays
        self.dynamic_ray_hit_mask = hit_mask
        self.dynamic_active_obstacle_id = active_obstacle_id
        self.rays = torch.minimum(self.rays, dynamic_rays)
        self._update_closing_rate_gt(
            static_rays, dynamic_rays, active_obstacle_id
        )

    def get_dynamic_obstacle_gt(self, relative_to_robot=False):
        """Return simulator ground truth for dynamic obstacles.

        The returned dictionary is intentionally separate from the actor
        observation.  It can be used by validators, privileged critics, or
        future teacher policies without changing the current checkpoint input
        layout.

        Returns:
            position: ``[num_envs, num_obstacles, 3]`` world or robot-relative
                obstacle centers.
            velocity: Backward-compatible alias for the physical root-state
                velocity.
            physical_root_velocity: ``[num_envs, num_obstacles, 3]`` raw
                Isaac Gym actor root-state velocity, retained as diagnostic.
            trajectory_velocity: ``[num_envs, num_obstacles, 3]`` reflected
                analytic trajectory velocity used as privileged GT.
            size: ``[num_envs, num_obstacles, 3]`` box dimensions ``[x,y,z]``.
            radius: ``[num_envs, num_obstacles, 1]`` horizontal bounding-circle
                radius used by the analytic ray query.
            actor_indices: simulator-domain actor indices.
        """
        position = self.dynamic_obstacle_states[..., 0:3].clone()
        trajectory_position, trajectory_velocity = (
            self._compute_dynamic_obstacle_states_at_time()
        )
        trajectory_position = torch.cat(
            (
                trajectory_position,
                torch.full_like(
                    trajectory_position[..., :1], self.dynamic_obstacle_height * 0.5
                ),
            ),
            dim=-1,
        )
        if relative_to_robot:
            position[..., :2] -= self.root_states[:, None, :2]
            trajectory_position[..., :2] -= self.root_states[:, None, :2]
        physical_root_velocity = self.dynamic_obstacle_states[..., 7:10].clone()
        trajectory_velocity = torch.cat(
            (
                trajectory_velocity,
                torch.zeros_like(trajectory_velocity[..., :1]),
            ),
            dim=-1,
        )
        size = self.dynamic_obstacle_size.view(1, 1, 3).expand(
            self.num_envs, self.num_dynamic_obstacles, -1
        )
        radius = torch.full(
            (self.num_envs, self.num_dynamic_obstacles, 1),
            float(self.dynamic_obstacle_radius),
            device=self.device,
            dtype=self.dynamic_obstacle_states.dtype,
        )
        return {
            'position': position,
            'trajectory_position': trajectory_position,
            'velocity': physical_root_velocity,
            'physical_root_velocity': physical_root_velocity,
            'trajectory_velocity': trajectory_velocity,
            'size': size,
            'radius': radius,
            'actor_indices': self.dynamic_obstacle_actor_indices,
        }

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.closing_rate_gt[env_ids] = 0.0
        self.lse_drift_gt[env_ids] = 0.0
        self.closing_rate_gt_future_fused_rays[env_ids] = float(
            self.cfg.sensors.ray2d.max_dist
        )
        self.future_dynamic_rays[env_ids] = float(
            self.cfg.sensors.ray2d.max_dist
        )
        self.dynamic_active_obstacle_id[env_ids] = -1
        self.future_dynamic_active_obstacle_id[env_ids] = -1
        self.current_fused_source_id[env_ids] = -1
        self.future_fused_source_id[env_ids] = -1
        self.source_switch_mask[env_ids] = False
        self.same_dynamic_source_mask[env_ids] = False
        self.active_obstacle_radial_velocity_gt[env_ids] = 0.0
        self.active_obstacle_radial_velocity_valid_mask[env_ids] = False
        self.active_obstacle_geometric_closing_rate_gt[env_ids] = 0.0
        self.active_obstacle_geometric_closing_valid_mask[env_ids] = False
        self.active_obstacle_ray_discriminant_sqrt[env_ids] = 0.0
