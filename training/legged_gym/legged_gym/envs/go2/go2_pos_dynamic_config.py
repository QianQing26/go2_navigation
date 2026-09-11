"""Configuration for the isolated dynamic-obstacle Go2 task."""

from .go2_pos_config import Go2PosRoughCfg, Go2PosRoughCfgPPO
from rsl_rl.utils.phase2 import (
    PHASE2_SPEED_FINAL,
    PHASE2_SPEED_START,
    PHASE2_SPEED_STEPS,
)


class Go2PosDynamicCfg(Go2PosRoughCfg):
    """First-stage dynamic-obstacle navigation configuration."""

    class env(Go2PosRoughCfg.env):
        class predictive_safety:
            # Phase-1 defaults to the historical delayed-ray shield.  The
            # frozen estimator modes are selected explicitly for evaluation or
            # an experiment config, so existing training remains unchanged.
            mode = 'original'
            estimator_checkpoint = ''
            calibration_delta = 0.0
            use_warmup_gate = True

    class terrain(Go2PosRoughCfg.terrain):
        # Keep a simple room as a static backdrop; moving boxes are the main
        # additional obstacles in this task.
        terrain_types = ['easy_room']
        terrain_proportions = [1.0]

    class dynamic_obstacles:
        # Six moving boxes provide a denser first-stage navigation scene.
        num_obstacles = 6
        size = [0.6, 0.6, 1.0]
        density = 1000.0
        mass = 50.0
        # Leave enough free area for six boxes plus the robot/goal clearance.
        # The room is 10 m wide; using the central 8 m x 7 m region avoids
        # the boundary walls while preventing overly dense initial layouts.
        bounds = [[-4.0, 4.0], [-3.5, 3.5]]
        # Phase-2 pilot support: keep the lower bound at 0.1 m/s and expand
        # the upper bound from 0.5 to 1.5 m/s over the shared curriculum.
        speed_range = list(PHASE2_SPEED_FINAL)
        class curriculum:
            enabled = True
            speed_start = list(PHASE2_SPEED_START)
            speed_steps = PHASE2_SPEED_STEPS
        min_robot_distance = 1.2
        min_goal_distance = 0.8
        obstacle_clearance = 0.1
        static_clearance = 0.1
        max_spawn_attempts = 512

    class motion_estimation:
        # Counterfactual horizon aligned with the 10 Hz exteroception rate.
        gt_horizon = 0.1
        gt_d_safe = 0.20
        gt_kappa = 10.0

    class replay(Go2PosRoughCfg.replay):
        # The existing replay buffer stores only robot state, not obstacle
        # trajectory history. The dynamic task now rewinds its deterministic
        # obstacle trajectories from their start/velocity parameters instead.
        enable_collision_replay = True
        enable_dynamic_obstacle_replay = True


class Go2PosDynamicCfgPPO(Go2PosRoughCfgPPO):
    class runner(Go2PosRoughCfgPPO.runner):
        experiment_name = 'go2_pos_dynamic'
        # Phase-2 starts with a deliberately bounded pilot.  A continuation
        # run must explicitly override this target to 2000 with --resume.
        max_iterations = 350
