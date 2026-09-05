"""Configuration for the isolated dynamic-obstacle Go2 task."""

from .go2_pos_config import Go2PosRoughCfg, Go2PosRoughCfgPPO


class Go2PosDynamicCfg(Go2PosRoughCfg):
    """First-stage dynamic-obstacle navigation configuration."""

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
        bounds = [[-2.5, 2.5], [-2.0, 2.0]]
        speed_range = [0.5, 2.5]
        min_robot_distance = 1.2
        min_goal_distance = 0.8

    class replay(Go2PosRoughCfg.replay):
        # The existing replay buffer stores only robot state, not obstacle
        # trajectory history. The dynamic task now rewinds its deterministic
        # obstacle trajectories from their start/velocity parameters instead.
        enable_collision_replay = True
        enable_dynamic_obstacle_replay = True


class Go2PosDynamicCfgPPO(Go2PosRoughCfgPPO):
    class runner(Go2PosRoughCfgPPO.runner):
        experiment_name = 'go2_pos_dynamic'
