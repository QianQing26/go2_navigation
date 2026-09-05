"""Smoke-test a configured low-level joint-position controller in Isaac Gym.

The test deliberately bypasses the navigation PPO policy.  It sends a fixed
sequence of ``[vx, vy, wz]`` commands through the task's normal ``step`` path,
so the exercised chain is:

    navigation command -> controller observation/inference/action -> PD -> sim

This is a wiring/numerical-validity test, not a locomotion-performance
benchmark.  The default task is ``go2_pos_rough`` and therefore uses the
controller selected by that task configuration (currently ``robogauge``).
"""

import argparse
import sys
from contextlib import nullcontext

from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403 - registers all tasks
from legged_gym.controllers import available_controllers
from legged_gym.utils import get_args, task_registry
import torch


def _parse_script_args():
    """Parse options owned by this script, preserving Isaac Gym options."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--phase_steps",
        type=int,
        default=150,
        help="Number of simulation steps for each command phase.",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=25,
        help="Print a controller status line every N steps.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the same test without opening the Isaac Gym viewer.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        choices=available_controllers(),
        help="Override the controller selected by the task configuration.",
    )
    script_args, remaining_args = parser.parse_known_args()
    # get_args() delegates to gymutil.parse_arguments(), which should only see
    # Isaac Gym and the repository's standard command-line arguments.
    sys.argv = [sys.argv[0]] + remaining_args
    return script_args


def _configure_for_controller_test(env_cfg):
    """Make the test deterministic and avoid unrelated episode resets."""

    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 60
    # The controller test only needs the robot and ground.  Navigation ray
    # debug lines are intentionally disabled because they can make the first
    # viewer frame very slow and obscure whether reset completed.
    env_cfg.env.debug_viz = False

    # Keep the navigation task's terrain-backed perception available, but make
    # every terrain cell geometrically flat.  A plane mesh would bypass the
    # terrain height/ray buffers used by LeggedRobotPos.
    env_cfg.terrain.mesh_type = "trimesh"
    env_cfg.terrain.terrain_types = ["flat"]
    env_cfg.terrain.terrain_proportions = [1.0]
    env_cfg.terrain.measure_heights = True
    # Room curriculum placement requires an obstacle-containing path and would
    # loop forever on a flat terrain.  The flat controller test has no terrain
    # curriculum by design.
    env_cfg.terrain.curriculum = False

    # The low-level policy has its own noise switch.  Disable both sources so
    # repeating a command produces a reproducible controller signal.
    env_cfg.noise.add_noise = False
    if hasattr(env_cfg, "controller"):
        env_cfg.controller.add_noise = False

    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.asset.terminate_after_contacts_on = []

    # Use one environment and avoid the collision replay path: replay is not
    # part of this controller test.
    if hasattr(env_cfg, "replay"):
        env_cfg.replay.enable_collision_replay = False
        if hasattr(env_cfg.replay, "enable_dynamic_obstacle_replay"):
            env_cfg.replay.enable_dynamic_obstacle_replay = False

    # One flat cell is enough for this test.  LeggedRobotPos handles the
    # one-row case explicitly when selecting the robot/goal origins.
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 1

    # Match the deterministic center of the single flat cell before reset;
    # _focus_camera_on_robot will refine this after the root state is written.
    env_cfg.viewer.pos = [9.0, 1.0, 5.0]
    env_cfg.viewer.lookat = [5.0, 5.0, 0.0]


def _command_sequence():
    """Return direction-complete navigation command phases."""

    return [
        ("stop", (0.0, 0.0, 0.0)),
        ("forward", (1.0, 0.0, 0.0)),
        ("backward", (-0.5, 0.0, 0.0)),
        ("lateral_left", (0.0, 0.6, 0.0)),
        ("lateral_right", (0.0, -0.6, 0.0)),
        ("turn_left", (0.0, 0.0, 0.8)),
        ("turn_right", (0.0, 0.0, -0.8)),
        ("stop", (0.0, 0.0, 0.0)),
    ]


def _focus_camera_on_robot(env):
    """Aim the viewer at the robot after its randomized spawn is known."""

    if env.viewer is None:
        return
    robot_pos = env.root_states[0, :3].detach().cpu().tolist()
    env.set_camera(
        [robot_pos[0] + 4.0, robot_pos[1] - 4.0, robot_pos[2] + 3.0],
        [robot_pos[0], robot_pos[1], robot_pos[2] + 0.2],
    )


def _finite(name, tensor):
    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
        raise RuntimeError("{} contains NaN/Inf or is not a Tensor".format(name))


def _status(env, phase_name, command, phase_step):
    """Print only signals useful for checking the controller wiring."""

    filtered = env.nav_actions_after_clip[0].detach().cpu().tolist()
    low_action = env.actions_orig[0].detach()
    base_lin = env.base_lin_vel[0].detach().cpu().tolist()
    base_ang = env.base_ang_vel[0].detach().cpu().tolist()
    print(
        "[{} {:>3}/{:<3}] cmd=({:+.2f},{:+.2f},{:+.2f}) "
        "filtered=({:+.2f},{:+.2f},{:+.2f}) "
        "action[min,max,mean_abs]=({:+.3f},{:+.3f},{:.3f}) "
        "base_v=({:+.2f},{:+.2f}) wz={:+.2f}".format(
            phase_name,
            phase_step,
            env._controller_test_phase_steps,
            *command,
            *filtered,
            float(low_action.min().cpu()),
            float(low_action.max().cpu()),
            float(low_action.abs().mean().cpu()),
            base_lin[0],
            base_lin[1],
            base_ang[2],
        ),
        flush=True,
    )


def run(args, script_args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    _configure_for_controller_test(env_cfg)
    if script_args.controller is not None:
        env_cfg.controller.name = script_args.controller

    # A controller test is intentionally single-environment even if a stale
    # --num_envs value is present in a shell alias or launch script.
    args.num_envs = None
    args.headless = script_args.headless
    args.wandb = False

    print(
        "[controller-test] task_template={} terrain={} mesh={} controller={} "
        "device={} headless={}".format(
            args.task,
            env_cfg.terrain.terrain_types[0],
            env_cfg.terrain.mesh_type,
            env_cfg.controller.name,
            args.sim_device,
            args.headless,
        ),
        flush=True,
    )
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env._controller_test_phase_steps = script_args.phase_steps

    try:
        if not hasattr(env, "controller"):
            raise RuntimeError("The task did not create a low-level controller")
        controller_path = getattr(env.controller, "policy_path", None)
        if controller_path is None and hasattr(env.controller, "model_dir"):
            controller_path = getattr(env.controller, "model_dir", None)
            controller_path = controller_path or "<default ctrl_model directory>"
        controller_path = controller_path or "configured controller backend"
        joint_mapping = getattr(env.controller, "joint_reindex", None)
        if joint_mapping is None:
            joint_mapping = "identity (Go2 URDF/model order)"
        else:
            joint_mapping = joint_mapping.detach().cpu().tolist()
        print(
            "[controller-test] loaded {} ({}) joint_mapping={}".format(
                env.controller.__class__.__name__,
                controller_path,
                joint_mapping,
            ),
            flush=True,
        )

        # BaseTask.reset() also performs one zero-command warm-up step.  This
        # is useful for stateful controllers because it verifies reset() and
        # first inference before any non-zero command is sent.
        print("[controller-test] resetting environment", flush=True)
        obs, _ = env.reset()
        _finite("reset observation", obs)
        _focus_camera_on_robot(env)
        print("[controller-test] reset completed", flush=True)

        total_steps = 0
        context = torch.no_grad() if torch.is_grad_enabled() else nullcontext()
        with context:
            for phase_name, command in _command_sequence():
                command_tensor = torch.tensor(
                    command, dtype=torch.float32, device=env.device
                ).repeat(env.num_envs, 1)
                print(
                    "[controller-test] phase={} command=({:+.2f}, {:+.2f}, {:+.2f})".format(
                        phase_name, *command
                    ),
                    flush=True,
                )

                for phase_step in range(1, script_args.phase_steps + 1):
                    obs, _, _, _, _ = env.step(command_tensor)
                    total_steps += 1

                    # These are the exact tensors at the controller boundary
                    # and at the PD action boundary in LeggedRobotPos.step().
                    _finite("observation", obs)
                    _finite("filtered navigation command", env.nav_actions_after_clip)
                    _finite("controller joint action", env.actions_orig)
                    _finite("dof position", env.dof_pos)
                    _finite("dof velocity", env.dof_vel)
                    if tuple(env.nav_actions_orig.shape) != (1, 3):
                        raise RuntimeError(
                            "Navigation command shape is {}, expected (1, 3)".format(
                                tuple(env.nav_actions_orig.shape)
                            )
                        )
                    if tuple(env.actions_orig.shape) != (1, env.num_actions):
                        raise RuntimeError(
                            "Controller action shape is {}, expected (1, {})".format(
                                tuple(env.actions_orig.shape), env.num_actions
                            )
                        )
                    filtered_low = env.nav_actions_after_clip < env.nav_clip_min - 1e-5
                    filtered_high = env.nav_actions_after_clip > env.nav_clip_max + 1e-5
                    if bool((filtered_low | filtered_high).any()):
                        raise RuntimeError(
                            "Filtered navigation command exceeded configured limits"
                        )

                    # The task clips the raw command before filtering; because
                    # all test commands are within the configured limits, this
                    # should be an exact pass-through.
                    raw_error = (env.nav_actions_orig - command_tensor).abs().max()
                    if float(raw_error) > 1e-5:
                        raise RuntimeError(
                            "Navigation command was changed before filtering: {}".format(
                                float(raw_error)
                            )
                        )

                    if phase_step == 1 or phase_step % script_args.print_every == 0:
                        _status(env, phase_name, command, phase_step)

        print(
            "[controller-test] PASS: {} steps, all controller observations/actions "
            "were finite and shaped correctly.".format(total_steps),
            flush=True,
        )
    finally:
        if getattr(env, "viewer", None) is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    script_args = _parse_script_args()
    if script_args.phase_steps <= 0:
        raise ValueError("--phase_steps must be positive")
    if script_args.print_every <= 0:
        raise ValueError("--print_every must be positive")
    run(get_args(), script_args)
