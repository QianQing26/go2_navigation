"""RoboGauge's batchable MoE-CTS joint-position controller."""

import os
from pathlib import Path

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR

from .base_controller import BaseController, ControllerState
from .registry import register_controller


def _resolve_model_path(path):
    """Resolve the configured checkpoint without copying the binary here."""

    path = str(path).replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    return str(Path(os.path.expandvars(os.path.expanduser(path))))


@register_controller("robogauge")
class RoboGaugeJointPositionController(BaseController):
    """Run the exported ``go2_moe_cts`` policy.

    The exported policy accepts one current 45-D observation and keeps the
    five-step observation history internally, so history is not duplicated by
    this wrapper.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_path = getattr(self.cfg, "model_path", None)
        if not model_path:
            raise ValueError("The 'robogauge' controller requires cfg.model_path")
        model_path = _resolve_model_path(model_path)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"RoboGauge policy checkpoint was not found: {model_path}"
            )

        self.policy_path = model_path
        self.policy = torch.jit.load(model_path, map_location=self.device)
        self.policy.eval()

        self.obs_dim = int(getattr(self.cfg, "obs_dim", 45))
        self.history_length = int(getattr(self.cfg, "history_length", 5))
        if self.obs_dim != 45:
            raise ValueError(
                f"RoboGauge policy expects a 45-D observation, got {self.obs_dim}"
            )
        exported_obs_dim = getattr(self.policy, "obs_dim", None)
        if exported_obs_dim is not None and int(exported_obs_dim) != self.obs_dim:
            raise ValueError(
                f"RoboGauge checkpoint expects {int(exported_obs_dim)} observations, "
                f"but config provides {self.obs_dim}"
            )
        exported_history_length = getattr(self.policy, "history_length", None)
        if exported_history_length is not None and int(exported_history_length) != self.history_length:
            raise ValueError(
                f"RoboGauge checkpoint uses history length {int(exported_history_length)}, "
                f"but config provides {self.history_length}"
            )

        # Values used by go2_rl_gym's GO2Cfg.
        self.scale_ang_vel = float(getattr(self.cfg, "obs_scale_ang_vel", 0.25))
        self.scale_dof_pos = float(getattr(self.cfg, "obs_scale_dof_pos", 1.0))
        self.scale_dof_vel = float(getattr(self.cfg, "obs_scale_dof_vel", 0.05))
        command_scale = getattr(self.cfg, "command_scale", [2.0, 2.0, 0.25])
        self.command_scale = torch.as_tensor(
            command_scale, dtype=torch.float32, device=self.device
        )
        if self.command_scale.numel() != 3:
            raise ValueError("RoboGauge command_scale must contain three values")

        # Source training noise: ang_vel, gravity, dof_pos, dof_vel.
        noise_scales = getattr(self.cfg, "noise_scales", [0.2, 0.05, 0.01, 1.5])
        self.noise_vec = torch.cat(
            (
                torch.ones(3) * float(noise_scales[0]) * self.scale_ang_vel,
                torch.ones(3) * float(noise_scales[1]),
                torch.zeros(3),
                torch.ones(12) * float(noise_scales[2]) * self.scale_dof_pos,
                torch.ones(12) * float(noise_scales[3]) * self.scale_dof_vel,
                torch.zeros(12),
            ),
            dim=0,
        ).to(self.device)
        self.noise_enabled = bool(getattr(self.cfg, "add_noise", False))
        self.noise_level = float(getattr(self.cfg, "noise_level", 1.0))

    def _reindex_joints(self, tensor):
        if self.joint_reindex is None:
            return tensor
        return tensor.index_select(1, self.joint_reindex)

    def build_observation(self, state: ControllerState) -> torch.Tensor:
        """Build the exact 45-D observation used by the source Go2 policy."""

        observation = torch.cat(
            (
                state.base_ang_vel * self.scale_ang_vel,
                state.projected_gravity,
                state.nav_command[:, :3] * self.command_scale,
                self._reindex_joints(
                    (state.dof_pos - state.default_dof_pos) * self.scale_dof_pos
                ),
                self._reindex_joints(state.dof_vel * self.scale_dof_vel),
                state.previous_action,
            ),
            dim=-1,
        )
        if observation.shape[1] != self.obs_dim:
            raise ValueError(
                f"RoboGauge observation dimension must be {self.obs_dim}, "
                f"got {observation.shape[1]}"
            )

        if self.noise_enabled:
            observation = observation + (
                2.0 * torch.rand_like(observation) - 1.0
            ) * self.noise_level * self.noise_vec
        return observation

    def inference(self, observation: torch.Tensor) -> torch.Tensor:
        """Run the stateful batchable TorchScript policy."""

        output = self.policy(observation)
        # Tolerate exporters that wrap the action in a one-element tuple.
        if isinstance(output, (tuple, list)):
            if len(output) != 1:
                raise ValueError(
                    "RoboGauge policy must return one action tensor, "
                    f"got {len(output)} outputs"
                )
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"RoboGauge policy returned {type(output).__name__}, expected Tensor"
            )
        return output

    def reset(self, env_ids=None):
        """Reset recurrent state, preserving other vectorized environments."""

        if env_ids is None:
            reset = getattr(self.policy, "reset", None)
            if reset is None:
                raise RuntimeError("RoboGauge policy does not expose reset()")
            reset()
            return

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        history = getattr(self.policy, "history", None)
        if (
            isinstance(history, torch.Tensor)
            and history.ndim == 3
            and history.shape[0] == self.num_envs
        ):
            history.index_fill_(0, env_ids, 0.0)
            return

        # Before first inference the exported module owns a one-row
        # placeholder; it resizes to the vectorized batch on first call.
        reset = getattr(self.policy, "reset", None)
        if reset is None:
            raise RuntimeError("RoboGauge policy does not expose reset()")
        reset()
