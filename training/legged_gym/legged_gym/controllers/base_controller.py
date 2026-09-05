"""Interfaces shared by vectorized joint-position controllers.

The navigation policy produces a small command (currently vx, vy and yaw).
This package turns that command and the simulator state into the normalized
12-dimensional joint-position action consumed by ``LeggedRobot.step``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class ControllerState:
    """The simulator state exposed to a low-level controller.

    All tensors are batched along dimension zero.  The state deliberately
    contains only common simulator quantities; observation layout and history
    belong to the concrete controller.
    """

    base_ang_vel: torch.Tensor
    projected_gravity: torch.Tensor
    nav_command: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    default_dof_pos: torch.Tensor
    previous_action: torch.Tensor
    episode_length: torch.Tensor


class BaseController(ABC):
    """Base class for a batched, joint-position low-level controller."""

    def __init__(
        self,
        controller_cfg,
        env_cfg,
        num_envs: int,
        num_actions: int,
        device,
        joint_reindex: Optional[Sequence[int]] = None,
    ):
        self.cfg = controller_cfg
        self.env_cfg = env_cfg
        self.num_envs = int(num_envs)
        self.num_actions = int(num_actions)
        self.device = torch.device(device)
        self.joint_reindex = None
        if joint_reindex is not None:
            self.joint_reindex = torch.as_tensor(
                list(joint_reindex), dtype=torch.long, device=self.device
            )

    @abstractmethod
    def build_observation(self, state: ControllerState) -> torch.Tensor:
        """Convert simulator state into the model input."""

    @abstractmethod
    def inference(self, observation: torch.Tensor) -> torch.Tensor:
        """Run the controller model and return its raw output."""

    def post_process_action(self, model_output: torch.Tensor) -> torch.Tensor:
        """Convert model output into the environment's action convention."""

        action = model_output
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device)

        if action.ndim != 2 or action.shape[0] != self.num_envs:
            raise ValueError(
                f"Controller action must have shape ({self.num_envs}, N), "
                f"got {tuple(action.shape)}"
            )
        if action.shape[1] != self.num_actions:
            raise ValueError(
                f"Controller action dimension must be {self.num_actions}, "
                f"got {action.shape[1]}"
            )

        clip_actions = getattr(self.env_cfg.normalization, "clip_actions", None)
        if clip_actions is not None:
            action = torch.clamp(action, -float(clip_actions), float(clip_actions))
        return action

    def action_to_sim(self, action: torch.Tensor) -> torch.Tensor:
        """Map controller/model joint order to Isaac Gym joint order.

        The legacy SEA-Nav TorchScript controller uses the configured
        sim-to-policy permutation.  Controllers imported from another
        project may already use the Isaac/URDF order and can set
        ``joint_reindex`` to ``None``.
        """

        if self.joint_reindex is None:
            return action
        return action.index_select(1, self.joint_reindex)

    @torch.no_grad()
    def get_action(self, state: ControllerState) -> torch.Tensor:
        """Build input, infer, and return a simulator-ready joint action."""

        observation = self.build_observation(state)
        model_output = self.inference(observation)
        return self.post_process_action(model_output)

    def reset(self, env_ids=None):
        """Reset controller-owned recurrent/history state.

        Stateless controllers can keep the default implementation.  A
        controller with history should override this method and accept either
        all environments (``None``) or a tensor/list of environment ids.
        """

        return None
