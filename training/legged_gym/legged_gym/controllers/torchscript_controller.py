"""TorchScript and ONNX implementations of the current Go2 controller."""

import os
from pathlib import Path

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR

from .base_controller import BaseController, ControllerState
from .registry import register_controller


def _resolve_model_path(model_dir, filename, default_dir):
    directory = model_dir or default_dir
    directory = str(directory).replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    directory = os.path.expanduser(directory)
    path = Path(directory) / str(filename)
    return str(path)


class _TorchScriptModel:
    def __init__(self, path, device):
        self.path = path
        self.module = torch.jit.load(path, map_location=device)
        self.module.eval()

    def __call__(self, tensor):
        return self.module(tensor)


class _OnnxModel:
    def __init__(self, path, device):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "The ONNX controller requires the optional 'onnxruntime' package"
            ) from exc

        self.path = path
        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if device.type == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, tensor):
        # ONNX Runtime accepts NumPy arrays.  Keeping this conversion here
        # makes the rest of the controller backend-independent.
        value = tensor.detach().to("cpu").numpy()
        output = self.session.run(None, {self.input_name: value})[0]
        return torch.as_tensor(output, dtype=tensor.dtype, device=tensor.device)


class _HistoryJointPositionController(BaseController):
    """Shared observation pipeline for the existing three-model controller."""

    model_extension = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        loco_cfg = self.env_cfg.loco
        normalization_cfg = self.env_cfg.normalization
        noise_cfg = self.env_cfg.noise

        self.history_length = int(self.env_cfg.env.his_len)
        self.obs_dim = int(getattr(loco_cfg, "num_obs_buf", 45))
        self.obs_history = torch.zeros(
            self.num_envs,
            self.history_length,
            self.obs_dim,
            dtype=torch.float,
            device=self.device,
        )

        loco_scales = loco_cfg.normalization.obs_scales
        self.scale_lin_vel = float(loco_scales.lin_vel)
        self.scale_ang_vel = float(loco_scales.ang_vel)
        self.scale_dof_pos = float(loco_scales.dof_pos)
        self.scale_dof_vel = float(loco_scales.dof_vel)

        self.noise_enabled = bool(noise_cfg.add_noise)
        noise_scales = noise_cfg.noise_scales
        self.noise_vec = torch.cat(
            (
                torch.ones(3) * float(noise_scales.ang_vel),
                torch.ones(3) * float(noise_scales.gravity),
                torch.zeros(3),
                torch.ones(12)
                * float(noise_scales.dof_pos)
                * float(normalization_cfg.obs_scales.dof_pos),
                torch.ones(12)
                * float(noise_scales.dof_vel)
                * float(normalization_cfg.obs_scales.dof_vel),
                torch.zeros(self.num_actions),
            ),
            dim=0,
        ).to(self.device)

        default_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "ctrl_model")
        self.model_dir = getattr(self.cfg, "model_dir", None)
        self._load_models(default_dir)

    def _filename(self, attr_name):
        filename = str(getattr(self.cfg, attr_name))
        if self.model_extension and filename.endswith(".jit"):
            filename = filename[:-4] + self.model_extension
        return filename

    def _load_models(self, default_dir):
        raise NotImplementedError

    def _reindex_joints(self, tensor):
        if self.joint_reindex is None:
            return tensor
        return tensor.index_select(1, self.joint_reindex)

    def build_observation(self, state: ControllerState) -> torch.Tensor:
        """Build the original 45-D SLR observation and its 10-frame history."""

        current = torch.cat(
            (
                state.base_ang_vel * self.scale_ang_vel,
                state.projected_gravity,
                state.nav_command[:, :3]
                * torch.tensor(
                    [self.scale_lin_vel, self.scale_lin_vel, self.scale_ang_vel],
                    device=self.device,
                ),
                self._reindex_joints(
                    (state.dof_pos - state.default_dof_pos) * self.scale_dof_pos
                ),
                self._reindex_joints(state.dof_vel * self.scale_dof_vel),
                state.previous_action,
            ),
            dim=-1,
        )
        # The original controller appended the un-noised yaw-rate component
        # separately to the body model input. Keep that distinction here so
        # extracting the observation pipeline does not alter the policy.
        self._raw_ang_vel_z = state.base_ang_vel[:, 2:] * self.scale_ang_vel

        if current.shape[1] != self.obs_dim:
            raise ValueError(
                f"Controller observation dimension must be {self.obs_dim}, "
                f"got {current.shape[1]}"
            )

        if self.noise_enabled:
            current = current + (2.0 * torch.rand_like(current) - 1.0) * 0.5 * self.noise_vec

        reset_mask = (state.episode_length <= 1).view(-1, 1, 1)
        self.obs_history = torch.where(
            reset_mask,
            torch.stack([current] * self.history_length, dim=1),
            torch.cat((self.obs_history[:, 1:], current.unsqueeze(1)), dim=1),
        )
        return self.obs_history.reshape(self.num_envs, -1)

    def inference(self, observation: torch.Tensor) -> torch.Tensor:
        """Run velocity encoder, latent encoder and the joint policy."""

        base_lin_vel_pred = self.encoder_vel(observation)
        latent = self.encoder_latent(observation)
        current = observation[:, -self.obs_dim :]
        actor_input = torch.cat(
            (base_lin_vel_pred, current, self._raw_ang_vel_z, latent), dim=-1
        )
        return self.body(actor_input)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.obs_history.zero_()
            return
        self.obs_history[env_ids] = 0.0


@register_controller("torchscript")
class TorchScriptJointPositionController(_HistoryJointPositionController):
    """The repository's original frozen TorchScript locomotion controller."""

    def _load_models(self, default_dir):
        self.encoder_vel = _TorchScriptModel(
            _resolve_model_path(
                self.model_dir, self._filename("encoder_vel_model"), default_dir
            ),
            self.device,
        )
        self.encoder_latent = _TorchScriptModel(
            _resolve_model_path(
                self.model_dir, self._filename("encoder_latent_model"), default_dir
            ),
            self.device,
        )
        self.body = _TorchScriptModel(
            _resolve_model_path(
                self.model_dir, self._filename("body_model"), default_dir
            ),
            self.device,
        )


@register_controller("onnx")
class OnnxJointPositionController(_HistoryJointPositionController):
    """ONNX variant using the same three-model interface as the JIT variant.

    The filenames default to the corresponding ``.onnx`` names when the task
    config still contains the current ``.jit`` defaults.  ``onnxruntime`` is an
    optional dependency and is imported only when this controller is selected.
    """

    model_extension = ".onnx"

    def _load_models(self, default_dir):
        self.encoder_vel = _OnnxModel(
            _resolve_model_path(
                self.model_dir, self._filename("encoder_vel_model"), default_dir
            ),
            self.device,
        )
        self.encoder_latent = _OnnxModel(
            _resolve_model_path(
                self.model_dir, self._filename("encoder_latent_model"), default_dir
            ),
            self.device,
        )
        self.body = _OnnxModel(
            _resolve_model_path(
                self.model_dir, self._filename("body_model"), default_dir
            ),
            self.device,
        )
