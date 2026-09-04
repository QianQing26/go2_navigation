"""Pluggable low-level locomotion controllers."""

from .base_controller import BaseController, ControllerState
from .registry import available_controllers, make_controller, register_controller

# Import built-ins so their decorators run during package import.
from .torchscript_controller import (  # noqa: F401,E402
    OnnxJointPositionController,
    TorchScriptJointPositionController,
)
from .robogauge_controller import (  # noqa: F401,E402
    RoboGaugeJointPositionController,
)

__all__ = [
    "BaseController",
    "ControllerState",
    "OnnxJointPositionController",
    "TorchScriptJointPositionController",
    "RoboGaugeJointPositionController",
    "available_controllers",
    "make_controller",
    "register_controller",
]
