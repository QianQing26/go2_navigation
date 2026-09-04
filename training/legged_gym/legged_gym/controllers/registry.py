"""Registry and factory for low-level controllers."""


_CONTROLLERS = {}


def register_controller(name):
    """Register a controller class under a configuration-facing name."""

    key = str(name).lower()

    def decorator(controller_cls):
        if key in _CONTROLLERS and _CONTROLLERS[key] is not controller_cls:
            raise ValueError(f"Controller '{name}' is already registered")
        _CONTROLLERS[key] = controller_cls
        return controller_cls

    return decorator


def available_controllers():
    """Return registered controller names in deterministic order."""

    return tuple(sorted(_CONTROLLERS))


def make_controller(
    name,
    controller_cfg,
    env_cfg,
    num_envs,
    num_actions,
    device,
    joint_reindex=None,
):
    """Construct a controller from the task configuration."""

    key = str(name).lower()
    if key not in _CONTROLLERS:
        choices = ", ".join(available_controllers())
        raise ValueError(f"Unknown controller '{name}'. Available: {choices}")

    return _CONTROLLERS[key](
        controller_cfg=controller_cfg,
        env_cfg=env_cfg,
        num_envs=num_envs,
        num_actions=num_actions,
        device=device,
        joint_reindex=joint_reindex,
    )
