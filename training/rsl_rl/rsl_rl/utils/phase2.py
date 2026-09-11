"""Pure helpers for the Phase-2 predictive-CBF pilot.

Keeping these small pieces independent from Isaac Gym makes the experiment
semantics easy to test on CPU and prevents reporting logic from being coupled
to simulator termination implementation details.
"""

from collections import Counter

import torch


PHASE2_SAFETY_MODES = (
    'original',
    'synchronized_static',
    'predictive',
)

PHASE2_SPEED_START = (0.1, 0.5)
PHASE2_SPEED_FINAL = (0.1, 1.5)
PHASE2_SPEED_STEPS = 50000

TERMINAL_OUTCOMES = (
    'collision_failure',
    'safe_success',
    'timeout_failure',
    'stuck_failure',
    'other_failure',
)


def classify_terminal_outcome(collision, goal_reached, timeout, stuck):
    """Return one mutually-exclusive terminal outcome.

    Collision has highest priority because a terminal frame can also satisfy
    the goal condition.  This function intentionally does not alter simulator
    flags or reward semantics.
    """

    if bool(collision):
        return 'collision_failure'
    if bool(goal_reached):
        return 'safe_success'
    if bool(timeout):
        return 'timeout_failure'
    if bool(stuck):
        return 'stuck_failure'
    return 'other_failure'


def outcome_rates(outcomes):
    """Return mutually-exclusive counts and rates for terminal outcomes."""

    counts = Counter(outcomes)
    total = float(len(outcomes))
    rates = {
        outcome: counts.get(outcome, 0) / total if total else 0.0
        for outcome in TERMINAL_OUTCOMES
    }
    return {
        'counts': {outcome: counts.get(outcome, 0) for outcome in TERMINAL_OUTCOMES},
        'rates': rates,
        'total_rate': sum(rates.values()),
    }


def curriculum_speed_range(progress, speed_start, speed_final):
    """Linearly interpolate a curriculum support and clamp progress to [0, 1]."""

    progress = min(max(float(progress), 0.0), 1.0)
    start_min, start_max = [float(value) for value in speed_start]
    final_min, final_max = [float(value) for value in speed_final]
    return (
        start_min + progress * (final_min - start_min),
        start_max + progress * (final_max - start_max),
    )


def remaining_iterations(max_iterations, current_iteration):
    """Translate a total target iteration into work left for ``runner.learn``."""

    max_iterations = int(max_iterations)
    current_iteration = int(current_iteration)
    if max_iterations < 0 or current_iteration < 0:
        raise ValueError('iteration targets must be non-negative')
    if current_iteration > max_iterations:
        raise ValueError(
            'checkpoint iteration {} exceeds target {}'.format(
                current_iteration, max_iterations
            )
        )
    return max_iterations - current_iteration


def snapshot_control_context(env, safety_drift, shield_rays, layer):
    """Snapshot all values used to produce one control action.

    The function is deliberately Isaac-Gym agnostic: it only consumes the
    tensor attributes exposed by the navigation environment and CBF layer.
    """

    def snapshot(name, fallback):
        value = getattr(layer, name, fallback)
        return value.reshape(-1).detach().clone()

    return {
        'exteroception_updated': env.exteroception_updated_mask.detach().clone(),
        'history_count': env.exteroception_history_count.detach().clone(),
        'safety_drift': safety_drift.reshape(-1).detach().clone(),
        'shield_rays_min': shield_rays.min(dim=-1).values.detach().clone(),
        'shield_rays_max': shield_rays.max(dim=-1).values.detach().clone(),
        'h_comp': snapshot('last_h_comp', torch.zeros_like(safety_drift)),
        'Lgh_u': snapshot('last_Lgh_u', torch.zeros_like(safety_drift)),
        'alpha_h': snapshot('last_alpha_h', torch.zeros_like(safety_drift)),
        'residual': snapshot('last_nominal_barrier_residual', safety_drift),
        'eta': snapshot('last_eta', safety_drift),
        'intervention_norm': snapshot('last_intervention_norm', safety_drift),
    }
