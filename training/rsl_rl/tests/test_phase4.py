"""CPU tests for Phase-4 oracle semantics and shadow diagnostics."""

import unittest

import torch

from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer
from rsl_rl.utils.phase4 import (
    extract_pre_collision_window,
    false_safe_decision,
    lse_drift_from_fused_rays,
    primary_failure_label,
    select_multi_horizon_drift,
    warning_time,
)


class Phase4SemanticsTest(unittest.TestCase):
    def test_gt_01_matches_existing_lse_formula(self):
        current = torch.tensor([[0.4, 1.0, 2.0]])
        future = torch.tensor([[0.2, 0.9, 2.0]])
        actual = lse_drift_from_fused_rays(
            current, future, 0.1, d_safe=0.20, kappa=10.0
        )
        expected = (
            -torch.logsumexp(-10.0 * (future - 0.20), dim=-1, keepdim=True) / 10.0
            + torch.logsumexp(-10.0 * (current - 0.20), dim=-1, keepdim=True) / 10.0
        ) / 0.1
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-7))

    def test_static_rays_are_held_in_counterfactual(self):
        current = torch.tensor([[0.3, 0.5, 2.0]])
        future_dynamic = torch.tensor([[0.1, 0.8, 2.0]])
        static = torch.tensor([[0.7, 0.7, 0.7]])
        future_fused = torch.minimum(static, future_dynamic)
        self.assertTrue(torch.equal(future_fused, torch.tensor([[0.1, 0.7, 0.7]])))

    def test_multi_horizon_uses_minimum_drift_and_selected_horizon(self):
        drift, horizon = select_multi_horizon_drift({0.1: 0.2, 0.3: -0.4, 0.5: -0.1})
        self.assertEqual(drift, -0.4)
        self.assertEqual(horizon, 0.3)

    def test_false_safe_and_warning_time(self):
        self.assertTrue(false_safe_decision(0.01, -0.01))
        self.assertFalse(false_safe_decision(-0.01, -0.01))
        rows = [
            {'time_s': 0.0, 'residual': 0.2},
            {'time_s': 0.1, 'residual': -0.1},
        ]
        self.assertAlmostEqual(warning_time(0.5, rows, 'residual'), 0.4)

    def test_collision_window_is_last_100_samples_and_annotated(self):
        rows = [{'time_s': index * 0.02, 'value': index} for index in range(120)]
        selected = extract_pre_collision_window(rows, 2.4, max_samples=100)
        self.assertEqual(len(selected), 100)
        self.assertEqual(selected[0]['value'], 20)
        self.assertAlmostEqual(selected[-1]['time_to_collision_s'], 0.02)

    def test_shadow_projection_matches_live_and_does_not_mutate(self):
        torch.manual_seed(3)
        layer = ExactLSECBFLayer()
        u_bar = torch.randn(2, 3)
        rays = torch.rand(2, 41) * 2.0 + 0.2
        alpha = torch.rand(2, 1) + 0.5
        drift = torch.tensor([[-0.2], [0.1]])
        live = layer(u_bar, rays, alpha, safety_drift=drift)
        snapshot = layer.last_intervention_norm.clone()
        shadow = layer.project(u_bar, rays, alpha, safety_drift=drift)
        self.assertTrue(torch.allclose(live, shadow['safe_action'], atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.equal(snapshot, layer.last_intervention_norm))

    def test_oracle_update_cadence_can_be_zoh(self):
        cache = torch.tensor([[0.2]])
        update_mask = [True, False, False, False, False, True]
        values = []
        for index, update in enumerate(update_mask):
            if update:
                cache = torch.tensor([[float(index)]])
            values.append(cache.clone())
        held = [float(value.reshape(-1)[0]) for value in values]
        self.assertEqual(held, [0.0, 0.0, 0.0, 0.0, 0.0, 5.0])

    def test_primary_label_precedence(self):
        self.assertEqual(
            primary_failure_label(['F3_short_horizon', 'F1_estimator_decision_miss']),
            'F1_estimator_decision_miss',
        )


if __name__ == '__main__':
    unittest.main()
