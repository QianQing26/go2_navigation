"""CPU tests for Phase-4.5 execution-path semantics."""

import unittest

import torch

from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer
from rsl_rl.utils.phase4_5 import (
    barrier_residual,
    distribution,
    reconstruct_execution_stages,
    shadow_reprojection,
    stage_deltas,
    stage_residuals,
    tracking_violation,
    violation_events,
)


class Phase45SemanticsTest(unittest.TestCase):
    def setUp(self):
        self.u = torch.tensor([[4.0, -2.0, 0.3], [1.0, 0.5, -1.5]])
        self.previous = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        self.minimum = torch.tensor([-0.5, -1.0, -1.0])
        self.maximum = torch.tensor([2.0, 1.0, 1.0])

    def test_reconstruction_raw_filter_bounds(self):
        stages = reconstruct_execution_stages(
            self.u, self.previous, 0.5, self.minimum, self.maximum
        )
        self.assertTrue(torch.equal(stages['rawclip'][0], torch.tensor([3.0, -2.0, 0.3])))
        self.assertTrue(torch.equal(stages['filter'][0], torch.tensor([1.5, -1.0, 0.15])))
        self.assertTrue(torch.equal(stages['cmd'][0], torch.tensor([1.5, -1.0, 0.15])))
        self.assertTrue(torch.equal(stages['cmd'][1], torch.tensor([0.75, 0.25, -0.75])))

    def test_beta_one_filter_equals_rawclip(self):
        stages = reconstruct_execution_stages(
            self.u, self.previous, 1.0, self.minimum, self.maximum
        )
        self.assertTrue(torch.equal(stages['filter'], stages['rawclip']))
        self.assertFalse(torch.equal(stages['cmd'], stages['rawclip']))

    def test_stage_delta_vectors_and_events(self):
        stages = reconstruct_execution_stages(
            self.u, self.previous, 0.5, self.minimum, self.maximum
        )
        bound_stages = reconstruct_execution_stages(
            torch.tensor([[4.0, 0.0, 0.0]]),
            torch.tensor([[2.0, 0.0, 0.0]]),
            1.0, self.minimum, self.maximum,
        )
        deltas = stage_deltas(stages)
        self.assertGreater(float(deltas['raw_clamp'].abs().sum()), 0.0)
        self.assertGreater(float(deltas['filter'].abs().sum()), 0.0)
        self.assertGreater(float(stage_deltas(bound_stages)['bound'].abs().sum()), 0.0)

    def test_residual_sign_and_stage_events(self):
        drift = torch.tensor([[-0.1], [0.1]])
        lgh = torch.tensor([[-1.0, 0.0], [-1.0, 0.0]])
        alpha_h = torch.tensor([[1.0], [1.0]])
        actions = {
            'cbf': torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            'rawclip': torch.tensor([[0.8, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            'filter': torch.tensor([[1.2, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            'cmd': torch.tensor([[1.2, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
        residuals = stage_residuals(drift, lgh, alpha_h, actions)
        self.assertAlmostEqual(float(residuals['cbf'][0]), 0.4)
        events = violation_events(residuals)
        self.assertFalse(bool(events['raw_clamp'][0]))
        self.assertTrue(bool(events['filter'][0]))

    def test_tracking_violation_definition(self):
        self.assertTrue(bool(tracking_violation(torch.tensor([[0.1]]), torch.tensor([[-0.1]]))[0]))
        self.assertFalse(bool(tracking_violation(torch.tensor([[-0.1]]), torch.tensor([[-0.2]]))[0]))

    def test_shadow_projection_is_stateless_and_matches_live(self):
        torch.manual_seed(4)
        layer = ExactLSECBFLayer()
        u_bar = torch.randn(2, 3)
        rays = torch.rand(2, 41) * 2.0 + 0.2
        alpha = torch.rand(2, 1) + 0.5
        drift = torch.tensor([[-0.2], [0.1]])
        live = layer(u_bar, rays, alpha, safety_drift=drift)
        snapshot = layer.last_h_comp.clone()
        result = shadow_reprojection(
            layer, live, rays, alpha, drift,
            torch.tensor([-0.5, -1.0, -1.0]),
            torch.tensor([2.0, 1.0, 1.0]),
        )
        self.assertTrue(torch.equal(snapshot, layer.last_h_comp))
        self.assertEqual(tuple(result['action'].shape), (2, 3))

    def test_distribution_includes_requested_percentiles(self):
        result = distribution([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(result['count'], 5)
        self.assertEqual(result['p10'], 1.4)
        self.assertEqual(result['p50'], 3.0)
        self.assertEqual(result['p90'], 4.6)


if __name__ == '__main__':
    unittest.main()
