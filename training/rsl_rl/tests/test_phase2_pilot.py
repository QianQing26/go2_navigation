"""CPU tests for Phase-2 pilot experiment and reporting semantics."""

import os
import tempfile
import types
import unittest

import torch

from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils.phase2 import (
    PHASE2_SPEED_FINAL,
    PHASE2_SPEED_START,
    PHASE2_SPEED_STEPS,
    classify_terminal_outcome,
    curriculum_speed_range,
    outcome_rates,
    remaining_iterations,
    snapshot_control_context,
)


class Phase2PilotTest(unittest.TestCase):
    def test_mutually_exclusive_outcome_priority_and_sum(self):
        self.assertEqual(
            classify_terminal_outcome(True, True, False, False),
            'collision_failure',
        )
        self.assertEqual(
            classify_terminal_outcome(False, True, True, False),
            'safe_success',
        )
        self.assertEqual(
            classify_terminal_outcome(False, False, True, True),
            'timeout_failure',
        )
        self.assertEqual(
            classify_terminal_outcome(False, False, False, True),
            'stuck_failure',
        )
        summary = outcome_rates([
            'collision_failure', 'safe_success', 'timeout_failure',
            'stuck_failure', 'other_failure',
        ])
        self.assertEqual(sum(summary['counts'].values()), 5)
        self.assertAlmostEqual(summary['total_rate'], 1.0)

    def test_curriculum_endpoints(self):
        start = PHASE2_SPEED_START
        final = PHASE2_SPEED_FINAL
        self.assertEqual(PHASE2_SPEED_STEPS, 50000)
        self.assertEqual(curriculum_speed_range(0.0, start, final), start)
        self.assertEqual(curriculum_speed_range(0.5, start, final), (0.1, 1.0))
        self.assertEqual(curriculum_speed_range(1.0, start, final), final)

    def test_weights_only_load_keeps_fresh_optimizer(self):
        torch.manual_seed(12)
        source = torch.nn.Linear(3, 2)
        source_optimizer = torch.optim.Adam(source.parameters(), lr=1.0e-3)
        source_optimizer.zero_grad()
        source(torch.ones(1, 3)).sum().backward()
        source_optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = os.path.join(directory, 'rough.pt')
            torch.save({
                'model_state_dict': source.state_dict(),
                'optimizer_state_dict': source_optimizer.state_dict(),
                'iter': 2000,
                'infos': None,
            }, checkpoint_path)

            target = torch.nn.Linear(3, 2)
            target_optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-3)
            target_optimizer.zero_grad()
            target(torch.ones(1, 3)).sum().backward()
            target_optimizer.step()
            self.assertGreater(len(target_optimizer.state), 0)
            runner = object.__new__(OnPolicyRunner)
            runner.alg = types.SimpleNamespace(
                actor_critic=target, optimizer=target_optimizer
            )
            runner.device = 'cpu'
            runner.current_learning_iteration = 999
            runner.load_weights(checkpoint_path)

            for actual, expected in zip(target.parameters(), source.parameters()):
                self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(len(target_optimizer.state), 0)
            self.assertEqual(runner.current_learning_iteration, 0)

    def test_resume_loads_optimizer_and_preserves_iteration(self):
        source = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(source.parameters(), lr=1.0e-3)
        optimizer.zero_grad()
        source(torch.ones(1, 3)).sum().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = os.path.join(directory, 'pilot_350.pt')
            torch.save({
                'model_state_dict': source.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': 350,
                'infos': None,
            }, checkpoint_path)
            target = torch.nn.Linear(3, 2)
            target_optimizer = torch.optim.Adam(target.parameters(), lr=1.0e-3)
            runner = object.__new__(OnPolicyRunner)
            runner.alg = types.SimpleNamespace(
                actor_critic=target, optimizer=target_optimizer
            )
            runner.device = 'cpu'
            runner.load(checkpoint_path, load_optimizer=True)
            self.assertEqual(runner.current_learning_iteration, 350)
            self.assertGreater(len(target_optimizer.state), 0)
            self.assertEqual(remaining_iterations(2000, 350), 1650)

    def test_ab_config_differs_only_in_safety_context(self):
        # This mirrors the three CLI-generated in-memory configs: all shared
        # training fields are copied unchanged and only the safety context is
        # changed per arm.  The curriculum constants are shared with the
        # dynamic-task config itself.
        common = {
            'reward': {'collision': -4.0, 'termination': -100.0},
            'algorithm': {'num_learning_epochs': 5, 'num_mini_batches': 4},
            'policy': {'init_noise_std': 1.5},
            'num_envs': 2048,
            'seed': 1,
            'max_iterations': 350,
            'curriculum': {
                'speed_start': PHASE2_SPEED_START,
                'speed_final': PHASE2_SPEED_FINAL,
                'speed_steps': PHASE2_SPEED_STEPS,
            },
        }
        configs = []
        for mode in ('original', 'synchronized_static', 'predictive'):
            config = {'shared': common, 'safety': {'mode': mode}}
            if mode == 'predictive':
                config['safety']['estimator_checkpoint'] = 'estimator.pt'
            configs.append(config)
        shared_snapshots = [config['shared'] for config in configs]
        self.assertEqual(shared_snapshots[0], shared_snapshots[1])
        self.assertEqual(shared_snapshots[1], shared_snapshots[2])
        self.assertEqual(
            [config['safety']['mode'] for config in configs],
            ['original', 'synchronized_static', 'predictive'],
        )

    def test_timeseries_context_is_pre_step_snapshot(self):
        env = types.SimpleNamespace(
            exteroception_updated_mask=torch.tensor([False]),
            exteroception_history_count=torch.tensor([3]),
        )
        layer = types.SimpleNamespace(
            last_h_comp=torch.tensor([[0.2]]),
            last_Lgh_u=torch.tensor([[-0.3]]),
            last_alpha_h=torch.tensor([[0.1]]),
            last_nominal_barrier_residual=torch.tensor([[-0.2]]),
            last_eta=torch.tensor([[0.4]]),
            last_intervention_norm=torch.tensor([[0.05]]),
        )
        snapshot = snapshot_control_context(
            env,
            torch.tensor([[0.0]]),
            torch.tensor([[0.2, 1.5]]),
            layer,
        )
        env.exteroception_updated_mask[:] = True
        env.exteroception_history_count[:] = 4
        self.assertFalse(bool(snapshot['exteroception_updated'][0]))
        self.assertEqual(int(snapshot['history_count'][0]), 3)
        self.assertAlmostEqual(float(snapshot['shield_rays_min'][0]), 0.2)
        self.assertAlmostEqual(float(snapshot['shield_rays_max'][0]), 1.5)

    def test_training_safety_metrics_use_update_only_for_online_gt(self):
        runner = object.__new__(OnPolicyRunner)
        runner.safety_mode = 'predictive'
        runner.env = types.SimpleNamespace(
            lse_drift_gt=torch.tensor([[-0.2], [0.1], [-0.1]])
        )
        runner.predictive_runtime = types.SimpleNamespace(
            last_inference_mask=torch.tensor([True, False, True])
        )
        runner.alg = types.SimpleNamespace(actor_critic=types.SimpleNamespace(
            cbf_layer=types.SimpleNamespace(
                last_intervention_norm=torch.tensor([[0.0], [0.2], [0.0]]),
                last_nominal_barrier_residual=torch.tensor([[-0.1], [0.2], [0.1]]),
                last_Lgh_u=torch.tensor([[-0.1], [0.0], [0.2]]),
                last_alpha_h=torch.tensor([[0.0], [0.2], [-0.1]]),
            )
        ))
        stats = runner._new_rollout_safety_stats()
        runner._record_rollout_safety_stats(stats, torch.tensor([[-0.2], [0.4], [-0.1]]))
        metrics = runner._rollout_safety_metrics(stats)
        self.assertAlmostEqual(metrics['intervention_frequency'], 1.0 / 3.0)
        self.assertEqual(metrics['estimator_online_count'], 2)
        self.assertAlmostEqual(metrics['estimator_online_mae'], 0.0)
        self.assertEqual(metrics['estimator_online_dangerous_count'], 2)
        self.assertAlmostEqual(metrics['estimator_online_false_safe_rate'], 0.0)


if __name__ == '__main__':
    unittest.main()
