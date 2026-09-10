"""CPU regression tests for Phase-1 predictive CBF integration."""

import os
import unittest

import torch

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.cbf_actor_critic import DifferentiableSafeActorCritic
from rsl_rl.modules.cbf_lse_layer import ExactLSECBFLayer
from rsl_rl.storage.rollout_storage import RolloutStorage
from motion_estimator.runtime import PredictiveDriftRuntime


REPOSITORY_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..', '..')
)
ESTIMATOR_CHECKPOINT = os.path.join(
    REPOSITORY_ROOT,
    'motion_estimator/artifacts/20260909_162237_v1_1_C/best.pt',
)


class _FakeEnv:
    def __init__(self, num_envs=3, history_length=10, num_rays=41):
        self.num_envs = num_envs
        self.rays_hist = torch.full(
            (num_envs, history_length, num_rays), 2.0
        )
        self.motion_ego_hist = torch.zeros(num_envs, history_length, 3)
        self.exteroception_updated_mask = torch.zeros(num_envs, dtype=torch.bool)
        self.exteroception_history_count = torch.zeros(num_envs, dtype=torch.long)


def _legacy_cbf(layer, u_bar, lidar_dists, alpha):
    u_2d = u_bar[:, :2]
    yaw_rate = u_bar[:, 2:]
    h_i = lidar_dists - layer.d_safe
    min_h, _ = torch.min(h_i, dim=1, keepdim=True)
    h_comp = min_h - (1.0 / layer.kappa) * torch.log(
        torch.sum(torch.exp(-layer.kappa * (h_i - min_h)), dim=1, keepdim=True)
    )
    lambda_i = torch.exp(-layer.kappa * (h_i - h_comp)).unsqueeze(-1)
    lgh = -torch.sum(
        lambda_i * layer.ray_unit_vectors.unsqueeze(0), dim=1
    )
    lgh_u = torch.sum(lgh * u_2d, dim=1, keepdim=True)
    eta = -(lgh_u + alpha * h_comp) / (
        torch.sum(lgh.square(), dim=1, keepdim=True) + layer.damping_factor
    )
    return torch.cat((u_2d + torch.relu(eta) * lgh, yaw_rate), dim=-1)


class PredictiveCbfIntegrationTest(unittest.TestCase):
    def test_backward_compatibility_and_zero_drift(self):
        torch.manual_seed(7)
        layer = ExactLSECBFLayer()
        u_bar = torch.randn(4, 3)
        rays = torch.rand(4, 41) * 2.8 + 0.1
        alpha = torch.rand(4, 1) + 0.5
        expected = _legacy_cbf(layer, u_bar, rays, alpha)
        actual = layer(u_bar, rays, alpha)
        zero = layer(u_bar, rays, alpha, safety_drift=torch.zeros(4, 1))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(actual, zero, atol=1e-7, rtol=1e-7))
        for name in (
            'last_h_comp', 'last_Lgh', 'last_Lgh_u', 'last_safety_drift',
            'last_alpha_h', 'last_eta', 'last_nominal_barrier_residual',
            'last_intervention_norm',
        ):
            self.assertFalse(getattr(layer, name).requires_grad)

    def test_negative_drift_increases_safety_correction(self):
        layer = ExactLSECBFLayer()
        u_bar = torch.zeros(1, 3)
        rays = torch.full((1, 41), 3.0)
        rays[:, 20] = 0.21
        alpha = torch.ones(1, 1)
        layer(u_bar, rays, alpha, safety_drift=torch.zeros(1, 1))
        eta_static = layer.last_eta.clone()
        intervention_static = layer.last_intervention_norm.clone()
        layer(u_bar, rays, alpha, safety_drift=torch.tensor([[-1.0]]))
        self.assertGreater(float(layer.last_eta), float(eta_static))
        self.assertGreater(
            float(layer.last_intervention_norm), float(intervention_static)
        )

    def test_actor_distribution_context_consistency_and_checkpoint_load(self):
        torch.manual_seed(3)
        actor = DifferentiableSafeActorCritic(
            num_actions=3, num_props=12, num_rays=41, his_len=10,
        )
        obs = torch.randn(5, 55 * 10)
        _, _, _, log_rays, _ = actor.extract(obs)
        shield_rays = torch.exp2(log_rays)
        zeros = torch.zeros(5, 1)
        old_mean = actor.forward(obs)
        context_mean = actor.forward(
            obs, safety_drift=zeros, shield_rays=shield_rays
        )
        self.assertTrue(torch.allclose(old_mean, context_mean, atol=1e-6, rtol=1e-6))

        if os.path.isfile(os.path.join(REPOSITORY_ROOT, 'training/legged_gym/logs')):
            policy_candidates = []
            for root, _, files in os.walk(
                os.path.join(REPOSITORY_ROOT, 'training/legged_gym/logs')
            ):
                for filename in files:
                    if filename.startswith('model_') and filename.endswith('.pt'):
                        policy_candidates.append(os.path.join(root, filename))
            if policy_candidates:
                checkpoint = torch.load(policy_candidates[0], map_location='cpu')
                actor.load_state_dict(checkpoint['model_state_dict'], strict=True)

    def test_storage_context_round_trip_and_ppo_distribution(self):
        storage = RolloutStorage(
            num_envs=2, num_transitions_per_env=2, obs_shape=[550],
            actions_shape=[3], device='cpu', shield_rays_shape=(41,),
        )
        transition = RolloutStorage.Transition()
        transition.observations = torch.randn(2, 550)
        transition.next_observations = torch.randn(2, 550)
        transition.actions = torch.randn(2, 3)
        transition.rewards = torch.ones(2)
        transition.dones = torch.zeros(2)
        transition.values = torch.zeros(2, 1)
        transition.actions_log_prob = torch.zeros(2)
        transition.action_mean = torch.zeros(2, 3)
        transition.action_sigma = torch.ones(2, 3)
        transition.safety_drift = torch.tensor([[0.1], [0.2]])
        transition.next_safety_drift = torch.tensor([[0.3], [0.4]])
        transition.shield_rays = torch.full((2, 41), 1.0)
        transition.next_shield_rays = torch.full((2, 41), 2.0)
        storage.add_transitions(transition)
        batch = next(storage.mini_batch_generator(1, 1))
        self.assertEqual(tuple(batch[9].shape), (4, 1))
        self.assertEqual(tuple(batch[10].shape), (4, 1))
        self.assertEqual(tuple(batch[11].shape), (4, 41))
        self.assertEqual(tuple(batch[12].shape), (4, 41))
        self.assertTrue(torch.equal(storage.safety_drift[0], transition.safety_drift))
        self.assertTrue(torch.equal(storage.next_shield_rays[0], transition.next_shield_rays))

        actor = DifferentiableSafeActorCritic(
            num_actions=3, num_props=12, num_rays=41, his_len=10,
        )
        ppo = PPO(actor, device='cpu', num_learning_epochs=1, num_mini_batches=1)
        ppo.init_storage(2, 1, [550], [3], shield_rays_shape=(41,))
        obs = torch.randn(2, 550)
        drift = torch.zeros(2, 1)
        rays = torch.ones(2, 41)
        ppo.act(obs, obs, safety_drift=drift, shield_rays=rays)
        stored_mean = ppo.transition.action_mean.clone()
        actor.update_distribution(obs, safety_drift=drift, shield_rays=rays)
        self.assertTrue(torch.allclose(stored_mean, actor.action_mean))

    def test_ppo_update_consumes_context_batches(self):
        torch.manual_seed(11)
        actor = DifferentiableSafeActorCritic(
            num_actions=3, num_props=12, num_rays=41, his_len=10,
            actor_hidden_dims=[32], critic_hidden_dims=[32],
            encoder_hidden_dims=[32, 16],
        )
        ppo = PPO(
            actor, device='cpu', num_learning_epochs=1, num_mini_batches=1,
            learning_rate=1.0e-4,
        )
        ppo.init_storage(2, 2, [550], [3], shield_rays_shape=(41,))
        obs = torch.randn(2, 550)
        for step in range(2):
            drift = torch.full((2, 1), 0.05 * step)
            rays = torch.full((2, 41), 1.5 + 0.1 * step)
            ppo.act(obs, obs, safety_drift=drift, shield_rays=rays)
            next_obs = torch.randn(2, 550)
            ppo.process_env_step(
                next_obs, torch.ones(2), torch.zeros(2), {},
                next_safety_drift=drift + 0.01,
                next_shield_rays=rays + 0.05,
            )
            obs = next_obs
        ppo.compute_returns(obs, {})
        losses = ppo.update()
        self.assertEqual(len(losses), 5)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses))

    @unittest.skipUnless(
        os.path.isfile(ESTIMATOR_CHECKPOINT),
        'repository estimator checkpoint is not available',
    )
    def test_runtime_parity_warmup_zoh_and_reset(self):
        runtime = PredictiveDriftRuntime(ESTIMATOR_CHECKPOINT, device='cpu')
        self.assertTrue(runtime.model.training is False)
        self.assertTrue(all(not parameter.requires_grad for parameter in runtime.model.parameters()))
        env = _FakeEnv(history_length=runtime.history_length, num_rays=runtime.num_rays)

        env.exteroception_updated_mask[:] = True
        env.exteroception_history_count[:] = runtime.history_length - 1
        drift, _ = runtime.update(env)
        self.assertTrue(torch.equal(drift, torch.zeros_like(drift)))

        env.exteroception_history_count[:] = runtime.history_length
        env.motion_ego_hist.uniform_(-0.2, 0.2)
        env.rays_hist.uniform_(0.2, 2.8)
        direct = runtime._predict_physical_drift(
            env.rays_hist, env.motion_ego_hist, env.rays_hist[:, -1]
        )
        drift, _ = runtime.update(env)
        self.assertTrue(torch.allclose(drift, direct, atol=1e-6, rtol=1e-6))

        cached = drift.clone()
        env.exteroception_updated_mask[:] = False
        env.rays_hist.uniform_(0.2, 2.8)
        held, _ = runtime.update(env)
        self.assertTrue(torch.equal(held, cached))

        runtime.reset(torch.tensor([1]))
        self.assertEqual(float(runtime.cached_drift[1]), 0.0)


if __name__ == '__main__':
    unittest.main()
