"""CPU tests for Phase-3 repeated-evaluation aggregation."""

import json
import os
import sys
import tempfile
import unittest


SCRIPTS_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../legged_gym/legged_gym/scripts'
))
sys.path.insert(0, SCRIPTS_DIR)
try:
    from aggregate_repeated_evaluations import (
        METHOD_CODES,
        REPEATS,
        build_difficulty_reports,
        build_multiseed_repeated_summary,
        build_paired_repeat_results,
        build_scenario_stability,
        sample_stats,
        validate_manifest_identity,
    )
finally:
    sys.path.pop(0)


def _episode(outcome, scenario_id=0, difficulty='low', speed=0.2):
    return {
        'scenario_id': scenario_id,
        'scenario_bank_hash': 'bank',
        'difficulty_bin': difficulty,
        'vmax_speed_mps': speed,
        'terminal_outcome': outcome,
        'goal_reached': int(outcome == 'safe_success'),
        'safe_success': int(outcome == 'safe_success'),
        'collision': int(outcome == 'collision_failure'),
        'stuck': int(outcome == 'stuck_failure'),
        'timeout': int(outcome == 'timeout_failure'),
        'other_failure': int(outcome == 'other_failure'),
        'episode_length': 10,
        'mean_speed_mps': 0.5,
    }


def _reports(outcome_by_seed_repeat_method=None):
    reports = {}
    for seed in (1, 2, 3):
        for method in METHOD_CODES:
            for repeat in REPEATS:
                outcome = 'safe_success'
                if outcome_by_seed_repeat_method:
                    outcome = outcome_by_seed_repeat_method.get(
                        (seed, method, repeat), outcome
                    )
                reports[(seed, method, repeat)] = {
                    'seed': seed, 'method': method, 'repeat': repeat,
                    'by_id': {
                        0: _episode(outcome, 0),
                        1: _episode('collision_failure', 1),
                    },
                    'summary': {
                        'safe_success_rate': 0.5,
                        'collision_rate': 0.5,
                        'stuck_rate': 0.0,
                        'timeout_rate': 0.0,
                        'mean_speed_mps': 0.5,
                        'intervention_frequency': 0.1,
                        'mean_intervention_norm': 0.2,
                        'mean_safety_drift': -0.1,
                        'negative_drift_rate': 0.2,
                        'drift_induced_intervention_rate': 0.05,
                    },
                }
    return reports


class Phase3RepeatedTest(unittest.TestCase):

    def test_sample_std_is_ddof_one(self):
        result = sample_stats([1.0, 2.0, 3.0])
        self.assertEqual(result['mean'], 2.0)
        self.assertEqual(result['std'], 1.0)
        self.assertEqual(result['std_ddof'], 1)
        self.assertEqual(result['range'], 2.0)

    def test_manifest_identity_strict_and_paired(self):
        first = {
            'policy_sha256': 'a', 'safety_mode': 'original',
            'scenario_bank_hash': 'bank', 'num_envs': 64,
        }
        second = dict(first)
        validate_manifest_identity([first, second])
        with self.assertRaisesRegex(ValueError, 'manifests differ'):
            validate_manifest_identity([first, dict(first, num_envs=32)])
        paired = dict(first, policy_sha256='c', safety_mode='predictive')
        validate_manifest_identity([first, paired], paired=True)
        with self.assertRaisesRegex(ValueError, 'manifests differ'):
            validate_manifest_identity([first, dict(paired, scenario_bank_hash='other')], paired=True)

    def test_scenario_stability_finds_stable_and_unstable(self):
        reports = _reports({(1, 'C', 2): 'stuck_failure'})
        with tempfile.TemporaryDirectory() as directory:
            _, summary = build_scenario_stability(reports, (1, 2, 3), directory)
            result = summary['1_C']
            self.assertEqual(result['num_scenarios'], 2)
            self.assertEqual(result['num_stable_scenarios'], 1)
            self.assertEqual(result['num_unstable_scenarios'], 1)
            self.assertEqual(result['outcome_disagreement_rate'], 0.5)
            self.assertTrue(os.path.isfile(os.path.join(directory, 'scenario_stability.csv')))

    def test_paired_repeat_grouping_does_not_pool_repeats(self):
        changes = {}
        for repeat in (1, 2, 3):
            changes[(1, 'A', repeat)] = 'collision_failure'
            changes[(1, 'B', repeat)] = 'collision_failure'
            changes[(1, 'C', repeat)] = 'safe_success'
        reports = _reports(changes)
        with tempfile.TemporaryDirectory() as directory:
            rows, transitions, _ = build_paired_repeat_results(
                reports, (1,), directory, bootstrap_resamples=100
            )
            self.assertEqual(len(rows), 6)  # 3 repeats x A/B comparisons
            self.assertTrue(all(row['num_pairs'] == 2 for row in rows))
            self.assertEqual(len(transitions), 3 * 2 * 25)
            self.assertEqual(rows[0]['fail_to_safe'], 1)

    def test_training_seed_aggregate_uses_three_seed_means(self):
        reports = _reports()
        for seed in (1, 2, 3):
            for repeat in (1, 2, 3):
                reports[(seed, 'C', repeat)]['summary']['safe_success_rate'] = seed / 10.0
        with tempfile.TemporaryDirectory() as directory:
            _, summary = build_multiseed_repeated_summary(reports, (1, 2, 3), directory)
            metric = summary['C']['safe_success_rate']
            self.assertAlmostEqual(metric['seed_means']['1'], 0.1)
            self.assertAlmostEqual(metric['seed_means']['2'], 0.2)
            self.assertAlmostEqual(metric['seed_means']['3'], 0.3)
            self.assertAlmostEqual(metric['across_seed_mean'], 0.2)
            self.assertAlmostEqual(metric['across_seed_std'], 0.1)
            self.assertTrue(os.path.isfile(os.path.join(directory, 'multiseed_repeated_aggregate.csv')))

    def test_difficulty_repeated_aggregation_has_two_levels(self):
        reports = _reports()
        with tempfile.TemporaryDirectory() as directory:
            _, summary = build_difficulty_reports(reports, (1, 2, 3), directory)
            self.assertIn('across_training_seeds', summary['A']['low'])
            self.assertTrue(os.path.isfile(os.path.join(directory, 'difficulty_repeated.csv')))
            self.assertTrue(os.path.isfile(os.path.join(directory, 'difficulty_repeated_summary.csv')))


if __name__ == '__main__':
    unittest.main()
