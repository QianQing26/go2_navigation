"""CPU tests for Phase-3 fixed-cohort data and paired statistics."""

import csv
import json
import os
import sys
import tempfile
import types
import unittest

import torch

from rsl_rl.utils.phase3 import (
    CANONICAL_CHECKPOINTS,
    SCENARIO_TENSOR_KEYS,
    deterministic_paired_bootstrap,
    diff_manifests,
    difficulty_bin,
    exact_mcnemar,
    fixed_cohort_batches,
    fixed_cohort_batches_for_indices,
    load_scenario_bank,
    parse_scenario_ids,
    save_scenario_bank,
    validate_fixed_cohort_rows,
)


def _small_scenarios(count=8):
    scenarios = {}
    for key in SCENARIO_TENSOR_KEYS:
        if key == 'scenario_id':
            scenarios[key] = torch.arange(count, dtype=torch.long)
        elif key == 'robot_root_state':
            scenarios[key] = torch.zeros(count, 13)
        elif key == 'robot_dof_state':
            scenarios[key] = torch.zeros(count, 12, 2)
        elif key in {'dynamic_obstacle_start', 'dynamic_obstacle_velocity'}:
            scenarios[key] = torch.zeros(count, 2, 2)
        elif key == 'scenario_vmax_speed_mps':
            scenarios[key] = torch.tensor(
                [0.2, 0.3, 0.6, 0.7, 1.1, 1.2, 0.4, 1.4]
            )
        elif key in {
            'position_target', 'env_origin', 'initial_commands',
            'initial_nav_actions_filtered', 'initial_actions_orig',
        }:
            scenarios[key] = torch.zeros(count, 3)
        elif key in {
            'dynamic_obstacle_effective_low',
            'dynamic_obstacle_effective_high',
            'dynamic_obstacle_current_speed_range',
        }:
            scenarios[key] = torch.zeros(count, 2)
        else:
            scenarios[key] = torch.zeros(count)
    return scenarios


class Phase3FixedCohortTest(unittest.TestCase):
    def test_bank_roundtrip_hash_and_reset_state(self):
        scenarios = _small_scenarios()
        metadata = {
            'generation_seed': 7,
            'num_scenarios': 8,
            'task': 'go2_pos_dynamic',
            'obstacle_count': 2,
            'obstacle_speed_range': [0.1, 1.5],
        }
        extras = {
            'terrain_height_field_raw': torch.zeros(4, 5, dtype=torch.int16),
            'terrain_env_origins': torch.zeros(2, 3, 3),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'bank.pt')
            bank_hash = save_scenario_bank(path, scenarios, metadata, extras)
            bank = load_scenario_bank(path)
            self.assertEqual(bank_hash, bank['metadata']['bank_hash'])
            self.assertEqual(bank['metadata']['num_scenarios'], 8)
            self.assertTrue(torch.equal(
                bank['scenarios']['dynamic_obstacle_velocity'],
                scenarios['dynamic_obstacle_velocity'],
            ))
            self.assertTrue(torch.equal(
                bank['scenarios']['scenario_vmax_speed_mps'],
                scenarios['scenario_vmax_speed_mps'],
            ))

            raw = torch.load(path, map_location='cpu')
            raw['scenarios']['position_target'][0, 0] = 1.0
            torch.save(raw, path)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_scenario_bank(path)

    def test_batches_cover_each_scenario_once(self):
        batches = fixed_cohort_batches(8, 4)
        self.assertEqual(batches, [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(
            sorted(index for batch in batches for index in batch), list(range(8))
        )
        with self.assertRaises(ValueError):
            fixed_cohort_batches(7, 4)
        self.assertEqual(
            fixed_cohort_batches_for_indices([7, 2, 5, 1], 2),
            [[7, 2], [5, 1]],
        )
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            fixed_cohort_batches_for_indices([1, 1], 2)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            fixed_cohort_batches_for_indices([1, 2, 3], 2)

    def test_scenario_selection_rejects_duplicates_and_out_of_range(self):
        self.assertEqual(parse_scenario_ids('3,1,7', 8), [3, 1, 7])
        self.assertEqual(parse_scenario_ids(None, 3), [0, 1, 2])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            parse_scenario_ids('1,1', 8)
        with self.assertRaisesRegex(ValueError, 'out of range'):
            parse_scenario_ids('8', 8)

    def test_manifest_equal_and_trajectory_diff(self):
        left = {
            'policy_sha256': 'policy',
            'scenario_bank_hash': 'bank',
            'provenance': {'evaluation_kind': 'fixed_cohort'},
        }
        right = {
            'policy_sha256': 'policy',
            'scenario_bank_hash': 'bank',
            'provenance': {'evaluation_kind': 'checkpoint_sweep'},
        }
        self.assertEqual(
            diff_manifests(
                left, right,
                ignored_paths=('provenance.evaluation_kind',),
            ),
            [],
        )
        left_changed = dict(left)
        left_changed['num_envs'] = 16
        right_changed = dict(right)
        right_changed['num_envs'] = 32
        differences = diff_manifests(
            left_changed, right_changed,
            ignored_paths=('provenance.evaluation_kind',),
        )
        self.assertEqual(differences[0]['path'], 'num_envs')

    def test_canonical_checkpoint_protocol(self):
        self.assertEqual(CANONICAL_CHECKPOINTS, (500, 1000, 1200, 1500, 2000))

    def test_multiseed_aggregation_treats_seed_as_replicate(self):
        scripts_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../../legged_gym/legged_gym/scripts'
        ))
        sys.path.insert(0, scripts_dir)
        try:
            from aggregate_multiseed_results import aggregate_seed_reports
        finally:
            sys.path.pop(0)
        aggregate_fields = [
            'mode', 'num_scenarios', 'safe_success_rate', 'collision_rate',
            'stuck_rate', 'timeout_rate', 'mean_robot_speed_mps',
            'intervention_frequency',
        ]
        difficulty_fields = [
            'mode', 'difficulty_bin', 'num_scenarios',
            'safe_success_rate', 'collision_rate',
        ]
        with tempfile.TemporaryDirectory() as directory:
            seed_directories = {}
            for seed in (1, 2, 3):
                seed_dir = os.path.join(directory, 'seed{}'.format(seed))
                os.makedirs(seed_dir)
                with open(os.path.join(seed_dir, 'aggregate.csv'), 'w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
                    writer.writeheader()
                    for mode, safe in zip(
                        ('A_original', 'B_synchronized_static', 'C_predictive'),
                        (0.1, 0.2, 0.4),
                    ):
                        writer.writerow({
                            'mode': mode, 'num_scenarios': 8,
                            'safe_success_rate': safe + seed * 0.01,
                            'collision_rate': 1.0 - safe,
                            'stuck_rate': 0.0, 'timeout_rate': 0.0,
                            'mean_robot_speed_mps': 0.5,
                            'intervention_frequency': 0.1,
                        })
                with open(os.path.join(seed_dir, 'difficulty_stratified.csv'), 'w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=difficulty_fields)
                    writer.writeheader()
                    for mode in ('A_original', 'B_synchronized_static', 'C_predictive'):
                        for difficulty_bin in ('low', 'medium', 'high'):
                            writer.writerow({
                                'mode': mode, 'difficulty_bin': difficulty_bin,
                                'num_scenarios': 2,
                                'safe_success_rate': 0.1 if mode != 'C_predictive' else 0.4,
                                'collision_rate': 0.8 if mode != 'C_predictive' else 0.5,
                            })
                seed_directories[seed] = seed_dir
            result = aggregate_seed_reports(
                seed_directories, os.path.join(directory, 'out')
            )
            self.assertEqual(result['training_seeds'], [1, 2, 3])
            self.assertEqual(result['replicate_unit'], 'training_seed')
            self.assertFalse(result['pooled_scenario_p_value'])
            self.assertTrue(
                result['effect_consistency']['A_to_C']['delta_safe_success']['all_same_nonzero_sign']
            )

    def test_transition_highlights_count_safe_to_failure(self):
        scripts_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../../legged_gym/legged_gym/scripts'
        ))
        sys.path.insert(0, scripts_dir)
        try:
            from compare_fixed_cohort import transition_highlights
        finally:
            sys.path.pop(0)
        transitions = {
            'A_to_C': [
                {'baseline_outcome': 'safe_success', 'candidate_outcome': 'safe_success', 'count': 3},
                {'baseline_outcome': 'safe_success', 'candidate_outcome': 'collision_failure', 'count': 2},
                {'baseline_outcome': 'safe_success', 'candidate_outcome': 'stuck_failure', 'count': 1},
                {'baseline_outcome': 'collision_failure', 'candidate_outcome': 'safe_success', 'count': 4},
            ],
            'B_to_C': [
                {'baseline_outcome': 'safe_success', 'candidate_outcome': 'safe_success', 'count': 2},
                {'baseline_outcome': 'safe_success', 'candidate_outcome': 'timeout_failure', 'count': 1},
                {'baseline_outcome': 'safe_success', 'candidate_outcome': 'other_failure', 'count': 3},
                {'baseline_outcome': 'timeout_failure', 'candidate_outcome': 'safe_success', 'count': 5},
            ],
        }
        highlights = transition_highlights(transitions, 10)
        self.assertEqual(highlights['A_safe_to_C_fail'], 3)
        self.assertEqual(highlights['B_safe_to_C_fail'], 4)
        self.assertEqual(highlights['A_fail_to_C_safe_success'], 4)
        self.assertEqual(highlights['B_fail_to_C_safe_success'], 5)

    def test_repeated_run_comparison_and_manifest_mismatch_rejection(self):
        scripts_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../../legged_gym/legged_gym/scripts'
        ))
        sys.path.insert(0, scripts_dir)
        try:
            from compare_evaluation_runs import compare_runs
        finally:
            sys.path.pop(0)
        episode_fields = [
            'scenario_id', 'terminal_outcome', 'episode_length',
            'goal_reached', 'collision', 'stuck', 'timeout',
        ]
        timeseries_fields = ['scenario_id', 'episode_step']
        with tempfile.TemporaryDirectory() as directory:
            left = os.path.join(directory, 'left')
            right = os.path.join(directory, 'right')
            for path in (left, right):
                os.makedirs(path)
                with open(os.path.join(path, 'evaluation_manifest.json'), 'w') as handle:
                    json.dump({
                        'policy_sha256': 'same', 'scenario_bank_hash': 'same',
                        'provenance': {'evaluation_kind': 'fixed_cohort'},
                    }, handle)
                with open(os.path.join(path, 'summary.json'), 'w') as handle:
                    json.dump({'ok': True}, handle)
                with open(os.path.join(path, 'episodes.csv'), 'w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=episode_fields)
                    writer.writeheader()
                    writer.writerow({
                        'scenario_id': 0, 'terminal_outcome': 'safe_success',
                        'episode_length': 4, 'goal_reached': 1,
                        'collision': 0, 'stuck': 0, 'timeout': 0,
                    })
                with open(os.path.join(path, 'timeseries.csv'), 'w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=timeseries_fields)
                    writer.writeheader()
                    writer.writerow({'scenario_id': 0, 'episode_step': 1})
            result = compare_runs(left, right, os.path.join(directory, 'comparison'))
            self.assertTrue(result['all_outcome_fields_equal'])
            self.assertTrue(result['all_diagnostics_equal_within_tolerance'])
            with open(os.path.join(right, 'evaluation_manifest.json'), 'w') as handle:
                json.dump({
                    'policy_sha256': 'same', 'scenario_bank_hash': 'different',
                    'provenance': {'evaluation_kind': 'checkpoint_sweep'},
                }, handle)
            with self.assertRaisesRegex(ValueError, 'manifests differ'):
                compare_runs(left, right, os.path.join(directory, 'rejected'))

    def test_difficulty_bins_and_paired_statistics_are_deterministic(self):
        self.assertEqual(difficulty_bin(0.1), 'low')
        self.assertEqual(difficulty_bin(0.5), 'medium')
        self.assertEqual(difficulty_bin(1.0), 'high')
        self.assertEqual(difficulty_bin(1.5), 'high')
        with self.assertRaises(ValueError):
            difficulty_bin(0.05)

        baseline = [False, True, False, True]
        candidate = [True, True, False, False]
        test = exact_mcnemar(baseline, candidate)
        self.assertEqual(test['baseline_fail_candidate_success'], 1)
        self.assertEqual(test['baseline_success_candidate_fail'], 1)
        self.assertEqual(test['exact_p_value'], 1.0)
        first = deterministic_paired_bootstrap(
            baseline, candidate, seed=11, resamples=1000
        )
        second = deterministic_paired_bootstrap(
            baseline, candidate, seed=11, resamples=1000
        )
        self.assertEqual(first, second)

    def test_result_ids_reject_duplicates(self):
        rows = [
            {'scenario_id': '0', 'scenario_bank_hash': 'abc'},
            {'scenario_id': '0', 'scenario_bank_hash': 'abc'},
        ]
        with self.assertRaisesRegex(ValueError, 'duplicated'):
            validate_fixed_cohort_rows(rows)

    def test_paired_comparison_rejects_mismatched_banks(self):
        scripts_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '../../legged_gym/legged_gym/scripts'
        ))
        sys.path.insert(0, scripts_dir)
        try:
            from compare_fixed_cohort import compare
        finally:
            sys.path.pop(0)

        fields = [
            'scenario_id', 'scenario_bank_hash', 'difficulty_bin',
            'vmax_speed_mps', 'terminal_outcome', 'goal_reached',
            'safe_success', 'collision', 'stuck', 'timeout', 'other_failure',
            'episode_length', 'duration_s', 'mean_speed_mps', 'total_reward',
            'intervention_frequency', 'mean_intervention_norm',
            'max_intervention_norm', 'residual_negative_probability',
            'mean_safety_drift', 'negative_drift_rate',
            'drift_induced_intervention_rate',
        ]

        def write_result(path, bank_hash):
            rows = []
            for scenario_id, outcome in enumerate(('safe_success', 'collision_failure')):
                rows.append({
                    'scenario_id': scenario_id,
                    'scenario_bank_hash': bank_hash,
                    'difficulty_bin': 'low',
                    'vmax_speed_mps': 0.2,
                    'terminal_outcome': outcome,
                    'goal_reached': int(outcome == 'safe_success'),
                    'safe_success': int(outcome == 'safe_success'),
                    'collision': int(outcome == 'collision_failure'),
                    'stuck': 0, 'timeout': 0, 'other_failure': 0,
                    'episode_length': 10, 'duration_s': 0.2,
                    'mean_speed_mps': 0.5, 'total_reward': 0.0,
                    'intervention_frequency': 0.0,
                    'mean_intervention_norm': 0.0,
                    'max_intervention_norm': 0.0,
                    'residual_negative_probability': 0.0,
                    'mean_safety_drift': 0.0, 'negative_drift_rate': 0.0,
                    'drift_induced_intervention_rate': 0.0,
                })
            with open(path, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)

        with tempfile.TemporaryDirectory() as directory:
            paths = [os.path.join(directory, name) for name in ('a.csv', 'b.csv', 'c.csv')]
            write_result(paths[0], 'same')
            write_result(paths[1], 'same')
            write_result(paths[2], 'different')
            args = types.SimpleNamespace(
                a=paths[0], b=paths[1], c=paths[2],
                output_dir=os.path.join(directory, 'out'),
                bootstrap_seed=1, bootstrap_resamples=100,
            )
            with self.assertRaisesRegex(ValueError, 'bank hash mismatch'):
                compare(args)


if __name__ == '__main__':
    unittest.main()
