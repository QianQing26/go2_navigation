"""Paired statistics for three fixed-cohort frozen-policy evaluations."""

import argparse
import csv
import json
import os

from rsl_rl.utils.phase2 import TERMINAL_OUTCOMES
from rsl_rl.utils.phase3 import (
    DIFFICULTY_BINS,
    deterministic_paired_bootstrap,
    exact_mcnemar,
    difficulty_bin,
    validate_fixed_cohort_rows,
)


FAILURE_OUTCOMES = (
    'collision_failure', 'stuck_failure', 'timeout_failure', 'other_failure',
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a', required=True, help='A/original episodes.csv')
    parser.add_argument('--b', required=True, help='B/synchronized-static episodes.csv')
    parser.add_argument('--c', required=True, help='C/predictive episodes.csv')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--bootstrap_seed', type=int, default=20260911)
    parser.add_argument('--bootstrap_resamples', type=int, default=10000)
    return parser.parse_args()


def _read_rows(path):
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('empty fixed-cohort result: {}'.format(path))
    return rows


def _float(row, key):
    return float(row.get(key, 0.0) or 0.0)


def _int(row, key):
    return int(float(row.get(key, 0) or 0))


def _validate_result(rows, label):
    by_id, bank_hash = validate_fixed_cohort_rows(rows)
    expected_ids = set(range(len(rows)))
    if set(by_id) != expected_ids:
        raise ValueError(
            '{} does not contain exactly scenario IDs [0, N): {}'.format(
                label, sorted(set(by_id) ^ expected_ids)
            )
        )
    for scenario_id, row in by_id.items():
        outcome = row.get('terminal_outcome')
        if outcome not in TERMINAL_OUTCOMES:
            raise ValueError(
                '{} scenario {} has invalid terminal_outcome={!r}'.format(
                    label, scenario_id, outcome
                )
            )
        expected = {
            'safe_success': int(outcome == 'safe_success'),
            'collision': int(outcome == 'collision_failure'),
            'stuck': int(outcome == 'stuck_failure'),
            'timeout': int(outcome == 'timeout_failure'),
            'other_failure': int(outcome == 'other_failure'),
        }
        if _int(row, 'goal_reached') not in (0, 1):
            raise ValueError(
                '{} scenario {} has invalid goal_reached flag'.format(
                    label, scenario_id
                )
            )
        for key, value in expected.items():
            if _int(row, key) not in (0, 1) or _int(row, key) != value:
                raise ValueError(
                    '{} scenario {} has inconsistent {} for {}'.format(
                        label, scenario_id, key, outcome
                    )
                )
        actual_bin = difficulty_bin(_float(row, 'vmax_speed_mps'))
        if row.get('difficulty_bin') != actual_bin:
            raise ValueError(
                '{} scenario {} has inconsistent difficulty_bin'.format(
                    label, scenario_id
                )
            )
    return by_id, bank_hash


def _aggregate(rows):
    count = float(len(rows))
    return {
        'num_scenarios': int(count),
        'safe_success_rate': sum(_int(row, 'safe_success') for row in rows) / count,
        'goal_reached_rate': sum(_int(row, 'goal_reached') for row in rows) / count,
        'collision_rate': sum(_int(row, 'collision') for row in rows) / count,
        'stuck_rate': sum(_int(row, 'stuck') for row in rows) / count,
        'timeout_rate': sum(_int(row, 'timeout') for row in rows) / count,
        'other_failure_rate': sum(_int(row, 'other_failure') for row in rows) / count,
        'mean_speed_mps': sum(_float(row, 'mean_speed_mps') for row in rows) / count,
        'mean_vmax_speed_mps': sum(_float(row, 'vmax_speed_mps') for row in rows) / count,
        'mean_episode_length': sum(_float(row, 'episode_length') for row in rows) / count,
        'mean_duration_s': sum(_float(row, 'duration_s') for row in rows) / count,
        'intervention_frequency': sum(_float(row, 'intervention_frequency') * _float(row, 'episode_length') for row in rows) / max(sum(_float(row, 'episode_length') for row in rows), 1.0),
        'mean_intervention_norm': sum(_float(row, 'mean_intervention_norm') * _float(row, 'episode_length') for row in rows) / max(sum(_float(row, 'episode_length') for row in rows), 1.0),
        'max_intervention_norm': max(_float(row, 'max_intervention_norm') for row in rows),
        'residual_negative_probability': sum(_float(row, 'residual_negative_probability') * _float(row, 'episode_length') for row in rows) / max(sum(_float(row, 'episode_length') for row in rows), 1.0),
        'mean_safety_drift': sum(_float(row, 'mean_safety_drift') * _float(row, 'episode_length') for row in rows) / max(sum(_float(row, 'episode_length') for row in rows), 1.0),
        'negative_drift_rate': sum(_float(row, 'negative_drift_rate') * _float(row, 'episode_length') for row in rows) / max(sum(_float(row, 'episode_length') for row in rows), 1.0),
        'drift_induced_intervention_rate': sum(_float(row, 'drift_induced_intervention_rate') * _float(row, 'episode_length') for row in rows) / max(sum(_float(row, 'episode_length') for row in rows), 1.0),
    }


def _transition(pair, baseline, candidate):
    rows = []
    total = float(len(baseline))
    for base_outcome in TERMINAL_OUTCOMES:
        for candidate_outcome in TERMINAL_OUTCOMES:
            count = sum(
                int(
                    baseline[scenario_id]['terminal_outcome'] == base_outcome
                    and candidate[scenario_id]['terminal_outcome'] == candidate_outcome
                )
                for scenario_id in baseline
            )
            rows.append({
                'pair': pair,
                'baseline_outcome': base_outcome,
                'candidate_outcome': candidate_outcome,
                'count': count,
                'rate': count / total,
            })
    return rows


def _paired_tests(pair, baseline, candidate, seed, resamples):
    ids = sorted(baseline)
    base_success = [_int(baseline[i], 'safe_success') for i in ids]
    cand_success = [_int(candidate[i], 'safe_success') for i in ids]
    base_collision = [_int(baseline[i], 'collision') for i in ids]
    cand_collision = [_int(candidate[i], 'collision') for i in ids]
    return {
        'pair': pair,
        'num_pairs': len(ids),
        'safe_success_mcnemar_exact': exact_mcnemar(base_success, cand_success),
        'collision_mcnemar_exact': exact_mcnemar(base_collision, cand_collision),
        'delta_safe_success_bootstrap': deterministic_paired_bootstrap(
            base_success, cand_success, seed=seed, resamples=resamples
        ),
        'delta_collision_bootstrap': deterministic_paired_bootstrap(
            base_collision, cand_collision, seed=seed, resamples=resamples
        ),
    }


def _difficulty_rows(results):
    rows = []
    for label, by_id in results.items():
        for name, lower, upper in DIFFICULTY_BINS:
            subset = [row for row in by_id.values() if row['difficulty_bin'] == name]
            count = float(len(subset))
            rows.append({
                'mode': label,
                'difficulty_bin': name,
                'lower_mps': lower,
                'upper_mps': upper,
                'num_scenarios': int(count),
                'safe_success_rate': sum(_int(row, 'safe_success') for row in subset) / count if count else 0.0,
                'collision_rate': sum(_int(row, 'collision') for row in subset) / count if count else 0.0,
                'stuck_rate': sum(_int(row, 'stuck') for row in subset) / count if count else 0.0,
                'timeout_rate': sum(_int(row, 'timeout') for row in subset) / count if count else 0.0,
                'mean_speed_mps': sum(_float(row, 'vmax_speed_mps') for row in subset) / count if count else 0.0,
                'mean_robot_speed_mps': sum(_float(row, 'mean_speed_mps') for row in subset) / count if count else 0.0,
            })
    return rows


def transition_highlights(transitions, num_scenarios):
    """Return human-readable paired highlights from the transition matrix."""

    def count(pair, baseline_outcome, candidate_outcome):
        return sum(
            int(row['baseline_outcome'] == baseline_outcome
                and row['candidate_outcome'] == candidate_outcome)
            * int(row['count'])
            for row in transitions[pair]
        )

    return {
        'A_fail_to_C_safe_success': sum(
            count('A_to_C', outcome, 'safe_success')
            for outcome in FAILURE_OUTCOMES
        ),
        'B_fail_to_C_safe_success': sum(
            count('B_to_C', outcome, 'safe_success')
            for outcome in FAILURE_OUTCOMES
        ),
        'A_safe_to_C_fail': sum(
            count('A_to_C', 'safe_success', outcome)
            for outcome in FAILURE_OUTCOMES
        ),
        'B_safe_to_C_fail': sum(
            count('B_to_C', 'safe_success', outcome)
            for outcome in FAILURE_OUTCOMES
        ),
    }


def _write_csv(path, rows, fields):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(script_args):
    results = {
        'A_original': _read_rows(script_args.a),
        'B_synchronized_static': _read_rows(script_args.b),
        'C_predictive': _read_rows(script_args.c),
    }
    validated = {}
    bank_hash = None
    for label, rows in results.items():
        by_id, result_hash = _validate_result(rows, label)
        if bank_hash is None:
            bank_hash = result_hash
        elif result_hash != bank_hash:
            raise ValueError(
                'scenario bank hash mismatch: {} has {}, expected {}'.format(
                    label, result_hash, bank_hash
                )
            )
        validated[label] = by_id
    counts = {label: len(by_id) for label, by_id in validated.items()}
    if len(set(counts.values())) != 1:
        raise ValueError('fixed-cohort result counts differ: {}'.format(counts))
    canonical_ids = set(next(iter(validated.values())))
    for label, by_id in validated.items():
        if set(by_id) != canonical_ids:
            raise ValueError('{} scenario IDs differ from the paired cohort'.format(label))

    output_dir = os.path.abspath(os.path.expanduser(script_args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    aggregate_rows = []
    aggregate = {}
    for label, by_id in validated.items():
        aggregate[label] = _aggregate(list(by_id.values()))
        aggregate_rows.append({'mode': label, **aggregate[label]})
    aggregate_fields = ['mode'] + list(aggregate_rows[0].keys())[1:]
    _write_csv(os.path.join(output_dir, 'aggregate.csv'), aggregate_rows, aggregate_fields)

    transitions = {}
    transition_rows = []
    for pair, baseline_label in (
        ('A_to_C', 'A_original'), ('B_to_C', 'B_synchronized_static')
    ):
        pair_rows = _transition(
            pair, validated[baseline_label], validated['C_predictive']
        )
        transitions[pair] = pair_rows
        transition_rows.extend(pair_rows)
    _write_csv(
        os.path.join(output_dir, 'paired_transition_matrix.csv'),
        transition_rows,
        ['pair', 'baseline_outcome', 'candidate_outcome', 'count', 'rate'],
    )

    paired_tests = {}
    for pair, baseline_label in (
        ('A_to_C', 'A_original'), ('B_to_C', 'B_synchronized_static')
    ):
        paired_tests[pair] = _paired_tests(
            pair, validated[baseline_label], validated['C_predictive'],
            script_args.bootstrap_seed, script_args.bootstrap_resamples,
        )
    with open(os.path.join(output_dir, 'paired_tests.json'), 'w') as handle:
        json.dump(paired_tests, handle, indent=2)

    difficulty = _difficulty_rows(validated)
    _write_csv(
        os.path.join(output_dir, 'difficulty_stratified.csv'), difficulty,
        ['mode', 'difficulty_bin', 'lower_mps', 'upper_mps', 'num_scenarios',
         'safe_success_rate', 'collision_rate', 'stuck_rate', 'timeout_rate',
         'mean_speed_mps', 'mean_robot_speed_mps'],
    )

    summary = {
        'scenario_bank_hash': bank_hash,
        'num_scenarios': len(canonical_ids),
        'scenario_ids_are_identical': True,
        'aggregate': aggregate,
        'paired_tests': paired_tests,
        'transition_highlights': transition_highlights(
            transitions, len(canonical_ids)
        ),
        'bootstrap_seed': int(script_args.bootstrap_seed),
        'bootstrap_resamples': int(script_args.bootstrap_resamples),
        'artifacts': {
            'aggregate_csv': 'aggregate.csv',
            'transition_matrix_csv': 'paired_transition_matrix.csv',
            'paired_tests_json': 'paired_tests.json',
            'difficulty_csv': 'difficulty_stratified.csv',
        },
    }
    with open(os.path.join(output_dir, 'comparison_summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    compare(_parse_args())
