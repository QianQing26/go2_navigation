"""Compare per-scenario outputs from two fixed-cohort evaluator runs."""

import argparse
import csv
import json
import math
import os
import sys

from compare_evaluation_manifests import compare_manifests


OUTCOME_FIELDS = (
    'terminal_outcome', 'episode_length', 'goal_reached', 'collision',
    'stuck', 'timeout',
)
DIAGNOSTIC_FIELDS = (
    'robot_xy', 'robot_yaw', 'goal_xy', 'obstacle_xy', 'obstacle_velocity',
    'shield_rays', 'safety_drift_value', 'u_bar', 'u_s', 'alpha',
    'Lgh_norm_sq', 'denominator', 'pre_clip_action', 'post_clip_action',
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left_output', required=True)
    parser.add_argument('--right_output', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--atol', type=float, default=1.0e-6)
    parser.add_argument('--rtol', type=float, default=1.0e-6)
    return parser.parse_args()


def _read_csv(path):
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('empty CSV: {}'.format(path))
    return rows


def _by_id(rows, label):
    result = {}
    for row in rows:
        scenario_id = int(row['scenario_id'])
        if scenario_id in result:
            raise ValueError('{} contains duplicate scenario {}'.format(label, scenario_id))
        result[scenario_id] = row
    return result


def _scalar(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _diagnostic_value(row, field):
    value = row.get(field, '')
    if value == '':
        return value
    if field in {'robot_yaw', 'safety_drift_value', 'Lgh_norm_sq', 'denominator'}:
        return _scalar(value)
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _close(left, right, atol, rtol):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b, atol, rtol) for a, b in zip(left, right)
        )
    return left == right


def _first_divergence(left_rows, right_rows, scenario_id, atol, rtol):
    left = {
        int(row['episode_step']): row for row in left_rows
        if int(row['scenario_id']) == scenario_id
    }
    right = {
        int(row['episode_step']): row for row in right_rows
        if int(row['scenario_id']) == scenario_id
    }
    for step in sorted(set(left) | set(right)):
        if step not in left or step not in right:
            return {
                'first_divergence_step': step,
                'first_divergence_field': 'diagnostic_row_missing',
                'left': left.get(step), 'right': right.get(step),
            }
        for field in DIAGNOSTIC_FIELDS:
            left_value = _diagnostic_value(left[step], field)
            right_value = _diagnostic_value(right[step], field)
            if not _close(left_value, right_value, atol, rtol):
                return {
                    'first_divergence_step': step,
                    'first_divergence_field': field,
                    'left': left[step], 'right': right[step],
                }
    return {
        'first_divergence_step': None,
        'first_divergence_field': None,
        'left': None, 'right': None,
    }


def compare_runs(left_output, right_output, output_dir, atol=1.0e-6, rtol=1.0e-6):
    left_output = os.path.abspath(os.path.expanduser(left_output))
    right_output = os.path.abspath(os.path.expanduser(right_output))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    manifest = compare_manifests(
        os.path.join(left_output, 'evaluation_manifest.json'),
        os.path.join(right_output, 'evaluation_manifest.json'),
    )
    if not manifest['equal']:
        raise ValueError(
            'evaluation manifests differ; refusing an unpaired comparison: {}'.format(
                manifest['differences']
            )
        )
    with open(os.path.join(left_output, 'summary.json')) as handle:
        left_summary = json.load(handle)
    with open(os.path.join(right_output, 'summary.json')) as handle:
        right_summary = json.load(handle)
    left_episodes = _by_id(_read_csv(os.path.join(left_output, 'episodes.csv')), 'left')
    right_episodes = _by_id(_read_csv(os.path.join(right_output, 'episodes.csv')), 'right')
    if set(left_episodes) != set(right_episodes):
        raise ValueError('scenario IDs differ between evaluation outputs')
    left_ts = _read_csv(os.path.join(left_output, 'timeseries.csv'))
    right_ts = _read_csv(os.path.join(right_output, 'timeseries.csv'))
    rows = []
    details = {}
    for scenario_id in sorted(left_episodes):
        left_episode = left_episodes[scenario_id]
        right_episode = right_episodes[scenario_id]
        outcome_equal = all(
            left_episode.get(field) == right_episode.get(field)
            for field in OUTCOME_FIELDS
        )
        divergence = _first_divergence(
            left_ts, right_ts, scenario_id, atol=atol, rtol=rtol
        )
        rows.append({
            'scenario_id': scenario_id,
            'fixed_outcome': left_episode.get('terminal_outcome'),
            'sweep_outcome': right_episode.get('terminal_outcome'),
            'fixed_length': left_episode.get('episode_length'),
            'sweep_length': right_episode.get('episode_length'),
            'outcome_fields_equal': int(outcome_equal),
            'first_divergence_step': divergence['first_divergence_step'],
            'first_divergence_field': divergence['first_divergence_field'] or '',
        })
        details[str(scenario_id)] = divergence
    result = {
        'manifest': manifest,
        'scenario_count': len(rows),
        'outcome_fields': list(OUTCOME_FIELDS),
        'diagnostic_fields': list(DIAGNOSTIC_FIELDS),
        'atol': atol,
        'rtol': rtol,
        'all_outcome_fields_equal': all(row['outcome_fields_equal'] for row in rows),
        'all_diagnostics_equal_within_tolerance': all(
            details[str(row['scenario_id'])]['first_divergence_step'] is None
            for row in rows
        ),
        'left_summary': left_summary,
        'right_summary': right_summary,
        'rows': rows,
        'first_divergence_details': details,
    }
    fields = list(rows[0]) if rows else []
    with open(os.path.join(output_dir, 'scenario_comparison.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(output_dir, 'scenario_comparison.json'), 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


if __name__ == '__main__':
    parsed = _parse_args()
    result = compare_runs(
        parsed.left_output, parsed.right_output, parsed.output_dir,
        atol=parsed.atol, rtol=parsed.rtol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0 if result['all_outcome_fields_equal'] else 1)
