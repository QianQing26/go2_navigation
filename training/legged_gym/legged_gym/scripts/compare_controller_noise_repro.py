"""Compare two independent fixed-cohort controller-noise evaluations.

The evaluator stores pre-action robot pose in ``timeseries.csv``.  This
utility compares those poses by scenario and control step, then compares the
terminal outcome rows.  It intentionally does not rerun the simulator.
"""

import argparse
import csv
import json
import math
import os


def _read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _read_json_vector(value):
    parsed = json.loads(value)
    return [float(item) for item in parsed]


def _pose_key(row):
    return int(row['scenario_id']), int(row['global_step'])


def _float_or_none(value):
    return None if value in ('', None) else float(value)


def _first_trajectory_divergence(rows_a, rows_b, xy_tolerance, yaw_tolerance):
    by_key_a = {_pose_key(row): row for row in rows_a}
    by_key_b = {_pose_key(row): row for row in rows_b}
    common = sorted(set(by_key_a).intersection(by_key_b), key=lambda key: (key[1], key[0]))
    missing = sorted(set(by_key_a).symmetric_difference(by_key_b), key=lambda key: (key[1], key[0]))
    for scenario_id, step in common:
        row_a = by_key_a[(scenario_id, step)]
        row_b = by_key_b[(scenario_id, step)]
        xy_a = _read_json_vector(row_a['robot_xy'])
        xy_b = _read_json_vector(row_b['robot_xy'])
        xy_delta = math.sqrt(sum((a - b) ** 2 for a, b in zip(xy_a, xy_b)))
        yaw_a = float(row_a['robot_yaw'])
        yaw_b = float(row_b['robot_yaw'])
        yaw_delta = abs(yaw_a - yaw_b)
        if xy_delta > xy_tolerance or yaw_delta > yaw_tolerance:
            return {
                'scenario_id': scenario_id,
                'global_step': step,
                'episode_step_a': int(row_a['episode_step']),
                'episode_step_b': int(row_b['episode_step']),
                'variable': 'robot_xy' if xy_delta > xy_tolerance else 'robot_yaw',
                'robot_xy_l2_delta': xy_delta,
                'robot_yaw_abs_delta': yaw_delta,
                'missing_pose_rows': len(missing),
            }
    if missing:
        scenario_id, step = missing[0]
        return {
            'scenario_id': scenario_id,
            'global_step': step,
            'episode_step_a': None,
            'episode_step_b': None,
            'variable': 'trajectory_row_presence',
            'robot_xy_l2_delta': None,
            'robot_yaw_abs_delta': None,
            'missing_pose_rows': len(missing),
        }
    return None


def compare(run_a, run_b, xy_tolerance=1.0e-7, yaw_tolerance=1.0e-7):
    episodes_a = _read_csv(os.path.join(run_a, 'episodes.csv'))
    episodes_b = _read_csv(os.path.join(run_b, 'episodes.csv'))
    timeseries_a = _read_csv(os.path.join(run_a, 'timeseries.csv'))
    timeseries_b = _read_csv(os.path.join(run_b, 'timeseries.csv'))

    by_id_a = {int(row['scenario_id']): row for row in episodes_a}
    by_id_b = {int(row['scenario_id']): row for row in episodes_b}
    scenario_ids = sorted(set(by_id_a).union(by_id_b))
    outcome_disagreement = [
        scenario_id for scenario_id in scenario_ids
        if (
            scenario_id not in by_id_a or scenario_id not in by_id_b
            or by_id_a[scenario_id]['terminal_outcome']
            != by_id_b[scenario_id]['terminal_outcome']
        )
    ]
    safe_success_disagreement = [
        scenario_id for scenario_id in scenario_ids
        if (
            scenario_id not in by_id_a or scenario_id not in by_id_b
            or by_id_a[scenario_id]['safe_success']
            != by_id_b[scenario_id]['safe_success']
        )
    ]
    collision_disagreement = [
        scenario_id for scenario_id in scenario_ids
        if (
            scenario_id not in by_id_a or scenario_id not in by_id_b
            or by_id_a[scenario_id]['collision']
            != by_id_b[scenario_id]['collision']
        )
    ]

    def rate(rows, field):
        return sum(int(row[field]) for row in rows) / max(len(rows), 1)

    first_divergence = _first_trajectory_divergence(
        timeseries_a, timeseries_b, xy_tolerance, yaw_tolerance
    )
    return {
        'run1': os.path.abspath(run_a),
        'run2': os.path.abspath(run_b),
        'scenario_count': len(scenario_ids),
        'outcome_disagreement_count': len(outcome_disagreement),
        'outcome_disagreement_scenarios': outcome_disagreement,
        'safe_success_disagreement_count': len(safe_success_disagreement),
        'safe_success_disagreement_scenarios': safe_success_disagreement,
        'collision_disagreement_count': len(collision_disagreement),
        'collision_disagreement_scenarios': collision_disagreement,
        'first_divergence': first_divergence,
        'run1_safe_success_rate': rate(episodes_a, 'safe_success'),
        'run2_safe_success_rate': rate(episodes_b, 'safe_success'),
        'run1_collision_rate': rate(episodes_a, 'collision'),
        'run2_collision_rate': rate(episodes_b, 'collision'),
        'xy_tolerance': xy_tolerance,
        'yaw_tolerance': yaw_tolerance,
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run1', required=True)
    parser.add_argument('--run2', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--xy-tolerance', type=float, default=1.0e-7)
    parser.add_argument('--yaw-tolerance', type=float, default=1.0e-7)
    return parser.parse_args()


def main(args=None):
    args = _parse_args() if args is None else args
    result = compare(
        args.run1, args.run2,
        xy_tolerance=args.xy_tolerance,
        yaw_tolerance=args.yaw_tolerance,
    )
    result['label'] = args.label
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == '__main__':
    main()
