"""Summarize a small diagnostic sample of repeat-unstable scenarios."""

import argparse
import csv
import json
import os


def _read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _read_json(path):
    with open(path) as handle:
        return json.load(handle)


def _json(value):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return []


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def select_unstable_scenarios(stability_csv, seed=1, method='C', limit=5):
    rows = [
        row for row in _read_csv(stability_csv)
        if int(row['seed']) == int(seed) and row['method'] == method
        and int(row['is_stable']) == 0
    ]
    rows.sort(key=lambda row: int(row['scenario_id']))
    return [int(row['scenario_id']) for row in rows[:max(0, int(limit))]]


def _scenario_diagnostics(directory, scenario_id):
    episodes = {
        int(row['scenario_id']): row
        for row in _read_csv(os.path.join(directory, 'episodes.csv'))
    }
    if scenario_id not in episodes:
        raise ValueError('{} has no scenario {}'.format(directory, scenario_id))
    summary = _read_json(os.path.join(directory, 'summary.json'))
    dt = _float(summary.get('control_dt_s'), 0.02)
    timeseries = [
        row for row in _read_csv(os.path.join(directory, 'timeseries.csv'))
        if int(row['scenario_id']) == int(scenario_id)
    ]
    first_intervention_time = None
    min_obstacle_distance = None
    safety_drift_values = []
    robot_xy_at_terminal = None
    robot_yaw_at_terminal = None
    for row in timeseries:
        step = int(row['episode_step'])
        if _float(row.get('intervention_norm')) > 1.0e-6 and first_intervention_time is None:
            first_intervention_time = (step - 1) * dt
        robot_xy = _json(row.get('robot_xy'))
        obstacle_xy = _json(row.get('obstacle_xy'))
        if len(robot_xy) >= 2 and obstacle_xy:
            distances = [
                ((float(point[0]) - float(robot_xy[0])) ** 2
                 + (float(point[1]) - float(robot_xy[1])) ** 2) ** 0.5
                for point in obstacle_xy if len(point) >= 2
            ]
            if distances:
                value = min(distances)
                min_obstacle_distance = value if min_obstacle_distance is None else min(min_obstacle_distance, value)
        safety_drift_values.append(_float(row.get('safety_drift_value')))
        robot_xy_at_terminal = robot_xy
        robot_yaw_at_terminal = _float(row.get('robot_yaw'))
    return {
        'directory': os.path.abspath(directory),
        'terminal_outcome': episodes[scenario_id].get('terminal_outcome'),
        'episode_length': int(float(episodes[scenario_id].get('episode_length', 0))),
        'first_intervention_time_s': first_intervention_time,
        'minimum_obstacle_center_distance_m': min_obstacle_distance,
        'safety_drift_min': min(safety_drift_values) if safety_drift_values else None,
        'safety_drift_mean': (
            sum(safety_drift_values) / len(safety_drift_values)
            if safety_drift_values else None
        ),
        'terminal_robot_xy': robot_xy_at_terminal,
        'terminal_robot_yaw': robot_yaw_at_terminal,
        'diagnostic_rows': len(timeseries),
    }


def audit_unstable_scenarios(stability_csv, diagnostic_dirs, output_dir,
                             seed=1, method='C', limit=5):
    scenario_ids = select_unstable_scenarios(stability_csv, seed, method, limit)
    if len(scenario_ids) < 3:
        raise ValueError(
            'expected at least 3 unstable scenarios, found {}'.format(len(scenario_ids))
        )
    if set(diagnostic_dirs) != {1, 2, 3}:
        raise ValueError('diagnostic_dirs must contain repeats 1, 2, and 3')
    rows = []
    report = {
        'seed': int(seed), 'method': method,
        'selected_scenario_ids': scenario_ids,
        'selection_rule': 'first {} scenario IDs with is_stable=0'.format(limit),
        'repeats': {},
    }
    for scenario_id in scenario_ids:
        report['repeats'][str(scenario_id)] = {}
        for repeat in (1, 2, 3):
            diagnostics = _scenario_diagnostics(diagnostic_dirs[repeat], scenario_id)
            report['repeats'][str(scenario_id)][str(repeat)] = diagnostics
            rows.append({
                'seed': int(seed), 'method': method, 'scenario_id': scenario_id,
                'repeat': repeat, **{
                    key: value for key, value in diagnostics.items()
                    if key != 'directory'
                },
            })
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    fields = list(rows[0])
    with open(os.path.join(output_dir, 'unstable_scenario_mechanism.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(output_dir, 'unstable_scenario_mechanism.json'), 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    return report


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stability_csv', required=True)
    parser.add_argument('--repeat1_dir', required=True)
    parser.add_argument('--repeat2_dir', required=True)
    parser.add_argument('--repeat3_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--method', default='C')
    parser.add_argument('--limit', type=int, default=5)
    return parser.parse_args()


if __name__ == '__main__':
    parsed = _parse_args()
    result = audit_unstable_scenarios(
        parsed.stability_csv,
        {1: parsed.repeat1_dir, 2: parsed.repeat2_dir, 3: parsed.repeat3_dir},
        parsed.output_dir, parsed.seed, parsed.method, parsed.limit,
    )
    print(json.dumps({
        'selected_scenario_ids': result['selected_scenario_ids'],
        'output_dir': os.path.abspath(parsed.output_dir),
    }, indent=2))
