"""Compare evaluation metrics from v1 and safety-calibrated runs."""

import argparse
import json
import os


METRICS = (
    ('closing_mae', ('local_dynamic_breakdown', 'approaching_gt_0.05', 'mae')),
    ('closing_underestimation_rate', ('local_dynamic_breakdown', 'approaching_gt_0.05', 'underestimation_rate')),
    ('drift_mae', ('scalar_drift', 'mae')),
    ('drift_rmse', ('scalar_drift', 'rmse')),
    ('false_safe_rate', ('scalar_drift', 'false_safe_rate')),
    ('optimistic_danger_rate', ('scalar_drift', 'optimistic_danger_rate')),
    ('severe_danger_optimistic', ('scalar_danger_breakdown', 'severe_danger', 'optimistic_danger_rate')),
    ('high_speed_mild_optimistic', ('speed_danger_breakdown', 'high_speed__mild_danger', 'optimistic_danger_rate')),
    ('high_speed_severe_optimistic', ('speed_danger_breakdown', 'high_speed__severe_danger', 'optimistic_danger_rate')),
    ('high_speed_optimistic', ('speed_danger_breakdown', 'high_speed__danger', 'optimistic_danger_rate')),
    ('switching_mae', ('scalar_switch_breakdown', 'switching', 'mae')),
    ('static_leakage', ('local_closing_no_switch', 'static_leakage_mean_abs_pred')),
    ('drift_error_p95', ('error_quantiles', 'all', 'p95')),
    ('drift_error_p99', ('error_quantiles', 'all', 'p99')),
    ('drift_error_p999', ('error_quantiles', 'all', 'p99.9')),
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run', action='append', required=True,
        help='Run specification LABEL=metrics.json; repeat for v1, B, C, D',
    )
    parser.add_argument('--output_dir', default='.')
    return parser.parse_args()


def _value(data, path):
    current = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _load_runs(specs):
    runs = {}
    for spec in specs:
        if '=' not in spec:
            raise ValueError('--run must use LABEL=metrics.json: {}'.format(spec))
        label, path = spec.split('=', 1)
        with open(path) as file:
            runs[label] = json.load(file)
    return runs


def main(args):
    runs = _load_runs(args.run)
    labels = list(runs)
    rows = []
    for metric, path in METRICS:
        values = {label: _value(runs[label], path) for label in labels}
        baseline = values[labels[0]]
        delta = {
            label: (
                value - baseline
                if isinstance(value, (int, float)) and isinstance(baseline, (int, float))
                else None
            )
            for label, value in values.items()
        }
        rows.append({'metric': metric, 'values': values, 'delta_vs_{}'.format(labels[0]): delta})
    result = {
        'baseline': labels[0],
        'runs': labels,
        'metrics': rows,
    }
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'v1_vs_v1_1.json'), 'w') as file:
        json.dump(result, file, indent=2)
    lines = ['# v1 vs v1.1 comparison', '', '| Metric | ' + ' | '.join(labels) + ' | ' +
             ' | '.join('{} delta'.format(label) for label in labels[1:]) + ' |',
             '|---|' + '---|' * (len(labels) + len(labels) - 1)]
    for row in rows:
        values = [row['values'].get(label) for label in labels]
        deltas = [row['delta_vs_{}'.format(labels[0])].get(label) for label in labels[1:]]
        def fmt(value):
            return 'n/a' if value is None else '{:.6f}'.format(value)
        lines.append('| {} | {} |'.format(
            row['metric'], ' | '.join(fmt(value) for value in values + deltas)
        ))
    with open(os.path.join(output_dir, 'v1_vs_v1_1.md'), 'w') as file:
        file.write('\n'.join(lines) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main(_parse_args())
