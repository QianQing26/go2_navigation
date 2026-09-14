"""Aggregate Phase-3 paired reports with training seed as the replicate."""

import argparse
import csv
import json
import os

import numpy as np


MODES = ('A_original', 'B_synchronized_static', 'C_predictive')
METRICS = (
    'safe_success_rate', 'collision_rate', 'stuck_rate', 'timeout_rate',
    'mean_robot_speed_mps', 'intervention_frequency',
)
DIFFICULTY_BINS = ('low', 'medium', 'high')


def _read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    if not rows:
        raise ValueError('cannot write an empty aggregate')
    fields = list(rows[0])
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_values(values):
    """Return mean and sample std across independent training-seed replicates."""

    values = np.asarray([float(value) for value in values], dtype=np.float64)
    if values.size == 0:
        raise ValueError('at least one replicate is required')
    return {
        'mean': float(values.mean()),
        # ddof=1 is the descriptive standard deviation of training replicates.
        'std': float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def _float(row, key):
    return float(row.get(key, 0.0) or 0.0)


def _load_seed(seed, directory):
    directory = os.path.abspath(os.path.expanduser(directory))
    aggregate = {
        row['mode']: row for row in _read_csv(os.path.join(directory, 'aggregate.csv'))
    }
    if set(aggregate) != set(MODES):
        raise ValueError('{} must contain exactly A/B/C aggregate rows'.format(directory))
    scenario_counts = {int(float(row.get('num_scenarios', 0) or 0)) for row in aggregate.values()}
    if len(scenario_counts) != 1 or not next(iter(scenario_counts)):
        raise ValueError('{} has inconsistent scenario counts'.format(directory))
    difficulty = _read_csv(os.path.join(directory, 'difficulty_stratified.csv'))
    difficulty_map = {
        (row['mode'], row['difficulty_bin']): row for row in difficulty
    }
    for mode in MODES:
        for difficulty_bin in DIFFICULTY_BINS:
            if (mode, difficulty_bin) not in difficulty_map:
                raise ValueError('missing {} {} difficulty row'.format(mode, difficulty_bin))
    return {
        'seed': int(seed), 'directory': directory, 'aggregate': aggregate,
        'num_scenarios': next(iter(scenario_counts)),
        'difficulty': difficulty_map,
    }


def aggregate_seed_reports(seed_directories, output_dir):
    """Write the three multi-seed artifacts and return their summary."""

    if len(seed_directories) < 2:
        raise ValueError('multi-seed aggregation requires at least two seeds')
    loaded = [
        _load_seed(seed, directory)
        for seed, directory in sorted(seed_directories.items())
    ]
    seeds = [item['seed'] for item in loaded]
    if len(set(seeds)) != len(seeds):
        raise ValueError('training seed IDs must be unique')
    scenario_counts = {item['num_scenarios'] for item in loaded}
    if len(scenario_counts) != 1:
        raise ValueError('training seeds use different scenario counts')
    os.makedirs(os.path.abspath(os.path.expanduser(output_dir)), exist_ok=True)
    output_dir = os.path.abspath(os.path.expanduser(output_dir))

    aggregate_rows = []
    mean_std = {}
    per_seed = {}
    for mode in MODES:
        per_seed[mode] = {}
        for item in loaded:
            per_seed[mode][str(item['seed'])] = {
                metric: _float(item['aggregate'][mode], metric)
                for metric in METRICS
            }
        mean_std[mode] = {}
        for metric in METRICS:
            values = [per_seed[mode][str(seed)][metric] for seed in seeds]
            mean_std[mode][metric] = aggregate_values(values)
        for seed in seeds:
            row = {
                'seed': seed, 'row_type': 'training_seed', 'mode': mode,
                **per_seed[mode][str(seed)],
            }
            aggregate_rows.append(row)
        aggregate_rows.append({
            'seed': 'mean', 'row_type': 'mean', 'mode': mode,
            **{metric: mean_std[mode][metric]['mean'] for metric in METRICS},
        })
        aggregate_rows.append({
            'seed': 'std', 'row_type': 'sample_std', 'mode': mode,
            **{metric: mean_std[mode][metric]['std'] for metric in METRICS},
        })
    _write_csv(os.path.join(output_dir, 'multiseed_aggregate.csv'), aggregate_rows)

    difficulty_rows = []
    difficulty_summary = {}
    for mode in MODES:
        difficulty_summary[mode] = {}
        for difficulty_bin in DIFFICULTY_BINS:
            key = (mode, difficulty_bin)
            values = {
                'safe_success_rate': [
                    _float(item['difficulty'][key], 'safe_success_rate')
                    for item in loaded
                ],
                'collision_rate': [
                    _float(item['difficulty'][key], 'collision_rate')
                    for item in loaded
                ],
            }
            stats = {
                metric: aggregate_values(items)
                for metric, items in values.items()
            }
            difficulty_summary[mode][difficulty_bin] = stats
            for item in loaded:
                source = item['difficulty'][key]
                difficulty_rows.append({
                    'seed': item['seed'], 'row_type': 'training_seed',
                    'mode': mode, 'difficulty_bin': difficulty_bin,
                    'num_scenarios': source.get('num_scenarios', 0),
                    'safe_success_rate': _float(source, 'safe_success_rate'),
                    'collision_rate': _float(source, 'collision_rate'),
                })
            difficulty_rows.extend([
                {
                    'seed': 'mean', 'row_type': 'mean', 'mode': mode,
                    'difficulty_bin': difficulty_bin,
                    'num_scenarios': '',
                    **{metric: stats[metric]['mean'] for metric in stats},
                },
                {
                    'seed': 'std', 'row_type': 'sample_std', 'mode': mode,
                    'difficulty_bin': difficulty_bin,
                    'num_scenarios': '',
                    **{metric: stats[metric]['std'] for metric in stats},
                },
            ])
    _write_csv(os.path.join(output_dir, 'multiseed_difficulty.csv'), difficulty_rows)

    effects = {}
    for seed in seeds:
        item = next(value for value in loaded if value['seed'] == seed)
        effects[str(seed)] = {}
        for baseline in ('A_original', 'B_synchronized_static'):
            suffix = 'A_to_C' if baseline == 'A_original' else 'B_to_C'
            effects[str(seed)][suffix] = {
                'delta_safe_success': _float(item['aggregate']['C_predictive'], 'safe_success_rate') - _float(item['aggregate'][baseline], 'safe_success_rate'),
                'delta_collision': _float(item['aggregate']['C_predictive'], 'collision_rate') - _float(item['aggregate'][baseline], 'collision_rate'),
            }
    consistency = {}
    for effect_name in ('A_to_C', 'B_to_C'):
        consistency[effect_name] = {}
        for metric in ('delta_safe_success', 'delta_collision'):
            values = [effects[str(seed)][effect_name][metric] for seed in seeds]
            consistency[effect_name][metric] = {
                'signs': [int(np.sign(value)) for value in values],
                'all_same_nonzero_sign': (
                    all(value > 0 for value in values)
                    or all(value < 0 for value in values)
                ),
                'mean': aggregate_values(values)['mean'],
                'std': aggregate_values(values)['std'],
            }
    summary = {
        'training_seeds': seeds,
        'replicate_unit': 'training_seed',
        'num_training_replicates': len(seeds),
        'scenario_count_per_seed': next(iter(scenario_counts)),
        'pooled_scenario_p_value': False,
        'metrics': list(METRICS),
        'per_seed': per_seed,
        'mean_std': mean_std,
        'effect_deltas_per_seed': effects,
        'effect_consistency': consistency,
        'difficulty_bins': list(DIFFICULTY_BINS),
        'difficulty_mean_std': difficulty_summary,
        'artifacts': {
            'aggregate_csv': 'multiseed_aggregate.csv',
            'difficulty_csv': 'multiseed_difficulty.csv',
        },
    }
    with open(os.path.join(output_dir, 'multiseed_summary.json'), 'w') as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed1', required=True)
    parser.add_argument('--seed2', required=True)
    parser.add_argument('--seed3', required=True)
    parser.add_argument('--output_dir', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    parsed = _parse_args()
    result = aggregate_seed_reports(
        {1: parsed.seed1, 2: parsed.seed2, 3: parsed.seed3},
        parsed.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
