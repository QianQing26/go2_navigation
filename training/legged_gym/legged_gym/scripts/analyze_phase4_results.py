"""Aggregate Phase-4 failure windows and oracle-repeat results."""

import argparse
import csv
import json
import math
import os
import statistics

from rsl_rl.utils.phase4 import aggregate_numeric


SEEDS = (1, 2)
MODES = (
    'predictive_learned', 'oracle_gt_0p1', 'oracle_gt_0p2',
    'oracle_gt_0p3', 'oracle_gt_0p5', 'oracle_gt_multi',
)


def _read_json(path):
    with open(path) as handle:
        return json.load(handle)


def _read_csv(path):
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _mean(values):
    values = [float(value) for value in values if not math.isnan(float(value))]
    return sum(values) / len(values) if values else float('nan')


def _std(values):
    values = [float(value) for value in values if not math.isnan(float(value))]
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else float('nan')


def _find_phase3_baseline(phase3_root, seed, method):
    if not phase3_root:
        return None
    candidates = []
    root = os.path.abspath(os.path.expanduser(phase3_root))
    for repeat in (1, 2, 3):
        path = os.path.join(root, 'seed{}'.format(seed), method, 'repeat{}'.format(repeat), 'summary.json')
        if os.path.isfile(path):
            candidates.append(_read_json(path))
    if not candidates:
        return None
    return {
        'safe_success_rate': _mean([item.get('safe_success_rate', item.get('success_rate', float('nan'))) for item in candidates]),
        'collision_rate': _mean([item.get('collision_rate', float('nan')) for item in candidates]),
        'repeats': len(candidates),
    }


def _load_oracle_records(oracle_root):
    rows = []
    for seed in SEEDS:
        for mode in MODES:
            for repeat in (1, 2, 3):
                path = os.path.join(
                    oracle_root, 'seed{}'.format(seed), mode,
                    'repeat{}'.format(repeat), 'summary.json'
                )
                if not os.path.isfile(path):
                    continue
                summary = _read_json(path)
                rows.append({
                    'seed': seed, 'mode': mode, 'repeat': repeat,
                    'safe_success_rate': summary.get('safe_success_rate', float('nan')),
                    'collision_rate': summary.get('collision_rate', float('nan')),
                    'stuck_rate': summary.get('stuck_rate', float('nan')),
                    'timeout_rate': summary.get('timeout_rate', float('nan')),
                    'mean_robot_speed_mps': summary.get('mean_robot_speed_mps', float('nan')),
                    'intervention_frequency': summary.get('intervention_frequency', float('nan')),
                    'mean_intervention_norm': summary.get('mean_intervention_norm', float('nan')),
                    'max_intervention_norm': summary.get('max_intervention_norm', float('nan')),
                    'collision_episode_count': summary.get('collision_episode_count', 0),
                    'output_dir': os.path.dirname(path),
                })
    return rows


def _aggregate_seed_mode(rows):
    result = []
    fields = (
        'safe_success_rate', 'collision_rate', 'stuck_rate', 'timeout_rate',
        'mean_robot_speed_mps', 'intervention_frequency',
        'mean_intervention_norm', 'max_intervention_norm',
    )
    for seed in SEEDS:
        for mode in MODES:
            subset = [row for row in rows if row['seed'] == seed and row['mode'] == mode]
            if not subset:
                continue
            item = {'seed': seed, 'mode': mode, 'repeats': len(subset)}
            for field in fields:
                values = [row[field] for row in subset]
                item[field + '_mean'] = _mean(values)
                item[field + '_std'] = _std(values)
                item[field + '_min'] = min(values)
                item[field + '_max'] = max(values)
            result.append(item)
    return result


def _difficulty_rows(oracle_root):
    rows = []
    for seed in SEEDS:
        for mode in MODES:
            for repeat in (1, 2, 3):
                path = os.path.join(
                    oracle_root, 'seed{}'.format(seed), mode,
                    'repeat{}'.format(repeat), 'summary.json'
                )
                if not os.path.isfile(path):
                    continue
                summary = _read_json(path)
                for item in summary.get('difficulty_stratified', []):
                    rows.append({
                        'seed': seed, 'mode': mode, 'repeat': repeat,
                        **item,
                    })
    return rows


def _aggregate_difficulty(rows):
    result = []
    for seed in SEEDS:
        for mode in MODES:
            for difficulty in ('low', 'medium', 'high'):
                subset = [
                    row for row in rows
                    if row['seed'] == seed and row['mode'] == mode
                    and row['difficulty_bin'] == difficulty
                ]
                if not subset:
                    continue
                result.append({
                    'seed': seed, 'mode': mode, 'difficulty_bin': difficulty,
                    'repeats': len(subset),
                    'safe_success_rate_mean': _mean([row['safe_success_rate'] for row in subset]),
                    'safe_success_rate_std': _std([row['safe_success_rate'] for row in subset]),
                    'collision_rate_mean': _mean([row['collision_rate'] for row in subset]),
                    'collision_rate_std': _std([row['collision_rate'] for row in subset]),
                    'stuck_rate_mean': _mean([row['stuck_rate'] for row in subset]),
                    'mean_robot_speed_mps': _mean([row['mean_robot_speed_mps'] for row in subset]),
                })
    return result


def _failure_rows(failure_root):
    rows = []
    taxonomy = {}
    for seed in SEEDS:
        seed_root = os.path.join(failure_root, 'seed{}'.format(seed))
        summary_path = os.path.join(seed_root, 'collision_episode_summary.csv')
        if os.path.isfile(summary_path):
            seed_rows = _read_csv(summary_path)
            for row in seed_rows:
                row['seed'] = seed
                rows.append(row)
                for tag in json.loads(row.get('failure_tags', '[]')):
                    taxonomy[tag] = taxonomy.get(tag, 0) + 1
    return rows, taxonomy


def _bottleneck_level(estimator_headroom, horizon_headroom, control_signal, geometry_signal):
    """A transparent descriptive label, not a statistical significance test."""

    if estimator_headroom > 0.15:
        estimator = 'HIGH'
    elif estimator_headroom > 0.05:
        estimator = 'MEDIUM'
    else:
        estimator = 'LOW'
    if horizon_headroom > 0.15:
        horizon = 'HIGH'
    elif horizon_headroom > 0.05:
        horizon = 'MEDIUM'
    else:
        horizon = 'LOW'
    control = 'HIGH' if control_signal > 0.15 else 'MEDIUM' if control_signal > 0.05 else 'LOW'
    geometry = 'HIGH' if geometry_signal < 0.15 else 'MEDIUM' if geometry_signal < 0.35 else 'LOW'
    return estimator, horizon, control, geometry


def analyze(oracle_root, failure_root, output_dir, phase3_root=None):
    oracle_root = os.path.abspath(os.path.expanduser(oracle_root))
    failure_root = os.path.abspath(os.path.expanduser(failure_root))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    records = _load_oracle_records(oracle_root)
    repeat_fields = list(records[0]) if records else []
    _write_csv(os.path.join(output_dir, 'oracle_repeat_results.csv'), records, repeat_fields)
    seed_summary = _aggregate_seed_mode(records)
    _write_csv(
        os.path.join(output_dir, 'oracle_seed_summary.csv'), seed_summary,
        list(seed_summary[0]) if seed_summary else [],
    )
    difficulty_raw = _difficulty_rows(oracle_root)
    difficulty_summary = _aggregate_difficulty(difficulty_raw)
    _write_csv(
        os.path.join(output_dir, 'oracle_difficulty_summary.csv'), difficulty_summary,
        list(difficulty_summary[0]) if difficulty_summary else [],
    )

    horizon_rows = []
    for seed in SEEDS:
        by_mode = {
            row['mode']: row for row in seed_summary if row['seed'] == seed
        }
        base = by_mode.get('oracle_gt_0p1')
        learned = by_mode.get('predictive_learned')
        if not base or not learned:
            continue
        for mode in ('oracle_gt_0p1', 'oracle_gt_0p2', 'oracle_gt_0p3', 'oracle_gt_0p5', 'oracle_gt_multi'):
            item = by_mode.get(mode)
            if not item:
                continue
            horizon_rows.append({
                'seed': seed, 'mode': mode,
                'safe_success_rate_mean': item['safe_success_rate_mean'],
                'collision_rate_mean': item['collision_rate_mean'],
                'safe_success_headroom_vs_learned': item['safe_success_rate_mean'] - learned['safe_success_rate_mean'],
                'collision_headroom_vs_learned': learned['collision_rate_mean'] - item['collision_rate_mean'],
                'safe_success_headroom_vs_gt0p1': item['safe_success_rate_mean'] - base['safe_success_rate_mean'],
                'collision_headroom_vs_gt0p1': base['collision_rate_mean'] - item['collision_rate_mean'],
            })
    _write_csv(
        os.path.join(output_dir, 'oracle_horizon_summary.csv'), horizon_rows,
        list(horizon_rows[0]) if horizon_rows else [],
    )

    failure_rows, taxonomy = _failure_rows(failure_root)
    learned_rates = []
    gt_rates = []
    for seed in SEEDS:
        for item in seed_summary:
            if item['seed'] == seed and item['mode'] == 'predictive_learned':
                learned_rates.append(item)
            if item['seed'] == seed and item['mode'] == 'oracle_gt_0p1':
                gt_rates.append(item)
    learned_sr = _mean([item['safe_success_rate_mean'] for item in learned_rates])
    gt_sr = _mean([item['safe_success_rate_mean'] for item in gt_rates])
    learned_cr = _mean([item['collision_rate_mean'] for item in learned_rates])
    gt_cr = _mean([item['collision_rate_mean'] for item in gt_rates])
    multi = [item for item in seed_summary if item['mode'] == 'oracle_gt_multi']
    gt_multi_sr = _mean([item['safe_success_rate_mean'] for item in multi])
    gt_multi_cr = _mean([item['collision_rate_mean'] for item in multi])
    estimator_headroom_sr = gt_sr - learned_sr
    estimator_headroom_cr = learned_cr - gt_cr
    horizon_headroom_sr = gt_multi_sr - gt_sr
    horizon_headroom_cr = gt_cr - gt_multi_cr
    failure_vs_oracle = {
        'validated_training_seeds': list(SEEDS),
        'failure_analysis_collision_count': len(failure_rows),
        'taxonomy_counts': taxonomy,
        'learned_mean': {'safe_success_rate': learned_sr, 'collision_rate': learned_cr},
        'oracle_gt_0p1_mean': {'safe_success_rate': gt_sr, 'collision_rate': gt_cr},
        'oracle_gt_multi_mean': {'safe_success_rate': gt_multi_sr, 'collision_rate': gt_multi_cr},
        'estimator_headroom': {
            'safe_success_rate': estimator_headroom_sr,
            'collision_rate': estimator_headroom_cr,
        },
        'horizon_headroom_multi_vs_gt0p1': {
            'safe_success_rate': horizon_headroom_sr,
            'collision_rate': horizon_headroom_cr,
        },
        'failure_vs_oracle_causality_note': (
            'Taxonomy labels are descriptive heuristics; causal attribution is '
            'based on the paired fixed-policy oracle experiments.'
        ),
    }
    with open(os.path.join(output_dir, 'failure_vs_oracle.json'), 'w') as handle:
        json.dump(failure_vs_oracle, handle, indent=2, sort_keys=True, allow_nan=True)

    control_signal = _mean([
        float(row.get('mean_tracking_error', 0.0)) for row in failure_rows
    ]) if failure_rows else 0.0
    geometry_signal = (
        max(0.0, 1.0 - gt_multi_sr) if not math.isnan(gt_multi_sr) else 1.0
    )
    levels = _bottleneck_level(
        max(estimator_headroom_sr, estimator_headroom_cr),
        max(horizon_headroom_sr, horizon_headroom_cr),
        control_signal, geometry_signal,
    )
    recommendation = 'A. Better estimator / uncertainty / asymmetric calibration'
    if levels[1] == 'HIGH' and levels[0] != 'HIGH':
        recommendation = 'B. Multi-horizon predictive barrier'
    elif levels[2] == 'HIGH':
        recommendation = 'C. Controller / locomotion-aware safety model'
    elif levels[3] == 'HIGH':
        recommendation = 'D. Barrier geometry / multi-obstacle representation'

    report_lines = [
        '# Phase-4 Residual Failure Analysis + Oracle Upper-Bound Study',
        '',
        'Validated training seeds for Phase-4: **1, 2**. Seed3 is excluded because its earlier evaluation was confirmed invalid.',
        '',
        '## Oracle fixed-policy results',
        '',
        '| Seed | Mode | Safe-success mean | Collision mean | Repeats |',
        '|---:|---|---:|---:|---:|',
    ]
    for row in seed_summary:
        report_lines.append(
            '| {} | {} | {:.4f} | {:.4f} | {} |'.format(
                row['seed'], row['mode'], row['safe_success_rate_mean'],
                row['collision_rate_mean'], row['repeats']
            )
        )
    report_lines += [
        '',
        '## Headroom',
        '',
        '- Estimator headroom (GT-0.1 vs learned): SR `{:.4f}`, CR reduction `{:.4f}`.'.format(estimator_headroom_sr, estimator_headroom_cr),
        '- Horizon headroom (GT-multi vs GT-0.1): SR `{:.4f}`, CR reduction `{:.4f}`.'.format(horizon_headroom_sr, horizon_headroom_cr),
        '- Failure taxonomy counts: `{}`.'.format(json.dumps(taxonomy, sort_keys=True)),
        '',
        '## Failure Attribution vs Oracle Causality',
        '',
        'The taxonomy is retained as a non-exclusive descriptive summary. The paired oracle results, not the heuristic primary labels, determine the causal interpretation.',
        '',
        '## Bottleneck assessment',
        '',
        '- ESTIMATOR BOTTLENECK: **{}**'.format(levels[0]),
        '- HORIZON BOTTLENECK: **{}**'.format(levels[1]),
        '- CONTROL/EXECUTION BOTTLENECK: **{}**'.format(levels[2]),
        '- GEOMETRY/CBF BOTTLENECK: **{}**'.format(levels[3]),
        '',
        '## Recommended next direction',
        '',
        '**{}**'.format(recommendation),
        '',
        'This recommendation is descriptive and based on the fixed-policy oracle gap; no oracle-trained policy was evaluated.',
    ]
    with open(os.path.join(output_dir, 'phase4_final_report.md'), 'w') as handle:
        handle.write('\n'.join(report_lines) + '\n')
    return failure_vs_oracle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle_root', required=True)
    parser.add_argument('--failure_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--phase3_root', default=None)
    args = parser.parse_args()
    result = analyze(args.oracle_root, args.failure_root, args.output_dir, args.phase3_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=True))


if __name__ == '__main__':
    main()
