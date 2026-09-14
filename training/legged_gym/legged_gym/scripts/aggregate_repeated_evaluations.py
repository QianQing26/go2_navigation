"""Aggregate Phase-3 repeated evaluations without pooling repeats as seeds.

The directory layout consumed by this script is::

    <root>/seed1/A/repeat1/{episodes.csv,summary.json,evaluation_manifest.json}
    <root>/seed1/A/repeat2/...

The three repeats are evaluation-level measurements.  A training seed is the
replicate used by the final multi-seed summary.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from compare_fixed_cohort import (
    _paired_tests,
    _transition,
    _validate_result,
    transition_highlights,
)
from rsl_rl.utils.phase3 import diff_manifests


METHODS = (
    ('A', 'A_original'),
    ('B', 'B_synchronized_static'),
    ('C', 'C_predictive'),
)
METHOD_CODES = tuple(item[0] for item in METHODS)
METHOD_NAMES = dict(METHODS)
REPEATS = (1, 2, 3)
BASE_METRICS = (
    'safe_success_rate', 'collision_rate', 'stuck_rate', 'timeout_rate',
    'mean_robot_speed_mps', 'intervention_frequency', 'mean_intervention_norm',
)
C_METRICS = ('mean_safety_drift', 'negative_drift_rate', 'drift_induced_intervention_rate')
DIFFICULTY_BINS = ('low', 'medium', 'high')

# These are the fields which are expected to differ when two different frozen
# policies/modes are paired on the same cohort.  All other manifest fields are
# protocol identity and must match.
METHOD_SPECIFIC_MANIFEST_PATHS = {
    'policy_path', 'policy_sha256', 'safety_mode',
    'estimator_checkpoint', 'estimator_sha256',
}

# ``git_diff_sha256`` covers the whole working tree, including unrelated
# analysis-script edits made between long-running repeats.  The evaluator
# source hash, git commit, and all trajectory-affecting config fields remain
# strict identity checks below; this broad working-tree provenance field is
# reported separately rather than rejecting otherwise valid measurements.
NON_TRAJECTORY_MANIFEST_PATHS = ('git_diff_sha256',)


def _read_json(path):
    with open(path) as handle:
        return json.load(handle)


def _read_csv(path):
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('empty CSV: {}'.format(path))
    return rows


def _write_csv(path, rows, fields):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value):
    return float(value or 0.0)


def _mean(values):
    values = [float(value) for value in values]
    return sum(values) / float(len(values)) if values else 0.0


def sample_stats(values):
    """Return descriptive repeat statistics with explicitly sample ddof=1."""

    values = np.asarray([float(value) for value in values], dtype=np.float64)
    if values.size < 1:
        raise ValueError('at least one value is required')
    result = {
        'mean': float(values.mean()),
        # With one smoke replicate the sample standard deviation is undefined;
        # retain a numeric zero while recording that ddof=1 needs >=2 values.
        'std': float(values.std(ddof=1)) if values.size > 1 else 0.0,
        'min': float(values.min()),
        'max': float(values.max()),
        'range': float(values.max() - values.min()),
        'std_ddof': 1 if values.size > 1 else None,
    }
    result.update({
        'repeat{}'.format(index): float(value)
        for index, value in enumerate(values, start=1)
    })
    return result


def _manifest_protocol_view(manifest):
    """Remove only policy/mode fields before a paired protocol comparison."""

    value = json.loads(json.dumps(manifest))
    for path in METHOD_SPECIFIC_MANIFEST_PATHS:
        value.pop(path, None)
    return value


def validate_manifest_identity(manifests, paired=False):
    """Validate exact repeat identity or common paired protocol identity."""

    if not manifests:
        raise ValueError('at least one manifest is required')
    reference = (
        _manifest_protocol_view(manifests[0]) if paired else manifests[0]
    )
    for index, manifest in enumerate(manifests[1:], start=1):
        candidate = _manifest_protocol_view(manifest) if paired else manifest
        differences = diff_manifests(
            reference, candidate, ignored_paths=NON_TRAJECTORY_MANIFEST_PATHS
        )
        if differences:
            mode = 'paired protocol' if paired else 'repeat'
            raise ValueError(
                '{} manifests differ at index {}: {}'.format(
                    mode, index, differences
                )
            )
    return True


def _metric_from_summary(summary, metric):
    summary_key = {
        'mean_robot_speed_mps': 'mean_speed_mps',
    }.get(metric, metric)
    if summary_key not in summary:
        raise ValueError('summary is missing metric {}'.format(summary_key))
    return float(summary[summary_key])


def _load_repeat(root, seed, method, repeat):
    directory = os.path.abspath(os.path.join(
        os.path.expanduser(root), 'seed{}'.format(int(seed)), method,
        'repeat{}'.format(int(repeat)),
    ))
    required = ('episodes.csv', 'summary.json', 'evaluation_manifest.json')
    missing = [name for name in required if not os.path.isfile(os.path.join(directory, name))]
    if missing:
        raise FileNotFoundError('{} missing {}'.format(directory, ', '.join(missing)))
    episodes = _read_csv(os.path.join(directory, 'episodes.csv'))
    summary = _read_json(os.path.join(directory, 'summary.json'))
    manifest = _read_json(os.path.join(directory, 'evaluation_manifest.json'))
    by_id, bank_hash = _validate_result(episodes, '{} seed{}'.format(method, seed))
    if manifest.get('scenario_bank_hash') != bank_hash:
        raise ValueError('{} manifest/episode bank hash mismatch'.format(directory))
    if int(manifest.get('scenario_count', -1)) != len(by_id):
        raise ValueError('{} manifest/episode scenario count mismatch'.format(directory))
    if set(manifest.get('scenario_ids', [])) != set(by_id):
        raise ValueError('{} manifest/episode scenario IDs mismatch'.format(directory))
    return {
        'seed': int(seed), 'method': method, 'method_name': METHOD_NAMES[method],
        'repeat': int(repeat), 'directory': directory, 'episodes': episodes,
        'by_id': by_id, 'summary': summary, 'manifest': manifest,
        'bank_hash': bank_hash,
    }


def _load_all(root, seeds, repeats=REPEATS, methods=METHOD_CODES):
    reports = {}
    for seed in seeds:
        for method in methods:
            method_reports = []
            for repeat in repeats:
                report = _load_repeat(root, seed, method, repeat)
                reports[(int(seed), method, int(repeat))] = report
                method_reports.append(report)
            validate_manifest_identity(
                [report['manifest'] for report in method_reports], paired=False
            )
    for seed in seeds:
        for repeat in repeats:
            validate_manifest_identity([
                reports[(int(seed), method, int(repeat))]['manifest']
                for method in methods
            ], paired=True)
    all_reports = list(reports.values())
    bank_hashes = {report['bank_hash'] for report in all_reports}
    manifest_hashes = {
        report['manifest'].get('scenario_bank_file_sha256') for report in all_reports
    }
    id_digests = {report['manifest'].get('scenario_ids_digest') for report in all_reports}
    if len(bank_hashes) != 1 or len(manifest_hashes) != 1 or len(id_digests) != 1:
        raise ValueError('scenario-bank identity differs across repeated evaluations')
    return reports


def aggregate_repeat_metrics(reports, seeds, repeats=REPEATS, methods=METHOD_CODES):
    """Create repeat1/2/3 plus sample-stat rows for each seed and method."""

    metrics = list(BASE_METRICS)
    if 'C' in methods:
        metrics.extend(C_METRICS)
    rows = []
    summary = {}
    for seed in seeds:
        for method in methods:
            key = '{}_{}'.format(seed, method)
            summary[key] = {}
            for metric in metrics:
                values = [
                    _metric_from_summary(
                        reports[(int(seed), method, repeat)]['summary'], metric
                    ) for repeat in repeats
                ]
                stats = sample_stats(values)
                summary[key][metric] = stats
                rows.append({
                    'seed': int(seed), 'method': method,
                    'method_name': METHOD_NAMES[method], 'metric': metric,
                    **stats,
                })
    return rows, summary


def _episode_metric(row, metric):
    key = {
        'safe_success_rate': 'safe_success',
        'collision_rate': 'collision',
        'stuck_rate': 'stuck',
        'timeout_rate': 'timeout',
        'mean_robot_speed_mps': 'mean_speed_mps',
    }[metric]
    return _float(row.get(key, 0.0))


def _difficulty_metrics(by_id, name):
    subset = [row for row in by_id.values() if row.get('difficulty_bin') == name]
    return {
        'num_scenarios': len(subset),
        'safe_success_rate': _mean([_episode_metric(row, 'safe_success_rate') for row in subset]),
        'collision_rate': _mean([_episode_metric(row, 'collision_rate') for row in subset]),
        'stuck_rate': _mean([_episode_metric(row, 'stuck_rate') for row in subset]),
        'timeout_rate': _mean([_episode_metric(row, 'timeout_rate') for row in subset]),
        'mean_robot_speed_mps': _mean([_episode_metric(row, 'mean_robot_speed_mps') for row in subset]),
    }


def build_difficulty_reports(reports, seeds, output_dir, repeats=REPEATS,
                            methods=METHOD_CODES):
    """Write per-repeat difficulty rows and the two requested summary levels."""

    detail_rows = []
    detail = {}
    for seed in seeds:
        for method in methods:
            for repeat in repeats:
                key = '{}_{}_{}'.format(seed, method, repeat)
                detail[key] = {}
                for difficulty in DIFFICULTY_BINS:
                    metrics = _difficulty_metrics(
                        reports[(int(seed), method, repeat)]['by_id'], difficulty
                    )
                    detail[key][difficulty] = metrics
                    detail_rows.append({
                        'seed': int(seed), 'method': method,
                        'method_name': METHOD_NAMES[method], 'repeat': int(repeat),
                        'difficulty_bin': difficulty, **metrics,
                    })
    detail_fields = [
        'seed', 'method', 'method_name', 'repeat', 'difficulty_bin',
        'num_scenarios', 'safe_success_rate', 'collision_rate', 'stuck_rate',
        'timeout_rate', 'mean_robot_speed_mps',
    ]
    _write_csv(os.path.join(output_dir, 'difficulty_repeated.csv'), detail_rows, detail_fields)

    summary_rows = []
    summary = {}
    summary_metrics = (
        'safe_success_rate', 'collision_rate', 'stuck_rate', 'timeout_rate',
        'mean_robot_speed_mps',
    )
    for method in methods:
        for difficulty in DIFFICULTY_BINS:
            for seed in seeds:
                values = {
                    metric: [detail['{}_{}_{}'.format(seed, method, repeat)][difficulty][metric]
                             for repeat in repeats]
                    for metric in summary_metrics
                }
                seed_means = {metric: _mean(items) for metric, items in values.items()}
                summary.setdefault(method, {}).setdefault(difficulty, {})[str(seed)] = seed_means
                summary_rows.append({
                    'aggregation_level': 'repeat_mean_within_training_seed',
                    'seed': int(seed), 'method': method,
                    'method_name': METHOD_NAMES[method],
                    'difficulty_bin': difficulty, 'num_scenarios': detail['{}_{}_{}'.format(seed, method, repeats[0])][difficulty]['num_scenarios'],
                    **seed_means,
                })
            for metric in summary_metrics:
                stats = sample_stats([
                    summary[method][difficulty][str(seed)][metric] for seed in seeds
                ])
                summary.setdefault(method, {}).setdefault(difficulty, {}).setdefault('across_training_seeds', {})[metric] = stats
            summary_rows.append({
                'aggregation_level': 'across_training_seeds', 'seed': 'all',
                'method': method, 'method_name': METHOD_NAMES[method],
                'difficulty_bin': difficulty, 'num_scenarios': '',
                **{
                    metric: summary[method][difficulty]['across_training_seeds'][metric]['mean']
                    for metric in summary_metrics
                },
            })
    summary_fields = [
        'aggregation_level', 'seed', 'method', 'method_name', 'difficulty_bin',
        'num_scenarios', *summary_metrics,
    ]
    _write_csv(os.path.join(output_dir, 'difficulty_repeated_summary.csv'), summary_rows, summary_fields)
    return detail, summary


def build_scenario_stability(reports, seeds, output_dir, repeats=REPEATS,
                             methods=METHOD_CODES):
    rows = []
    summary = {}
    for seed in seeds:
        for method in methods:
            report_rows = [reports[(int(seed), method, repeat)]['by_id'] for repeat in repeats]
            scenario_ids = sorted(report_rows[0])
            unstable = 0
            outcome_unstable = 0
            safe_unstable = 0
            collision_unstable = 0
            for scenario_id in scenario_ids:
                outcomes = [report[scenario_id]['terminal_outcome'] for report in report_rows]
                safe_flags = [int(report[scenario_id].get('safe_success', 0)) for report in report_rows]
                collision_flags = [int(report[scenario_id].get('collision', 0)) for report in report_rows]
                is_stable = len(set(outcomes)) == 1
                unstable += int(not is_stable)
                outcome_unstable += int(not is_stable)
                safe_unstable += int(len(set(safe_flags)) != 1)
                collision_unstable += int(len(set(collision_flags)) != 1)
                rows.append({
                    'seed': int(seed), 'method': method,
                    'method_name': METHOD_NAMES[method], 'scenario_id': int(scenario_id),
                    **{
                        'repeat{}_outcome'.format(repeat): outcome
                        for repeat, outcome in zip(repeats, outcomes)
                    },
                    'safe_probability': sum(safe_flags) / float(len(repeats)),
                    'collision_probability': sum(collision_flags) / float(len(repeats)),
                    'is_stable': int(is_stable),
                })
            count = float(len(scenario_ids))
            summary['{}_{}'.format(seed, method)] = {
                'seed': int(seed), 'method': method,
                'method_name': METHOD_NAMES[method], 'num_scenarios': int(count),
                'num_stable_scenarios': int(count - unstable),
                'num_unstable_scenarios': unstable,
                'outcome_disagreement_rate': outcome_unstable / count,
                'safe_disagreement_rate': safe_unstable / count,
                'collision_disagreement_rate': collision_unstable / count,
            }
    fields = [
        'seed', 'method', 'method_name', 'scenario_id',
        *['repeat{}_outcome'.format(repeat) for repeat in repeats],
        'safe_probability',
        'collision_probability', 'is_stable',
    ]
    _write_csv(os.path.join(output_dir, 'scenario_stability.csv'), rows, fields)
    return rows, summary


def build_paired_repeat_results(reports, seeds, output_dir, bootstrap_seed=20260911,
                                bootstrap_resamples=10000, repeats=REPEATS,
                                methods=METHOD_CODES):
    rows = []
    transition_rows = []
    details = {}
    for seed in seeds:
        for repeat in repeats:
            loaded = {
                method: reports[(int(seed), method, repeat)]['by_id']
                for method in methods
            }
            pair_defs = [('A_to_C', 'A')]
            if 'B' in methods:
                pair_defs.append(('B_to_C', 'B'))
            for pair, baseline_method in pair_defs:
                candidate = loaded['C']
                baseline = loaded[baseline_method]
                tests = _paired_tests(
                    pair, baseline, candidate,
                    seed=int(bootstrap_seed) + int(seed) * 100 + int(repeat),
                    resamples=bootstrap_resamples,
                )
                transitions = _transition(
                    pair, baseline, candidate
                )
                transition_rows.extend([{
                    'seed': int(seed), 'repeat': int(repeat), **row
                } for row in transitions])
                other_pair = 'B_to_C' if pair == 'A_to_C' else 'A_to_C'
                highlights = transition_highlights(
                    {pair: transitions, other_pair: []}, len(baseline)
                )
                base_aggregate = {
                    'safe_success': _mean([
                        int(row['safe_success']) for row in baseline.values()
                    ]),
                    'collision': _mean([
                        int(row['collision']) for row in baseline.values()
                    ]),
                }
                candidate_aggregate = {
                    'safe_success': _mean([
                        int(row['safe_success']) for row in candidate.values()
                    ]),
                    'collision': _mean([
                        int(row['collision']) for row in candidate.values()
                    ]),
                }
                safe_bootstrap = tests['delta_safe_success_bootstrap']
                collision_bootstrap = tests['delta_collision_bootstrap']
                rows.append({
                    'seed': int(seed), 'repeat': int(repeat), 'comparison': pair,
                    'baseline_method': baseline_method, 'candidate_method': 'C',
                    'num_pairs': len(baseline),
                    'baseline_safe_success': base_aggregate['safe_success'],
                    'candidate_safe_success': candidate_aggregate['safe_success'],
                    'delta_safe_success': candidate_aggregate['safe_success'] - base_aggregate['safe_success'],
                    'baseline_collision': base_aggregate['collision'],
                    'candidate_collision': candidate_aggregate['collision'],
                    'delta_collision': candidate_aggregate['collision'] - base_aggregate['collision'],
                    'fail_to_safe': highlights['{}_fail_to_C_safe_success'.format(baseline_method)],
                    'safe_to_fail': highlights['{}_safe_to_C_fail'.format(baseline_method)],
                    'mcnemar_safe_p': tests['safe_success_mcnemar_exact']['exact_p_value'],
                    'mcnemar_collision_p': tests['collision_mcnemar_exact']['exact_p_value'],
                    'bootstrap_safe_ci_low': safe_bootstrap['ci_95_low'],
                    'bootstrap_safe_ci_high': safe_bootstrap['ci_95_high'],
                    'bootstrap_collision_ci_low': collision_bootstrap['ci_95_low'],
                    'bootstrap_collision_ci_high': collision_bootstrap['ci_95_high'],
                })
                details['{}_{}_{}'.format(seed, repeat, pair)] = {
                    'tests': tests, 'transition_highlights': highlights,
                }
    fields = list(rows[0])
    _write_csv(os.path.join(output_dir, 'paired_repeat_results.csv'), rows, fields)
    _write_csv(
        os.path.join(output_dir, 'paired_transition_counts.csv'), transition_rows,
        ['seed', 'repeat', 'pair', 'baseline_outcome', 'candidate_outcome', 'count', 'rate'],
    )
    return rows, transition_rows, details


def build_multiseed_repeated_summary(reports, seeds, output_dir, repeats=REPEATS,
                                     methods=METHOD_CODES):
    metrics = list(BASE_METRICS) + list(C_METRICS)
    rows = []
    summary = {}
    for method in methods:
        summary[method] = {}
        for metric in metrics:
            seed_means = {}
            for seed in seeds:
                values = [
                    _metric_from_summary(
                        reports[(int(seed), method, repeat)]['summary'], metric
                    ) for repeat in repeats
                ]
                seed_means[str(seed)] = _mean(values)
            stats = sample_stats(list(seed_means.values()))
            summary[method][metric] = {
                'seed_means': seed_means,
                'across_seed_mean': stats['mean'],
                'across_seed_std': stats['std'],
                'across_seed_std_ddof': 1,
            }
            rows.append({
                'method': method, 'method_name': METHOD_NAMES[method],
                'metric': metric,
                **{'seed{}_mean'.format(seed): seed_means[str(seed)] for seed in seeds},
                'across_seed_mean': stats['mean'],
                'across_seed_std': stats['std'],
            })
    fields = ['method', 'method_name', 'metric'] + [
        'seed{}_mean'.format(seed) for seed in seeds
    ] + ['across_seed_mean', 'across_seed_std']
    _write_csv(os.path.join(output_dir, 'multiseed_repeated_aggregate.csv'), rows, fields)
    return rows, summary


def _effect_noise_summary(paired_rows, repeat_summary, seeds, pair_defs):
    effects = {}
    direction = {}
    for comparison, baseline_method in pair_defs:
        effects[comparison] = {}
        direction[comparison] = {}
        for metric, field in (
            ('delta_safe_success', 'delta_safe_success'),
            ('delta_collision', 'delta_collision'),
        ):
            effects[comparison][metric] = {}
            for seed in seeds:
                values = [
                    float(row[field]) for row in paired_rows
                    if int(row['seed']) == int(seed) and row['comparison'] == comparison
                ]
                stats = sample_stats(values)
                baseline_std = repeat_summary['{}_{}'.format(seed, baseline_method)][
                    'safe_success_rate' if metric == 'delta_safe_success' else 'collision_rate'
                ]['std']
                candidate_std = repeat_summary['{}_C'.format(seed)][
                    'safe_success_rate' if metric == 'delta_safe_success' else 'collision_rate'
                ]['std']
                pooled_std = max(float(baseline_std), float(candidate_std), 1.0e-12)
                stats['pooled_evaluation_std'] = pooled_std
                stats['effect_to_eval_std'] = abs(stats['mean']) / pooled_std
                effects[comparison][metric][str(seed)] = stats
            all_values = [effects[comparison][metric][str(seed)]['mean'] for seed in seeds]
            expected = 1 if metric == 'delta_safe_success' else -1
            signs = [int(np.sign(value)) for value in all_values]
            direction[comparison][metric] = {
                'expected_sign': expected, 'signs_by_seed_mean': signs,
                'positive_or_negative_expected_count': sum(
                    int((value > 0) if expected > 0 else (value < 0))
                    for value in all_values
                ),
                'seed_count': len(seeds),
                'all_expected_direction': all(
                    (value > 0) if expected > 0 else (value < 0)
                    for value in all_values
                ),
            }
    noise_values = []
    noise_ranges = []
    for key, metric_map in repeat_summary.items():
        for metric in ('safe_success_rate', 'collision_rate'):
            noise_values.append(metric_map[metric]['std'])
            noise_ranges.append(metric_map[metric]['range'])
    effect_values = [
        abs(item['mean'])
        for comparison in effects.values()
        for metric in comparison.values()
        for item in metric.values()
    ]
    return {
        'effects': effects,
        'direction_consistency': direction,
        'evaluation_variance_scale': {
            'mean_within_model_sr_std': _mean([
                repeat_summary[key]['safe_success_rate']['std'] for key in repeat_summary
            ]),
            'mean_within_model_cr_std': _mean([
                repeat_summary[key]['collision_rate']['std'] for key in repeat_summary
            ]),
            'max_within_model_sr_range': max([
                repeat_summary[key]['safe_success_rate']['range'] for key in repeat_summary
            ]),
            'max_within_model_cr_range': max([
                repeat_summary[key]['collision_rate']['range'] for key in repeat_summary
            ]),
            'max_within_model_std': max(noise_values),
            'max_within_model_range': max(noise_ranges),
        },
        'method_effect_scale': {
            'mean_absolute_paired_effect': _mean(effect_values),
            'max_absolute_paired_effect': max(effect_values),
        },
    }


def _assessment(reports, repeat_summary, multiseed_summary, effects, seeds, methods):
    direction = effects['direction_consistency']
    seed_checks = {}
    unsupported = []
    for seed in seeds:
        means = {
            method: {
                metric: multiseed_summary[method][metric]['seed_means'][str(seed)]
                for metric in ('safe_success_rate', 'collision_rate')
            } for method in methods
        }
        checks = {}
        for baseline in ('A', 'B'):
            if baseline not in methods:
                continue
            checks.update({
                'C_safe_gt_{}'.format(baseline): means['C']['safe_success_rate'] > means[baseline]['safe_success_rate'],
                'C_collision_lt_{}'.format(baseline): means['C']['collision_rate'] < means[baseline]['collision_rate'],
            })
        seed_checks[str(seed)] = {'metrics': means, 'checks': checks}
        if not all(checks.values()):
            unsupported.append(int(seed))
    multi_seed_pass = not unsupported
    return {
        'configuration_reproducibility': 'PASS',
        'scenario_bank_reproducibility': 'PASS',
        'evaluator_semantic_consistency': 'PASS',
        # Prior CUDA0 repeats already established a diagnostic divergence from
        # robot yaw; full formal repeats intentionally omit large timeseries.
        'bitwise_gpu_physics_reproducibility': 'FAIL',
        'bitwise_gpu_physics_evidence': 'seed1 C repeated CUDA0 audit diverged from robot yaw in single-env and batched runs',
        'seed_mean_checks': seed_checks,
        'unsupported_training_seeds': unsupported,
        'multi_seed_main_hypothesis_pass': multi_seed_pass,
        'effect_direction_consistency_by_seed_mean': direction,
    }


def aggregate_repeated_evaluations(root, output_dir, seeds=(1, 2, 3),
                                   bootstrap_seed=20260911, bootstrap_resamples=10000,
                                   repeats=REPEATS, methods=METHOD_CODES):
    root = os.path.abspath(os.path.expanduser(root))
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    seeds = [int(seed) for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError('training seed IDs must be unique')
    repeats = tuple(int(repeat) for repeat in repeats)
    if len(repeats) < 2 or len(set(repeats)) != len(repeats):
        raise ValueError('at least two unique repeats are required')
    methods = tuple(methods)
    if 'C' not in methods or 'A' not in methods:
        raise ValueError('the aggregate requires at least methods A and C')
    if len(set(methods)) != len(methods):
        raise ValueError('method IDs must be unique')
    reports = _load_all(root, seeds, repeats, methods)
    git_diff_hashes = sorted({
        report['manifest'].get('git_diff_sha256', '')
        for report in reports.values()
    })
    repeat_rows, repeat_summary = aggregate_repeat_metrics(
        reports, seeds, repeats, methods
    )
    _write_csv(
        os.path.join(output_dir, 'repeated_eval_summary.csv'), repeat_rows,
        list(repeat_rows[0]),
    )
    _, stability_summary = build_scenario_stability(
        reports, seeds, output_dir, repeats, methods
    )
    paired_rows, _, paired_details = build_paired_repeat_results(
        reports, seeds, output_dir, bootstrap_seed, bootstrap_resamples, repeats,
        methods,
    )
    _, multiseed_summary = build_multiseed_repeated_summary(
        reports, seeds, output_dir, repeats, methods
    )
    _, difficulty_summary = build_difficulty_reports(
        reports, seeds, output_dir, repeats, methods
    )
    pair_defs = [('A_to_C', 'A')]
    if 'B' in methods:
        pair_defs.append(('B_to_C', 'B'))
    effects = _effect_noise_summary(paired_rows, repeat_summary, seeds, pair_defs)
    # Attach all repeat-level directions explicitly; these are not pooled
    # samples and are used only for a transparent direction-count diagnostic.
    for comparison, baseline in pair_defs:
        for metric, field, expected in (
            ('delta_safe_success', 'delta_safe_success', 1),
            ('delta_collision', 'delta_collision', -1),
        ):
            values = [
                float(row[field]) for row in paired_rows if row['comparison'] == comparison
            ]
            expected_count = sum(int(value > 0 if expected > 0 else value < 0) for value in values)
            effects['direction_consistency'][comparison][metric].update({
                'expected_sign': expected,
                'expected_direction_count': expected_count,
                'total_seed_repeat_combinations': len(values),
                'expected_direction_fraction': expected_count / float(len(values)),
                'all_seed_repeat_combinations_expected_direction': expected_count == len(values),
            })
    assessment = _assessment(
        reports, repeat_summary, multiseed_summary, effects, seeds, methods
    )
    all_direction = all(
        item['all_seed_repeat_combinations_expected_direction']
        for comparison in effects['direction_consistency'].values()
        for item in comparison.values()
    )
    max_ratio = max(
        item['effect_to_eval_std']
        for comparison in effects['effects'].values()
        for metric in comparison.values()
        for item in metric.values()
    )
    assessment['direction_all_9_combinations_pass'] = all_direction
    assessment['max_effect_to_eval_std'] = max_ratio
    if not all_direction:
        assessment['statistical_evaluation_reproducibility'] = 'FAIL'
    elif max_ratio >= 2.0:
        assessment['statistical_evaluation_reproducibility'] = 'PASS'
    else:
        assessment['statistical_evaluation_reproducibility'] = 'INCONCLUSIVE'
    assessment['multi_seed_method_reproducibility'] = (
        'PASS' if assessment['multi_seed_main_hypothesis_pass'] and all_direction
        else 'FAIL' if not assessment['multi_seed_main_hypothesis_pass'] else 'INCONCLUSIVE'
    )
    statuses = {
        key: assessment[key] for key in (
            'configuration_reproducibility', 'scenario_bank_reproducibility',
            'evaluator_semantic_consistency', 'bitwise_gpu_physics_reproducibility',
            'statistical_evaluation_reproducibility', 'multi_seed_method_reproducibility',
        )
    }
    if statuses['multi_seed_method_reproducibility'] == 'PASS' and statuses['statistical_evaluation_reproducibility'] == 'PASS':
        final_judgment = 'SUPPORTED'
    elif statuses['multi_seed_method_reproducibility'] == 'FAIL':
        final_judgment = 'NOT SUPPORTED'
    else:
        final_judgment = 'PARTIALLY SUPPORTED'
    result = {
        'protocol': {
            'root': root, 'training_seeds': seeds,
            'methods': list(methods), 'repeats': list(repeats),
            'replicate_hierarchy': [
                'scenario', 'evaluation_repeat', 'training_seed'
            ],
            'pooled_repeat_as_independent_sample': False,
            'bootstrap_seed': int(bootstrap_seed),
            'bootstrap_resamples': int(bootstrap_resamples),
            'manifest_git_diff_sha256_values': git_diff_hashes,
            'manifest_git_diff_changed_during_collection': len(git_diff_hashes) > 1,
        },
        'repeated_eval_summary': repeat_summary,
        'scenario_disagreement': stability_summary,
        'paired_repeat_details': paired_details,
        'multiseed_repeated_summary': multiseed_summary,
        'difficulty_repeated_summary': difficulty_summary,
        'effect_noise': effects,
        'reproducibility_assessment': assessment,
        'status': statuses,
        'final_research_judgment': final_judgment,
        'artifacts': {
            'repeated_eval_summary_csv': 'repeated_eval_summary.csv',
            'scenario_stability_csv': 'scenario_stability.csv',
            'paired_repeat_results_csv': 'paired_repeat_results.csv',
            'paired_transition_counts_csv': 'paired_transition_counts.csv',
            'difficulty_repeated_csv': 'difficulty_repeated.csv',
            'difficulty_repeated_summary_csv': 'difficulty_repeated_summary.csv',
            'multiseed_repeated_aggregate_csv': 'multiseed_repeated_aggregate.csv',
        },
    }
    with open(os.path.join(output_dir, 'repeated_eval_summary.json'), 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    with open(os.path.join(output_dir, 'reproducibility_assessment.json'), 'w') as handle:
        json.dump({
            'status': statuses,
            'assessment': assessment,
            'final_research_judgment': final_judgment,
        }, handle, indent=2, sort_keys=True)
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--repeats', nargs='+', type=int, default=list(REPEATS))
    parser.add_argument('--methods', nargs='+', choices=METHOD_CODES,
                        default=list(METHOD_CODES))
    parser.add_argument('--bootstrap_seed', type=int, default=20260911)
    parser.add_argument('--bootstrap_resamples', type=int, default=10000)
    return parser.parse_args()


if __name__ == '__main__':
    parsed = _parse_args()
    result = aggregate_repeated_evaluations(
        parsed.root, parsed.output_dir, parsed.seeds,
        parsed.bootstrap_seed, parsed.bootstrap_resamples, parsed.repeats,
        parsed.methods,
    )
    print(json.dumps({
        'status': result['status'],
        'final_research_judgment': result['final_research_judgment'],
    }, indent=2, sort_keys=True))
