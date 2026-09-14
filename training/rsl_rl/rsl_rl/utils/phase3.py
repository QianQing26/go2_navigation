"""Pure data and statistics helpers for Phase-3 fixed-cohort validation."""

import hashlib
import json
import math
import os
from collections import Counter

import numpy as np
import torch


SCENARIO_BANK_VERSION = 'phase3.fixed_cohort.v1'
SCENARIO_TENSOR_KEYS = (
    'scenario_id',
    'robot_root_state',
    'robot_dof_state',
    'position_target',
    'env_origin',
    'terrain_level',
    'terrain_type',
    'goal_level',
    'dynamic_obstacle_start',
    'dynamic_obstacle_velocity',
    'dynamic_obstacle_time',
    'dynamic_obstacle_effective_low',
    'dynamic_obstacle_effective_high',
    'dynamic_obstacle_current_speed_range',
    'episode_length',
    'initial_commands',
    'initial_nav_actions_filtered',
    'initial_actions_orig',
    'scenario_vmax_speed_mps',
)

DIFFICULTY_BINS = (
    ('low', 0.1, 0.5),
    ('medium', 0.5, 1.0),
    ('high', 1.0, 1.5),
)

# The training job writes checkpoints every 100 iterations.  Keep this list
# in one pure-Python module so the evaluator, sweep wrapper, tests, and report
# all use the same pre-registered checkpoint protocol.
CANONICAL_CHECKPOINTS = (500, 1000, 1200, 1500, 2000)


def _canonical_metadata(metadata):
    metadata = dict(metadata)
    metadata.pop('bank_hash', None)
    return json.dumps(
        metadata, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')


def _update_tensor_hash(digest, key, value):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest.update(key.encode('utf-8'))
    digest.update(json.dumps({
        'dtype': str(tensor.dtype),
        'shape': list(tensor.shape),
    }, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    digest.update(tensor.numpy().tobytes(order='C'))


def compute_scenario_bank_hash(metadata, scenarios, extras=None):
    """Hash metadata and tensor contents independent of torch pickle layout."""

    digest = hashlib.sha256()
    digest.update(_canonical_metadata(metadata))
    for key in sorted(scenarios):
        _update_tensor_hash(digest, key, scenarios[key])
    for key in sorted(extras or {}):
        _update_tensor_hash(digest, 'extra:' + key, extras[key])
    return digest.hexdigest()


def _validate_scenarios(scenarios, metadata):
    missing = [key for key in SCENARIO_TENSOR_KEYS if key not in scenarios]
    if missing:
        raise ValueError('scenario bank is missing tensors: {}'.format(missing))
    ids = torch.as_tensor(scenarios['scenario_id']).reshape(-1).long()
    count = int(metadata['num_scenarios'])
    if ids.numel() != count:
        raise ValueError(
            'scenario_id count {} does not match num_scenarios {}'.format(
                ids.numel(), count
            )
        )
    if torch.unique(ids).numel() != count:
        raise ValueError('scenario_id contains duplicates')
    if sorted(ids.tolist()) != list(range(count)):
        raise ValueError('scenario_id must be the complete range [0, N)')
    for key, value in scenarios.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError('scenario tensor {} is not a torch.Tensor'.format(key))
        if value.shape[0] != count:
            raise ValueError(
                'scenario tensor {} has first dimension {}, expected {}'.format(
                    key, value.shape[0], count
                )
            )


def save_scenario_bank(path, scenarios, metadata, extras=None):
    """Save a validated CPU tensor bank and its stable content hash."""

    metadata = dict(metadata)
    metadata.setdefault('version', SCENARIO_BANK_VERSION)
    if 'num_scenarios' not in metadata:
        metadata['num_scenarios'] = int(torch.as_tensor(
            scenarios['scenario_id']
        ).numel())
    scenarios = {
        key: torch.as_tensor(value).detach().cpu().clone()
        for key, value in scenarios.items()
    }
    extras = {
        key: torch.as_tensor(value).detach().cpu().clone()
        for key, value in (extras or {}).items()
    }
    _validate_scenarios(scenarios, metadata)
    metadata['bank_hash'] = compute_scenario_bank_hash(
        metadata, scenarios, extras
    )
    path = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save({
        'metadata': metadata, 'scenarios': scenarios, 'extras': extras,
    }, path)
    return metadata['bank_hash']


def load_scenario_bank(path):
    """Load and integrity-check a scenario bank."""

    path = os.path.abspath(os.path.expanduser(path))
    bank = torch.load(path, map_location='cpu')
    if not isinstance(bank, dict) or 'metadata' not in bank or 'scenarios' not in bank:
        raise ValueError('invalid scenario bank structure: {}'.format(path))
    metadata = dict(bank['metadata'])
    scenarios = dict(bank['scenarios'])
    extras = dict(bank.get('extras', {}))
    _validate_scenarios(scenarios, metadata)
    actual_hash = compute_scenario_bank_hash(metadata, scenarios, extras)
    expected_hash = metadata.get('bank_hash')
    if expected_hash and expected_hash != actual_hash:
        raise ValueError(
            'scenario bank hash mismatch: expected {}, got {}'.format(
                expected_hash, actual_hash
            )
        )
    metadata['bank_hash'] = actual_hash
    metadata['path'] = path
    return {'metadata': metadata, 'scenarios': scenarios, 'extras': extras}


def sha256_file(path, chunk_size=1024 * 1024):
    """Return a stable SHA256 for an on-disk artifact."""

    path = os.path.abspath(os.path.expanduser(path))
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def scenario_ids_digest(scenario_ids):
    """Hash the ordered scenario-id list used by one evaluation."""

    payload = json.dumps(
        [int(value) for value in scenario_ids],
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def parse_scenario_ids(value, num_scenarios):
    """Parse and validate a comma-separated scenario-id selection.

    ``None`` or an empty value means the complete bank.  The evaluator keeps
    the order supplied by the caller, while duplicate and out-of-range IDs
    are rejected because either would invalidate paired comparisons.
    """

    count = int(num_scenarios)
    if count < 1:
        raise ValueError('num_scenarios must be positive')
    if value is None or str(value).strip() == '':
        ids = list(range(count))
    else:
        tokens = str(value).replace(',', ' ').split()
        if not tokens:
            ids = list(range(count))
        else:
            try:
                ids = [int(token) for token in tokens]
            except ValueError as exc:
                raise ValueError(
                    'scenario IDs must be integers separated by commas'
                ) from exc
    if not ids:
        raise ValueError('scenario selection must not be empty')
    if len(set(ids)) != len(ids):
        raise ValueError('scenario selection contains duplicate IDs')
    invalid = [value for value in ids if value < 0 or value >= count]
    if invalid:
        raise ValueError(
            'scenario IDs out of range [0, {}): {}'.format(count, invalid)
        )
    return ids


def difficulty_bin(speed_mps):
    """Map analytic scenario vmax to the shared low/medium/high bins."""

    speed = float(speed_mps)
    if 0.1 <= speed < 0.5:
        return 'low'
    if 0.5 <= speed < 1.0:
        return 'medium'
    if 1.0 <= speed <= 1.5:
        return 'high'
    raise ValueError('scenario speed {:.6f} is outside [0.1, 1.5]'.format(speed))


def validate_fixed_cohort_rows(rows):
    """Validate one per-scenario CSV representation and return its ID map."""

    if not rows:
        raise ValueError('fixed-cohort result is empty')
    hashes = {row.get('scenario_bank_hash', '') for row in rows}
    if len(hashes) != 1 or '' in hashes:
        raise ValueError('fixed-cohort rows do not contain one bank hash')
    by_id = {}
    for row in rows:
        scenario_id = int(row['scenario_id'])
        if scenario_id in by_id:
            raise ValueError('duplicated scenario_id {}'.format(scenario_id))
        by_id[scenario_id] = row
    return by_id, next(iter(hashes))


def exact_mcnemar(baseline_positive, candidate_positive):
    """Return discordants, chi-square-style statistic and exact p-value."""

    baseline = np.asarray(baseline_positive, dtype=bool)
    candidate = np.asarray(candidate_positive, dtype=bool)
    if baseline.shape != candidate.shape:
        raise ValueError('paired McNemar inputs must have equal shape')
    baseline_fail_candidate_success = int((~baseline & candidate).sum())
    baseline_success_candidate_fail = int((baseline & ~candidate).sum())
    discordant = baseline_fail_candidate_success + baseline_success_candidate_fail
    statistic = (
        (baseline_fail_candidate_success - baseline_success_candidate_fail) ** 2
        / float(discordant)
        if discordant else 0.0
    )
    if discordant == 0:
        p_value = 1.0
    else:
        try:
            from scipy.stats import binomtest
            p_value = float(binomtest(
                min(baseline_fail_candidate_success, baseline_success_candidate_fail),
                discordant, 0.5, alternative='two-sided',
            ).pvalue)
        except ImportError:
            # Exact two-sided binomial probability without a scipy dependency.
            probabilities = [
                math.comb(discordant, k) * (0.5 ** discordant)
                for k in range(discordant + 1)
            ]
            observed = probabilities[
                min(baseline_fail_candidate_success, baseline_success_candidate_fail)
            ]
            p_value = min(1.0, sum(p for p in probabilities if p <= observed + 1e-15))
    return {
        'baseline_fail_candidate_success': baseline_fail_candidate_success,
        'baseline_success_candidate_fail': baseline_success_candidate_fail,
        'discordant': discordant,
        'statistic': statistic,
        'exact_p_value': p_value,
    }


def deterministic_paired_bootstrap(baseline, candidate, seed=20260911, resamples=10000):
    """Bootstrap candidate-minus-baseline mean difference with fixed RNG."""

    baseline = np.asarray(baseline, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if baseline.shape != candidate.shape or baseline.size == 0:
        raise ValueError('paired bootstrap inputs must have equal non-empty shape')
    difference = candidate - baseline
    rng = np.random.RandomState(int(seed))
    # Process in chunks to keep memory bounded for larger future cohorts.
    bootstrap_means = np.empty(int(resamples), dtype=np.float64)
    chunk_size = 512
    for start in range(0, int(resamples), chunk_size):
        end = min(start + chunk_size, int(resamples))
        indices = rng.randint(0, difference.size, size=(end - start, difference.size))
        bootstrap_means[start:end] = difference[indices].mean(axis=1)
    return {
        'observed_difference': float(difference.mean()),
        'ci_95_low': float(np.percentile(bootstrap_means, 2.5)),
        'ci_95_high': float(np.percentile(bootstrap_means, 97.5)),
        'bootstrap_seed': int(seed),
        'bootstrap_resamples': int(resamples),
    }


def outcome_counts(rows):
    counts = Counter(row['terminal_outcome'] for row in rows)
    total = float(len(rows))
    return {
        'counts': dict(counts),
        'rates': {
            key: counts.get(key, 0) / total if total else 0.0
            for key in (
                'collision_failure', 'safe_success', 'timeout_failure',
                'stuck_failure', 'other_failure',
            )
        },
    }


def fixed_cohort_batches(num_scenarios, num_envs):
    """Return complete, non-overlapping batches for a fixed cohort."""

    num_scenarios = int(num_scenarios)
    num_envs = int(num_envs)
    if num_scenarios < 1 or num_envs < 1:
        raise ValueError('num_scenarios and num_envs must be positive')
    if num_scenarios % num_envs:
        raise ValueError(
            'num_scenarios={} must be divisible by num_envs={}'.format(
                num_scenarios, num_envs
            )
        )
    return [
        list(range(start, start + num_envs))
        for start in range(0, num_scenarios, num_envs)
    ]


def fixed_cohort_batches_for_indices(scenario_indices, num_envs):
    """Batch an explicit, duplicate-free scenario selection exactly once."""

    indices = [int(value) for value in scenario_indices]
    if not indices:
        raise ValueError('scenario selection must be non-empty')
    if len(set(indices)) != len(indices):
        raise ValueError('scenario selection contains duplicate IDs')
    num_envs = int(num_envs)
    if num_envs < 1 or len(indices) % num_envs:
        raise ValueError(
            'selected scenarios ({}) must be divisible by num_envs ({})'.format(
                len(indices), num_envs
            )
        )
    return [
        indices[start:start + num_envs]
        for start in range(0, len(indices), num_envs)
    ]


def diff_manifests(left, right, ignored_paths=()):
    """Recursively compare two evaluation manifests.

    Paths are dot-separated.  This intentionally compares every field by
    default; callers may ignore provenance fields such as output directory
    and evaluation kind when comparing fixed and sweep invocations.
    """

    ignored = set(ignored_paths)
    differences = []

    def walk(left_value, right_value, path):
        if path in ignored:
            return
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            for key in sorted(set(left_value) | set(right_value)):
                child = '{}.{}'.format(path, key) if path else str(key)
                if key not in left_value:
                    differences.append({'path': child, 'left': None, 'right': right_value[key]})
                elif key not in right_value:
                    differences.append({'path': child, 'left': left_value[key], 'right': None})
                else:
                    walk(left_value[key], right_value[key], child)
            return
        if isinstance(left_value, list) and isinstance(right_value, list):
            if len(left_value) != len(right_value):
                differences.append({
                    'path': path, 'left': left_value, 'right': right_value,
                })
                return
            for index, (left_item, right_item) in enumerate(zip(left_value, right_value)):
                walk(left_item, right_item, '{}[{}]'.format(path, index))
            return
        if left_value != right_value:
            differences.append({
                'path': path, 'left': left_value, 'right': right_value,
            })

    walk(left, right, '')
    return differences
