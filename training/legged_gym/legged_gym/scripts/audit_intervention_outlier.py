"""Extract and classify the largest intervention from one evaluation output."""

import argparse
import csv
import json
import os


AUDIT_FIELDS = (
    'scenario_id', 'episode_step', 'global_step', 'h_comp', 'Lgh_u',
    'alpha_h', 'residual', 'eta', 'Lgh_norm_sq', 'denominator',
    'intervention_norm', 'shield_rays_min', 'robot_xy', 'robot_yaw',
    'goal_xy', 'obstacle_xy', 'obstacle_velocity', 'shield_rays',
    'safety_drift_value', 'u_bar', 'u_s', 'alpha', 'pre_clip_action',
    'post_clip_action',
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation_dir', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


def _json_value(row, key):
    value = row.get(key, '')
    if value == '':
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except ValueError:
            return value


def audit(evaluation_dir, output):
    evaluation_dir = os.path.abspath(os.path.expanduser(evaluation_dir))
    with open(os.path.join(evaluation_dir, 'episodes.csv'), newline='') as handle:
        episodes = list(csv.DictReader(handle))
    if not episodes:
        raise ValueError('episodes.csv is empty')
    episode = max(episodes, key=lambda row: float(row['max_intervention_norm']))
    scenario_id = episode['scenario_id']
    with open(os.path.join(evaluation_dir, 'timeseries.csv'), newline='') as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get('scenario_id') == scenario_id
        ]
    if not rows:
        raise ValueError(
            'timeseries.csv does not contain diagnostic rows for scenario {}'.format(
                scenario_id
            )
        )
    row = max(rows, key=lambda item: float(item['intervention_norm']))
    u_bar = _json_value(row, 'u_bar') or []
    alpha = _json_value(row, 'alpha') or []
    lgh_norm_sq = float(row['Lgh_norm_sq'])
    if any(abs(float(value)) > 10.0 for value in u_bar):
        classification = 'policy_numerical_outlier'
        reason = 'nominal policy action u_bar contains a value with magnitude > 10'
    elif alpha and min(abs(float(value)) for value in alpha) < 1.0e-8:
        classification = 'alpha_outlier'
        reason = 'policy alpha output is effectively zero'
    elif lgh_norm_sq < 1.0e-8:
        classification = 'cbf_geometry_degeneracy'
        reason = '||Lgh||^2 is effectively zero'
    else:
        classification = 'not_classified_as_policy_or_geometry_outlier'
        reason = 'no numerical policy/alpha/geometry threshold was triggered'
    result = {
        'evaluation_dir': evaluation_dir,
        'scenario_id': int(scenario_id),
        'episode_max_intervention_norm': float(episode['max_intervention_norm']),
        'diagnostic_max_intervention_norm': float(row['intervention_norm']),
        'episode_terminal_outcome': episode['terminal_outcome'],
        'episode_length': int(float(episode['episode_length'])),
        'classification': classification,
        'classification_reason': reason,
        'geometry_is_degenerate': bool(lgh_norm_sq < 1.0e-8),
        'fields': {
            key: _json_value(row, key) for key in AUDIT_FIELDS
        },
        'downstream_action_clipping': {
            'pre_clip_action': _json_value(row, 'pre_clip_action'),
            'post_clip_action': _json_value(row, 'post_clip_action'),
        },
    }
    output = os.path.abspath(os.path.expanduser(output))
    with open(output, 'w') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return result


if __name__ == '__main__':
    parsed = _parse_args()
    print(json.dumps(audit(parsed.evaluation_dir, parsed.output), indent=2, sort_keys=True))
