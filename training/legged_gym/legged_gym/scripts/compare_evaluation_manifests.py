"""Deterministically diff two Phase-3 evaluation manifests."""

import argparse
import json
import os
import sys

from rsl_rl.utils.phase3 import diff_manifests


IGNORED_PROVENANCE_PATHS = (
    'provenance.evaluation_kind',
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left')
    parser.add_argument('right')
    parser.add_argument('--output', default=None)
    parser.add_argument(
        '--include_provenance', action='store_true',
        help='Also compare evaluation-kind provenance fields.',
    )
    return parser.parse_args()


def compare_manifests(left_path, right_path, ignored_paths=IGNORED_PROVENANCE_PATHS):
    with open(os.path.abspath(os.path.expanduser(left_path))) as handle:
        left = json.load(handle)
    with open(os.path.abspath(os.path.expanduser(right_path))) as handle:
        right = json.load(handle)
    differences = diff_manifests(left, right, ignored_paths=ignored_paths)
    return {
        'equal': not differences,
        'ignored_paths': list(ignored_paths),
        'left': os.path.abspath(os.path.expanduser(left_path)),
        'right': os.path.abspath(os.path.expanduser(right_path)),
        'differences': differences,
    }


def main(args=None):
    parsed = _parse_args() if args is None else args
    ignored = () if parsed.include_provenance else IGNORED_PROVENANCE_PATHS
    result = compare_manifests(parsed.left, parsed.right, ignored_paths=ignored)
    if parsed.output:
        with open(os.path.abspath(os.path.expanduser(parsed.output)), 'w') as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['equal'] else 1


if __name__ == '__main__':
    sys.exit(main())
