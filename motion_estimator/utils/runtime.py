"""Runtime helpers shared by the standalone MotionEstimator entrypoints."""

import os
import sys


MOTION_ESTIMATOR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
REPOSITORY_ROOT = os.path.abspath(os.path.join(MOTION_ESTIMATOR_DIR, '..'))


def ensure_local_imports():
    """Make ``data``, ``models`` and ``utils`` importable in direct execution."""
    if MOTION_ESTIMATOR_DIR not in sys.path:
        sys.path.insert(0, MOTION_ESTIMATOR_DIR)


def project_path(value):
    """Resolve a path relative to the repository root unless already absolute."""
    value = os.path.expanduser(value)
    return value if os.path.isabs(value) else os.path.join(REPOSITORY_ROOT, value)
