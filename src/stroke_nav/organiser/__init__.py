"""Compatibility layer with the organiser's HCMR-EV stack (OpenNG3 + EndoNav/CatForce + openng3_rl).

Everything a policy trained here needs in order to run in the organiser's pipeline:

* ``spec``     their robot/device parameters, RL environment and SAC settings, scenario manifest
* ``frame``    rigid transform between our VMR anatomy (RAS mm) and their world frame (m)
* ``fluoro``   a 1x128x128 "fluoro" image comparable to their noisy WebGL render
* ``env``      ``OrganiserCompatEnv``: their exact observation/action spaces, reward and success rule,
               driven by our fast kinematic simulator
* ``export``   writes a run folder in their ``HCMR-EV-RL/runs/<run_id>/`` layout, plus trajectory
               exports for their ROS 2 ``DeviceCommand`` / Twist interfaces

See docs/ORGANISER_SETUP.md (their system) and docs/HANDOVER.md (how to hand work over).
"""

from __future__ import annotations

import sys
from pathlib import Path

COMPAT_DIR = Path(__file__).resolve().parents[3] / "compat"


def ensure_policy_importable() -> str:
    """Make ``openng3_rl.policy.FluoroStateExtractor`` importable.

    SB3 stores the feature-extractor class by module path. Models must therefore reference
    ``openng3_rl.policy.FluoroStateExtractor`` to load in the organiser's evaluation. If their
    package is installed we use it; otherwise our architecture-identical copy in ``compat/``.
    Returns the module file that provides the class.
    """
    try:
        import openng3_rl.policy as mod  # noqa: F401
    except ImportError:
        if str(COMPAT_DIR) not in sys.path:
            sys.path.insert(0, str(COMPAT_DIR))
        import openng3_rl.policy as mod  # noqa: F401
    return str(Path(mod.__file__))
