import numpy as np
import pytest

from stroke_nav.data.anatomy import Anatomy
from stroke_nav.data.centerline import build_graph


def _arc(p0, p1, n=60, bulge=(0.0, 0.0, 0.0)):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, b = map(np.asarray, (p0, p1, bulge))
    return (1 - t) * p0 + t * p1 + 4 * t * (1 - t) * b


@pytest.fixture(scope="session")
def toy_anatomy() -> Anatomy:
    """Trunk ('aorta') along +z with two 45-degree side branches and a straight continuation."""
    paths = {
        "aorta": _arc([0, 0, 0], [0, 0, 120]),
        "left_branch": _arc([0, 0, 60], [35, 0, 95], bulge=(0, 0, 3)),
        "right_branch": _arc([0.4, 0.3, 80], [-35, 0, 115], bulge=(0, 0, 3)),
        "back_branch": _arc([0.3, 0, 40], [0, 35, 75]),
    }
    g = build_graph(paths, root="aorta", spacing=1.0)
    g.radius[:] = 2.0
    g.meta.update({"model": "toy"})
    return Anatomy(g)
