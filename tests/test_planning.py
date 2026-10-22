import numpy as np
import pytest

from stroke_nav.agents.follower import TrajectoryFollower
from stroke_nav.agents.scripted import run_episode
from stroke_nav.data.anatomy import Anatomy
from stroke_nav.data.centerline import CenterlineGraph, build_graph, trim_outside_leaves
from stroke_nav.env.vessel_env import EnvConfig, VesselNavEnv
from stroke_nav.planning.constraints import DeviceSpec, VesselGeometry, get_device, menger_curvature
from stroke_nav.planning.planner import PathPlanner, PlannerWeights
from stroke_nav.planning.trajectory import build_trajectory


def _poly(*pts, n=40):
    pts = [np.asarray(p, float) for p in pts]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append((1 - t) * a + t * b)
    out.append(pts[-1][None])
    return np.concatenate(out)


def _gentle_detour(z0, z1, depth, n=120):
    """Leaves and rejoins the z axis tangentially; max curvature = depth * (2*pi/L)^2 / 2."""
    z = np.linspace(z0, z1, n)
    x = -depth * (1 - np.cos(2 * np.pi * (z - z0) / (z1 - z0))) / 2
    return np.stack([x, np.zeros(n), z], axis=1)


@pytest.fixture(scope="module")
def loop_anatomy() -> Anatomy:
    """Trunk, then two routes that rejoin: 'short_kinked' (V-shaped apex, bend radius ~5 mm,
    63.2 mm long) and 'long_smooth' (tangential detour, bend radius ~18 mm, ~64 mm long), then a
    distal segment holding the target."""
    paths = {
        "aorta": _poly([0, 0, 0], [0, 0, 60]),
        "short_kinked": _poly([0, 0, 60], [10, 0, 90], [0, 0, 120]),
        "long_smooth": _gentle_detour(60, 120, depth=10),
        "distal": _poly([0, 0, 120], [0, 0, 160]),
    }
    g = build_graph(paths, root="aorta", spacing=1.0)
    g.radius[:] = 2.0
    g.meta.update({"model": "loop"})
    return Anatomy(g)


def _narrow(anat: Anatomy, branch: str, radius: float, lo: float = 10, hi: float = 14) -> Anatomy:
    g = anat.graph
    g2 = CenterlineGraph(g.nodes.copy(), g.branch_id.copy(), g.arc.copy(), list(g.branch_names), g.edges.copy(),
                         g.radius.copy(), dict(g.meta))
    nodes = anat.branch_nodes(branch)
    arcs = g2.arc[nodes] - g2.arc[nodes].min()
    g2.radius[nodes[(arcs >= lo) & (arcs <= hi)]] = radius
    return Anatomy(g2)


def test_menger_curvature_of_circle():
    r = 5.0
    pts = [np.array([r * np.cos(a), r * np.sin(a), 0.0]) for a in (0.0, 0.3, 0.6)]
    assert abs(menger_curvature(*pts) - 1 / r) < 1e-9
    assert menger_curvature(np.zeros(3), np.ones(3), 2 * np.ones(3)) == 0.0


def test_device_limits():
    mc = get_device("microcatheter")
    assert mc.min_lumen_radius_mm == pytest.approx(0.5 * 1.2 * 0.57)
    assert get_device("guidewire").min_lumen_radius_mm < mc.min_lumen_radius_mm


def test_astar_equals_dijkstra_without_penalties(loop_anatomy):
    a = loop_anatomy
    w = PlannerWeights(narrow=0, curvature=0, branch=0, takeoff=0)
    p = PathPlanner(a, "guidewire", w)
    start, target = a.default_start(), a.resolve("distal@-1")
    astar, dij = p.plan(start, target), p.shortest(start, target)
    assert astar.feasible and astar.length_mm == pytest.approx(dij.length_mm, abs=1e-6)
    assert "short_kinked" in astar.branches()


def test_radius_constraint_reroutes_microcatheter_only(loop_anatomy):
    a = _narrow(loop_anatomy, "short_kinked", 0.25)  # passes 0.014" wire (0.216 mm) but not 1.7F (0.342 mm)
    start, target = a.default_start(), a.resolve("distal@-1")
    length_only = PlannerWeights(narrow=0, curvature=0, branch=0, takeoff=0)  # isolate the hard limit
    wire = PathPlanner(a, "guidewire", length_only).plan(start, target)
    cath = PathPlanner(a, "microcatheter", length_only).plan(start, target)
    base = PathPlanner(a, "microcatheter").shortest(start, target)
    assert wire.feasible and "short_kinked" in wire.branches()
    assert cath.feasible and "long_smooth" in cath.branches() and "short_kinked" not in cath.branches()
    # with the default soft penalties even the wire prefers the wide route when it is barely longer
    assert "long_smooth" in PathPlanner(a, "guidewire").plan(start, target).branches()
    assert cath.radius.min() >= get_device("microcatheter").min_lumen_radius_mm
    assert not base.feasible and "narrower" in base.message


def test_no_feasible_route_is_reported(loop_anatomy):
    a = _narrow(_narrow(loop_anatomy, "short_kinked", 0.1), "long_smooth", 0.1, 30, 34)
    plan = PathPlanner(a, "microcatheter").plan(a.default_start(), a.resolve("distal@-1"))
    assert not plan.feasible and not plan.nodes and "narrower" in plan.message


def test_curvature_awareness(loop_anatomy):
    a = loop_anatomy
    start, target = a.default_start(), a.resolve("distal@-1")
    stiff = DeviceSpec("stiff", outer_diameter_mm=0.5, min_bend_radius_mm=8.0)
    geo = VesselGeometry(a)
    shortest = PathPlanner(a, stiff, PlannerWeights(curvature=0, branch=0, takeoff=0)).plan(start, target)
    hard = PathPlanner(a, stiff, PlannerWeights(hard_curvature=True), geo).plan(start, target)
    assert "short_kinked" in shortest.branches()
    assert hard.feasible and "long_smooth" in hard.branches()
    assert (1 / np.maximum(hard.curvature, 1e-9)).min() >= stiff.min_bend_radius_mm - 1e-6
    assert (1 / np.maximum(shortest.curvature, 1e-9)).min() < stiff.min_bend_radius_mm


def test_trajectory_respects_limits(loop_anatomy):
    a = loop_anatomy
    plan = PathPlanner(a, "microcatheter").plan(a.default_start(), a.resolve("distal@-3"))
    tr = build_trajectory(plan)
    assert np.all(tr.v_ref <= tr.v_limit + 1e-9)
    ds = np.diff(tr.s)
    dv2 = np.diff(tr.v_ref ** 2)
    assert np.all(np.abs(dv2) <= 2 * tr.params.a_max_mm_s2 * ds + 1e-6)
    assert tr.v_ref[-1] == 0.0
    assert np.all(np.diff(tr.L_wire) >= -1e-9)  # the reference never retracts
    assert np.all(tr.L_cath <= tr.L_wire + 1e-6)  # catheter stays behind the wire
    assert tr.L_cath[-1] == pytest.approx(tr.s_target, abs=0.3)
    v_max = 40.0
    assert np.all(np.diff(tr.L_wire) <= v_max * tr.dt + 1e-6)
    assert all(e["reachable"] for e in tr.roll_events)


def test_follower_succeeds_where_shortest_path_gets_stuck(loop_anatomy):
    a = _narrow(loop_anatomy, "short_kinked", 0.25)
    env = VesselNavEnv(EnvConfig(target="distal@-3", max_steps=400), anatomy=a)
    planned = [run_episode(env, TrajectoryFollower(), seed=s) for s in range(5)]
    naive = [run_episode(env, TrajectoryFollower(use_shortest_path=True), seed=s) for s in range(3)]
    assert all(r["success"] for r in planned)
    assert not any(r["success"] for r in naive)
    assert all(r["wall_pushes_microcatheter"] > 0 for r in naive)  # catheter blocked at the narrowing


def test_simulator_blocks_device_at_narrow_lumen(loop_anatomy):
    a = _narrow(loop_anatomy, "short_kinked", 0.25)
    env = VesselNavEnv(EnvConfig(target="distal@-3", plan_routes=False), anatomy=a)
    env.reset(seed=0)
    sim = env.sim
    # with the shortest route the catheter's reachable depth ends before the narrowing
    for _ in range(80):
        env.step(np.array([1.0, 0.0, 1.0, 0.0]))
    if "short_kinked" in {a.branch_of(n) for n in sim.track}:
        assert sim.passable_limit(1) < sim.passable_limit(0)


def test_occlusion_detection():
    nodes = np.stack([np.zeros(60), np.zeros(60), np.arange(60.0)], axis=1)
    small = np.stack([np.arange(1, 31.0), np.zeros(30), np.full(30, 30.0)], axis=1)
    g = CenterlineGraph(
        nodes=np.concatenate([nodes, small]), branch_id=np.array([0] * 60 + [1] * 30),
        arc=np.concatenate([np.arange(60.0), np.arange(30.0)]), branch_names=["big", "small"],
        edges=np.array([(i, i + 1) for i in range(59)] + [(30, 60)] + [(60 + i, 61 + i) for i in range(29)]),
        radius=np.concatenate([np.full(60, 3.0), np.full(30, 0.6)]), meta={},
    )
    g.radius[20:26] = np.nan  # 6 mm gap in a big vessel -> occlusion
    g.radius[70:74] = np.nan  # 4 mm gap in a small vessel -> interpolated artefact
    trim_outside_leaves(g)
    assert np.all(g.radius[20:26] == 0.0)
    assert np.all(g.radius[70:74] > 0.5)
    assert g.meta["occlusions"] and g.meta["occlusions"][0]["branch"] == "big"


def test_follower_densify_bounds_spacing():
    from stroke_nav.agents.follower import _densify

    pts = np.array([[0.0, 0, 0], [0, 0, 10.0], [0, 0, 10.2]])
    d = _densify(pts, 0.5)
    assert np.allclose(d[0], pts[0]) and np.allclose(d[-1], pts[-1])
    assert np.linalg.norm(np.diff(d, axis=0), axis=1).max() <= 0.5 + 1e-9
