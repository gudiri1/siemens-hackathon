"""Device-aware path planning on the centerline graph.

Search space: directed edges (u -> v) of the centerline graph, so the cost of entering node w can
depend on where the device came from (curvature and bifurcation take-off angle need the incoming
direction). Search: A* with the straight-line distance to the target as heuristic (admissible,
because every edge costs at least its length).

Cost of moving v -> w after arriving from u:
    length(v, w) * (1 + w_narrow * narrowness(w)            # lumen tighter than comfortable
                      + w_curve * (kappa(u, v, w) / kappa_max) ** 2)   # bending effort
    + w_branch                 if v is a bifurcation and w leaves the current vessel
    + w_takeoff * excess angle if the turn at a bifurcation is sharper than takeoff_comfort_deg

Hard constraints (edge removed):
    lumen radius at w < device.min_lumen_radius_mm         (always)
    w lies in a forbidden branch                            (optional, e.g. collateral connectors)
    kappa(u, v, w) > device.max_curvature_per_mm            (only with hard_curvature=True)

Dijkstra (plain shortest path, no device limits) is kept as the baseline for comparison.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from stroke_nav.data.anatomy import Anatomy
from stroke_nav.planning.constraints import DeviceSpec, VesselGeometry, get_device


@dataclass
class PlannerWeights:
    narrow: float = 4.0
    curvature: float = 1.0
    branch: float = 5.0  # mm-equivalent cost of every vessel change at a bifurcation
    takeoff: float = 0.2  # mm-equivalent per degree above takeoff_comfort_deg
    takeoff_comfort_deg: float = 60.0
    hard_curvature: bool = False
    forbidden_branches: tuple[str, ...] = ()  # e.g. ("anterior_connector",) to exclude collateral routes


@dataclass
class Plan:
    """A planned route plus everything needed to judge and execute it."""

    anatomy: Anatomy
    device: DeviceSpec
    nodes: list[int]
    method: str
    cost: float
    feasible: bool
    message: str = ""
    s: np.ndarray = field(default_factory=lambda: np.zeros(0))  # arc length (mm)
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    curvature: np.ndarray = field(default_factory=lambda: np.zeros(0))
    junctions: list[dict] = field(default_factory=list)  # bifurcations passed on the route
    expanded: int = 0

    @property
    def length_mm(self) -> float:
        return float(self.s[-1]) if len(self.s) else 0.0

    def branches(self) -> list[str]:
        out: list[str] = []
        for n in self.nodes:
            b = self.anatomy.branch_of(n)
            if not out or out[-1] != b:
                out.append(b)
        return out

    def summary(self) -> dict:
        d = self.device
        if not self.nodes:
            return {"method": self.method, "device": d.name, "feasible": False, "message": self.message}
        bend_r = 1.0 / np.maximum(self.curvature, 1e-9)
        return {
            "method": self.method,
            "device": d.name,
            "feasible": self.feasible,
            "message": self.message,
            "length_mm": round(self.length_mm, 1),
            "cost": round(self.cost, 1),
            "branches": " > ".join(self.branches()),
            "bifurcations": len(self.junctions),
            "vessel_changes": sum(1 for j in self.junctions if j["changes_vessel"]),
            "min_lumen_radius_mm": round(float(self.radius.min()), 2),
            "required_lumen_radius_mm": round(d.min_lumen_radius_mm, 3),
            "min_bend_radius_mm": round(float(bend_r.min()), 2),
            "device_min_bend_radius_mm": d.min_bend_radius_mm,
            "tight_bends": int(np.sum(bend_r < d.min_bend_radius_mm)),
            "max_takeoff_deg": round(max((j["turn_deg"] for j in self.junctions), default=0.0), 0),
            "nodes_expanded": self.expanded,
        }


class PathPlanner:
    def __init__(self, anatomy: Anatomy, device: str | DeviceSpec = "microcatheter",
                 weights: PlannerWeights | None = None, geometry: VesselGeometry | None = None):
        self.anat = anatomy
        self.device = get_device(device)
        self.w = weights or PlannerWeights()
        self.geo = geometry or VesselGeometry(anatomy)

    # ------------------------------------------------------------------ public API
    def plan(self, start: int, target: int) -> Plan:
        """Device-aware A* from start to target."""
        start, target = int(start), int(target)
        a, dev = self.anat, self.device
        r_min = dev.min_lumen_radius_mm
        if a.radius[start] < r_min:
            return self._fail(f"start node lumen ({a.radius[start]:.2f} mm) is narrower than the device needs ({r_min:.2f} mm)")
        if a.radius[target] < r_min:
            return self._fail(f"target lumen ({a.radius[target]:.2f} mm) is narrower than the device needs ({r_min:.2f} mm)")

        goal = a.nodes[target]
        h = lambda n: float(np.linalg.norm(a.nodes[n] - goal))  # noqa: E731
        # state = (prev, node); prev = -1 at the start
        g: dict[tuple[int, int], float] = {(-1, start): 0.0}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {(-1, start): None}
        heap = [(h(start), 0.0, -1, start)]
        expanded = 0
        while heap:
            f, gc, prev, node = heapq.heappop(heap)
            state = (prev, node)
            if gc > g.get(state, np.inf) + 1e-9:
                continue
            expanded += 1
            if node == target:
                path = self._reconstruct(parent, state)
                return self._make_plan(path, gc, "astar", expanded)
            for nxt in a.neighbors[node]:
                if nxt == prev:
                    continue
                step = self.transition_cost(prev, node, nxt)
                if not np.isfinite(step):
                    continue
                ns = (node, nxt)
                ng = gc + step
                if ng < g.get(ns, np.inf) - 1e-9:
                    g[ns] = ng
                    parent[ns] = state
                    heapq.heappush(heap, (ng + h(nxt), ng, node, nxt))
        reason = self._why_unreachable(start, target)
        p = self._fail(reason)
        p.expanded = expanded
        return p

    def shortest(self, start: int, target: int) -> Plan:
        """Plain Dijkstra shortest path that ignores device limits (baseline)."""
        try:
            path = self.anat.route(int(start), int(target))
        except ValueError as e:
            return self._fail(str(e), method="dijkstra")
        plan = self._make_plan(path, float("nan"), "dijkstra", 0)
        plan.cost = self.path_cost(path)
        violations = int(np.sum(plan.radius < self.device.min_lumen_radius_mm))
        if violations:
            plan.feasible = False
            plan.message = (f"{violations} nodes narrower than {self.device.min_lumen_radius_mm:.2f} mm "
                            f"(min {plan.radius.min():.2f} mm in {self.anat.branch_of(path[int(np.argmin(plan.radius))])})")
        return plan

    # ------------------------------------------------------------------ costs
    def transition_cost(self, u: int, v: int, w: int) -> float:
        a, dev, wt = self.anat, self.device, self.w
        r = a.radius[w]
        if r < dev.min_lumen_radius_mm:
            return np.inf
        if wt.forbidden_branches and a.branch_of(w) in wt.forbidden_branches:
            return np.inf
        length = float(np.linalg.norm(a.nodes[w] - a.nodes[v]))
        r_ok = dev.comfort_lumen_radius_mm
        narrow = max(0.0, (r_ok - r) / (r_ok - dev.min_lumen_radius_mm)) if r < r_ok else 0.0
        kappa = self.geo.turn_curvature(u, v, w) if u >= 0 else 0.0
        if wt.hard_curvature and kappa > dev.max_curvature_per_mm:
            return np.inf
        bend = (kappa / dev.max_curvature_per_mm) ** 2
        cost = length * (1.0 + wt.narrow * narrow + wt.curvature * bend)
        if u >= 0 and a.degree[v] >= 3:
            if a.branch_id[w] != a.branch_id[u]:
                cost += wt.branch
            turn = self.geo.turn_angle_deg(u, v, w)
            cost += wt.takeoff * max(0.0, turn - wt.takeoff_comfort_deg)
        return cost

    def path_cost(self, path: list[int]) -> float:
        c = 0.0
        for i in range(1, len(path)):
            u = path[i - 2] if i >= 2 else -1
            step = self.transition_cost(u, path[i - 1], path[i])
            c += step if np.isfinite(step) else float(np.linalg.norm(self.anat.nodes[path[i]] - self.anat.nodes[path[i - 1]])) * 100
        return c

    # ------------------------------------------------------------------ helpers
    def _reconstruct(self, parent, state) -> list[int]:
        out = []
        while state is not None:
            out.append(state[1])
            state = parent[state]
        return out[::-1]

    def _make_plan(self, path: list[int], cost: float, method: str, expanded: int) -> Plan:
        a = self.anat
        pts = a.nodes[path]
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        kappa = self.geo.path_curvature(path)
        junctions = []
        for i in range(1, len(path) - 1):
            v = path[i]
            if a.degree[v] < 3:
                continue
            u, w = path[i - 1], path[i + 1]
            junctions.append({
                "index": i,
                "node": int(v),
                "s_mm": float(s[i]),
                "from_branch": a.branch_of(u),
                "to_branch": a.branch_of(w),
                "changes_vessel": bool(a.branch_id[w] != a.branch_id[u]),
                "turn_deg": self.geo.turn_angle_deg(u, v, w),
                "options": [a.branch_of(c) for c in a.neighbors[v] if c != u],
            })
        return Plan(a, self.device, list(map(int, path)), method, float(cost), True, "", s, pts.copy(),
                    a.radius[path].copy(), kappa, junctions, expanded)

    def _fail(self, message: str, method: str = "astar") -> Plan:
        return Plan(self.anat, self.device, [], method, float("inf"), False, message)

    def _why_unreachable(self, start: int, target: int) -> str:
        a = self.anat
        try:
            path = a.route(start, target)
        except ValueError:
            return "target is not connected to the start in the vessel graph"
        r = a.radius[path]
        i = int(np.argmin(r))
        forbidden = self.w.forbidden_branches
        extra = f" or uses a forbidden vessel ({', '.join(forbidden)})" if forbidden else ""
        return (f"no route satisfies the device limits: every route is narrower than "
                f"{self.device.min_lumen_radius_mm:.2f} mm somewhere{extra} (shortest route: {r[i]:.2f} mm in "
                f"{a.branch_of(path[i])})")


def compare_plans(plans: list[Plan]) -> str:
    """Markdown table comparing plans (e.g. constrained A* vs plain Dijkstra, or devices)."""
    rows = [p.summary() for p in plans]
    keys = ["method", "device", "feasible", "length_mm", "branches", "bifurcations", "vessel_changes",
            "min_lumen_radius_mm", "required_lumen_radius_mm", "min_bend_radius_mm", "tight_bends",
            "max_takeoff_deg", "cost", "message"]
    head = "| " + " | ".join(keys) + " |"
    sep = "|" + "---|" * len(keys)
    lines = [head, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)
