"""Follow a planned joint-space trajectory in the simulator.

The follower only uses what a real robot would have:
* encoder readings of its own joints (inserted length and roll of each device), and
* the observed device tip position (fluoroscopy / tracking) to detect a wrong turn.

Each step it commands the velocity that brings each joint to the next reference sample. If the
guidewire tip strays from the planned path (the simulator's steering noise picked another vessel),
it retracts both devices to just before the last bifurcation and replays the reference from there.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from stroke_nav.env.devices import CATH, WIRE
from stroke_nav.env.vessel_env import VesselNavEnv
from stroke_nav.planning.planner import PathPlanner
from stroke_nav.planning.trajectory import Trajectory, TrajectoryParams, build_trajectory


class TrajectoryFollower:
    def __init__(self, trajectory: Trajectory | None = None, device: str | None = None,
                 params: TrajectoryParams | None = None, use_shortest_path: bool = False,
                 deviation_mm: float = 2.5, max_retries: int = 20):
        """``trajectory=None`` plans one per episode from the env's start and target (A* for
        ``device``, default the env's success device; ``use_shortest_path`` uses plain Dijkstra)."""
        self.fixed = trajectory
        self.device = device
        self.params = params
        self.use_shortest_path = use_shortest_path
        self.deviation_mm = deviation_mm
        self.max_retries = max_retries

    # ------------------------------------------------------------------ episode
    def reset(self, env: VesselNavEnv) -> None:
        if self.fixed is not None:
            self.traj = self.fixed
        else:
            spec = self.device or env.cfg.sim.device(env.success_dev).spec
            planner = PathPlanner(env.anat, spec)
            plan = (planner.shortest if self.use_shortest_path else planner.plan)(env.start_node, env.target_node)
            if not plan.nodes:
                raise ValueError(f"planning failed: {plan.message}")
            plan.feasible = True  # a shortest-path baseline is executed even if it violates limits
            self.traj = build_trajectory(plan, env.cfg.sim, self.params)
        self.k = 0
        self.retries = 0
        self.recovering: float | None = None
        self._tree = cKDTree(self.traj.points)
        self._junction_s = [e["s_mm"] for e in self.traj.roll_events]

    def act(self, env: VesselNavEnv, obs=None) -> np.ndarray:
        sim, p = env.sim, env.cfg.sim
        tr = self.traj
        step_w = p.guidewire.v_max_mm_s * p.dt
        step_c = p.microcatheter.v_max_mm_s * p.dt
        rot_w = np.deg2rad(p.guidewire.w_max_deg_s) * p.dt
        rot_c = np.deg2rad(p.microcatheter.w_max_deg_s) * p.dt
        action = np.zeros(4)

        # wrong turn? (observed tip far from the planned path)
        tip = sim.position_at(sim.L[WIRE])
        dist, _ = self._tree.query(tip)
        if self.recovering is None and dist > self.deviation_mm and self.retries < self.max_retries:
            passed = [s for s in self._junction_s if s < sim.L[WIRE]]
            self.recovering = (passed[-1] if passed else sim.L[WIRE] - 10.0) - 3.0
            self.retries += 1
        if self.recovering is not None:
            back = self.recovering
            if sim.L[WIRE] > back + 0.1:
                action[0] = -np.clip((sim.L[WIRE] - back) / step_w, 0.1, 1.0)
            if sim.L[CATH] > back - 5.0 + 0.1:
                action[2] = -np.clip((sim.L[CATH] - (back - 5.0)) / step_c, 0.1, 1.0)
            if action[0] == 0 and action[2] == 0:
                # replay the reference from the sample where the wire was at this depth
                self.k = int(np.searchsorted(tr.L_wire, sim.L[WIRE]))
                self.k = max(0, self.k - 1)
                self.recovering = None
            else:
                return action

        k = min(self.k + 1, len(tr.t) - 1)
        action[0] = np.clip((tr.L_wire[k] - sim.L[WIRE]) / step_w, -1.0, 1.0)
        action[1] = np.clip(_wrap(tr.phi_wire[k] - sim.phi[WIRE]) / rot_w, -1.0, 1.0)
        action[2] = np.clip((tr.L_cath[k] - sim.L[CATH]) / step_c, -1.0, 1.0)
        action[3] = np.clip(_wrap(tr.phi_cath[k] - sim.phi[CATH]) / rot_c, -1.0, 1.0)
        self.k = k
        return action


def _wrap(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)
