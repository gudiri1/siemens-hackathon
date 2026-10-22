"""3D scene for the playground: vessel surface, centerlines, devices, start/target markers, HUD."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stroke_nav.env.devices import CATH, WIRE
from stroke_nav.env.vessel_env import VesselNavEnv

WIRE_COLOR = "#1f2937"
CATH_COLOR = "#f59e0b"
TARGET_COLOR = "#16a34a"
START_COLOR = "#2563eb"
VESSEL_COLOR = "#e8a0a0"


def load_surface(env: VesselNavEnv, target_cells: int = 80_000):
    """Load the anatomy's surface model in mm, decimated and cached next to the processed graph."""
    import pyvista as pv

    meta = env.anat.meta
    src = meta.get("surface_file")
    if not src or not Path(src).exists():
        return None
    cache = Path("data/processed") / f"{Path(src).stem}_{meta.get('model', 'model')}_display.vtp"
    if cache.exists():
        return pv.read(str(cache))
    surf = pv.read(src).extract_surface(algorithm="dataset_surface").triangulate()
    surf.points = surf.points * float(meta.get("unit_scale", 1.0))
    if surf.n_cells > target_cells:
        surf = surf.decimate(1.0 - target_cells / surf.n_cells)
    surf = surf.compute_normals(auto_orient_normals=True)
    surf.save(str(cache))
    return surf


class Scene:
    def __init__(self, env: VesselNavEnv, plotter, show_centerlines: bool = True):
        import pyvista as pv

        self.pv = pv
        self.env = env
        self.pl = plotter
        self.surface = load_surface(env)
        if self.surface is not None:
            self.pl.add_mesh(self.surface, color=VESSEL_COLOR, opacity=0.22, smooth_shading=True,
                             pickable=True, name="vessel")
        a = env.anat
        lines = []
        for u, v in a.graph.edges:
            lines += [2, int(u), int(v)]
        self.centerlines = pv.PolyData(a.nodes.copy(), lines=np.array(lines))
        self.centerlines["branch"] = a.branch_id.astype(float)
        self.cl_actor = self.pl.add_mesh(self.centerlines, scalars="branch", cmap="tab20", line_width=1.5,
                                         opacity=0.55, show_scalar_bar=False, pickable=True, name="centerlines")
        self.cl_actor.SetVisibility(show_centerlines)
        self.hud = None

    # ------------------------------------------------------------------ updates
    def update(self, status: str = "", keys_help: str | None = None) -> None:
        env, sim, pv = self.env, self.env.sim, self.pv
        for dev, color, radius, name in ((WIRE, WIRE_COLOR, 0.35, "wire"), (CATH, CATH_COLOR, 0.8, "cath")):
            pts = sim.device_polyline(dev)
            if len(pts) >= 2:
                tube = pv.lines_from_points(pts).tube(radius=radius, n_sides=10)
                self.pl.add_mesh(tube, color=color, name=name, pickable=False, reset_camera=False)
            tip = sim.tracking_points(dev)[0]
            self.pl.add_mesh(pv.Sphere(radius=radius * 1.6, center=tip), color=color, name=name + "_tip",
                             pickable=False, reset_camera=False)
        thr = env.cfg.success_threshold_mm
        self.pl.add_mesh(pv.Sphere(radius=thr / 2, center=env.target_pos), color=TARGET_COLOR, opacity=0.8,
                         name="target", pickable=False, reset_camera=False)
        self.pl.add_mesh(pv.Sphere(radius=3.0, center=env.anat.nodes[env.start_node]), color=START_COLOR,
                         name="start", pickable=False, reset_camera=False)
        self.pl.add_text(self.hud_text(status), position="upper_left", font_size=9, name="hud", color="black")
        if keys_help:
            self.pl.add_text(keys_help, position="lower_left", font_size=8, name="keys", color="#374151")

    def hud_text(self, status: str) -> str:
        env, sim = self.env, self.env.sim
        a = env.anat
        lines = [
            f"{a.name}",
            f"target: {a.branch_of(env.target_node)} (node {env.target_node})   start: {a.branch_of(env.start_node)}",
            f"step {env.steps}   time {env.steps * env.cfg.sim.dt:5.1f} s   "
            f"distance to target {env.D:6.1f} mm (geodesic, {env.cfg.success_device} tip)",
            f"guidewire     L {sim.L[WIRE]:6.1f} mm  roll {np.degrees(sim.phi[WIRE]):5.0f} deg  "
            f"in {a.branch_of(sim.node_at(sim.L[WIRE]))}",
            f"microcatheter L {sim.L[CATH]:6.1f} mm  roll {np.degrees(sim.phi[CATH]):5.0f} deg  "
            f"in {a.branch_of(sim.node_at(sim.L[CATH]))}",
            f"wrong-branch entries {env._ep['wrong_branch_events']}   last action "
            + "[" + ", ".join(f"{x:+.1f}" for x in env.prev_action) + "]",
        ]
        if status:
            lines.append(status)
        return "\n".join(lines)

    def set_front_view(self, zoom: float = 1.35) -> None:
        """Anterior (AP) view, head up, patient's left on screen right (VMR models use RAS axes)."""
        a = self.env.anat
        c = a.center
        d = 3.0 * a.half_extent
        self.pl.camera_position = [(c[0], c[1] + d, c[2]), tuple(c), (0, 0, 1)]
        self.pl.reset_camera(bounds=(*_interleave(a.bbox_min, a.bbox_max),))
        self.pl.camera.zoom(zoom)


def _interleave(lo: np.ndarray, hi: np.ndarray) -> list[float]:
    return [float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2])]
