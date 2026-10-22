"""A 1x128x128 uint8 image comparable to the organiser's policy input.

Their "fluoro" is not an X-ray simulation: it is a 480x480 WebGL perspective render (dark
background, dark-red vessel, devices, goal marker) that ``openng3_rl.ros_io`` converts to greyscale
and resizes to 128x128. We render the static vessel once with a camera estimated from a recorded
frame (silhouette match, approximate), then draw the devices and goal per step with numpy.

Grey levels were measured on their policy input: background ~13, vessel ~52, brightest ~124.
This keeps the image in the same range and layout; it is not pixel-identical (no grid, no
lighting model, approximate camera), so expect a visual domain gap when transferring policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stroke_nav.organiser.spec import IMAGE_SHAPE


@dataclass(frozen=True)
class CameraSpec:
    """Perspective camera in the organiser world frame (metres), estimated from a recorded frame."""

    azimuth_deg: float = 105.0
    elevation_deg: float = 45.0
    distance_m: float = 0.8
    view_angle_deg: float = 30.0
    focal_point_m: tuple[float, float, float] = (0.0312, 0.0632, 0.1487)  # their mesh centre
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class GreyLevels:
    background: int = 13
    vessel_dark: int = 38
    vessel_bright: int = 66
    guidewire: int = 118
    catheter: int = 96
    goal: int = 124


class FluoroRenderer:
    def __init__(self, surface_world_m, camera: CameraSpec | None = None, levels: GreyLevels | None = None,
                 size: int = IMAGE_SHAPE[1]):
        """``surface_world_m``: a pyvista PolyData of the vessel already in organiser world metres."""
        import pyvista as pv

        self.camera = camera or CameraSpec()
        self.levels = levels or GreyLevels()
        self.size = size
        pl = pv.Plotter(off_screen=True, window_size=(size, size))
        pl.set_background("black")
        pl.add_mesh(surface_world_m, color="white", smooth_shading=True)
        c = np.asarray(self.camera.focal_point_m, dtype=float)
        a, e = np.radians(self.camera.azimuth_deg), np.radians(self.camera.elevation_deg)
        d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        pl.camera.position = tuple(c + self.camera.distance_m * d)
        pl.camera.focal_point = tuple(c)
        pl.camera.up = self.camera.up
        pl.camera.view_angle = self.camera.view_angle_deg
        pl.render()
        shade = pl.screenshot(return_img=True)[..., :3].mean(-1) / 255.0
        cam = pl.renderer.GetActiveCamera()
        m = cam.GetCompositeProjectionTransformMatrix(1.0, -1.0, 1.0)
        self._proj = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
        pl.close()
        lv = self.levels
        vessel = shade > 0.02
        bg = np.full((size, size), float(lv.background))
        bg[vessel] = lv.vessel_dark + (lv.vessel_bright - lv.vessel_dark) * np.clip(shade[vessel], 0, 1)
        self.background = np.clip(bg, 0, 255).astype(np.uint8)

    def project(self, pts_world_m: np.ndarray) -> np.ndarray:
        """World points (N,3) -> pixel coordinates (N,2) as (col, row); NaN behind the camera."""
        p = np.atleast_2d(np.asarray(pts_world_m, dtype=float))
        h = np.c_[p, np.ones(len(p))] @ self._proj.T
        w = h[:, 3:4]
        ndc = np.where(w > 1e-9, h[:, :3] / np.where(w > 1e-9, w, 1.0), np.nan)
        col = (ndc[:, 0] + 1.0) * 0.5 * self.size
        row = (1.0 - ndc[:, 1]) * 0.5 * self.size
        return np.stack([col, row], axis=1)

    def render(self, devices: list[tuple[np.ndarray, int]], goal_world_m=None) -> np.ndarray:
        """devices: list of (polyline world metres (N,3), grey level). Returns uint8 (1, H, W)."""
        img = self.background.copy()
        for pts, level in devices:
            if pts is None or len(pts) < 2:
                continue
            px = self.project(pts)
            self._polyline(img, px, level)
        if goal_world_m is not None:
            g = self.project(np.asarray(goal_world_m)[None])[0]
            if np.all(np.isfinite(g)):
                rr, cc = np.ogrid[: self.size, : self.size]
                img[(rr - g[1]) ** 2 + (cc - g[0]) ** 2 <= 2.5] = self.levels.goal
        return img[None]

    def _polyline(self, img: np.ndarray, px: np.ndarray, level: int) -> None:
        ok = np.all(np.isfinite(px), axis=1)
        px = px[ok]
        if len(px) < 2:
            return
        seg = np.diff(px, axis=0)
        n = np.maximum(np.ceil(np.linalg.norm(seg, axis=1) * 2).astype(int), 1)
        t = np.concatenate([np.linspace(0, 1, k, endpoint=False) for k in n])
        idx = np.repeat(np.arange(len(seg)), n)
        pts = px[idx] + seg[idx] * t[:, None]
        pts = np.vstack([pts, px[-1:]])
        c = np.round(pts[:, 0]).astype(int)
        r = np.round(pts[:, 1]).astype(int)
        inside = (c >= 0) & (c < self.size) & (r >= 0) & (r < self.size)
        img[r[inside], c[inside]] = np.maximum(img[r[inside], c[inside]], level)
