"""Build a centerline graph from SimVascular path files (``*.pth``) and a surface model (``*.vtp``).

Pipeline:
    parse every ``Paths/*.pth``  ->  convert cm to mm  ->  resample each path to ~1 mm
    ->  grow a tree outward from the root path (aorta), attaching each branch endpoint to the
        nearest node of the already-connected tree
    ->  add short loop edges (e.g. circle of Willis) between endpoints that touch another path
    ->  lumen radius per node from the surface model (distance from centerline to wall)
    ->  save ``data/processed/{model}[_{variant}].npz``
"""

from __future__ import annotations

import json
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_PROCESSED_DIR = Path("data/processed")


# --------------------------------------------------------------------------------------- parsing
def parse_pth_text(text: str) -> np.ndarray:
    """Return the interpolated path points ``(N, 3)`` of a SimVascular ``.pth`` file, in file units."""
    start = text.find("<path")
    if start < 0:
        raise ValueError("not a SimVascular path file (no <path> element)")
    root = ET.fromstring(text[start:])
    pts = [
        [float(pos.get("x")), float(pos.get("y")), float(pos.get("z"))]
        for pp in root.iter("path_point")
        for pos in pp.iter("pos")
    ]
    if not pts:  # some files only carry control points
        pts = [[float(p.get(k)) for k in "xyz"] for p in root.iter("point")]
    if len(pts) < 2:
        raise ValueError("path has fewer than 2 points")
    return np.asarray(pts, dtype=np.float64)


def parse_pth(path: Path | str) -> np.ndarray:
    return parse_pth_text(Path(path).read_text(encoding="utf-8", errors="replace"))


def infer_unit_scale(points: np.ndarray) -> float:
    """SimVascular models are usually in cm. Return 10 for cm, 1 for mm, based on the overall extent."""
    extent = float(np.max(points.max(0) - points.min(0)))
    return 10.0 if extent < 80.0 else 1.0


def resample_polyline(points: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Resample a polyline at (approximately) uniform arc-length spacing, keeping both endpoints."""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    points = points[keep]
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(round(s[-1] / spacing)) + 1)
    s_new = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(s_new, s, points[:, k]) for k in range(3)], axis=1)


# ----------------------------------------------------------------------------------------- graph
@dataclass
class CenterlineGraph:
    nodes: np.ndarray  # (N, 3) mm
    branch_id: np.ndarray  # (N,) int
    arc: np.ndarray  # (N,) arc length along own branch, mm
    branch_names: list[str]
    edges: np.ndarray  # (E, 2) int
    radius: np.ndarray  # (N,) mm, NaN if unknown
    meta: dict = field(default_factory=dict)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            nodes=self.nodes.astype(np.float32),
            branch_id=self.branch_id.astype(np.int32),
            arc=self.arc.astype(np.float32),
            branch_names=np.array(self.branch_names),
            edges=self.edges.astype(np.int32),
            radius=self.radius.astype(np.float32),
            meta=np.array(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: Path | str) -> "CenterlineGraph":
        z = np.load(path, allow_pickle=False)
        return cls(
            nodes=z["nodes"].astype(np.float64),
            branch_id=z["branch_id"],
            arc=z["arc"].astype(np.float64),
            branch_names=[str(s) for s in z["branch_names"]],
            edges=z["edges"],
            radius=z["radius"].astype(np.float64),
            meta=json.loads(str(z["meta"])),
        )


def build_graph(
    paths: dict[str, np.ndarray],
    root: str | None = None,
    spacing: float = 1.0,
    attach_max_mm: float = 25.0,
    loop_max_mm: float = 2.5,
) -> CenterlineGraph:
    """Connect resampled paths (already in mm) into one graph.

    Tree growth: starting from ``root`` (default: a path named 'aorta', else the longest path),
    repeatedly attach the unconnected path whose endpoint is closest to any connected node.
    Loops: afterwards, every endpoint within ``loop_max_mm`` of another path gets an extra edge.
    """
    names = sorted(paths)
    if root is None:
        root = "aorta" if "aorta" in paths else max(names, key=lambda n: _length(paths[n]))
    if root not in paths:
        raise KeyError(f"root path {root!r} not found; available: {names}")

    pts_list, bid_list, arc_list, offsets = [], [], [], {}
    offset = 0
    for i, n in enumerate(names):
        p = resample_polyline(paths[n], spacing)
        offsets[n] = (offset, offset + len(p))
        pts_list.append(p)
        bid_list.append(np.full(len(p), i))
        arc_list.append(np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]))
        offset += len(p)
    nodes = np.concatenate(pts_list)
    branch_id = np.concatenate(bid_list)
    arc = np.concatenate(arc_list)

    edges: set[tuple[int, int]] = set()
    for n in names:
        a, b = offsets[n]
        edges.update((i, i + 1) for i in range(a, b - 1))

    def endpoints(n: str) -> tuple[int, int]:
        a, b = offsets[n]
        return a, b - 1

    connected = {root}
    attach_info: dict[str, dict] = {}
    unconnected = set(names) - connected
    while unconnected:
        conn_idx = np.concatenate([np.arange(*offsets[c]) for c in sorted(connected)])
        tree = cKDTree(nodes[conn_idx])
        best = None
        for n in sorted(unconnected):
            for e in endpoints(n):
                d, j = tree.query(nodes[e])
                if best is None or d < best[0]:
                    best = (d, n, e, int(conn_idx[j]))
        d, n, e, j = best
        if d > attach_max_mm:
            warnings.warn(f"paths {sorted(unconnected)} are > {attach_max_mm} mm from the tree; dropped")
            break
        edges.add((min(e, j), max(e, j)))
        attach_info[n] = {
            "parent": names[branch_id[j]],
            "parent_arc_mm": round(float(arc[j]), 1),
            "distance_mm": round(float(d), 2),
            "endpoint": "start" if e == offsets[n][0] else "end",
        }
        connected.add(n)
        unconnected.discard(n)

    kept = sorted(connected)
    # Loop edges between endpoints and other paths (only among connected paths).
    loops = []
    all_idx = np.concatenate([np.arange(*offsets[c]) for c in kept])
    tree_all = cKDTree(nodes[all_idx])
    for n in kept:
        for e in endpoints(n):
            for j_local in tree_all.query_ball_point(nodes[e], loop_max_mm):
                j = int(all_idx[j_local])
                if branch_id[j] == branch_id[e]:
                    continue
                key = (min(e, j), max(e, j))
                if key in edges:
                    continue
                # Only one loop edge per (endpoint, other branch): keep the closest node.
                others = [
                    int(all_idx[k])
                    for k in tree_all.query_ball_point(nodes[e], loop_max_mm)
                    if branch_id[all_idx[k]] == branch_id[j]
                ]
                closest = min(others, key=lambda k: np.linalg.norm(nodes[k] - nodes[e]))
                key = (min(e, closest), max(e, closest))
                if key not in edges and not _already_linked(edges, e, branch_id, branch_id[closest]):
                    edges.add(key)
                    loops.append((n, names[branch_id[closest]], round(float(np.linalg.norm(nodes[closest] - nodes[e])), 2)))

    # Drop nodes of paths that could not be attached and re-index.
    keep_mask = np.isin(branch_id, [names.index(n) for n in kept])
    remap = -np.ones(len(nodes), dtype=np.int64)
    remap[keep_mask] = np.arange(keep_mask.sum())
    edge_arr = np.array([(remap[a], remap[b]) for a, b in sorted(edges) if keep_mask[a] and keep_mask[b]])
    new_names = kept
    name_map = {names.index(n): i for i, n in enumerate(new_names)}
    bid_new = np.array([name_map[b] for b in branch_id[keep_mask]])

    meta = {"root": root, "spacing_mm": spacing, "attachments": attach_info, "loops": loops}
    return CenterlineGraph(
        nodes=nodes[keep_mask],
        branch_id=bid_new,
        arc=arc[keep_mask],
        branch_names=new_names,
        edges=edge_arr,
        radius=np.full(int(keep_mask.sum()), np.nan),
        meta=meta,
    )


def _length(p: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _already_linked(edges: set, node: int, branch_id: np.ndarray, other_branch: int) -> bool:
    for a, b in edges:
        if a == node and branch_id[b] == other_branch:
            return True
        if b == node and branch_id[a] == other_branch:
            return True
    return False


# ---------------------------------------------------------------------------------------- radius
def lumen_radius(nodes_mm: np.ndarray, surface_path: Path | str, unit_scale: float) -> tuple[np.ndarray, float]:
    """Distance from each centerline node to the vessel wall, in mm.

    Returns (radius, inside_fraction). Uses VTK's implicit poly-data distance on the surface model.
    """
    import pyvista as pv

    surf = pv.read(str(surface_path))
    surf = surf.extract_surface(algorithm="dataset_surface").triangulate()
    surf.points = surf.points * unit_scale
    probe = pv.PolyData(nodes_mm.astype(np.float64))
    probe = probe.compute_implicit_distance(surf)
    d = np.asarray(probe["implicit_distance"], dtype=np.float64)
    inside_frac = float(np.mean(d < 0))
    if inside_frac < 0.5:  # normals point inward: flip the convention
        d = -d
        inside_frac = 1.0 - inside_frac
    radius = np.where(d < 0, -d, np.nan)  # nodes outside the wall get NaN
    return radius, inside_frac


def trim_outside_leaves(graph: CenterlineGraph, min_run: int = 3, min_radius: float = 0.3) -> dict[str, float]:
    """Remove branch tails that run outside the surface model and fill interior radius gaps.

    Centerlines often extend past the truncated surface model. A run of outside (NaN-radius) nodes
    touching a leaf endpoint (degree 1) is removed; remaining NaNs are interpolated along the branch.
    Returns {branch: trimmed length in mm}. Modifies ``graph`` in place.
    """
    n = len(graph.nodes)
    degree = np.bincount(graph.edges.ravel(), minlength=n)
    drop = np.zeros(n, dtype=bool)
    trimmed: dict[str, float] = {}
    for b, name in enumerate(graph.branch_names):
        idx = np.where(graph.branch_id == b)[0]
        outside = ~np.isfinite(graph.radius[idx])
        for ordered in (idx, idx[::-1]):
            if degree[ordered[0]] != 1:
                continue
            out = ~np.isfinite(graph.radius[ordered])
            run = int(np.argmin(out)) if not out.all() else len(out)
            if run >= min_run and run < len(ordered):
                drop[ordered[:run]] = True
                trimmed[name] = trimmed.get(name, 0.0) + float(abs(graph.arc[ordered[run - 1]] - graph.arc[ordered[0]]))
        del outside
    keep = ~drop
    remap = -np.ones(n, dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    e = graph.edges
    e = e[keep[e[:, 0]] & keep[e[:, 1]]]
    graph.edges = remap[e]
    graph.nodes = graph.nodes[keep]
    graph.branch_id = graph.branch_id[keep]
    graph.arc = graph.arc[keep]
    graph.radius = graph.radius[keep]

    # Interpolate remaining interior gaps along each branch, then clamp.
    for b in range(len(graph.branch_names)):
        idx = np.where(graph.branch_id == b)[0]
        r = graph.radius[idx]
        ok = np.isfinite(r)
        if ok.any() and not ok.all():
            graph.radius[idx] = np.interp(graph.arc[idx], graph.arc[idx][ok], r[ok])
        elif not ok.any():
            graph.radius[idx] = min_radius
    graph.radius = np.maximum(graph.radius, min_radius)
    return trimmed


# ----------------------------------------------------------------------------------------- build
def processed_path(model: str, variant: str | None = None, processed_dir: Path | str = DEFAULT_PROCESSED_DIR) -> Path:
    stem = model if variant is None else f"{model}_{variant}"
    return Path(processed_dir) / f"{stem}.npz"


def build_model(
    model: str,
    variant: str | None = None,
    raw_dir: Path | str = "data/raw",
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    root: str | None = None,
    spacing: float = 1.0,
    compute_radius: bool = True,
    verbose: bool = True,
) -> CenterlineGraph:
    """Parse a fetched VMR model folder and save its centerline graph as ``.npz``."""
    folder = Path(raw_dir) / model
    pth_files = sorted((folder / "Paths").glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"no .pth files in {folder / 'Paths'}; run scripts/fetch_data.py --model {model}")
    raw = {f.stem: parse_pth(f) for f in pth_files}
    scale = infer_unit_scale(np.concatenate(list(raw.values())))
    paths_mm = {k: v * scale for k, v in raw.items()}
    graph = build_graph(paths_mm, root=root, spacing=spacing)
    graph.meta.update({"model": model, "variant": variant, "unit_scale": scale})

    vtps = sorted((folder / "Models").glob("*.vtp"))
    if variant is not None:
        vtps = [v for v in vtps if v.stem == variant]
    if compute_radius and vtps:
        try:
            radius, inside = lumen_radius(graph.nodes, vtps[0], scale)
            graph.radius = radius
            graph.meta.update({"surface": vtps[0].name, "inside_fraction_raw": round(inside, 4)})
            trimmed = trim_outside_leaves(graph)
            graph.meta["trimmed_mm"] = {k: round(v, 1) for k, v in trimmed.items()}
            if verbose and trimmed:
                print("[build] trimmed tails outside the surface: "
                      + ", ".join(f"{k} {v:.0f} mm" for k, v in trimmed.items()))
        except Exception as e:  # pragma: no cover - depends on VTK install
            warnings.warn(f"radius computation failed ({e}); radius left as NaN")
    graph.meta["surface_file"] = str(vtps[0]) if vtps else None

    out = graph.save(processed_path(model, variant or graph.meta.get("surface", "")[:-4] or None, processed_dir))
    if verbose:
        ext = graph.nodes.max(0) - graph.nodes.min(0)
        print(f"[build] {model}: {len(graph.branch_names)} branches, {len(graph.nodes)} nodes, "
              f"{len(graph.edges)} edges, extent {np.round(ext, 1)} mm, unit scale x{scale:g}")
        for n, info in graph.meta["attachments"].items():
            print(f"[build]   {n:<26} -> {info['parent']:<26} at {info['parent_arc_mm']:6.1f} mm "
                  f"({info['endpoint']}, gap {info['distance_mm']} mm)")
        for a, b, d in graph.meta["loops"]:
            print(f"[build]   loop {a} <-> {b} (gap {d} mm)")
        if np.isfinite(graph.radius).any():
            r = graph.radius[np.isfinite(graph.radius)]
            print(f"[build]   radius: median {np.median(r):.2f} mm, range {r.min():.2f}-{r.max():.2f} mm, "
                  f"raw inside fraction {graph.meta.get('inside_fraction_raw')}")
        print(f"[build] saved {out}")
    return graph
