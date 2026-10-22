"""START evaluation metrics for mechanical-thrombectomy navigation.

Follows the in-silico navigation metrics of the START position statement (Robertshaw et al.,
"A Position Statement on Endovascular Models and Effectiveness Metrics for Mechanical
Thrombectomy Navigation", JAHA 2026, doi:10.1161/JAHA.125.044931), plus the path ratio used in
Robertshaw et al. IJCARS 2024/2025.

Per-episode raw quantities come from ``VesselNavEnv.episode_metrics()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# (key, label, unit, how to aggregate: "rate" | "mean" | "mean_success")
START_METRICS: list[tuple[str, str, str, str]] = [
    ("success", "Success rate", "%", "rate"),
    ("procedure_time_s", "Procedure time (successful)", "s", "mean_success"),
    ("fluoroscopy_time_s", "Fluoroscopy time", "s", "mean"),
    ("steps", "Number of steps", "", "mean"),
    ("phases_passed", "Phases completed (route bifurcations passed)", "", "mean"),
    ("wrong_branch_events", "Failures: wrong-branch entries", "", "mean"),
    ("handling_errors", "Handling errors (retractions + wall pushes)", "", "mean"),
    ("path_length_m_guidewire", "Path length, guidewire", "m", "mean"),
    ("path_length_m_microcatheter", "Path length, microcatheter", "m", "mean"),
    ("tip_speed_m_s", "Instrument tip speed (mean)", "m/s", "mean"),
    ("centerline_error_pct", "Path-following error vs centerline", "%", "mean"),
    ("contact_force_N", "Contact force", "N", "mean"),
    ("wall_pushes", "Wall pushes (contact proxy)", "", "mean"),
    ("path_ratio", "Path ratio (progress toward target)", "%", "mean"),
]


def episode_start_metrics(raw: dict) -> dict:
    """Convert raw env quantities into START metrics (SI units where the statement asks for them)."""
    t = raw["steps"] * raw["dt"]
    d0 = max(raw["initial_distance_mm"], 1e-9)
    return {
        "success": float(raw["success"]),
        "procedure_time_s": t,
        "fluoroscopy_time_s": t,  # assumption: continuous fluoroscopy during navigation
        "steps": raw["steps"],
        "phases_passed": raw["phases_passed"],
        "phases_total": raw["phases_total"],
        "wrong_branch_events": raw["wrong_branch_events"],
        "handling_errors": raw["retractions_guidewire"] + raw["retractions_microcatheter"]
        + raw["wall_pushes_guidewire"] + raw["wall_pushes_microcatheter"],
        "path_length_m_guidewire": raw["path_length_mm_guidewire"] / 1000.0,
        "path_length_m_microcatheter": raw["path_length_mm_microcatheter"] / 1000.0,
        "tip_speed_m_s": raw["tip_speed_mean_mm_s"] / 1000.0,
        "centerline_error_pct": raw["centerline_error_pct"],
        "contact_force_N": raw["contact_force_N"],
        "wall_pushes": raw["wall_pushes_guidewire"] + raw["wall_pushes_microcatheter"],
        "path_ratio": 100.0 * (1.0 if raw["success"] else (d0 - raw["min_distance_mm"]) / d0),
        "target_branch": raw["target_branch"],
        "target_node": raw["target_node"],
        "initial_distance_mm": raw["initial_distance_mm"],
        "final_distance_mm": raw["final_distance_mm"],
    }


def wilson_ci(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Summary:
    policy: str
    n: int
    values: dict[str, str]  # metric key -> formatted cell


def summarize(policy: str, episodes: list[dict]) -> Summary:
    n = len(episodes)
    succ = np.array([e["success"] for e in episodes], dtype=float)
    cells: dict[str, str] = {}
    for key, _label, unit, how in START_METRICS:
        if how == "rate":
            lo, hi = wilson_ci(succ.sum(), n)
            cells[key] = f"{100 * succ.mean():.0f} [{100 * lo:.0f}-{100 * hi:.0f}]"
            continue
        vals = np.array([e[key] for e in episodes], dtype=float)
        if how == "mean_success":
            vals = vals[succ > 0]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            cells[key] = "n/a"
        else:
            prec = 3 if unit in ("m", "m/s", "N") else 1
            cells[key] = f"{vals.mean():.{prec}f} ± {vals.std():.{prec}f}"
    return Summary(policy, n, cells)


def markdown_table(summaries: list[Summary], title: str | None = None) -> str:
    head = "| START metric | unit | " + " | ".join(f"{s.policy} (n={s.n})" for s in summaries) + " |"
    sep = "|---|---|" + "---|" * len(summaries)
    rows = [head, sep]
    for key, label, unit, how in START_METRICS:
        unit_cell = unit + (" [95% CI]" if how == "rate" else "")
        rows.append(f"| {label} | {unit_cell} | " + " | ".join(s.values[key] for s in summaries) + " |")
    notes = (
        "\nValues are mean ± sd unless noted. Success rate uses a Wilson 95% interval. "
        "Fluoroscopy time assumes continuous imaging. Path-following error is 0 and contact force is "
        "not available because devices ride the centerline in the current kinematic model; wall pushes "
        "(forward pushes against a dead end) are the contact proxy."
    )
    out = (f"### {title}\n\n" if title else "") + "\n".join(rows) + "\n" + notes + "\n"
    return out
