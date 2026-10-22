"""Side-by-side demo: a robot following the plain shortest path vs one following the device-aware plan.

Both runs use the same anatomy, start and target, and the same trajectory follower; only the route
differs. Saves an animated GIF, its final frame as PNG, and prints START metrics for both runs.

    python scripts/compare_routes.py                                   # occluded right ICA, right M1
    python scripts/compare_routes.py --variant 40Left85Right           # 85% stenosis instead
    python scripts/compare_routes.py --target left_anterior@20 --out results/demo/left_A1.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyvista as pv

from stroke_nav.agents.follower import TrajectoryFollower
from stroke_nav.env.vessel_env import EnvConfig, VesselNavEnv
from stroke_nav.eval.start_metrics import episode_start_metrics, markdown_table, summarize
from stroke_nav.viz.scene import Scene

TITLES = ("Shortest path (Dijkstra)", "Device-aware plan (A*)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="0248_H_AOCERE_CAS")
    ap.add_argument("--variant", default="10Left100Right")
    ap.add_argument("--target", default="right_carotid@-5")
    ap.add_argument("--max-steps", type=int, default=260)
    ap.add_argument("--out", default="results/demo/shortest_vs_planned.gif")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(model=args.model, variant=args.variant, target=args.target, max_steps=args.max_steps)
    envs = [VesselNavEnv(EnvConfig(**cfg)), VesselNavEnv(EnvConfig(**cfg))]
    agents = [TrajectoryFollower(use_shortest_path=True), TrajectoryFollower()]

    pl = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1500, 900), border=False)
    pl.set_background("white")
    scenes = []
    for i, (env, agent) in enumerate(zip(envs, agents)):
        env.reset(seed=0)
        agent.reset(env)
        pl.subplot(0, i)
        scene = Scene(env, pl, show_centerlines=False)
        path = pv.lines_from_points(agent.traj.points).tube(radius=0.35)
        pl.add_mesh(path, color="#2563eb" if i else "#9ca3af", opacity=0.35, name="plan", reset_camera=False)
        blocked = env.anat.radius <= 0.0
        if blocked.any():
            c = env.anat.nodes[blocked].mean(0)
            pl.add_mesh(pv.Sphere(radius=4.0, center=c), color="#dc2626", opacity=0.85, name="occ", reset_camera=False)
            pl.add_point_labels([c + [6, 0, 0]], ["occlusion"], font_size=12, text_color="#dc2626",
                                shape_opacity=0, show_points=False, name="occ_label", always_visible=True)
        pl.add_text(TITLES[i], position="upper_edge", font_size=13, color="black", name="title")
        scene.set_front_view(zoom=1.25)
        scenes.append(scene)

    infos: list[dict | None] = [None, None]

    def status(i: int) -> str:
        env = envs[i]
        lines = [f"t = {env.steps * env.cfg.sim.dt:4.1f} s    distance to target {env.D:5.0f} mm",
                 f"microcatheter in: {env.anat.branch_of(env.sim.node_at(env.sim.L[1]))}",
                 f"blocked pushes: {int(env.sim.wall_pushes.sum())}"]
        if infos[i] is not None:
            ok = infos[i]["episode_metrics"]["success"]
            lines.append("REACHED TARGET" if ok else "STUCK - did not reach target")
        return "\n".join(lines)

    def draw() -> None:
        for i, scene in enumerate(scenes):
            pl.subplot(0, i)
            scene.update(show_hud=False)
            tip = envs[i].sim.position_at(envs[i].sim.L[1])
            pl.add_mesh(pv.Sphere(radius=2.5, center=tip), color="#f59e0b", name="cath_marker", reset_camera=False)
            pl.add_text(status(i), position="lower_left", font_size=10, color="#111827", name="status")

    pl.open_gif(str(out), fps=12)
    draw()
    pl.write_frame()
    done = [False, False]
    while not all(done):
        for i in range(2):
            if done[i]:
                continue
            _, _, term, trunc, info = envs[i].step(agents[i].act(envs[i]))
            if term or trunc:
                done[i], infos[i] = True, info
        if envs[1].steps % 2 == 0 or all(done):
            draw()
            pl.write_frame()
    for _ in range(18):
        pl.write_frame()
    final_png = out.with_name(out.stem + "_final.png")
    pl.screenshot(str(final_png))
    pl.close()

    rows = [summarize(t, [episode_start_metrics(inf["episode_metrics"])]) for t, inf in zip(TITLES, infos)]
    print(markdown_table(rows))
    print(f"saved {out} and {final_png}")


if __name__ == "__main__":
    main()
