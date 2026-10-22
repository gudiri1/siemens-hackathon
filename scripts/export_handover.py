"""Package work for the organiser: a trained run and/or a planned trajectory, in their formats.

    python scripts/export_handover.py --run-dir runs/<run_id>
    python scripts/export_handover.py --plan-target right_carotid_external@21
    python scripts/export_handover.py --run-dir runs/<run_id> --plan-target left_carotid@60 --name team-handover

Creates handover/<name>/ and handover/<name>.zip containing:
    runs/<run_id>/                  drop into HCMR-EV-RL/runs/ (validated with check_run_dir)
    trajectory/device_commands.csv  DeviceCommand setpoints (mm, deg; 20 Hz) for /endovascular/<dev>/command
    trajectory/twists.csv           Twist velocities (mm/s, rad/s; 5 Hz) for /control/autonomous/module_<n>/cmd_vel
    trajectory/policy_actions.csv   the same as normalised RL actions [-1, 1]^4 at 5 Hz
    trajectory/goals.txt            the target in their mesh frame (cm), for a scenario goals file
    trajectory/plan.json            path, speed profile and roll events (our frame) + summary
    ros2/*.py                       players for their ROS 2 stack (untested outside it)
    organiser_frame.json            transform between our anatomy and their world frame
    compat/openng3_rl/policy.py     the feature extractor the model references
    HANDOVER.md, ORGANISER_SETUP.md, MANIFEST.json (sha256 of every file)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--plan-target", default=None, help="target spec in our anatomy (see README), e.g. left_carotid@60")
    ap.add_argument("--plan-device", default="radifocus_035", help="device the path must fit (organiser guidewire by default)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "handover")
    args = ap.parse_args()
    if not args.run_dir and not args.plan_target:
        ap.error("give --run-dir and/or --plan-target")

    from stroke_nav.organiser.export import (check_run_dir, trajectory_device_commands, trajectory_policy_actions,
                                             trajectory_twists, write_csv, write_goal_for_target)
    from stroke_nav.organiser.frame import default_frame

    frame = default_frame()
    name = args.name or (args.run_dir.name if args.run_dir else "trajectory-" + args.plan_target.replace("@", "-at-"))
    out = args.out / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    if args.run_dir:
        problems = check_run_dir(args.run_dir)
        if problems:
            raise SystemExit(f"run folder would be rejected by the organiser: {problems}")
        dst = out / "runs" / args.run_dir.name
        shutil.copytree(args.run_dir, dst, ignore=shutil.ignore_patterns("tensorboard"))
        print(f"[export] run folder {args.run_dir} -> {dst}")

    if args.plan_target:
        export_trajectory(args.plan_target, args.plan_device, out / "trajectory", frame,
                          trajectory_device_commands, trajectory_twists, trajectory_policy_actions, write_csv, write_goal_for_target)

    (out / "ros2").mkdir()
    for f in (ROOT / "src/stroke_nav/organiser/templates").glob("*.py"):
        shutil.copy(f, out / "ros2" / f.name)
    frame.save(out / "organiser_frame.json")
    (out / "compat/openng3_rl").mkdir(parents=True)
    for f in ("__init__.py", "policy.py"):
        shutil.copy(ROOT / "compat/openng3_rl" / f, out / "compat/openng3_rl" / f)
    for doc in ("HANDOVER.md", "ORGANISER_SETUP.md"):
        if (ROOT / "docs" / doc).is_file():
            shutil.copy(ROOT / "docs" / doc, out / doc)
    manifest = {str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    print(f"[export] wrote {out} ({len(manifest)} files) and {archive}")


def export_trajectory(target_spec, plan_device, tdir: Path, frame, device_commands, twists, policy_actions, write_csv, write_goal):
    from stroke_nav.data.anatomy import Anatomy
    from stroke_nav.organiser.env import organiser_sim_params
    from stroke_nav.planning.constraints import get_device
    from stroke_nav.planning.planner import PathPlanner
    from stroke_nav.planning.trajectory import TrajectoryParams, build_trajectory

    anat = Anatomy.load("0248_H_AOCERE_CAS")
    start, target = anat.default_start(), anat.resolve(target_spec)
    plan = PathPlanner(anat, plan_device).plan(start, target)
    if not plan.feasible:
        raise SystemExit(f"no feasible path for {plan_device}: {plan.message}")
    # the organiser's 5F catheter cannot follow the wire into narrow vessels: stop it where it no longer fits
    cath_r = get_device("judkins_5f").min_lumen_radius_mm
    narrow = np.where(plan.radius < cath_r)[0]
    cath_limit = float(plan.s[narrow[0]] - 5.0) if len(narrow) else None
    sim = organiser_sim_params()
    traj = build_trajectory(plan, sim, TrajectoryParams(catheter_limit_mm=cath_limit))
    tdir.mkdir(parents=True, exist_ok=True)
    write_csv(device_commands(traj), tdir / "device_commands.csv")
    rows, report = twists(traj)
    write_csv(rows, tdir / "twists.csv")
    acts = policy_actions(traj)
    write_csv([{"t_s": round(i / 5.0, 3), "gw_trans": a[0], "gw_rot": a[1], "cath_trans": a[2], "cath_rot": a[3]} for i, a in enumerate(acts)],
              tdir / "policy_actions.csv")
    write_goal(tdir / "goals.txt", frame, anat.nodes[target], f"stroke-nav target {target_spec} ({anat.branch_of(target)})")
    traj.to_json(tdir / "plan.json")
    summary = {"target": target_spec, "target_branch": anat.branch_of(target),
               "target_world_m": frame.vmr_mm_to_world_m(anat.nodes[target]).round(5).tolist(),
               "plan": plan.summary(), "trajectory": traj.summary(), "catheter_stops_at_mm": cath_limit,
               "twist_limits": report,
               "note": "Insertion setpoints transfer directly. Roll setpoints come from our kinematic tip model and are a "
                       "starting point only; their rod physics needs closed-loop correction of rotation."}
    (tdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(f"[export] trajectory to {target_spec}: {plan.summary()['branches']}, {traj.summary()['duration_s']} s, "
          f"catheter stops at {cath_limit} mm; twist limits {report['max_abs']}")


if __name__ == "__main__":
    main()
