# stroke-nav

A playground for training controllers that steer a **guidewire + microcatheter** through
patient-specific vessel models toward a stroke target. The setting is the mechanical-thrombectomy
access route: aorta, aortic arch, carotid, ICA, then M1 or A1.

- **Anatomy:** any model from the [Vascular Model Repository](https://www.vascularmodel.com/dataset.html) (VMR) that has centerline paths.
  The default is `0248_H_AOCERE_CAS`, a CT model from the aortic root to the circle of Willis that
  comes in 30 carotid-stenosis variants.
- **Robot:** 4 learnable action parameters, `[v_guidewire, w_guidewire, v_microcatheter, w_microcatheter]`.
- **Evaluation:** START consensus metrics (Robertshaw et al., JAHA 2026).

## Setup

```bash
py -3.12 -m venv --system-site-packages .venv
.venv\Scripts\pip install -e .[viz,dev]
```

## 1. Download a model

```bash
python scripts/fetch_data.py --list --anatomy cerebral        # browse models that have centerlines
python scripts/fetch_data.py --model 0248_H_AOCERE_CAS         # ~14 MB of a 717 MB zip
python scripts/fetch_data.py --model 0248_H_AOCERE_CAS --list-variants
python scripts/fetch_data.py --model 0248_H_AOCERE_CAS --variant 10Left100Right
python scripts/fetch_data.py --model 0050_H_CERE_H
```

Only the centerlines (`Paths/*.pth`), the surface model (`Models/*.vtp`) and the license are pulled.
The script uses HTTP range requests, so the full zip is never downloaded. The script then builds
`data/processed/<model>_<variant>.npz`, which holds the centerline graph (1 mm nodes, junctions,
loops) and the lumen radius per node:

- The radius is half the median wall-to-wall chord across the vessel, from rays cast perpendicular to the centerline.
  This stays accurate where the centerline runs off-centre, such as the carotid siphon.
- Centerline tails that run outside the surface model are trimmed.
- A gap of 3 mm or more inside a large vessel (median radius 1.5 mm or more) is marked as an occlusion, with radius 0.
  The 100% stenosis variants of 0248 show up this way.

```bash
python scripts/plot_anatomy.py --model 0248_H_AOCERE_CAS      # results/<model>_anatomy.png
```

## 2. Pick a target and drive

```bash
python scripts/playground.py --model 0248_H_AOCERE_CAS
python scripts/playground.py --model 0248_H_AOCERE_CAS --target right_anterior@20 --record demo.gif
```

| Key | Action |
|---|---|
| Up / Down | guidewire advance / retract |
| Left / Right | guidewire rotate |
| i / k | microcatheter advance / retract |
| n / m | microcatheter rotate |
| b | advance both |
| p | pick the point under the mouse as the target (press g to switch to setting the start) |
| o / space | one oracle step / oracle autopilot on-off |
| l | plan a device-feasible path to the target and show it, coloured by planned speed |
| h | follow the planned trajectory (robot autopilot) on-off |
| x | reset episode |
| c | toggle centerlines |
| z | print START metrics |

Picked points snap to the nearest centerline node and are saved to `data/targets/<model>.json`.

**Target specs.** The same format works everywhere (`--target`, `EnvConfig.target`, `reset(options={"target": ...})`):

| Spec | Meaning |
|---|---|
| `left_carotid@120` | 120 mm along a branch |
| `left_carotid@-5` | 5 mm from the branch's distal end |
| `random:left_carotid@-15--1\|right_carotid@-15--1` | a new random node per episode (here M1) |
| `xyz:-120,110,-200` | nearest node to a point (mm) |
| `1234` | a node id |
| `file:data/targets/0248_H_AOCERE_CAS.json` | a saved pick |

For 0248, the carotid paths run CCA, ICA, then M1; `*_anterior` is A1; `*_carotid_external` is the ECA.

## 3. Evaluate with START metrics

```bash
python scripts/evaluate.py --policies oracle,random,forward --episodes 100 \
    --target "random:left_carotid@-15--1|right_carotid@-15--1"
```

The script writes `results/<name>/start_summary.md` and `episodes.csv`. The reported metrics are:

- success rate with a Wilson 95% interval
- procedure time and fluoroscopy time
- number of steps
- phases (route bifurcations passed)
- wrong-branch failures
- handling errors (retractions plus wall pushes)
- path length per device
- tip speed
- path-following error
- contact force
- path ratio

Policies get identical episode seeds, so the comparison is paired. A trained Stable-Baselines3
policy plugs in as `ppo:path/to/model.zip`.

## 4. Plan a feasible path and a robot trajectory

```bash
python scripts/plan_path.py --target right_carotid@-5 --follow 20 --record
python scripts/plan_path.py --variant 10Left100Right --target right_carotid@-5 --follow 20
python scripts/plan_path.py --target right_carotid@-5 --forbid anterior_connector,left_communicating,right_communicating,right_posterior
python scripts/plan_path.py --target left_anterior@20 --device guidewire --hard-curvature
python scripts/compare_routes.py        # side-by-side GIF: shortest path vs device-aware plan (occluded right ICA)
```

The pipeline has four parts (`src/stroke_nav/planning/`):

1. **Device limits** (`constraints.py`). Each device has an outer diameter, a minimum bend radius and a clearance factor.
   Presets: 0.014" guidewire (0.36 mm), 1.7F microcatheter (0.57 mm) and 6F guide catheter (2.0 mm).
   A device may only enter lumen with diameter at least 1.2 times its outer diameter.
2. **Constrained A\*** (`planner.py`). The search runs over directed edges, so each step knows the incoming direction.
   - Hard limits: lumen clearance, and optionally bend radius (`--hard-curvature`) and forbidden vessels (`--forbid`).
   - Soft costs: length, narrow lumen, bending effort (curvature relative to the device limit), each vessel change, and sharp bifurcation take-off angles.
   - Plain Dijkstra is reported next to it as the baseline.
3. **Trajectory** (`trajectory.py`).
   - The speed profile is limited by curvature, narrow lumen and a slow zone before each bifurcation, then by acceleration.
   - For each bifurcation, the guidewire roll that makes the pre-curved tip enter the planned vessel is computed with the simulator's own tip model.
   - The output is a joint-space reference at 7.5 Hz: inserted length and roll for the guidewire and the microcatheter.
     The wire pauses before a junction until it has rotated, and the catheter trails 10 mm behind, then advances to the target.
4. **Follower** (`agents/follower.py`). The follower tracks the reference using only joint encoders and the observed tip position.
   If the tip leaves the planned path, it backs up past the last bifurcation and retries.

Each run writes `results/plans/<name>/` with these files:

| File | Contents |
|---|---|
| `report.md` | route comparison, bifurcation and roll table, trajectory summary, START metrics |
| `profiles.png` | radius, bend radius and speed along the path; joint references over time |
| `path3d.png` | 3D view: planned path coloured by speed, plain shortest path in grey |
| `trajectory.csv` | joint-space reference: `t_s, L_guidewire_mm, roll_guidewire_deg, L_microcatheter_mm, roll_microcatheter_deg` |
| `plan.json` | path nodes with geometry and speed, plus roll events |
| `follow.gif` | the follower executing the plan (with `--record`) |

In code:

```python
from stroke_nav.data.anatomy import Anatomy
from stroke_nav.planning.planner import PathPlanner
from stroke_nav.planning.trajectory import build_trajectory

anat = Anatomy.load("0248_H_AOCERE_CAS")
plan = PathPlanner(anat, "microcatheter").plan(anat.default_start(), anat.resolve("right_carotid@-5"))
traj = build_trajectory(plan)          # traj.t, traj.L_wire, traj.phi_wire, traj.L_cath, traj.phi_cath
traj.to_csv("trajectory.csv")
```

Example results on 0248, reference variant with 85% right ICA stenosis, target right M1. Each simulator row is 20 episodes.

| Route | Result |
|---|---|
| Guidewire plan | Straight up the right ICA. The 0.26 mm narrowest lumen clears the 0.22 mm the wire needs. |
| Microcatheter, plain Dijkstra | Infeasible, because the stenosis is narrower than the 0.34 mm the catheter needs. In the simulator it gets stuck at the stenosis: 0/20. |
| Microcatheter, constrained A\* | Up the left carotid, across the anterior communicating connector, then to the right M1. 44 mm longer. Executed 20/20 in 26.5 s. |
| Same plan, collaterals forbidden | No feasible path. In that case only the wire can cross; the catheter cannot. |

The detour through the circle of Willis is geometrically feasible, but it is not a standard clinical approach.
Use `--forbid` to exclude collateral vessels.

## Environment (`stroke_nav.env.vessel_env.VesselNavEnv`)

- **Action** `Box(-1, 1, (4,))`: speeds scaled by `v_max` (default 40 mm/s) and `w_max` (default 180 °/s) per device, at 7.5 Hz.
  Tune the limits in `EnvConfig.sim.guidewire` and `EnvConfig.sim.microcatheter` (`DeviceParams`), which also set the tip angle.
- **Observation** (46 floats): 3 tracking points per device tip, now and at the previous step, plus the target, the target minus the tip, and the previous action.
  These are the inputs used by Robertshaw et al. 2024 and 2025.
- **Reward:** `-0.005 - 0.01 * Δ(geodesic distance to target, mm) + 1.0 on success`.
  The papers use a progress coefficient of 0.001; set `progress_coef` to match.
- **Success:** the microcatheter tip (or the guidewire tip, via `success_device`) is within 5 mm of the target along the centerlines.
- **Device model** (`env/devices.py`): both devices share one track through the centerline graph. The leading device's pre-curved tip chooses the vessel at each bifurcation, based on its roll angle.
  Rotation is therefore required: pushing forward alone never reaches the carotids. A catheter close behind the wire reduces steering noise.
  The model is kinematic, with no wall forces yet, so contact force reports n/a and wall pushes stand in as the proxy.
- **Lumen limits:** a device cannot enter lumen narrower than its clearance requires (`SimParams.enforce_lumen`).
  For example, the microcatheter stops at a tight stenosis that the guidewire can cross.
  Episode routes, used by the oracle and for wrong-branch counting, are planned by the constrained A\* for the success device (`EnvConfig.plan_routes`).

## Baselines and tests

- `agents/scripted.py`: `ScriptedOracle` uses privileged knowledge of the route and the branch rule. It reaches 100% of targets in every tested region of 0248 (CCA, ICA, M1, A1, ECA, subclavian, vertebral) and of 0050.
  `RandomAgent` is the lower bound.
- `agents/follower.py`: `TrajectoryFollower` executes planned trajectories.
  It reaches 100% of random CCA, ICA, M1, A1 and vertebral targets on both the stenosis and the occlusion variant, at about 1.5 times the oracle's time, because it respects the speed profile.
- `pytest -q`: 32 tests on synthetic vessel trees. They cover:
  - env API conformance, determinism and the parser
  - oracle success on every branch and the need for rotation
  - START metrics
  - planner rerouting around a narrowing for the catheter but not the wire, and A\* matching Dijkstra without penalties
  - curvature-aware rerouting
  - trajectory speed, acceleration and ordering limits
  - the follower succeeding where shortest-path execution gets stuck
  - occlusion detection

## Data license and acknowledgement

VMR data is free for research and development. Keep each model's `LICENSE` file with any copy of the data.
Publications should acknowledge:

> The data used herein was provided in whole or in part with Federal funds from the National Library of
> Medicine under Grant No. R01LM013120, and the National Heart, Lung, and Blood Institute, National Institutes
> of Health, Department of Health and Human Services, under Contract No. HHSN268201100035C.

Also cite N.M. Wilson, A.K. Ortiz, A.B. Johnson, "The Vascular Model Repository: A Public Resource of Medical
Imaging Data and Blood Flow Simulation Results," J. Med. Devices 7(4), 040923 (2013), doi:10.1115/1.4025983.

## References

- Robertshaw H. et al. Autonomous navigation of catheters and guidewires in mechanical thrombectomy using inverse reinforcement learning. IJCARS 19, 1569–1578 (2024).
- Robertshaw H. et al. Reinforcement learning for safe autonomous two-device navigation of cerebral vessels in mechanical thrombectomy. IJCARS 20, 1077–1086 (2025).
- Robertshaw H. et al. (START). A position statement on endovascular models and effectiveness metrics for mechanical thrombectomy navigation. JAHA 15(7), e044931 (2026).
