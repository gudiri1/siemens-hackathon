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
loops) and the lumen radius per node. Centerline tails that run outside the surface model are trimmed.

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

## Baselines and tests

- `agents/scripted.py`: `ScriptedOracle` uses privileged knowledge of the route and the branch rule. It reaches 100% of targets in every tested region of 0248 (CCA, ICA, M1, A1, ECA, subclavian, vertebral) and of 0050.
  `RandomAgent` is the lower bound.
- `pytest -q`: 22 tests on a synthetic vessel tree covering env API conformance, determinism, oracle success on every branch, the need for rotation, the parser and the metrics.

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
