import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from stroke_nav.organiser import ensure_policy_importable
from stroke_nav.organiser.env import CompatConfig, OrganiserCompatEnv, compute_reward, organiser_sim_params
from stroke_nav.organiser.export import (check_run_dir, trajectory_device_commands, trajectory_policy_actions,
                                         trajectory_twists, write_run_dir)
from stroke_nav.organiser.frame import OrganiserFrame, read_goals_txt, write_goals_txt
from stroke_nav.organiser.spec import (CAS_REFERENCE_GOALS_CM, DEFAULT_SPEC, IMAGE_SHAPE, STATE_SIZE,
                                       device_parameters_hash, validate_device_parameters)
from stroke_nav.planning.planner import PathPlanner
from stroke_nav.planning.trajectory import TrajectoryParams, build_trajectory

ROOT = Path(__file__).resolve().parents[1]
ORGANISER_RUN_HASH = "38b1ea532f0a929ed5d36111e2e0f0efb28ca1610a40ade0bccc987813a65fbc"


def test_device_parameters_match_organiser_run():
    assert device_parameters_hash(DEFAULT_SPEC.device_parameters()) == ORGANISER_RUN_HASH
    bad = DEFAULT_SPEC.device_parameters()
    del bad["controls"]["diameter"]
    with pytest.raises(ValueError):
        validate_device_parameters(bad)


def test_spec_values():
    s = DEFAULT_SPEC
    assert s.dt == pytest.approx(0.2)
    assert s.action.max_translation_m_s == 0.01 and s.action.max_rotation_rad_s == 1.0
    assert s.environment.max_steps == 600 and s.environment.success_steps == 3
    assert s.guidewire_geometry()["outer_diameter_mm"] == 0.89 and s.catheter_geometry()["outer_diameter_mm"] == 1.67
    cfg = s.online_rl_config()
    assert cfg["schema_version"] == 1 and cfg["training"]["buffer_size"] == 50_000


def test_frame_round_trip_and_known_points():
    f = OrganiserFrame()
    assert np.allclose(f.M @ f.M.T, np.eye(3), atol=1e-9) and np.linalg.det(f.M) > 0
    p = np.array([[-100.6, 39.0, -385.4], [-84.7, 123.7, -212.1]])
    assert np.allclose(f.world_m_to_vmr_mm(f.vmr_mm_to_world_m(p)), p, atol=1e-9)
    # our default start (distal descending aorta) is their device entry, the world origin
    assert np.linalg.norm(f.vmr_mm_to_world_m(p[0])) < 0.005
    goal = np.asarray(CAS_REFERENCE_GOALS_CM[0])
    assert np.allclose(f.vmr_mm_to_obj_cm(f.obj_cm_to_vmr_mm(goal)), goal, atol=1e-9)


def test_goals_txt_round_trip(tmp_path):
    g = np.array([[7.001, 9.756, 15.35], [1.0, 2.0, 3.0]])
    path = write_goals_txt(tmp_path / "goals.txt", g, "test")
    assert np.allclose(read_goals_txt(path), g, atol=1e-3)


def test_reward_matches_organiser_formula():
    t = compute_reward(0.100, 0.095, 0.0, 0.2, [1, 0, 0, 0], [0, 0, 0, 0], False, False)
    assert t["progress"] == pytest.approx(0.5)
    assert t["action_change"] == pytest.approx(-0.02 * 0.25)
    assert t["total"] == pytest.approx(0.5 - 0.005 - 0.001)
    t = compute_reward(0.01, 0.0, 2.0, 0.2, [0] * 4, [0] * 4, True, False)
    assert t["progress"] == 1.0 and t["contact"] == pytest.approx(-40.0) and t["terminal"] == 10.0


@pytest.fixture()
def compat_env(toy_anatomy):
    cfg = CompatConfig(goal_mode="target", target="left_branch@-3", image_mode="blank", noise=False)
    return OrganiserCompatEnv(cfg, anatomy=toy_anatomy, frame=OrganiserFrame())


def test_compat_env_spaces_and_api(compat_env):
    env = compat_env
    assert env.observation_space["image"].shape == IMAGE_SHAPE and env.observation_space["state"].shape == (STATE_SIZE,)
    assert env.action_space.shape == (4,)
    check_env(env, skip_render_check=True)


def test_compat_env_action_scaling_and_state_layout(compat_env):
    env = compat_env
    obs, _ = env.reset(seed=0)
    L0 = env.inner.sim.L.copy()
    obs, _, _, _, _ = env.step(np.array([1.0, 0.0, 0.0, 0.0]))
    assert env.inner.sim.L[0] - L0[0] == pytest.approx(2.0, abs=1e-6)  # 10 mm/s * 0.2 s
    state = obs["state"]
    tip = env._true_tip_world(0)
    assert np.allclose(state[0:3], tip, atol=1e-6)  # noise off: guidewire tip position in world metres
    assert np.linalg.norm(state[3:7]) == pytest.approx(1.0, abs=1e-5)  # unit quaternion
    assert state[17] == pytest.approx(env.inner.sim.L[0] / 1000.0, abs=1e-6)  # insertion depth (m)
    assert np.allclose(state[36:39], env.goal - tip, atol=1e-6)
    # Like openng3_rl.env.OpenNG3Env, the observation after step t carries the action of step t-1
    # (their builder runs before _prev_action is updated).
    assert np.allclose(state[39:43], 0.0)
    obs, _, _, _, _ = env.step(np.array([0.0, 0.5, 0.0, 0.0]))
    assert np.allclose(obs["state"][39:43], [1.0, 0.0, 0.0, 0.0])


def test_compat_env_success_rule(compat_env):
    env = compat_env
    env.reset(seed=0)
    tip = env._true_tip_world(0)
    env.goal = tip.copy()  # put the goal on the tip: success after 3 consecutive steps in range
    env._prev_dist = 0.0
    results = [env.step(np.zeros(4)) for _ in range(3)]
    assert [r[4]["success"] for r in results] == [False, False, True]
    assert results[-1][2] is True and results[-1][1] > 9.0  # terminated with the success bonus


def test_run_dir_validates_and_detects_tampering(tmp_path):
    run = write_run_dir(tmp_path / "run-1", DEFAULT_SPEC)
    (run / "final-model.zip").write_bytes(b"placeholder")
    assert check_run_dir(run) == []
    cfg = json.loads((run / "resolved-config.json").read_text())
    assert cfg["device_parameters"]["sha256"] == ORGANISER_RUN_HASH and cfg["scenario"]["name"] == "cas-reference"
    doc = json.loads((run / "device-parameters.json").read_text())
    doc["controls"]["diameter"] = 0.36
    (run / "device-parameters.json").write_text(json.dumps(doc))
    assert any("hash" in p for p in check_run_dir(run))


def test_trajectory_exports_respect_limits(toy_anatomy):
    plan = PathPlanner(toy_anatomy, "radifocus_035").plan(toy_anatomy.default_start(), toy_anatomy.resolve("left_branch@-3"))
    traj = build_trajectory(plan, organiser_sim_params(), TrajectoryParams(catheter_limit_mm=30.0))
    assert traj.L_cath.max() <= 30.0 + 1e-6
    cmds = trajectory_device_commands(traj)
    assert {c["device"] for c in cmds} == {"guidewire", "catheter"} and all(c["frames"] == 3 for c in cmds)
    rows, report = trajectory_twists(traj)
    assert not report["exceeds_rl_action_limit"] and not report["exceeds_bridge_limit"]
    acts = trajectory_policy_actions(traj)
    assert acts.shape[1] == 4 and np.all(np.abs(acts) <= 1.0)


def test_sb3_model_is_loadable_by_organiser(tmp_path, compat_env):
    ensure_policy_importable()
    from openng3_rl.policy import FluoroStateExtractor
    from stable_baselines3 import SAC

    model = SAC("MultiInputPolicy", compat_env, policy_kwargs={"features_extractor_class": FluoroStateExtractor},
                buffer_size=200, learning_starts=1000, batch_size=8, device="cpu", seed=0)
    path = tmp_path / "final-model.zip"
    model.save(path)
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("data"))
    assert "openng3_rl.policy.FluoroStateExtractor" in data["policy_kwargs"]["features_extractor_class"]
    import base64

    for key in ("observation_space", "action_space"):
        assert b"numpy._core" not in base64.b64decode(data[key][":serialized:"]), "NumPy 2 pickle: organiser cannot load"
    loaded = SAC.load(path, env=compat_env, device="cpu")
    obs, _ = compat_env.reset(seed=1)
    a, _ = loaded.predict(obs, deterministic=True)
    assert a.shape == (4,)


ORGANISER_MODEL = ROOT / "existing code base" / "sac_210000_steps.zip"


@pytest.mark.skipif(not ORGANISER_MODEL.is_file(), reason="organiser checkpoint not present")
def test_organiser_checkpoint_loads_in_our_env(compat_env):
    ensure_policy_importable()
    from stable_baselines3 import SAC

    model = SAC.load(ORGANISER_MODEL, env=compat_env, device="cpu")
    assert model.observation_space == compat_env.observation_space
    obs, _ = compat_env.reset(seed=0)
    assert model.predict(obs, deterministic=True)[0].shape == (4,)


@pytest.mark.skipif(not (ROOT / "data/processed/0248_H_AOCERE_CAS_40Left85Right.npz").is_file(), reason="VMR data not fetched")
def test_organiser_goal_lies_in_our_vessel():
    from stroke_nav.data.anatomy import Anatomy

    a = Anatomy.load("0248_H_AOCERE_CAS")
    g = OrganiserFrame().obj_cm_to_vmr_mm(CAS_REFERENCE_GOALS_CM[0])
    n = a.nearest_node(g)
    assert np.linalg.norm(g - a.nodes[n]) < a.radius[n]
    assert a.branch_of(n) == "right_carotid_external"
