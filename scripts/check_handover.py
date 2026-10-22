"""Check that a run folder will load and run in the organiser's pipeline.

    python scripts/check_handover.py --run-dir runs/<run_id>
    python scripts/check_handover.py --run-dir runs/<run_id> --episodes 5
    python scripts/check_handover.py --organiser-model "existing code base/sac_210000_steps.zip"

Checks:
1. run-folder files and hashes exactly as their evaluation validates them
2. the model references ``openng3_rl.policy.FluoroStateExtractor`` (their class path)
3. the model's pickles are readable by NumPy 1.26 (their container) - no ``numpy._core`` references
4. SB3 loads the model against the organiser-compatible env (identical spaces) and it runs
Exit code 0 when everything passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from stroke_nav.organiser import ensure_policy_importable


def inspect_model_zip(path: Path) -> list[str]:
    problems = []
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("data"))
        sysinfo = z.read("system_info.txt").decode(errors="replace") if "system_info.txt" in z.namelist() else ""
    pk = data.get("policy_kwargs", {})
    if "openng3_rl.policy.FluoroStateExtractor" not in str(pk.get("features_extractor_class", "")):
        problems.append(f"policy uses {pk.get('features_extractor_class')!r}, not openng3_rl.policy.FluoroStateExtractor")
    import base64

    for key in ("observation_space", "action_space", "policy_kwargs"):
        blob = data.get(key, {}).get(":serialized:")
        if blob and b"numpy._core" in base64.b64decode(blob):
            problems.append(f"{key} was pickled with NumPy 2 (numpy._core); the organiser's NumPy 1.26 cannot load it")
    for line in sysinfo.splitlines():
        if any(k in line for k in ("Stable-Baselines3", "Numpy", "Gymnasium", "PyTorch")):
            print("   ", line.strip())
    return problems


def run_model(model_path: Path, episodes: int) -> None:
    from stable_baselines3 import SAC

    from stroke_nav.organiser.env import OrganiserCompatEnv

    env = OrganiserCompatEnv()
    model = SAC.load(str(model_path), env=env, device="cpu")
    assert model.observation_space == env.observation_space and model.action_space == env.action_space
    print(f"   loaded with identical spaces; trained for {model.num_timesteps} steps")
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        print(f"   episode {ep}: success {info['success']}, steps {info['episode_steps']}, "
              f"final distance {info['eval/true_distance_m'] * 1000:.1f} mm")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--model", default="final-model.zip")
    ap.add_argument("--organiser-model", type=Path, help="also load one of the organiser's own checkpoints")
    ap.add_argument("--episodes", type=int, default=2)
    args = ap.parse_args()
    print("FluoroStateExtractor:", ensure_policy_importable())
    from stroke_nav.organiser.export import check_run_dir

    failed = False
    if args.run_dir:
        print(f"[1] run folder {args.run_dir}")
        problems = check_run_dir(args.run_dir)
        print("   OK" if not problems else "\n".join("   FAIL " + p for p in problems))
        failed |= bool(problems)
        model = args.run_dir / args.model
        if model.is_file():
            print(f"[2-3] model file {model}")
            problems = inspect_model_zip(model)
            print("   OK" if not problems else "\n".join("   FAIL " + p for p in problems))
            failed |= bool(problems)
            print("[4] load and run in the organiser-compatible env")
            run_model(model, args.episodes)
    if args.organiser_model:
        print(f"[x] organiser model {args.organiser_model}")
        inspect_model_zip(args.organiser_model)
        run_model(args.organiser_model, args.episodes)
    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
