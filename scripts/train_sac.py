"""Train SAC in our simulator with the organiser's exact interface, and write a hand-over run folder.

Same algorithm, policy class, hyperparameters and checkpoint naming as the organiser's
``openng3-rl-train`` (HCMR-EV-RL), so the result drops into their ``runs/`` folder:

    python scripts/train_sac.py --total-timesteps 50000
    python scripts/train_sac.py --goal-mode curriculum --curriculum-group near --total-timesteps 20000
    python scripts/train_sac.py --goal-mode target --target "random:right_carotid@20-60" --buffer-size 20000

Writes runs/<run_id>/ with:
    checkpoints/sac_<N>_steps.zip   every --checkpoint-every steps (their naming)
    final-model.zip                 the trained policy
    resolved-config.json, device-parameters.json, scenario.json   (their schema; validated)
    stroke-nav.json                 provenance: frame transform, our env settings, START summary
    tensorboard/                    learning curves

The image replay buffer is large: buffer_size x 2 x 16 KB (50k -> about 1.7 GB of RAM).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from stroke_nav.organiser import ensure_policy_importable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total-timesteps", type=int, default=50_000, help="organiser default is 1,000,000")
    ap.add_argument("--goal-mode", default="scenario", choices=["scenario", "curriculum", "target"])
    ap.add_argument("--curriculum-group", default="all", choices=["near", "medium", "far", "all"])
    ap.add_argument("--target", default="random:left_carotid@20-70|right_carotid@20-70")
    ap.add_argument("--image-mode", default="render", choices=["render", "blank"])
    ap.add_argument("--no-noise", action="store_true")
    ap.add_argument("--force-proxy-n", type=float, default=0.0)
    ap.add_argument("--buffer-size", type=int, default=None, help="override the organiser's 50,000")
    ap.add_argument("--learning-starts", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=None)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    ensure_policy_importable()
    import torch
    from openng3_rl.policy import FluoroStateExtractor
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback

    from stroke_nav.organiser.env import CompatConfig, OrganiserCompatEnv
    from stroke_nav.organiser.export import check_run_dir, new_run_id, write_run_dir
    from stroke_nav.organiser.spec import DEFAULT_SPEC

    overrides = {"total_timesteps": args.total_timesteps, "seed": args.seed, "device": args.device}
    if args.buffer_size:
        overrides["buffer_size"] = args.buffer_size
    if args.learning_starts is not None:
        overrides["learning_starts"] = args.learning_starts
    if args.checkpoint_every:
        overrides["checkpoint_every_steps"] = args.checkpoint_every
    spec = DEFAULT_SPEC.with_training(**overrides)
    tr = spec.training
    torch.set_num_threads(tr.torch_threads)

    cfg = CompatConfig(goal_mode=args.goal_mode, curriculum_group=args.curriculum_group, target=args.target,
                       image_mode=args.image_mode, noise=not args.no_noise, force_proxy_n=args.force_proxy_n, spec=spec)
    env = OrganiserCompatEnv(cfg)
    run_dir = Path(args.run_root) / (args.run_id or new_run_id())
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    print(f"[train] run {run_dir}  goal_mode={cfg.goal_mode}  steps={tr.total_timesteps}  buffer={tr.buffer_size}")

    model = SAC(
        "MultiInputPolicy", env,
        policy_kwargs={"features_extractor_class": FluoroStateExtractor},
        learning_rate=tr.learning_rate, buffer_size=tr.buffer_size, learning_starts=tr.learning_starts,
        batch_size=tr.batch_size, gamma=tr.gamma, tau=tr.tau, train_freq=(1, "step"), gradient_steps=1,
        tensorboard_log=str(run_dir / "tensorboard"), seed=tr.seed, device=tr.device, verbose=1,
    )
    ckpt = CheckpointCallback(save_freq=tr.checkpoint_every_steps, save_path=str(run_dir / "checkpoints"), name_prefix="sac")
    t0 = time.time()
    model.learn(total_timesteps=tr.total_timesteps, callback=ckpt, progress_bar=False)
    train_s = time.time() - t0
    model.save(run_dir / "final-model.zip")

    # quick deterministic evaluation in our simulator (organiser success rule + START inputs)
    eval_env = OrganiserCompatEnv(cfg, anatomy=env.anat, frame=env.frame, renderer=env.renderer)
    results = []
    for i in range(args.eval_episodes):
        obs, _ = eval_env.reset(seed=10_000 + i)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = eval_env.step(act)
            done = term or trunc
        m = info["episode_metrics"]
        results.append({"success": bool(info["success"]), "steps": info["episode_steps"],
                        "final_distance_m": info["eval/true_distance_m"], "wrong_branch_events": m["wrong_branch_events"]})
    summary = {
        "episodes": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])) if results else None,
        "mean_final_distance_m": float(np.mean([r["final_distance_m"] for r in results])) if results else None,
        "mean_steps": float(np.mean([r["steps"] for r in results])) if results else None,
    }
    write_run_dir(run_dir, spec, frame=env.frame, extra={
        "trained_in": "stroke-nav OrganiserCompatEnv",
        "train_seconds": round(train_s, 1),
        "compat_config": {k: (v if not hasattr(v, "__dataclass_fields__") else "organiser spec") for k, v in asdict(cfg).items() if k != "spec"},
        "evaluation_in_our_sim": summary,
    })
    (run_dir / "evaluation-in-our-sim.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=2), encoding="utf-8")
    problems = check_run_dir(run_dir)
    print(f"[train] done in {train_s / 60:.1f} min; eval in our sim: {summary}")
    print("[train] organiser run-folder check:", "OK" if not problems else problems)
    print(f"[train] hand over: copy {run_dir} into HCMR-EV-RL/runs/ (see docs/HANDOVER.md)")


if __name__ == "__main__":
    main()
