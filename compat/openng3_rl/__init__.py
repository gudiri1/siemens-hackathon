"""Minimal stand-in for the organiser's ``openng3_rl`` package.

Only ``openng3_rl.policy.FluoroStateExtractor`` is provided, so Stable-Baselines3 models trained in
stroke-nav pickle the same class path as models trained by the organiser and load in either
environment. If the organiser's real package is installed it takes precedence (see
``stroke_nav.organiser.ensure_policy_importable``). Do not install this directory as a package.
"""
