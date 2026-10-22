"""The system map reads thresholds from source; they must match what the code actually uses."""

import dataclasses
import importlib.util
import json
import re
from pathlib import Path

from stroke_nav.safety.faults import FaultModel
from stroke_nav.safety.monitor import SafetyConfig

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_system_map", ROOT / "scripts" / "build_system_map.py")
bsm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bsm)


def _norm(x):
    return json.loads(json.dumps(x, default=list))


def test_parsed_safety_config_matches_code():
    parsed = bsm.dataclass_defaults(ROOT / "src/stroke_nav/safety/monitor.py", "SafetyConfig")
    real = dataclasses.asdict(SafetyConfig())
    for k in ("speed_zones", "junction_fraction", "force_limit_N", "backoff_steps", "backoff_speed",
              "corridor_mm", "hold_after_dropouts", "overshoot_mm"):
        assert _norm(parsed[k]) == _norm(real[k]), k


def test_parsed_fault_presets_match_code():
    parsed = bsm.fault_presets(ROOT / "src/stroke_nav/safety/faults.py")
    for name in ("clean", "organiser", "stress"):
        real = dataclasses.asdict(getattr(FaultModel, name)())
        assert {k: parsed[name][k] for k in real} == real, name


def test_page_has_data_block_and_build_fills_it(tmp_path):
    out = tmp_path / "map.html"
    page = ROOT / "docs" / "system_map.html"
    html = page.read_text(encoding="utf-8")
    assert bsm.BLOCK.search(html)
    data = bsm.collect(None, None)
    payload = json.dumps(bsm._finite(data), allow_nan=False)
    out.write_text(bsm.BLOCK.sub(lambda m: m.group(1) + payload + m.group(3), html, count=1), encoding="utf-8")
    embedded = json.loads(re.search(r'id="repo-data">(.*?)</script>', out.read_text(encoding="utf-8"), re.S).group(1))
    assert "src/stroke_nav/safety/monitor.py" in embedded["inventory"]
    assert embedded["tests"]["defined"] > 0
