"""Refresh the data block of the interactive system map (docs/system_map.html) from the repository.

The page's layout and explanations are hand-written; everything that can go stale is read from the
repo here and embedded as JSON:

- commit, branch and date
- tests per file (defined in source; passed/failed when a JUnit report is given)
- SafetyConfig defaults and FaultModel presets (parsed from source, so thresholds on the page always
  match the code)
- the module inventory (the page flags modules that are not on the map and map blocks whose code is gone)
- the safety evaluation summary (results/<name>/safety_summary.json), if given

    python scripts/build_system_map.py                                   # refresh docs/system_map.html in place
    pytest --junitxml=pytest.xml
    python scripts/evaluate_safety.py --synthetic --episodes 20 --name ci
    python scripts/build_system_map.py --junit pytest.xml --safety results/ci/safety_summary.json --out site/index.html

The GitHub workflow .github/workflows/system-map.yml runs the last three lines on every push to main
and publishes the page with GitHub Pages. Only the standard library is used.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "system_map.html"
BLOCK = re.compile(r'(<script type="application/json" id="repo-data">)(.*?)(</script>)', re.S)


def _class(tree: ast.Module, name: str) -> ast.ClassDef | None:
    return next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name), None)


def dataclass_defaults(path: Path, name: str) -> dict:
    """Literal default values of a dataclass's fields (fields with non-literal defaults are skipped)."""
    cls = _class(ast.parse(path.read_text(encoding="utf-8")), name)
    out = {}
    for n in cls.body if cls else []:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.value is not None:
            try:
                out[n.target.id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


def fault_presets(path: Path) -> dict:
    """FaultModel defaults merged with the keyword arguments of each ``@classmethod`` preset."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    base = dataclass_defaults(path, "FaultModel")
    cls = _class(tree, "FaultModel")
    out = {}
    for fn in cls.body if cls else []:
        if not (isinstance(fn, ast.FunctionDef) and any(getattr(d, "id", None) == "classmethod" for d in fn.decorator_list)):
            continue
        ret = next((n.value for n in ast.walk(fn) if isinstance(n, ast.Return)), None)
        if isinstance(ret, ast.Call) and getattr(ret.func, "id", None) == "cls":
            kw = {}
            for k in ret.keywords:
                try:
                    kw[k.arg] = ast.literal_eval(k.value)
                except ValueError:
                    pass
            out[fn.name] = {**base, **kw}
    return out


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def tests_defined() -> dict:
    out = {}
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        n = sum(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name.startswith("test_") for x in ast.walk(tree))
        out[f.stem] = {"defined": n}
    return out


def tests_from_junit(path: Path, files: dict) -> dict:
    """Adds passed/failed/skipped per test file. Parametrised tests count once per case, as pytest runs them."""
    for v in files.values():
        v.update(passed=0, failed=0, skipped=0)
    for case in ET.parse(path).getroot().iter("testcase"):
        stem = (case.get("classname") or "").split(".")
        stem = next((s for s in stem if s.startswith("test_")), None)
        if stem is None:
            continue
        v = files.setdefault(stem, {"defined": 0, "passed": 0, "failed": 0, "skipped": 0})
        tags = {c.tag for c in case}
        if tags & {"failure", "error"}:
            v["failed"] += 1
        elif "skipped" in tags:
            v["skipped"] += 1
        else:
            v["passed"] += 1
    return files


def inventory() -> list[str]:
    files = [p for p in (ROOT / "src" / "stroke_nav").rglob("*.py") if p.name != "__init__.py"]
    files += list((ROOT / "scripts").glob("*.py"))
    return sorted(p.relative_to(ROOT).as_posix() for p in files)


def collect(junit: Path | None, safety: Path | None) -> dict:
    tests = tests_defined()
    ran = junit is not None and junit.exists()
    if ran:
        tests = tests_from_junit(junit, tests)
    total = {k: sum(v.get(k, 0) for v in tests.values()) for k in ("defined", "passed", "failed", "skipped")}
    summary = json.loads(safety.read_text(encoding="utf-8")) if safety and safety.exists() else None
    episodes = None
    if summary:
        episodes = next(iter(summary.values())).get("n")
    return {
        "built": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "ci": bool(os.environ.get("GITHUB_ACTIONS")),
        "commit": {
            "sha": os.environ.get("GITHUB_SHA") or git("rev-parse", "HEAD"),
            "date": git("log", "-1", "--format=%cI"),
            "subject": git("log", "-1", "--format=%s"),
            "branch": os.environ.get("GITHUB_REF_NAME") or git("rev-parse", "--abbrev-ref", "HEAD"),
        },
        "tests": {"ran": ran, "files": tests, **total},
        "config": dataclass_defaults(ROOT / "src/stroke_nav/safety/monitor.py", "SafetyConfig"),
        "faults": fault_presets(ROOT / "src/stroke_nav/safety/faults.py"),
        "inventory": inventory(),
        "safety": summary,
        "safety_episodes": episodes,
    }


def _finite(x):
    """NaN/inf -> None (JSON.parse in the browser rejects NaN); tuples -> lists."""
    if isinstance(x, float):
        return x if x == x and abs(x) != float("inf") else None
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--junit", type=Path, help="pytest --junitxml report")
    p.add_argument("--safety", type=Path, default=ROOT / "results/demo_arch_safety/safety_summary.json",
                   help="safety_summary.json from scripts/evaluate_safety.py --synthetic")
    p.add_argument("--page", type=Path, default=PAGE, help="page template (default: docs/system_map.html)")
    p.add_argument("--out", type=Path, default=None, help="output file (default: overwrite --page)")
    args = p.parse_args()

    html = args.page.read_text(encoding="utf-8")
    if not BLOCK.search(html):
        raise SystemExit(f"{args.page}: no <script id=\"repo-data\"> block found")
    data = collect(args.junit, args.safety)
    payload = json.dumps(_finite(data), ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = BLOCK.sub(lambda m: m.group(1) + payload + m.group(3), html, count=1)
    out = args.out or args.page
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    t = data["tests"]
    tests = f"{t['passed']}/{t['passed'] + t['failed']} tests passed" if t["ran"] else f"{t['defined']} tests defined (not run)"
    print(f"[map] {out}: commit {data['commit']['sha'][:7]}, {len(data['inventory'])} modules, {tests}, "
          f"safety evaluation: {'yes' if data['safety'] else 'no'}")


if __name__ == "__main__":
    main()
