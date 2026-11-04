"""Compare experiments/results/*.json with the copies in results/first_run/.

Run: .venv\\Scripts\\python experiments\\compare_runs.py
Writes experiments/results/reproducibility.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
IGNORE = {"runtime_seconds", "fit_seconds"}


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE:
                continue
            if k not in a or k not in b:
                out.append((f"{path}/{k}", a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                out += diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append((path, a, b))
    return out


def main():
    report = {}
    for name in ("lid_results", "dataset_audit", "sentiment_results"):
        a = json.loads((RES / "first_run" / f"{name}.json").read_text(encoding="utf-8"))
        b = json.loads((RES / f"{name}.json").read_text(encoding="utf-8"))
        d = diff(a, b)
        report[name] = {"n_differences": len(d), "differences": [{"path": p, "first_run": x, "final": y} for p, x, y in d]}
        print(f"{name}: {len(d)} differing values")
    s = report["sentiment_results"]["differences"]
    # Which sentiment runs changed at the headline level?
    runs = sorted({p.split("/runs/")[1].split("/")[0] for p, *_ in [(x["path"],) for x in s] if "/runs/" in p})
    report["sentiment_runs_with_any_change"] = runs
    headline = [x for x in s if x["path"].endswith("/test/macro_f1") and "/by_" not in x["path"]]
    report["sentiment_test_macro_f1_changes"] = headline
    (RES / "reproducibility.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print("runs with any change:", runs)
    for h in headline:
        print(f"  {h['path']}: {h['first_run']} -> {h['final']}")


if __name__ == "__main__":
    main()
