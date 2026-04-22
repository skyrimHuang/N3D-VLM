#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
EXP453 = ROOT / "scripts/exp453_chair_pose_search_demo.py"
BASE_OUT = ROOT / "outputs/sample_label_demo/pose_search_diagnostics"


@dataclass
class Case:
    name: str
    args: list[str]


def run_case(case: Case) -> dict:
    out_dir = BASE_OUT / case.name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [str(PY), str(EXP453), "--out_dir", str(out_dir), *case.args]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)

    summary_path = out_dir / "pose_search_summary.json"
    if not summary_path.exists():
        return {
            "case": case.name,
            "ok": False,
            "returncode": proc.returncode,
            "stderr": proc.stderr.strip(),
            "stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-10:]),
        }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    counts = summary.get("counts", {})
    support_ref = summary.get("support_reference", {})
    hyper = summary.get("hyperparams", {})

    total = max(int(counts.get("total_candidates", 0)), 1)
    support = int(counts.get("support_pass", 0))
    collision = int(counts.get("collision_pass", 0))
    rule = int(counts.get("rule_pass", 0))
    feasible = int(counts.get("feasible_after_dedup", 0))

    return {
        "case": case.name,
        "ok": True,
        "returncode": proc.returncode,
        "counts": counts,
        "support_reference": support_ref,
        "hyperparams": {
            "support_probe_start_height_fraction": hyper.get("support_probe_start_height_fraction"),
            "support_probe_depth": hyper.get("support_probe_depth"),
            "support_count_ratio_threshold": hyper.get("support_count_ratio_threshold"),
            "rule_depth": hyper.get("rule_depth"),
            "rule_radius": hyper.get("rule_radius"),
        },
        "drop_rates": {
            "support_drop": 1.0 - (support / total),
            "collision_drop_after_support": 0.0 if support == 0 else 1.0 - (collision / support),
            "rule_drop_after_collision": 0.0 if collision == 0 else 1.0 - (rule / collision),
            "dedup_drop_after_rule": 0.0 if rule == 0 else 1.0 - (feasible / rule),
        },
        "stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-12:]),
    }


def main() -> None:
    cases = [
        Case("baseline", []),
        Case("rule_off", ["--rule_depth", "0.0", "--rule_radius", "0.0"]),
        Case("rule_relaxed_r03", ["--rule_depth", "1.0", "--rule_radius", "0.3"]),
        Case("rule_relaxed_r05", ["--rule_depth", "1.0", "--rule_radius", "0.5"]),
        Case("support_ratio_03", ["--support_count_ratio_threshold", "0.3"]),
        Case("support_ratio_02_rule_off", ["--support_count_ratio_threshold", "0.2", "--rule_depth", "0.0", "--rule_radius", "0.0"]),
        Case("probe_start_00", ["--support_probe_start_height_fraction", "0.0"]),
    ]

    BASE_OUT.mkdir(parents=True, exist_ok=True)
    results = [run_case(c) for c in cases]

    report = {
        "workspace": str(ROOT),
        "script": str(EXP453),
        "results": results,
    }
    report_path = BASE_OUT / "diagnostic_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== exp453 filter diagnosis ===")
    for r in results:
        if not r.get("ok"):
            print(f"[FAIL] {r['case']} rc={r.get('returncode')} stderr={r.get('stderr')}")
            continue

        counts = r["counts"]
        dr = r["drop_rates"]
        print(
            f"[OK] {r['case']}: total={counts['total_candidates']} "
            f"support={counts['support_pass']} collision={counts['collision_pass']} "
            f"rule={counts['rule_pass']} feasible={counts['feasible_after_dedup']}"
        )
        print(
            f"     drop support={dr['support_drop']:.3f}, "
            f"collision={dr['collision_drop_after_support']:.3f}, "
            f"rule={dr['rule_drop_after_collision']:.3f}, "
            f"dedup={dr['dedup_drop_after_rule']:.3f}"
        )

    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
