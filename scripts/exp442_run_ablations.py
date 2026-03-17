from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _load_table_metric(table_path: Path) -> dict:
    lines = [line.strip() for line in table_path.read_text().splitlines() if line.strip()]
    # Expect header + separator + rows
    metrics = {}
    for line in lines[2:]:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        if len(parts) >= 4:
            metrics[parts[0]] = {
                "aabb": parts[1],
                "obb": parts[2],
                "centroid": parts[3],
            }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablations for exp4.4.2")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--single_view_pred", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/exp442_ablations")
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    ablations = [
        {
            "name": "B1_multiview_only",
            "params": {"blend": 0.0, "cluster_eps": 0.15, "min_cluster_size": 35},
        },
        {
            "name": "B2_multiview_cluster",
            "params": {"blend": 0.5, "cluster_eps": 0.12, "min_cluster_size": 28},
        },
        {
            "name": "B3_full_refine",
            "params": {"blend": 0.8, "cluster_eps": 0.10, "min_cluster_size": 35},
        },
        {
            "name": "B4_support_strict",
            "params": {"blend": 0.8, "cluster_eps": 0.10, "min_cluster_size": 60},
        },
    ]

    summary_rows = []
    for ab in ablations:
        name = ab["name"]
        pred_out = out_root / f"{name}.json"
        diag_out = out_root / f"{name}.diagnostics.json"
        bench_out = out_root / name

        _run(
            [
                args.python_executable,
                str(PROJECT_ROOT / "scripts" / "exp442_refine_predictions.py"),
                "--manifest",
                args.manifest,
                "--input_pred",
                args.single_view_pred,
                "--output_pred",
                str(pred_out),
                "--output_diag",
                str(diag_out),
                "--blend",
                str(ab["params"]["blend"]),
                "--cluster_eps",
                str(ab["params"]["cluster_eps"]),
                "--min_cluster_size",
                str(ab["params"]["min_cluster_size"]),
            ]
        )

        _run(
            [
                args.python_executable,
                str(PROJECT_ROOT / "scripts" / "exp442_benchmark.py"),
                "--manifest",
                args.manifest,
                "--single_view_pred",
                args.single_view_pred,
                "--multiview_pca_pred",
                str(pred_out),
                "--output_dir",
                str(bench_out),
                "--num_qualitative",
                "2",
            ]
        )

        table_metrics = _load_table_metric(bench_out / "table_4_8.md")
        method_row = table_metrics.get("Multi-view + PCA/OBB", {})
        summary_rows.append(
            {
                "ablation": name,
                "obb": method_row.get("obb", ""),
                "aabb": method_row.get("aabb", ""),
                "centroid": method_row.get("centroid", ""),
            }
        )

    with (out_root / "ablation_matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ablation", "aabb", "obb", "centroid"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Saved ablations to: {out_root}")


if __name__ == "__main__":
    main()
