from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import (
    Box3D,
    centroid_error_cm,
    hungarian_match,
    iou_3d_aabb,
    iou_3d_obb_yaw,
    recall_at_threshold,
    summarize,
)


def load_scene_to_boxes(pred_path: Path) -> Dict[str, List[Box3D]]:
    payload = json.loads(pred_path.read_text())
    scene_to_boxes: Dict[str, List[Box3D]] = {}

    if isinstance(payload, dict):
        for scene_id, boxes in payload.items():
            if isinstance(boxes, dict) and "boxes" in boxes:
                boxes = boxes["boxes"]
            scene_to_boxes[scene_id] = [Box3D.from_dict(x) for x in boxes]
        return scene_to_boxes

    if isinstance(payload, list):
        for entry in payload:
            scene_id = entry["scene_id"]
            boxes = entry.get("boxes", [])
            scene_to_boxes[scene_id] = [Box3D.from_dict(x) for x in boxes]
        return scene_to_boxes

    raise ValueError(f"Unsupported prediction format in {pred_path}")


def load_gt_scene_boxes(scene_entry: dict) -> Tuple[List[Box3D], List[Box3D]]:
    gt_payload = json.loads(Path(scene_entry["gt_boxes"]).read_text())
    gt_aabb = [Box3D.from_dict(obj["aabb"]) for obj in gt_payload["objects"]]
    gt_obb = [Box3D.from_dict(obj["obb"]) for obj in gt_payload["objects"]]
    return gt_aabb, gt_obb


def evaluate_method(
    manifest: Iterable[dict],
    scene_preds: Dict[str, List[Box3D]],
    iou_threshold: float,
    progress_desc: str = "Evaluating",
) -> dict:
    manifest = list(manifest)
    all_aabb_iou: List[float] = []
    all_obb_iou: List[float] = []
    all_centroid: List[float] = []
    scene_rows: List[dict] = []

    for scene in tqdm(manifest, desc=progress_desc, leave=False):
        scene_id = scene["scene_id"]
        preds = scene_preds.get(scene_id, [])
        gt_aabb, gt_obb = load_gt_scene_boxes(scene)

        matched_aabb = hungarian_match(
            preds=preds,
            gts=gt_aabb,
            iou_fn=iou_3d_aabb,
            centroid_fn=centroid_error_cm,
            iou_threshold=iou_threshold,
            class_aware=True,
        )
        matched_obb = hungarian_match(
            preds=preds,
            gts=gt_obb,
            iou_fn=iou_3d_obb_yaw,
            centroid_fn=centroid_error_cm,
            iou_threshold=iou_threshold,
            class_aware=True,
        )

        aabb_iou_vals = [m.iou for m in matched_aabb]
        obb_iou_vals = [m.iou for m in matched_obb]
        cen_vals = [m.centroid_error_cm for m in matched_aabb]

        all_aabb_iou.extend(aabb_iou_vals)
        all_obb_iou.extend(obb_iou_vals)
        all_centroid.extend(cen_vals)

        row = {
            "scene_id": scene_id,
            "num_gt": len(gt_aabb),
            "num_pred": len(preds),
            "matched": len(matched_aabb),
            "aabb_iou_mean": float(np.mean(aabb_iou_vals)) if aabb_iou_vals else 0.0,
            "obb_iou_mean": float(np.mean(obb_iou_vals)) if obb_iou_vals else 0.0,
            "centroid_cm_mean": float(np.mean(cen_vals)) if cen_vals else 0.0,
            "recall_025": recall_at_threshold(
                preds=preds,
                gts=gt_aabb,
                iou_fn=iou_3d_aabb,
                centroid_fn=centroid_error_cm,
                iou_threshold=0.25,
                class_aware=True,
            ),
            "recall_050": recall_at_threshold(
                preds=preds,
                gts=gt_aabb,
                iou_fn=iou_3d_aabb,
                centroid_fn=centroid_error_cm,
                iou_threshold=0.5,
                class_aware=True,
            ),
        }
        scene_rows.append(row)

    summary = {
        "aabb_iou": summarize(all_aabb_iou),
        "obb_iou": summarize(all_obb_iou),
        "centroid_cm": summarize(all_centroid),
        "scene_rows": scene_rows,
    }
    return summary


def write_scene_csv(rows: List[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        csv_path.write_text("")
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compare_scene_metric(base_rows: List[dict], method_rows: List[dict], key: str) -> dict:
    base_map = {r["scene_id"]: r[key] for r in base_rows}
    method_map = {r["scene_id"]: r[key] for r in method_rows}
    common_ids = sorted(set(base_map).intersection(method_map))
    if len(common_ids) == 0:
        return {"n": 0, "p_value": 1.0}
    base_vals = np.asarray([base_map[sid] for sid in common_ids], dtype=np.float64)
    method_vals = np.asarray([method_map[sid] for sid in common_ids], dtype=np.float64)
    if np.allclose(base_vals, method_vals):
        return {"n": len(common_ids), "p_value": 1.0}
    stat = wilcoxon(base_vals, method_vals, alternative="two-sided", zero_method="wilcox")
    return {"n": len(common_ids), "p_value": float(stat.pvalue)}


def write_markdown_table(table_path: Path, rows: List[Tuple[str, dict]]) -> None:
    lines = [
        "| Method | 3D-IoU(AABB) | 3D-IoU(OBB) | Centroid Error (cm) |",
        "|---|---:|---:|---:|",
    ]
    for name, metric in rows:
        lines.append(
            "| {} | {:.4f}±{:.4f} | {:.4f}±{:.4f} | {:.2f}±{:.2f} |".format(
                name,
                metric["aabb_iou"]["mean"],
                metric["aabb_iou"]["std"],
                metric["obb_iou"]["mean"],
                metric["obb_iou"]["std"],
                metric["centroid_cm"]["mean"],
                metric["centroid_cm"]["std"],
            )
        )
    table_path.write_text("\n".join(lines) + "\n")


def _box_corners_xy(box: Box3D) -> np.ndarray:
    half = box.size[:2] / 2.0
    local = np.array([[-half[0], -half[1]], [half[0], -half[1]], [half[0], half[1]], [-half[0], half[1]]])
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + box.center[:2]


def plot_scene_compare(
    scene_id: str,
    points: np.ndarray,
    gt_boxes: List[Box3D],
    base_boxes: List[Box3D],
    method_boxes: List[Box3D],
    save_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 6))
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    for ax, title, boxes in [
        (axes[0], "Single-view VLM (AABB rough)", base_boxes),
        (axes[1], "Multi-view + PCA/OBB (refined)", method_boxes),
    ]:
        if points.shape[0] > 20000:
            idx = np.random.choice(points.shape[0], 20000, replace=False)
            pts = points[idx]
        else:
            pts = points
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c="lightgray", alpha=0.2)

        for gt in gt_boxes:
            corners = _box_corners_xy(gt)
            z0 = gt.center[2] - gt.size[2] / 2.0
            z1 = gt.center[2] + gt.size[2] / 2.0
            closed = np.vstack([corners, corners[0]])
            ax.plot(closed[:, 0], closed[:, 1], z0, color="green", linewidth=1.0)
            ax.plot(closed[:, 0], closed[:, 1], z1, color="green", linewidth=1.0)

        for pred in boxes:
            corners = _box_corners_xy(pred)
            z0 = pred.center[2] - pred.size[2] / 2.0
            z1 = pred.center[2] + pred.size[2] / 2.0
            closed = np.vstack([corners, corners[0]])
            ax.plot(closed[:, 0], closed[:, 1], z0, color="red", linewidth=1.0)
            ax.plot(closed[:, 0], closed[:, 1], z1, color="red", linewidth=1.0)

        ax.set_title(f"{scene_id}\n{title}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark for thesis exp 4.4.2")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--single_view_pred", type=str, required=True)
    parser.add_argument("--multiview_pca_pred", type=str, required=True)
    parser.add_argument("--ablation_pred", type=str, nargs="*", default=[])
    parser.add_argument("--ablation_name", type=str, nargs="*", default=[])
    parser.add_argument("--output_dir", type=str, default="outputs/exp442_results")
    parser.add_argument("--iou_threshold", type=float, default=0.25)
    parser.add_argument("--num_qualitative", type=int, default=3)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_pred = load_scene_to_boxes(Path(args.single_view_pred))
    method_pred = load_scene_to_boxes(Path(args.multiview_pca_pred))
    ablations = [load_scene_to_boxes(Path(p)) for p in args.ablation_pred]

    print("[Stage] Evaluating single-view baseline...")
    base_metric = evaluate_method(
        manifest=manifest,
        scene_preds=base_pred,
        iou_threshold=args.iou_threshold,
        progress_desc="Eval baseline",
    )
    print("[Stage] Evaluating multi-view + PCA/OBB...")
    method_metric = evaluate_method(
        manifest=manifest,
        scene_preds=method_pred,
        iou_threshold=args.iou_threshold,
        progress_desc="Eval method",
    )

    rows = [
        ("Single-view VLM", base_metric),
        ("Multi-view + PCA/OBB", method_metric),
    ]
    for idx, abl in enumerate(ablations):
        name = args.ablation_name[idx] if idx < len(args.ablation_name) else f"Ablation-{idx + 1}"
        print(f"[Stage] Evaluating ablation: {name}")
        metric = evaluate_method(
            manifest=manifest,
            scene_preds=abl,
            iou_threshold=args.iou_threshold,
            progress_desc=f"Eval {name}",
        )
        rows.append((name, metric))

    write_markdown_table(output_dir / "table_4_8.md", rows)
    write_scene_csv(base_metric["scene_rows"], output_dir / "single_view_scene_metrics.csv")
    write_scene_csv(method_metric["scene_rows"], output_dir / "multiview_pca_scene_metrics.csv")

    pvals = {
        "aabb_iou": compare_scene_metric(base_metric["scene_rows"], method_metric["scene_rows"], "aabb_iou_mean"),
        "obb_iou": compare_scene_metric(base_metric["scene_rows"], method_metric["scene_rows"], "obb_iou_mean"),
        "centroid_cm": compare_scene_metric(base_metric["scene_rows"], method_metric["scene_rows"], "centroid_cm_mean"),
    }

    summary = {
        "single_view": {k: v for k, v in base_metric.items() if k != "scene_rows"},
        "multiview_pca": {k: v for k, v in method_metric.items() if k != "scene_rows"},
        "significance": pvals,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    plotted = 0
    print("[Stage] Rendering qualitative figures...")
    for scene in tqdm(manifest, desc="Qualitative render", leave=False):
        if plotted >= args.num_qualitative:
            break
        scene_id = scene["scene_id"]
        if scene_id not in base_pred or scene_id not in method_pred:
            continue
        point_path = Path(scene["scene_point_cloud"])
        if not point_path.exists():
            continue
        points = np.load(point_path)["points"]
        gt_aabb, _ = load_gt_scene_boxes(scene)
        plot_scene_compare(
            scene_id=scene_id,
            points=points,
            gt_boxes=gt_aabb,
            base_boxes=base_pred[scene_id],
            method_boxes=method_pred[scene_id],
            save_path=output_dir / f"figure_4_5_{scene_id}.png",
        )
        plotted += 1

    print(f"Saved benchmark outputs to: {output_dir}")
    print(f"Generated qualitative figures: {plotted}")


if __name__ == "__main__":
    main()