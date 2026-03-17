from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import Box3D, centroid_error_cm, hungarian_match, iou_3d_obb_yaw


def _boxes_from_json(path: Path) -> Dict[str, List[Box3D]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return {k: [Box3D.from_dict(x) for x in v] for k, v in payload.items()}
    if isinstance(payload, list):
        return {row["scene_id"]: [Box3D.from_dict(x) for x in row.get("boxes", [])] for row in payload}
    raise ValueError(f"Unsupported prediction format: {path}")


def _load_gt(scene_entry: dict) -> List[Box3D]:
    payload = json.loads(Path(scene_entry["gt_boxes"]).read_text())
    return [Box3D.from_dict(obj["obb"]) for obj in payload["objects"]]


def _yaw_error_deg(pred: float, gt: float) -> float:
    d = (pred - gt + np.pi) % (2 * np.pi) - np.pi
    return abs(float(np.degrees(d)))


def _size_errors(pred: Box3D, gt: Box3D) -> tuple[float, float]:
    abs_err = float(np.mean(np.abs(pred.size - gt.size)))
    rel_err = float(np.mean(np.abs(pred.size - gt.size) / np.maximum(gt.size, 1e-6)))
    return abs_err, rel_err


def evaluate_extended(manifest: List[dict], preds: Dict[str, List[Box3D]], iou_threshold: float) -> dict:
    rows: List[dict] = []
    ious: List[float] = []
    cents: List[float] = []
    yaws: List[float] = []
    size_abs: List[float] = []
    size_rel: List[float] = []

    for scene in manifest:
        scene_id = scene["scene_id"]
        gt_boxes = _load_gt(scene)
        pred_boxes = preds.get(scene_id, [])
        matches = hungarian_match(
            preds=pred_boxes,
            gts=gt_boxes,
            iou_fn=iou_3d_obb_yaw,
            centroid_fn=centroid_error_cm,
            iou_threshold=iou_threshold,
            class_aware=True,
        )

        for m in matches:
            pb = pred_boxes[m.pred_idx]
            gb = gt_boxes[m.gt_idx]
            yaw_err = _yaw_error_deg(pb.yaw, gb.yaw)
            s_abs, s_rel = _size_errors(pb, gb)
            row = {
                "scene_id": scene_id,
                "label": gb.label,
                "iou_obb": m.iou,
                "centroid_cm": m.centroid_error_cm,
                "yaw_err_deg": yaw_err,
                "size_abs": s_abs,
                "size_rel": s_rel,
            }
            rows.append(row)
            ious.append(m.iou)
            cents.append(m.centroid_error_cm)
            yaws.append(yaw_err)
            size_abs.append(s_abs)
            size_rel.append(s_rel)

    def _stats(vals: List[float]) -> dict:
        if not vals:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "count": int(arr.size)}

    by_label: Dict[str, List[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    grouped = {}
    for label, group in by_label.items():
        grouped[label] = {
            "iou_obb": _stats([r["iou_obb"] for r in group]),
            "centroid_cm": _stats([r["centroid_cm"] for r in group]),
            "yaw_err_deg": _stats([r["yaw_err_deg"] for r in group]),
            "size_abs": _stats([r["size_abs"] for r in group]),
            "size_rel": _stats([r["size_rel"] for r in group]),
        }

    return {
        "overall": {
            "iou_obb": _stats(ious),
            "centroid_cm": _stats(cents),
            "yaw_err_deg": _stats(yaws),
            "size_abs": _stats(size_abs),
            "size_rel": _stats(size_rel),
        },
        "grouped_by_label": grouped,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended evaluator for exp4.4.2")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/exp442_extended")
    parser.add_argument("--iou_threshold", type=float, default=0.25)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    preds = _boxes_from_json(Path(args.pred))
    result = evaluate_extended(manifest=manifest, preds=preds, iou_threshold=args.iou_threshold)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary_extended.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    rows = result["rows"]
    with (out / "rows_extended.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "label", "iou_obb", "centroid_cm", "yaw_err_deg", "size_abs", "size_rel"])
        writer.writeheader()
        writer.writerows(rows)

    with (out / "table_size_orientation.md").open("w") as f:
        overall = result["overall"]
        f.write("| Metric | Mean | Std | Count |\n")
        f.write("|---|---:|---:|---:|\n")
        for key in ["iou_obb", "centroid_cm", "yaw_err_deg", "size_abs", "size_rel"]:
            stat = overall[key]
            f.write(f"| {key} | {stat['mean']:.4f} | {stat['std']:.4f} | {stat['count']} |\\n")

    print(f"Saved extended evaluation to: {out}")


if __name__ == "__main__":
    main()
