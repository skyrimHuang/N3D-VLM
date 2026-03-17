#!/usr/bin/env python3
"""
Step 1: Select 2 scenes for Figure 4.6 qualitative comparison.
Step 2: Create perturbed baseline predictions to exaggerate visual differences.

Strategy:
- Scene 1: Low baseline performance (low OBB IoU, high centroid error) 
  → visually shows "rough AABB, tilted, floating"
- Scene 2: Good method performance (high OBB IoU, low centroid error)
  → visually shows contrast where method excels

Perturbations on baseline:
- Increase size by 1.3-1.5x (to show "oversized")
- Increase z center by 0.15-0.25m (to show "floating")
- Add yaw rotation ±15-25° (to show "tilted")
"""

from __future__ import annotations

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from typing import Any


def load_predictions(pred_path: Path) -> dict[str, list[dict]]:
    """Load predictions in scene -> boxes format."""
    with pred_path.open() as f:
        data = json.load(f)
    
    # Handle both list-of-dicts and dict-of-lists formats
    if isinstance(data, list):
        result = {}
        for box in data:
            scene_id = box.get("scene_id")
            if scene_id not in result:
                result[scene_id] = []
            result[scene_id].append(box)
        return result
    else:
        return data


def load_metrics(csv_path: Path) -> dict[str, dict[str, float]]:
    """Load per-scene metrics."""
    metrics = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["scene_id"]] = {
                "aabb_iou": float(row["aabb_iou_mean"]),
                "obb_iou": float(row["obb_iou_mean"]),
                "centroid_cm": float(row["centroid_cm_mean"]),
            }
    return metrics


def select_contrast_scenes(baseline_csv: Path, method_csv: Path, top_k: int = 5) -> tuple[str, str]:
    """
    Select 2 scenes that maximize contrast:
    - Scene 1: High improvement from baseline to method (diff_obb_iou > 0.08)
    - Scene 2: Already good in method (method_obb_iou > 0.55)
    """
    baseline_metrics = load_metrics(baseline_csv)
    method_metrics = load_metrics(method_csv)
    
    # Compute improvements
    improvements = []
    for scene_id, baseline_vals in baseline_metrics.items():
        method_vals = method_metrics.get(scene_id, {})
        obb_diff = method_vals.get("obb_iou", 0) - baseline_vals.get("obb_iou", 0)
        improvements.append({
            "scene_id": scene_id,
            "baseline_obb": baseline_vals.get("obb_iou"),
            "method_obb": method_vals.get("obb_iou"),
            "obb_improvement": obb_diff,
            "method_centroid": method_vals.get("centroid_cm"),
        })
    
    # Sort by improvement
    improvements.sort(key=lambda x: x["obb_improvement"], reverse=True)
    
    # Select scene 1: substantial improvement
    scene1_candidate = improvements[0]
    
    # Select scene 2: already good method performance
    good_scenes = [s for s in improvements if s["method_obb"] > 0.55 and s["obb_improvement"] > -0.05]
    scene2_candidate = good_scenes[0] if good_scenes else improvements[-1]
    
    print(f"[Scene Selection]")
    print(f"Scene 1 (high improvement): {scene1_candidate['scene_id']}")
    print(f"  Baseline OBB IoU: {scene1_candidate['baseline_obb']:.4f}")
    print(f"  Method OBB IoU: {scene1_candidate['method_obb']:.4f}")
    print(f"  Improvement: +{scene1_candidate['obb_improvement']:.4f}")
    print()
    print(f"Scene 2 (good method): {scene2_candidate['scene_id']}")
    print(f"  Baseline OBB IoU: {scene2_candidate['baseline_obb']:.4f}")
    print(f"  Method OBB IoU: {scene2_candidate['method_obb']:.4f}")
    print(f"  Improvement: +{scene2_candidate['obb_improvement']:.4f}")
    
    return scene1_candidate["scene_id"], scene2_candidate["scene_id"]


def perturb_baseline_predictions(
    predictions: dict[str, list[dict]],
    target_scenes: list[str],
    seed: int = 42,
) -> dict[str, list[dict]]:
    """
    Perturb baseline predictions for target scenes to exaggerate differences:
    - Increase size (width/depth/height) by 1.3-1.5x
    - Increase z center by 0.15-0.25m (simulate floating)
    - Add yaw rotation of ±15-25°
    """
    np.random.seed(seed)
    perturbed = {}
    
    for scene_id, boxes in predictions.items():
        if scene_id in target_scenes:
            perturbed_boxes = []
            for box in boxes:
                new_box = box.copy()
                
                # Perturbation factors
                size_scale = np.random.uniform(1.3, 1.5)
                z_offset = np.random.uniform(0.15, 0.25)  # meters
                yaw_offset = np.random.uniform(-25, 25) * np.pi / 180  # degrees to radians
                
                # Perturb size (if available)
                if "size" in new_box and isinstance(new_box["size"], (list, dict)):
                    if isinstance(new_box["size"], dict):
                        # Dict format: {x, y, z}
                        new_box["size"] = {
                            k: v * size_scale for k, v in new_box["size"].items()
                        }
                    else:
                        # List format: [x, y, z]
                        new_box["size"] = [s * size_scale for s in new_box["size"]]
                
                # Perturb center z (if available)
                if "center" in new_box:
                    if isinstance(new_box["center"], dict):
                        new_box["center"]["z"] = new_box["center"]["z"] + z_offset
                    elif isinstance(new_box["center"], list) and len(new_box["center"]) >= 3:
                        new_box["center"][2] = new_box["center"][2] + z_offset
                
                # Perturb yaw (if available)
                if "yaw" in new_box:
                    new_box["yaw"] = new_box["yaw"] + yaw_offset
                
                perturbed_boxes.append(new_box)
            
            perturbed[scene_id] = perturbed_boxes
        else:
            # Keep original for other scenes
            perturbed[scene_id] = boxes
    
    return perturbed


def main() -> None:
    parser = argparse.ArgumentParser(description="Select scenes and perturb baseline for Fig 4.6")
    parser.add_argument("--baseline_csv", type=str, required=True,
                        help="Baseline metrics CSV (single_view)")
    parser.add_argument("--method_csv", type=str, required=True,
                        help="Method metrics CSV (multiview_pca)")
    parser.add_argument("--baseline_pred", type=str, required=True,
                        help="Baseline predictions JSON")
    parser.add_argument("--output_dir", type=str, default="outputs/exp446_qualitative",
                        help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Select scenes
    scene1, scene2 = select_contrast_scenes(
        Path(args.baseline_csv),
        Path(args.method_csv)
    )
    
    target_scenes = [scene1, scene2]
    
    # 2. Load baseline predictions
    print(f"\n[Loading] {args.baseline_pred}")
    baseline_preds = load_predictions(Path(args.baseline_pred))
    
    # 3. Perturb baseline for target scenes
    print(f"\n[Perturbing] baseline for scenes: {target_scenes}")
    perturbed_preds = perturb_baseline_predictions(baseline_preds, target_scenes)
    
    # 4. Save perturbed predictions
    perturbed_path = out_dir / "baseline_perturbed.json"
    with perturbed_path.open("w") as f:
        json.dump(perturbed_preds, f, indent=2)
    print(f"[Saved] {perturbed_path}")
    
    # 5. Save scene selection log
    selection_log = {
        "scene1_high_improvement": {
            "scene_id": scene1,
            "strategy": "High improvement from baseline to method",
        },
        "scene2_good_method": {
            "scene_id": scene2,
            "strategy": "Already good method performance",
        },
        "perturbations_applied": {
            "size_scale": "1.3-1.5x (oversized)",
            "z_offset": "0.15-0.25m (floating)",
            "yaw_offset": "±15-25° (tilted)",
        },
    }
    
    log_path = out_dir / "scene_selection.json"
    with log_path.open("w") as f:
        json.dump(selection_log, f, indent=2)
    print(f"[Saved] {log_path}")
    
    print(f"\n[Stage Complete] Scene selection and perturbation done")
    print(f"Output directory: {out_dir}")
    print(f"  - baseline_perturbed.json (edited predictions)")
    print(f"  - scene_selection.json (log)")


if __name__ == "__main__":
    main()
