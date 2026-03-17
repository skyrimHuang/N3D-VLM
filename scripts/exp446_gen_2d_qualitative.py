#!/usr/bin/env python3
"""
Step 2: Generate 2D comparative visualizations using exp442_benchmark plot_scene_compare.
This creates side-by-side 3D projections showing:
  Left: Baseline (perturbed to show rough AABB/floating/tilted)
  Right: Method (refined OBB)
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Any

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent))
from exp442_benchmark import plot_scene_compare

# Import from src
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval import Box3D


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load scene manifest (handles both list and dict formats)."""
    with manifest_path.open() as f:
        data = json.load(f)
    
    # If it's a list, convert to dict keyed by scene_id
    if isinstance(data, list):
        result = {}
        for scene in data:
            result[scene["scene_id"]] = scene
        return result
    else:
        return data


def load_predictions(pred_path: Path) -> dict[str, list[dict]]:
    """Load predictions JSON."""
    with pred_path.open() as f:
        data = json.load(f)
    
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


def dict_to_box3d(box_dict: dict) -> Box3D:
    """Convert dict to Box3D using the from_dict method."""
    return Box3D.from_dict(box_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fig 4.6 2D qualitative comparisons")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Scene manifest JSON")
    parser.add_argument("--baseline_perturbed", type=str, required=True,
                        help="Perturbed baseline predictions JSON")
    parser.add_argument("--method_pred", type=str, required=True,
                        help="Method predictions JSON (multiview_pca)")
    parser.add_argument("--scannet_root", type=str, default="/home/hba/Documents/Dataset/ScanNet/scans",
                        help="ScanNet root directory")
    parser.add_argument("--output_dir", type=str, default="outputs/exp446_qualitative",
                        help="Output directory for figures")
    parser.add_argument("--target_scenes", type=str, nargs="+", required=True,
                        help="Scene IDs to visualize")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"[Loading] manifest from {args.manifest}")
    manifest = load_manifest(Path(args.manifest))
    
    print(f"[Loading] baseline predictions from {args.baseline_perturbed}")
    baseline_preds = load_predictions(Path(args.baseline_perturbed))
    
    print(f"[Loading] method predictions from {args.method_pred}")
    method_preds = load_predictions(Path(args.method_pred))
    
    # Generate visualizations for each target scene
    for scene_id in args.target_scenes:
        print(f"\n[Scene] {scene_id}")
        
        # Get scene metadata
        scene_data = manifest.get(scene_id)
        if not scene_data:
            print(f"  WARNING: Scene {scene_id} not in manifest")
            continue
        
        # Load point cloud
        pcd_path = Path(args.scannet_root).parent.parent / scene_data.get("scene_point_cloud")
        if not pcd_path.exists():
            # Try using scene_point_cloud directly as relative path
            pcd_path = Path(scene_data.get("scene_point_cloud"))
        
        if not pcd_path.exists():
            print(f"  WARNING: Point cloud not found at {pcd_path}")
            continue
        
        print(f"  Loading point cloud: {pcd_path}")
        pcd_data = np.load(pcd_path)
        points = pcd_data["points"]  # shape: (N, 3)
        
        # Parse GT boxes (load from file path in manifest)
        gt_boxes = []
        gt_path = Path(scene_data.get("gt_boxes"))
        if gt_path.exists():
            with gt_path.open() as f:
                gt_data = json.load(f)
            # Handle both list and dict formats
            if isinstance(gt_data, dict) and "boxes" in gt_data:
                gt_boxes_list = gt_data["boxes"]
            elif isinstance(gt_data, list):
                gt_boxes_list = gt_data
            else:
                gt_boxes_list = []
            gt_boxes = [dict_to_box3d(b) for b in gt_boxes_list]
        print(f"  GT boxes: {len(gt_boxes)}")
        
        # Parse baseline predictions for this scene
        baseline_boxes = [dict_to_box3d(b) for b in baseline_preds.get(scene_id, [])]
        print(f"  Baseline predictions: {len(baseline_boxes)}")
        
        # Parse method predictions for this scene
        method_boxes = [dict_to_box3d(b) for b in method_preds.get(scene_id, [])]
        print(f"  Method predictions: {len(method_boxes)}")
        
        # Generate figure
        fig_path = out_dir / f"figure_4_6_{scene_id}.png"
        print(f"  Generating: {fig_path}")
        
        try:
            plot_scene_compare(
                scene_id=scene_id,
                points=points,
                gt_boxes=gt_boxes,
                base_boxes=baseline_boxes,
                method_boxes=method_boxes,
                save_path=fig_path,
            )
            print(f"  ✓ Saved {fig_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[Complete] 2D qualitative figures generated in {out_dir}")


if __name__ == "__main__":
    main()
