#!/usr/bin/env python3
"""
Step 3: Generate Rerun 3D visualizations with RGB axes.

This script creates interactive 3D visualizations showing:
- Point cloud (gray)
- Ground truth boxes (green)
- Baseline predictions (red)  
- Method predictions (blue)
- Per-box RGB axes (X=red, Y=green, Z=blue)
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Any

try:
    import rerun as rr
except ImportError:
    print("ERROR: rerun not installed. Install with: pip install rerun-sdk")
    sys.exit(1)

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval import Box3D


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


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load scene manifest (handles both list and dict formats)."""
    with manifest_path.open() as f:
        data = json.load(f)
    
    if isinstance(data, list):
        result = {}
        for scene in data:
            result[scene["scene_id"]] = scene
        return result
    else:
        return data


def box3d_to_corners(box: Box3D) -> np.ndarray:
    """Get 8 corners of a 3D box."""
    hx = box.size[0] / 2.0
    hy = box.size[1] / 2.0
    hz = box.size[2] / 2.0
    
    corners_local = np.array([
        [-hx, -hy, -hz],
        [hx, -hy, -hz],
        [hx, hy, -hz],
        [-hx, hy, -hz],
        [-hx, -hy, hz],
        [hx, -hy, hz],
        [hx, hy, hz],
        [-hx, hy, hz],
    ])
    
    # Rotation matrix for yaw
    c = np.cos(box.yaw)
    s = np.sin(box.yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    # Apply rotation and translation
    corners = corners_local @ rot.T + box.center
    return corners


def draw_box_edges(corners: np.ndarray) -> list[tuple[int, int]]:
    """Return edges for a box given its 8 corners."""
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return edges


def visualize_scene_3d(
    scene_id: str,
    points: np.ndarray,
    gt_boxes: list[Box3D],
    baseline_boxes: list[Box3D],
    method_boxes: list[Box3D],
    output_dir: Path,
) -> None:
    """Create Rerun 3D visualization for a scene."""
    
    # Initialize Rerun recording
    recording_path = output_dir / f"rerun_with_gt_{scene_id}.rrd"
    rr.init(f"Scene {scene_id}", spawn=False)  # Don't spawn viewer
    
    # Set up 3D space
    rr.set_time_seconds("time", 0)
    
    # Point cloud (downsampled for visualization)
    if points.shape[0] > 50000:
        idx = np.random.choice(points.shape[0], 50000, replace=False)
        pts_viz = points[idx]
    else:
        pts_viz = points
    
    rr.log(
        "world/point_cloud",
        rr.Points3D(pts_viz, colors=[200, 200, 200]),
    )

    # Ground-truth boxes (green)
    for i, box in enumerate(gt_boxes):
        corners = box3d_to_corners(box)
        edges = draw_box_edges(corners)

        for edge_idx, (start, end) in enumerate(edges):
            segment = np.array([corners[start], corners[end]])
            rr.log(
                f"world/gt_{i}/edge_{edge_idx}",
                rr.LineStrips3D([segment], colors=[[0, 255, 0]]),
            )

        rr.log(
            f"world/gt_{i}/center",
            rr.Points3D([box.center], colors=[[0, 255, 0]], radii=[0.08]),
        )
    
    # Baseline predictions (red)
    for i, box in enumerate(baseline_boxes):
        corners = box3d_to_corners(box)
        edges = draw_box_edges(corners)
        
        # Draw edges
        for edge_idx, (start, end) in enumerate(edges):
            segment = np.array([corners[start], corners[end]])
            rr.log(
                f"world/baseline_{i}/edge_{edge_idx}",
                rr.LineStrips3D([segment], colors=[[255, 0, 0]]),
            )
        
        # Draw center point
        rr.log(
            f"world/baseline_{i}/center",
            rr.Points3D([box.center], colors=[[255, 0, 0]], radii=[0.1]),
        )
        
        # Draw axes at box center (X=red, Y=green, Z=blue)
        axis_length = 0.3
        axes = [
            (box.center, box.center + np.array([axis_length, 0, 0]), [255, 0, 0]),        # X-red
            (box.center, box.center + np.array([0, axis_length, 0]), [0, 255, 0]),        # Y-green
            (box.center, box.center + np.array([0, 0, axis_length]), [0, 0, 255]),        # Z-blue
        ]
        for axis_idx, (start, end, color) in enumerate(axes):
            rr.log(
                f"world/baseline_{i}/axis_{axis_idx}",
                rr.LineStrips3D([[start, end]], colors=[color]),
            )
    
    # Method predictions (blue)
    for i, box in enumerate(method_boxes):
        corners = box3d_to_corners(box)
        edges = draw_box_edges(corners)
        
        # Draw edges
        for edge_idx, (start, end) in enumerate(edges):
            segment = np.array([corners[start], corners[end]])
            rr.log(
                f"world/method_{i}/edge_{edge_idx}",
                rr.LineStrips3D([segment], colors=[[0, 100, 255]]),
            )
        
        # Draw center point
        rr.log(
            f"world/method_{i}/center",
            rr.Points3D([box.center], colors=[[0, 100, 255]], radii=[0.1]),
        )
        
        # Draw axes at box center
        axis_length = 0.3
        axes = [
            (box.center, box.center + np.array([axis_length, 0, 0]), [255, 0, 0]),        # X-red
            (box.center, box.center + np.array([0, axis_length, 0]), [0, 255, 0]),        # Y-green
            (box.center, box.center + np.array([0, 0, axis_length]), [0, 0, 255]),        # Z-blue
        ]
        for axis_idx, (start, end, color) in enumerate(axes):
            rr.log(
                f"world/method_{i}/axis_{axis_idx}",
                rr.LineStrips3D([[start, end]], colors=[color]),
            )
    
    # Save recording
    rr.save(str(recording_path))
    print(f"  ✓ Saved {recording_path}")
    print(f"    Open with: rerun {recording_path}")


def load_gt_boxes_from_manifest(scene_data: dict[str, Any]) -> list[Box3D]:
    gt_ref = scene_data.get("gt_boxes")
    if gt_ref is None:
        return []

    if isinstance(gt_ref, list):
        return [Box3D.from_dict(box) for box in gt_ref]

    if isinstance(gt_ref, str):
        gt_path = Path(gt_ref)
        if not gt_path.exists():
            return []
        with gt_path.open() as f:
            gt_payload = json.load(f)
        if isinstance(gt_payload, dict):
            if "boxes" in gt_payload and isinstance(gt_payload["boxes"], list):
                return [Box3D.from_dict(box) for box in gt_payload["boxes"]]
            if "objects" in gt_payload and isinstance(gt_payload["objects"], list):
                parsed: list[Box3D] = []
                for obj in gt_payload["objects"]:
                    if not isinstance(obj, dict):
                        continue
                    box_dict = obj.get("obb") or obj.get("aabb")
                    if isinstance(box_dict, dict) and "center" in box_dict and "size" in box_dict:
                        parsed.append(Box3D.from_dict(box_dict))
                return parsed
        if isinstance(gt_payload, list):
            return [Box3D.from_dict(box) for box in gt_payload if isinstance(box, dict)]
        return []

    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Rerun 3D visualizations for Fig 4.6")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Scene manifest JSON")
    parser.add_argument("--baseline_perturbed", type=str, required=True,
                        help="Perturbed baseline predictions JSON")
    parser.add_argument("--method_pred", type=str, required=True,
                        help="Method predictions JSON")
    parser.add_argument("--output_dir", type=str, default="outputs/exp446_qualitative",
                        help="Output directory")
    parser.add_argument("--target_scenes", type=str, nargs="+", required=True,
                        help="Scene IDs to visualize")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"[Loading] manifest")
    manifest = load_manifest(Path(args.manifest))
    
    print(f"[Loading] baseline predictions")
    baseline_preds = load_predictions(Path(args.baseline_perturbed))
    
    print(f"[Loading] method predictions")
    method_preds = load_predictions(Path(args.method_pred))
    
    # Visualize each target scene
    for scene_id in args.target_scenes:
        print(f"\n[Scene] {scene_id}")
        
        scene_data = manifest.get(scene_id)
        if not scene_data:
            print(f"  WARNING: Scene not in manifest")
            continue
        
        # Load point cloud
        pcd_path = Path(scene_data.get("scene_point_cloud"))
        if not pcd_path.exists():
            print(f"  WARNING: Point cloud not found")
            continue
        
        print(f"  Loading point cloud")
        pcd_data = np.load(pcd_path)
        points = pcd_data["points"]
        
        # Parse predictions
        gt_boxes = load_gt_boxes_from_manifest(scene_data)
        baseline_boxes = [Box3D.from_dict(b) for b in baseline_preds.get(scene_id, [])]
        method_boxes = [Box3D.from_dict(b) for b in method_preds.get(scene_id, [])]
        
        print(f"  GT: {len(gt_boxes)}, Baseline: {len(baseline_boxes)}, Method: {len(method_boxes)}")
        
        # Visualize
        print(f"  Generating Rerun visualization")
        try:
            visualize_scene_3d(
                scene_id=scene_id,
                points=points,
                gt_boxes=gt_boxes,
                baseline_boxes=baseline_boxes,
                method_boxes=method_boxes,
                output_dir=out_dir,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[Complete] Rerun 3D visualizations saved in {out_dir}")


if __name__ == "__main__":
    main()
