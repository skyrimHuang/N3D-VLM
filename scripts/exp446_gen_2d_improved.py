#!/usr/bin/env python3
"""
Improved 2D qualitative visualization for Figure 4.6.

Fixes:
1. Display real RGB colors from point cloud (opacity 0.75)
2. Show only top 4-5 boxes by confidence score
3. Fix vertical connecting lines in box visualization
4. Adjust axis scaling to prevent crowding
5. Generate pairs: boxes + no-boxes (same angle, opacity 1.0)
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval import Box3D


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load scene manifest."""
    with manifest_path.open() as f:
        data = json.load(f)

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


def draw_box_3d(ax, box: Box3D, color: str, linewidth: float = 1.5) -> None:
    """Draw a complete 3D box with all edges (top, bottom, and vertical)."""
    
    # Get 4 corners at bottom (z0)
    half_x = box.size[0] / 2.0
    half_y = box.size[1] / 2.0
    half_z = box.size[2] / 2.0
    
    # Local corners (unrotated)
    local_corners = np.array([
        [-half_x, -half_y],
        [half_x, -half_y],
        [half_x, half_y],
        [-half_x, half_y],
    ])
    
    # Rotation for yaw
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    rot = np.array([[c, -s], [s, c]])
    
    # Apply rotation and translation
    corners_xy = local_corners @ rot.T + box.center[:2]
    
    z0 = box.center[2] - half_z
    z1 = box.center[2] + half_z
    
    # Bottom face (z0)
    bottom = np.vstack([corners_xy, corners_xy[0]])
    ax.plot(bottom[:, 0], bottom[:, 1], z0, color=color, linewidth=linewidth)
    
    # Top face (z1)
    top = np.vstack([corners_xy, corners_xy[0]])
    ax.plot(top[:, 0], top[:, 1], z1, color=color, linewidth=linewidth)
    
    # Vertical edges connecting top and bottom (4 edges)
    for i in range(4):
        x_vals = [corners_xy[i, 0], corners_xy[i, 0]]
        y_vals = [corners_xy[i, 1], corners_xy[i, 1]]
        z_vals = [z0, z1]
        ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=linewidth)


def plot_scene_with_boxes(
    scene_id: str,
    points: np.ndarray,
    colors: np.ndarray,
    base_boxes: list[Box3D],
    method_boxes: list[Box3D],
    output_dir: Path,
    max_boxes: int = 4,
) -> None:
    """Generate comparison figure with both boxes and point cloud visualization."""
    
    # Downsample points for faster visualization
    if points.shape[0] > 30000:
        idx = np.random.choice(points.shape[0], 30000, replace=False)
        pts_viz = points[idx]
        colors_viz = colors[idx]
    else:
        pts_viz = points
        colors_viz = colors
    
    # Normalize colors to [0, 1]
    colors_normalized = colors_viz.astype(np.float32) / 255.0
    
    # Select top boxes by confidence
    base_boxes_sorted = sorted(base_boxes, key=lambda b: b.score, reverse=True)[:max_boxes]
    method_boxes_sorted = sorted(method_boxes, key=lambda b: b.score, reverse=True)[:max_boxes]
    
    # ========== Figure 1: With Bounding Boxes ==========
    fig = plt.figure(figsize=(16, 7))
    
    # Adjust for equal aspect ratio
    mean_extent = np.max(np.std(points, axis=0))
    
    # Left: Baseline
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(pts_viz[:, 0], pts_viz[:, 1], pts_viz[:, 2],
                c=colors_normalized, s=1.5, alpha=0.75, edgecolors='none')
    
    # Draw baseline boxes (red)
    for box in base_boxes_sorted:
        draw_box_3d(ax1, box, color='red', linewidth=2.0)
    
    # Set equal aspect ratio
    ax1.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax1.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax1.set_zlim(points[:, 2].min(), points[:, 2].max())
    
    ax1.set_xlabel("X (m)", fontsize=10)
    ax1.set_ylabel("Y (m)", fontsize=10)
    ax1.set_zlabel("Z (m)", fontsize=10)
    ax1.set_title(f"{scene_id}\nSingle-view VLM (AABB rough)\n({len(base_boxes_sorted)}/{len(base_boxes)} top boxes shown)",
                  fontsize=11)
    
    # Right: Method
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(pts_viz[:, 0], pts_viz[:, 1], pts_viz[:, 2],
                c=colors_normalized, s=1.5, alpha=0.75, edgecolors='none')
    
    # Draw method boxes (blue)
    for box in method_boxes_sorted:
        draw_box_3d(ax2, box, color='blue', linewidth=2.0)
    
    # Set equal aspect ratio
    ax2.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax2.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax2.set_zlim(points[:, 2].min(), points[:, 2].max())
    
    ax2.set_xlabel("X (m)", fontsize=10)
    ax2.set_ylabel("Y (m)", fontsize=10)
    ax2.set_zlabel("Z (m)", fontsize=10)
    ax2.set_title(f"{scene_id}\nMulti-view + PCA/OBB (refined)\n({len(method_boxes_sorted)}/{len(method_boxes)} top boxes shown)",
                  fontsize=11)
    
    # Match viewing angles between left and right
    ax1.view_init(elev=20, azim=45)
    ax2.view_init(elev=20, azim=45)
    
    fig.tight_layout()
    
    # Save with boxes
    output_path_with_boxes = output_dir / f"figure_4_6_{scene_id}_with_boxes.png"
    fig.savefig(output_path_with_boxes, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved {output_path_with_boxes.name}")
    plt.close(fig)
    
    # ========== Figure 2: Point Cloud Only (No Boxes) ==========
    fig = plt.figure(figsize=(16, 7))
    
    # Left: Baseline scene (no boxes)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(pts_viz[:, 0], pts_viz[:, 1], pts_viz[:, 2],
                c=colors_normalized, s=1.5, alpha=1.0, edgecolors='none')
    
    ax1.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax1.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax1.set_zlim(points[:, 2].min(), points[:, 2].max())
    
    ax1.set_xlabel("X (m)", fontsize=10)
    ax1.set_ylabel("Y (m)", fontsize=10)
    ax1.set_zlabel("Z (m)", fontsize=10)
    ax1.set_title(f"{scene_id}\nOriginal Point Cloud (Scene Context)",
                  fontsize=11)
    
    # Right: Method scene (no boxes)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(pts_viz[:, 0], pts_viz[:, 1], pts_viz[:, 2],
                c=colors_normalized, s=1.5, alpha=1.0, edgecolors='none')
    
    ax2.set_xlim(points[:, 0].min(), points[:, 0].max())
    ax2.set_ylim(points[:, 1].min(), points[:, 1].max())
    ax2.set_zlim(points[:, 2].min(), points[:, 2].max())
    
    ax2.set_xlabel("X (m)", fontsize=10)
    ax2.set_ylabel("Y (m)", fontsize=10)
    ax2.set_zlabel("Z (m)", fontsize=10)
    ax2.set_title(f"{scene_id}\nOriginal Point Cloud (Scene Context)",
                  fontsize=11)
    
    # Match viewing angles
    ax1.view_init(elev=20, azim=45)
    ax2.view_init(elev=20, azim=45)
    
    fig.tight_layout()
    
    # Save without boxes
    output_path_no_boxes = output_dir / f"figure_4_6_{scene_id}_point_cloud.png"
    fig.savefig(output_path_no_boxes, dpi=200, bbox_inches='tight')
    print(f"  ✓ Saved {output_path_no_boxes.name}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Improved Fig 4.6 2D qualitative visualizations")
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
    parser.add_argument("--max_boxes", type=int, default=4,
                        help="Maximum number of top boxes to display per method")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"[Loading] manifest from {args.manifest}")
    manifest = load_manifest(Path(args.manifest))
    
    print(f"[Loading] baseline predictions")
    baseline_preds = load_predictions(Path(args.baseline_perturbed))
    
    print(f"[Loading] method predictions")
    method_preds = load_predictions(Path(args.method_pred))
    
    # Visualize each scene
    for scene_id in args.target_scenes:
        print(f"\n[Scene] {scene_id}")
        
        scene_data = manifest.get(scene_id)
        if not scene_data:
            print(f"  WARNING: Scene not in manifest")
            continue
        
        # Load point cloud with colors
        pcd_path = Path(scene_data.get("scene_point_cloud"))
        if not pcd_path.exists():
            print(f"  WARNING: Point cloud not found")
            continue
        
        print(f"  Loading point cloud with colors")
        pcd_data = np.load(pcd_path)
        points = pcd_data["points"]
        colors = pcd_data.get("colors", np.ones_like(points, dtype=np.uint8) * 128)
        
        # Parse predictions
        baseline_boxes = [Box3D.from_dict(b) for b in baseline_preds.get(scene_id, [])]
        method_boxes = [Box3D.from_dict(b) for b in method_preds.get(scene_id, [])]
        
        print(f"  Points: {len(points)}, Baseline: {len(baseline_boxes)}, Method: {len(method_boxes)}")
        
        # Visualize
        try:
            plot_scene_with_boxes(
                scene_id=scene_id,
                points=points,
                colors=colors,
                base_boxes=baseline_boxes,
                method_boxes=method_boxes,
                output_dir=out_dir,
                max_boxes=args.max_boxes,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[Complete] Improved 2D visualizations saved in {out_dir}")


if __name__ == "__main__":
    main()
