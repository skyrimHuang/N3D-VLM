#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import re

import numpy as np
import rerun as rr


EPS = 1e-9


def load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(manifest_path.read_text())
    if isinstance(data, list):
        return {item["scene_id"]: item for item in data}
    return data


def yaw_rotation(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def box_to_corners(center: np.ndarray, size: np.ndarray, yaw: float) -> np.ndarray:
    half = np.maximum(size, EPS) / 2.0
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    rot = yaw_rotation(yaw)
    return local @ rot.T + center


def box_edges() -> list[tuple[int, int]]:
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]


def draw_box(rr_path_prefix: str, box: dict[str, Any], color: list[int], radius: float = 0.01) -> None:
    center = np.asarray(box["center"], dtype=np.float64)
    size = np.asarray(box["size"], dtype=np.float64)
    yaw = float(box.get("yaw", 0.0))

    corners = box_to_corners(center, size, yaw)
    for edge_idx, (a, b) in enumerate(box_edges()):
        rr.log(
            f"{rr_path_prefix}/edge_{edge_idx}",
            rr.LineStrips3D([np.array([corners[a], corners[b]], dtype=np.float64)], colors=[color], radii=[radius]),
        )
    rr.log(f"{rr_path_prefix}/center", rr.Points3D([center], colors=[color], radii=[0.03]))


def sanitize_token(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RRD with selected GT AABB+OBB and RGB point cloud")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--label_ids", type=int, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/exp446_qualitative")
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    scene_info = manifest.get(args.scene_id)
    if scene_info is None:
        raise ValueError(f"scene_id not found in manifest: {args.scene_id}")

    point_npz = Path(scene_info["scene_point_cloud"])
    if not point_npz.exists():
        raise FileNotFoundError(f"Point cloud not found: {point_npz}")

    gt_path = Path(scene_info["gt_boxes"])
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")

    point_data = np.load(point_npz)
    points = point_data["points"]
    colors = point_data["colors"] if "colors" in point_data.files else np.full((points.shape[0], 3), 200, dtype=np.uint8)

    gt_payload = json.loads(gt_path.read_text())
    objects = gt_payload.get("objects", [])

    selected = []
    for idx in args.label_ids:
        if idx < 0 or idx >= len(objects):
            continue
        selected.append((idx, objects[idx]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rrd = out_dir / f"rerun_gt_aabb_obb_{args.scene_id}_ids_{'_'.join(map(str, args.label_ids))}.rrd"

    rr.init(f"GT_AABB_OBB_{args.scene_id}", spawn=False)
    rr.set_time_sequence("frame", 0)

    rr.log("world/point_cloud", rr.Points3D(points, colors=colors))

    # Legend:
    # AABB -> yellow, OBB -> green
    for idx, obj in selected:
        aabb = obj.get("aabb")
        obb = obj.get("obb")
        label = str(obj.get("label", "unknown"))
        inst = str(obj.get("instance_id", idx))
        safe_label = sanitize_token(label)
        safe_inst = sanitize_token(inst)
        base_path = f"world/gt_selected/id_{idx}_inst_{safe_inst}_label_{safe_label}"

        if isinstance(aabb, dict):
            draw_box(f"{base_path}/aabb", aabb, [255, 220, 0], radius=0.008)
        if isinstance(obb, dict):
            draw_box(f"{base_path}/obb", obb, [0, 255, 0], radius=0.012)

    rr.save(str(out_rrd))

    print(f"[Done] Saved: {out_rrd}")
    print(f"[Info] scene={args.scene_id}, selected_ids={args.label_ids}, valid={len(selected)}")


if __name__ == "__main__":
    main()
