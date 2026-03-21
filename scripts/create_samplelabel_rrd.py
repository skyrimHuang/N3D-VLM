#!/usr/bin/env python
"""
Create two RRD files from SampleLabel data:
1) labeled_scene.rrd: point cloud + object label boxes with names.
2) chair_highlight.rrd: same scene but points inside Chair label (and within 1.0m
   around fire extinguisher center) highlighted in yellow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R


def load_binary_ply_xyzrgba(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with ply_path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY: missing end_header")
            header_lines.append(line)
            if line.strip() == b"end_header":
                break
        data_start = f.tell()

    header_text = b"".join(header_lines).decode("ascii", errors="ignore")
    if "format binary_little_endian 1.0" not in header_text:
        raise ValueError("Only binary_little_endian PLY is supported")

    vertex_count = None
    for line in header_text.splitlines():
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
            break
    if vertex_count is None:
        raise ValueError("Cannot parse vertex count from PLY header")

    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ]
    )

    with ply_path.open("rb") as f:
        f.seek(data_start)
        verts = np.fromfile(f, dtype=dtype, count=vertex_count)

    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
    rgb = np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def parse_objects(label_json_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(label_json_path.read_text(encoding="utf-8"))
    objects = payload.get("objects", [])
    if not objects:
        raise ValueError(f"No objects found in {label_json_path}")
    return objects


def object_to_obb(obj: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.array(
        [
            float(obj["centroid"]["x"]),
            float(obj["centroid"]["y"]),
            float(obj["centroid"]["z"]),
        ],
        dtype=np.float32,
    )
    size = np.array(
        [
            float(obj["dimensions"]["length"]),
            float(obj["dimensions"]["width"]),
            float(obj["dimensions"]["height"]),
        ],
        dtype=np.float32,
    )
    rot = R.from_euler(
        "xyz",
        [
            float(obj["rotations"].get("x", 0.0)),
            float(obj["rotations"].get("y", 0.0)),
            float(obj["rotations"].get("z", 0.0)),
        ],
        degrees=True,
    ).as_matrix().astype(np.float32)
    return center, size, rot


def points_in_obb(points_xyz: np.ndarray, center: np.ndarray, size: np.ndarray, rot_mat: np.ndarray) -> np.ndarray:
    local = (points_xyz - center[None, :]) @ rot_mat
    half = size / 2.0
    inside = np.all(np.abs(local) <= half[None, :], axis=1)
    return inside


def log_obb_as_lines(entity_path: str, center: np.ndarray, size: np.ndarray, rot_mat: np.ndarray, color: np.ndarray, label: str) -> None:
    sx, sy, sz = size.tolist()
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    corners = local @ rot_mat.T + center[None, :]
    strips = [
        np.array([corners[0], corners[1], corners[2], corners[3], corners[0]], dtype=np.float32),
        np.array([corners[4], corners[5], corners[6], corners[7], corners[4]], dtype=np.float32),
        np.array([corners[0], corners[4]], dtype=np.float32),
        np.array([corners[1], corners[5]], dtype=np.float32),
        np.array([corners[2], corners[6]], dtype=np.float32),
        np.array([corners[3], corners[7]], dtype=np.float32),
    ]
    rr.log(
        entity_path,
        rr.LineStrips3D(
            strips,
            colors=color[None, :],
            labels=[label],
        ),
    )


def export_rrd_with_labels(points_xyz: np.ndarray, colors_rgb: np.ndarray, objects: list[dict[str, Any]], out_rrd: Path) -> None:
    rr.init("SampleLabel Demo", spawn=False)
    rr.log("scene/points", rr.Points3D(points_xyz, colors=colors_rgb))

    palette = {
        "chair": np.array([80, 180, 255], dtype=np.uint8),
        "fire extinguisher": np.array([255, 80, 80], dtype=np.uint8),
    }

    for idx, obj in enumerate(objects):
        name = str(obj["name"])
        center, size, rot = object_to_obb(obj)
        color = palette.get(name.lower(), np.array([255, 200, 0], dtype=np.uint8))
        log_obb_as_lines(
            entity_path=f"scene/labels/{idx}_{name.replace(' ', '_')}",
            center=center,
            size=size,
            rot_mat=rot,
            color=color,
            label=name,
        )
        rr.log(
            f"scene/label_centers/{idx}_{name.replace(' ', '_')}",
            rr.Points3D(center[None, :], colors=color[None, :], labels=[name]),
        )

    rr.save(str(out_rrd))


def export_rrd_with_chair_highlight(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    chair_mask: np.ndarray,
    around_fire_mask: np.ndarray,
    objects: list[dict[str, Any]],
    out_rrd: Path,
) -> None:
    rr.init("SampleLabel Chair Highlight", spawn=False)

    colors = colors_rgb.copy()
    highlight_mask = chair_mask & around_fire_mask
    colors[highlight_mask] = np.array([255, 255, 0], dtype=np.uint8)

    rr.log("scene/points", rr.Points3D(points_xyz, colors=colors))

    if np.any(highlight_mask):
        rr.log(
            "scene/violations/chair_in_1m",
            rr.Points3D(
                points_xyz[highlight_mask],
                colors=np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (int(np.sum(highlight_mask)), 1)),
                labels=["chair_in_1m_of_extinguisher"],
            ),
        )

    for idx, obj in enumerate(objects):
        name = str(obj["name"])
        center, size, rot = object_to_obb(obj)
        color = np.array([0, 255, 0], dtype=np.uint8)
        log_obb_as_lines(
            entity_path=f"scene/labels/{idx}_{name.replace(' ', '_')}",
            center=center,
            size=size,
            rot_mat=rot,
            color=color,
            label=name,
        )

    rr.save(str(out_rrd))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create two RRD files from SampleLabel point cloud + labels")
    parser.add_argument(
        "--label_json",
        default="data/SampleLabel/SamplePointMapWithRightPose.json",
        help="Path to label JSON file",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Radius around fire extinguisher center (meters)",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/sample_label_demo",
        help="Output directory for RRD files",
    )
    args = parser.parse_args()

    label_json_path = Path(args.label_json)
    payload = json.loads(label_json_path.read_text(encoding="utf-8"))
    ply_rel = payload.get("path") or payload.get("filename")
    if not ply_rel:
        raise ValueError("Label JSON has no 'path' or 'filename' field")

    ply_path = (label_json_path.parent / ply_rel).resolve()
    if not ply_path.exists():
        alt = (label_json_path.parent / "pointclouds" / Path(ply_rel).name).resolve()
        if alt.exists():
            ply_path = alt
        else:
            raise FileNotFoundError(f"Point cloud file not found: {ply_path}")

    objects = parse_objects(label_json_path)
    points_xyz, colors_rgb = load_binary_ply_xyzrgba(ply_path)

    chair_obj = next((o for o in objects if str(o["name"]).lower() == "chair"), None)
    fire_obj = next((o for o in objects if "fire" in str(o["name"]).lower()), None)
    if chair_obj is None or fire_obj is None:
        raise ValueError("Both 'Chair' and 'Fire extinguisher' labels are required")

    chair_center, chair_size, chair_rot = object_to_obb(chair_obj)
    fire_center, _, _ = object_to_obb(fire_obj)

    chair_mask = points_in_obb(points_xyz, chair_center, chair_size, chair_rot)
    around_fire_mask = np.linalg.norm(points_xyz - fire_center[None, :], axis=1) <= float(args.radius)
    chair_in_1m_mask = chair_mask & around_fire_mask

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_rrd_labels = out_dir / "sample_with_labels.rrd"
    out_rrd_highlight = out_dir / "sample_chair_in_1m_yellow.rrd"

    export_rrd_with_labels(points_xyz, colors_rgb, objects, out_rrd_labels)
    export_rrd_with_chair_highlight(
        points_xyz=points_xyz,
        colors_rgb=colors_rgb,
        chair_mask=chair_mask,
        around_fire_mask=around_fire_mask,
        objects=objects,
        out_rrd=out_rrd_highlight,
    )

    summary = {
        "label_json": str(label_json_path),
        "pointcloud": str(ply_path),
        "radius_m": float(args.radius),
        "total_points": int(points_xyz.shape[0]),
        "chair_points": int(np.sum(chair_mask)),
        "points_within_1m_of_fire_extinguisher": int(np.sum(around_fire_mask)),
        "chair_points_within_1m_of_fire_extinguisher": int(np.sum(chair_in_1m_mask)),
        "rrd_with_labels": str(out_rrd_labels),
        "rrd_chair_yellow": str(out_rrd_highlight),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Saved: {out_rrd_labels}")
    print(f"Saved: {out_rrd_highlight}")
    print(f"Saved: {summary_path}")
    print(
        f"chair points in 1.0m around fire extinguisher: "
        f"{summary['chair_points_within_1m_of_fire_extinguisher']}"
    )


if __name__ == "__main__":
    main()
