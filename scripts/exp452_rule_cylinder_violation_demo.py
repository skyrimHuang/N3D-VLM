#!/usr/bin/env python3
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


def choose_forward_normal_from_plus_y(rot_mat: np.ndarray) -> tuple[np.ndarray, int, float]:
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    best_idx = 0
    best_dot = -1.0
    best_normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    for i in range(3):
        axis_i = rot_mat[:, i].astype(np.float32)
        axis_i = axis_i / (np.linalg.norm(axis_i) + 1e-12)
        cand = axis_i if float(np.dot(axis_i, y_axis)) >= 0.0 else -axis_i
        score = float(np.dot(cand, y_axis))
        if score > best_dot:
            best_dot = score
            best_idx = i
            best_normal = cand

    return best_normal, best_idx, best_dot


def points_in_forward_cylinder(
    points_xyz: np.ndarray,
    cylinder_base: np.ndarray,
    axis_unit: np.ndarray,
    depth_limit: float,
    radius_limit: float,
) -> np.ndarray:
    rel = points_xyz - cylinder_base[None, :]
    t = rel @ axis_unit
    axial_ok = (t >= 0.0) & (t <= depth_limit)

    radial_vec = rel - t[:, None] * axis_unit[None, :]
    radial_dist = np.linalg.norm(radial_vec, axis=1)
    radial_ok = radial_dist <= radius_limit
    return axial_ok & radial_ok


def cylinder_axis_segment(base: np.ndarray, axis_unit: np.ndarray, depth_limit: float) -> np.ndarray:
    tip = base + depth_limit * axis_unit
    return np.stack([base, tip], axis=0).astype(np.float32)


def build_cylinder_wireframe(
    base: np.ndarray,
    axis_unit: np.ndarray,
    depth_limit: float,
    radius_limit: float,
    num_segments: int = 48,
) -> list[np.ndarray]:
    axis = axis_unit / (np.linalg.norm(axis_unit) + 1e-12)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(axis, ref))) > 0.95:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    u = np.cross(axis, ref)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(axis, u)
    v = v / (np.linalg.norm(v) + 1e-12)

    top = base + depth_limit * axis
    angles = np.linspace(0.0, 2.0 * np.pi, num_segments, endpoint=False)
    circle_base = np.array(
        [base + radius_limit * (np.cos(a) * u + np.sin(a) * v) for a in angles],
        dtype=np.float32,
    )
    circle_top = np.array(
        [top + radius_limit * (np.cos(a) * u + np.sin(a) * v) for a in angles],
        dtype=np.float32,
    )

    strips: list[np.ndarray] = []
    strips.append(np.vstack([circle_base, circle_base[0:1]]).astype(np.float32))
    strips.append(np.vstack([circle_top, circle_top[0:1]]).astype(np.float32))

    for idx in range(0, num_segments, max(1, num_segments // 12)):
        strips.append(np.vstack([circle_base[idx], circle_top[idx]]).astype(np.float32))

    return strips


def build_obb_wireframe(center: np.ndarray, size: np.ndarray, rot_mat: np.ndarray) -> list[np.ndarray]:
    hx, hy, hz = (size / 2.0).astype(np.float32)
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
    return strips


def export_rrd_with_violation(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    violation_mask: np.ndarray,
    extinguisher_mask: np.ndarray,
    fire_center: np.ndarray,
    fire_size: np.ndarray,
    fire_rot: np.ndarray,
    cyl_base: np.ndarray,
    cyl_axis_unit: np.ndarray,
    depth_limit: float,
    radius_limit: float,
    out_rrd: Path,
) -> None:
    rr.init("Rule Cylinder Violation Demo", spawn=False)

    colors = colors_rgb.copy()
    colors[violation_mask] = np.array([255, 255, 0], dtype=np.uint8)
    rr.log("scene/points", rr.Points3D(points_xyz, colors=colors))

    axis_seg = cylinder_axis_segment(cyl_base, cyl_axis_unit, depth_limit)
    cyl_wire = build_cylinder_wireframe(cyl_base, cyl_axis_unit, depth_limit, radius_limit)

    rr.log(
        "scene/rule/cylinder_axis",
        rr.LineStrips3D([axis_seg], colors=np.array([[255, 140, 0]], dtype=np.uint8), labels=["Violation cylinder axis"]),
    )
    rr.log(
        "scene/rule/cylinder_wireframe",
        rr.LineStrips3D(cyl_wire, colors=np.array([[255, 165, 0]], dtype=np.uint8)),
    )

    fire_obb = build_obb_wireframe(fire_center, fire_size, fire_rot)
    rr.log(
        "scene/extinguisher/obb",
        rr.LineStrips3D(fire_obb, colors=np.array([[0, 220, 255]], dtype=np.uint8)),
    )

    rr.log(
        "scene/rule/cylinder_base_tip",
        rr.Points3D(
            np.stack([cyl_base, axis_seg[1]], axis=0),
            colors=np.array([[255, 140, 0], [255, 140, 0]], dtype=np.uint8),
        ),
    )

    rr.log(
        "scene/extinguisher/points",
        rr.Points3D(
            points_xyz[extinguisher_mask],
            colors=np.tile(np.array([[255, 80, 80]], dtype=np.uint8), (int(np.sum(extinguisher_mask)), 1)),
        ),
    )

    rr.log(
        "scene/violations/points",
        rr.Points3D(
            points_xyz[violation_mask],
            colors=np.tile(np.array([[255, 255, 0]], dtype=np.uint8), (int(np.sum(violation_mask)), 1)),
            labels=["Violation points"],
        ),
    )

    rr.log(
        "scene/extinguisher/center",
        rr.Points3D(fire_center[None, :], colors=np.array([[255, 80, 80]], dtype=np.uint8)),
    )

    rr.save(str(out_rrd))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rule-driven cylinder violation coloring demo for fire extinguisher")
    parser.add_argument(
        "--ply",
        default="data/SampleLabel/pointclouds/SamplePointMapWithRightPose_scaled_x2.75.ply",
        help="Input point cloud path",
    )
    parser.add_argument(
        "--label_json",
        default="outputs/sample_label_demo/SamplePointMapWithRightPose_labels_scaled_x2.75.json",
        help="Label JSON containing object OBBs",
    )
    parser.add_argument(
        "--depth",
        type=float,
        default=1.0,
        help="Cylinder depth limit in meters",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.8,
        help="Cylinder radius limit in meters",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/sample_label_demo",
        help="Output directory",
    )
    parser.add_argument(
        "--out_rrd_name",
        default="sample_fire_rule_cylinder_violation.rrd",
        help="Output RRD file name",
    )
    parser.add_argument(
        "--out_summary_name",
        default="sample_fire_rule_cylinder_violation_summary.json",
        help="Output JSON summary file name",
    )
    args = parser.parse_args()

    ply_path = Path(args.ply)
    label_json_path = Path(args.label_json)
    if not ply_path.exists():
        raise FileNotFoundError(f"Point cloud not found: {ply_path}")
    if not label_json_path.exists():
        raise FileNotFoundError(f"Label JSON not found: {label_json_path}")

    objects = parse_objects(label_json_path)
    fire_obj = next((o for o in objects if "fire" in str(o.get("name", "")).lower()), None)
    if fire_obj is None:
        raise ValueError("Cannot find Fire extinguisher object from label JSON")

    points_xyz, colors_rgb = load_binary_ply_xyzrgba(ply_path)
    fire_center, fire_size, fire_rot = object_to_obb(fire_obj)

    forward_unit, axis_idx, dot_y = choose_forward_normal_from_plus_y(fire_rot)
    cylinder_base = fire_center.copy()

    in_cylinder_mask = points_in_forward_cylinder(
        points_xyz=points_xyz,
        cylinder_base=cylinder_base,
        axis_unit=forward_unit,
        depth_limit=float(args.depth),
        radius_limit=float(args.radius),
    )
    extinguisher_mask = points_in_obb(points_xyz, fire_center, fire_size, fire_rot)
    violation_mask = in_cylinder_mask & (~extinguisher_mask)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rrd = out_dir / args.out_rrd_name
    out_summary = out_dir / args.out_summary_name

    export_rrd_with_violation(
        points_xyz=points_xyz,
        colors_rgb=colors_rgb,
        violation_mask=violation_mask,
        extinguisher_mask=extinguisher_mask,
        fire_center=fire_center,
        fire_size=fire_size,
        fire_rot=fire_rot,
        cyl_base=cylinder_base,
        cyl_axis_unit=forward_unit,
        depth_limit=float(args.depth),
        radius_limit=float(args.radius),
        out_rrd=out_rrd,
    )

    summary = {
        "rule": "灭火器前方1米内不得有遮挡物",
        "parameterized_tuple": {
            "E_target": "Fire extinguisher",
            "Z_space": {
                "type": "cylinder",
                "depth_limit_m": float(args.depth),
                "radius_limit_m": float(args.radius),
            },
            "C_logic": "cylinder 内非灭火器点视为违规",
        },
        "input": {
            "pointcloud": str(ply_path),
            "label_json": str(label_json_path),
        },
        "forward_axis_selection": {
            "axis_index": int(axis_idx),
            "forward_unit": forward_unit.tolist(),
            "dot_with_global_plus_y": float(dot_y),
            "global_plus_y": [0.0, 1.0, 0.0],
        },
        "extinguisher_obb": {
            "center": fire_center.tolist(),
            "size": fire_size.tolist(),
            "rotation_matrix": fire_rot.tolist(),
            "cylinder_base_center": cylinder_base.tolist(),
        },
        "statistics": {
            "total_points": int(points_xyz.shape[0]),
            "extinguisher_points": int(np.sum(extinguisher_mask)),
            "points_in_cylinder": int(np.sum(in_cylinder_mask)),
            "violation_points": int(np.sum(violation_mask)),
            "violation_ratio": float(np.sum(violation_mask) / max(points_xyz.shape[0], 1)),
        },
        "outputs": {
            "rrd": str(out_rrd),
            "summary_json": str(out_summary),
        },
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("完成：")
    print(f"- 输入点云: {ply_path}")
    print(f"- 标签文件: {label_json_path}")
    print(f"- 前向轴索引: {axis_idx}, 与+Y点积: {dot_y:.6f}")
    print(f"- 前向向量: {forward_unit}")
    print(f"- 圆柱底面中心(质心): {cylinder_base}")
    print(f"- 参数: depth={float(args.depth):.3f}m, radius={float(args.radius):.3f}m")
    print(f"- 违规点数量: {int(np.sum(violation_mask))}")
    print(f"- RRD: {out_rrd}")
    print(f"- Summary: {out_summary}")


if __name__ == "__main__":
    main()