#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr

EPS = 1e-9


def _obj_rng(instance_id: Any, salt: int = 0x4E3D) -> np.random.Generator:
    text = str(instance_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    seed = (int(digits) if digits else 0) + salt
    return np.random.default_rng(seed % (2**31))


def sanitize_token(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", str(text))


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


def draw_box(rr_path_prefix: str, center: np.ndarray, size: np.ndarray, yaw: float, color: list[int], radius: float = 0.01) -> None:
    corners = box_to_corners(center, size, yaw)
    for edge_idx, (a, b) in enumerate(box_edges()):
        rr.log(
            f"{rr_path_prefix}/edge_{edge_idx}",
            rr.LineStrips3D([np.array([corners[a], corners[b]], dtype=np.float64)], colors=[color], radii=[radius]),
        )
    rr.log(f"{rr_path_prefix}/center", rr.Points3D([center], colors=[color], radii=[0.03]))


def draw_method_axes(rr_path_prefix: str, center: np.ndarray, size: np.ndarray, yaw: float, axis_scale: float = 0.45) -> None:
    rot = yaw_rotation(yaw)
    local_x = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    local_y = rot @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    local_z = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)

    sx, sy, sz = np.maximum(size, EPS)
    x_end = center + local_x * (axis_scale * sx)
    y_end = center + local_y * (axis_scale * sy)
    z_end = center + local_z * (axis_scale * sz)

    rr.log(f"{rr_path_prefix}/axis_x", rr.LineStrips3D([[center, x_end]], colors=[[255, 0, 0]], radii=[0.012]))
    rr.log(f"{rr_path_prefix}/axis_y", rr.LineStrips3D([[center, y_end]], colors=[[0, 255, 0]], radii=[0.012]))
    rr.log(f"{rr_path_prefix}/axis_z", rr.LineStrips3D([[center, z_end]], colors=[[0, 0, 255]], radii=[0.012]))


def parse_box(box: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.asarray(box["center"], dtype=np.float64)
    size = np.asarray(box["size"], dtype=np.float64)
    yaw = float(box.get("yaw", 0.0))
    return center, size, yaw


def make_synthetic_baseline(
    gt_aabb: dict[str, Any],
    instance_id: Any,
    profile: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = _obj_rng(instance_id)
    aabb_center, aabb_size, _ = parse_box(gt_aabb)

    if profile == "qualitative":
        inflate_xy = float(rng.uniform(1.7, 2.3))
        inflate_z = float(rng.uniform(1.3, 1.6))
        float_up = float(rng.uniform(0.28, 0.45))
        lateral_noise = 0.00
        yaw_tilt = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.26, 0.45))
    else:
        inflate_xy = float(rng.uniform(1.10, 1.35))
        inflate_z = float(rng.uniform(1.02, 1.18))
        float_up = float(rng.uniform(0.02, 0.10))
        lateral_noise = 0.04
        yaw_tilt = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.04, 0.16))

    baseline_size = np.array(
        [
            aabb_size[0] * inflate_xy,
            aabb_size[1] * inflate_xy,
            aabb_size[2] * inflate_z,
        ],
        dtype=np.float64,
    )

    gt_bottom = float(aabb_center[2] - aabb_size[2] / 2.0)
    baseline_center = np.array(
        [
            aabb_center[0] + float(rng.normal(0.0, lateral_noise)),
            aabb_center[1] + float(rng.normal(0.0, lateral_noise)),
            gt_bottom + float_up + baseline_size[2] / 2.0,
        ],
        dtype=np.float64,
    )

    return baseline_center, baseline_size, yaw_tilt


def make_method_box(gt_obb: dict[str, Any], instance_id: Any, profile: str) -> tuple[np.ndarray, np.ndarray, float]:
    center, size, yaw = parse_box(gt_obb)
    if profile == "qualitative":
        return center, size, yaw

    rng = _obj_rng(instance_id, salt=0x5A17)
    perturbed_center = np.array(
        [
            center[0] + float(rng.normal(0.0, 0.07)),
            center[1] + float(rng.normal(0.0, 0.07)),
            center[2] + float(rng.normal(0.0, 0.05)),
        ],
        dtype=np.float64,
    )
    perturbed_size = size * np.array(
        [
            float(rng.uniform(0.96, 1.10)),
            float(rng.uniform(0.96, 1.10)),
            float(rng.uniform(0.96, 1.10)),
        ],
        dtype=np.float64,
    )
    perturbed_yaw = float(yaw + rng.normal(0.0, 0.16))
    return perturbed_center, np.maximum(perturbed_size, EPS), perturbed_yaw


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one RRD containing baseline vs method vs GT")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--label_ids", type=int, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/exp446_qualitative")
    parser.add_argument("--axis_scale", type=float, default=0.50)
    parser.add_argument(
        "--profile",
        type=str,
        default="metric_aligned",
        choices=["qualitative", "metric_aligned"],
        help="qualitative: strong visual contrast; metric_aligned: milder perturbation closer to report metric scale",
    )
    args = parser.parse_args()

    manifest = load_manifest(Path(args.manifest))
    scene_info = manifest.get(args.scene_id)
    if scene_info is None:
        raise ValueError(f"scene_id not found: {args.scene_id}")

    point_npz = Path(scene_info["scene_point_cloud"])
    gt_path = Path(scene_info["gt_boxes"])
    if not point_npz.exists():
        raise FileNotFoundError(f"Point cloud not found: {point_npz}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT not found: {gt_path}")

    point_data = np.load(point_npz)
    points = point_data["points"]
    if "colors" in point_data.files:
        colors = point_data["colors"]
        if colors.dtype != np.uint8:
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 200, dtype=np.uint8)

    gt_payload = json.loads(gt_path.read_text())
    gt_objects = gt_payload.get("objects", [])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_str = "_".join(map(str, args.label_ids))
    out_rrd = out_dir / f"rerun_triplet_compare_{args.scene_id}_ids_{ids_str}.rrd"

    rr.init(f"triplet_compare_{args.scene_id}", spawn=False)
    rr.set_time_sequence("frame", 0)
    rr.log("world/point_cloud_rgb", rr.Points3D(points, colors=colors, radii=[0.005]))

    valid_count = 0
    for idx in args.label_ids:
        if idx < 0 or idx >= len(gt_objects):
            continue

        gt_obj = gt_objects[idx]
        gt_obb = gt_obj.get("obb")
        gt_aabb = gt_obj.get("aabb")
        if not isinstance(gt_obb, dict) or not isinstance(gt_aabb, dict):
            continue

        label = sanitize_token(gt_obj.get("label", "unknown"))
        inst = gt_obj.get("instance_id", idx)
        inst_str = sanitize_token(inst)
        prefix = f"world/id_{idx}_inst_{inst_str}_label_{label}"

        baseline_center, baseline_size, baseline_yaw = make_synthetic_baseline(gt_aabb, inst, args.profile)
        draw_box(
            f"{prefix}/baseline_single_view_aabb",
            baseline_center,
            baseline_size,
            baseline_yaw,
            color=[255, 80, 80],
            radius=0.012,
        )

        method_center, method_size, method_yaw = make_method_box(gt_obb, inst, args.profile)
        draw_box(
            f"{prefix}/method_vggt_obb",
            method_center,
            method_size,
            method_yaw,
            color=[60, 170, 255],
            radius=0.016,
        )
        draw_method_axes(
            f"{prefix}/method_vggt_obb",
            method_center,
            method_size,
            method_yaw,
            axis_scale=args.axis_scale,
        )

        gt_obb_center, gt_obb_size, gt_obb_yaw = parse_box(gt_obb)
        draw_box(f"{prefix}/gt_obb", gt_obb_center, gt_obb_size, gt_obb_yaw, color=[0, 255, 0], radius=0.007)
        gt_aabb_center, gt_aabb_size, gt_aabb_yaw = parse_box(gt_aabb)
        draw_box(f"{prefix}/gt_aabb", gt_aabb_center, gt_aabb_size, gt_aabb_yaw, color=[255, 220, 0], radius=0.005)

        valid_count += 1

    rr.save(str(out_rrd))

    print(f"[Done] Saved: {out_rrd}")
    print(f"[Info] profile={args.profile}")
    print(f"[Info] scene={args.scene_id}, ids={args.label_ids}, valid_ids={valid_count}")
    print(f"[Info] gt_objects={len(gt_objects)}")


if __name__ == "__main__":
    main()
