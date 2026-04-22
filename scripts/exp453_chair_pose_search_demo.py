#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


@dataclass
class PoseCandidate:
    idx_xy: tuple[int, int]
    center: np.ndarray
    rot_mat: np.ndarray
    yaw_deg: float
    yaw_delta_deg: float
    support_score: int
    collision_points: int
    rule_hit_points: int
    distance_cost: float


def load_binary_ply_xyzrgba(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with ply_path.open("rb") as f:
        header_lines: list[bytes] = []
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


def object_to_obb(obj: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
    yaw_deg = float(obj["rotations"].get("z", 0.0))
    rot = R.from_euler(
        "xyz",
        [
            float(obj["rotations"].get("x", 0.0)),
            float(obj["rotations"].get("y", 0.0)),
            yaw_deg,
        ],
        degrees=True,
    ).as_matrix().astype(np.float32)
    return center, size, rot, yaw_deg


def points_in_obb(points_xyz: np.ndarray, center: np.ndarray, size: np.ndarray, rot_mat: np.ndarray) -> np.ndarray:
    local = (points_xyz - center[None, :]) @ rot_mat
    half = size / 2.0
    return np.all(np.abs(local) <= half[None, :], axis=1)


def log_obb_as_lines(entity_path: str, center: np.ndarray, size: np.ndarray, rot_mat: np.ndarray, color: np.ndarray) -> None:
    hx, hy, hz = (size / 2.0).tolist()
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
    rr.log(entity_path, rr.LineStrips3D(strips, colors=color[None, :]))


def choose_forward_normal_from_plus_y(rot_mat: np.ndarray) -> np.ndarray:
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    best = y_axis.copy()
    best_dot = -1.0
    for i in range(3):
        axis_i = rot_mat[:, i].astype(np.float32)
        axis_i = axis_i / (np.linalg.norm(axis_i) + 1e-12)
        cand = axis_i if float(np.dot(axis_i, y_axis)) >= 0 else -axis_i
        d = float(np.dot(cand, y_axis))
        if d > best_dot:
            best_dot = d
            best = cand
    return best


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


def angular_diff_deg(a: float, b: float) -> float:
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def infer_floor_height(points_xyz: np.ndarray, q: float = 1.0) -> float:
    z = points_xyz[:, 2]
    robust_q = max(float(q), 5.0)
    z_cut = float(np.percentile(z, robust_q))
    low_band = z[z <= z_cut]
    if low_band.size == 0:
        return float(np.percentile(z, q))
    return float(np.median(low_band))


def build_grid(
    points_xyz: np.ndarray,
    step: float,
    extent_mode: str,
    p_low: float,
    p_high: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    x_min, x_max = float(np.min(points_xyz[:, 0])), float(np.max(points_xyz[:, 0]))
    y_min, y_max = float(np.min(points_xyz[:, 1])), float(np.max(points_xyz[:, 1]))

    if extent_mode == "full":
        x0, x1 = x_min, x_max
        y0, y1 = y_min, y_max
    else:
        x0, x1 = np.percentile(points_xyz[:, 0], [p_low, p_high])
        y0, y1 = np.percentile(points_xyz[:, 1], [p_low, p_high])

    x0 -= margin
    x1 += margin
    y0 -= margin
    y1 += margin

    xs = np.arange(x0, x1 + 1e-6, step, dtype=np.float32)
    ys = np.arange(y0, y1 + 1e-6, step, dtype=np.float32)
    bounds = {
        "scene_xy_full_minmax": [x_min, x_max, y_min, y_max],
        "grid_xy_bounds_used": [float(x0), float(x1), float(y0), float(y1)],
    }
    return xs, ys, bounds


def sample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def transform_template(local_pts: np.ndarray, center: np.ndarray, rot_mat: np.ndarray) -> np.ndarray:
    return (local_pts @ rot_mat.T + center[None, :]).astype(np.float32)


def support_probe_count_for_grid_point(
    support_points: np.ndarray,
    support_tree: cKDTree,
    center: np.ndarray,
    support_rot_mat: np.ndarray,
    size: np.ndarray,
    footprint_expand: float,
    probe_start_height_fraction: float,
    probe_depth: float,
) -> int:
    bottom_z = float(center[2] - size[2] / 2.0)
    # Probe slab upper face is lifted from the bottom; lower face still extends below bottom.
    probe_top_z = bottom_z + float(np.clip(probe_start_height_fraction, 0.0, 1.0) * size[2])
    probe_low_z = bottom_z - float(probe_depth)
    z_min = min(probe_low_z, probe_top_z)
    z_max = max(probe_low_z, probe_top_z)
    half_x = float(size[0] / 2.0 + footprint_expand)
    half_y = float(size[1] / 2.0 + footprint_expand)
    # Use circumscribed radius for KDTree prefilter, then apply exact rectangular footprint test.
    radius_xy = float(np.sqrt(half_x * half_x + half_y * half_y))

    slab_height = float(max(z_max - z_min, 1e-6))
    q_center = np.array([center[0], center[1], (z_max + z_min) * 0.5], dtype=np.float32)
    q_radius = float(np.sqrt(radius_xy**2 + (slab_height * 0.5) ** 2))
    idx = support_tree.query_ball_point(q_center, q_radius)
    if len(idx) == 0:
        return 0

    pts = support_points[np.asarray(idx, dtype=np.int64)]
    z_ok = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
    local = (pts - center[None, :]) @ support_rot_mat
    xy_ok = (np.abs(local[:, 0]) <= half_x) & (np.abs(local[:, 1]) <= half_y)
    return int(np.sum(z_ok & xy_ok))


def collision_points_for_pose(
    env_points: np.ndarray,
    center: np.ndarray,
    rot_mat: np.ndarray,
    size: np.ndarray,
    inflate: float,
    ignore_bottom_fraction: float,
) -> int:
    if env_points.shape[0] == 0:
        return 0

    s = size + 2.0 * inflate
    local = (env_points - center[None, :]) @ rot_mat
    half = s / 2.0
    inside = np.all(np.abs(local) <= half[None, :], axis=1)

    if float(ignore_bottom_fraction) > 0.0:
        z_cut = -half[2] + float(ignore_bottom_fraction) * s[2]
        inside = inside & (local[:, 2] > z_cut)

    return int(np.sum(inside))


def rrd_points_with_alpha(points_xyz: np.ndarray, rgb: np.ndarray, alpha: int) -> rr.Points3D:
    rgba = np.concatenate([rgb.astype(np.uint8), np.full((rgb.shape[0], 1), alpha, dtype=np.uint8)], axis=1)
    return rr.Points3D(points_xyz, colors=rgba)


def export_rrd_1_grid(
    scene_xyz: np.ndarray,
    scene_rgb: np.ndarray,
    grid_xy: np.ndarray,
    floor_z: float,
    grid_point_radius: float,
    grid_post_height: float,
    chair_center: np.ndarray,
    chair_size: np.ndarray,
    chair_rot: np.ndarray,
    out_rrd: Path,
) -> None:
    rr.init("Pose Search Grid", spawn=False)
    rr.log("scene/points", rr.Points3D(scene_xyz, colors=scene_rgb))
    rr.log(
        "scene/search_grid",
        rr.Points3D(
            np.concatenate([grid_xy, np.full((grid_xy.shape[0], 1), floor_z, dtype=np.float32)], axis=1),
            colors=np.tile(np.array([[80, 210, 255]], dtype=np.uint8), (grid_xy.shape[0], 1)),
            radii=float(grid_point_radius),
        ),
    )

    z0 = floor_z - 0.02
    z1 = floor_z + float(grid_post_height)
    posts = [
        np.array([[p[0], p[1], z0], [p[0], p[1], z1]], dtype=np.float32)
        for p in grid_xy
    ]
    rr.log(
        "scene/search_grid_posts",
        rr.LineStrips3D(posts, colors=np.array([[70, 185, 255]], dtype=np.uint8)),
    )

    log_obb_as_lines("scene/chair_original_obb", chair_center, chair_size, chair_rot, np.array([0, 255, 255], dtype=np.uint8))
    rr.save(str(out_rrd))


def export_rrd_2_all_feasible(
    scene_xyz: np.ndarray,
    scene_rgb: np.ndarray,
    local_render_pts: np.ndarray,
    candidates: list[PoseCandidate],
    chair_size: np.ndarray,
    out_rrd: Path,
) -> None:
    rr.init("Pose Search All Feasible", spawn=False)
    rr.log("scene/points", rr.Points3D(scene_xyz, colors=scene_rgb))
    for i, cand in enumerate(candidates):
        pts = transform_template(local_render_pts, cand.center, cand.rot_mat)
        rr.log(f"scene/feasible_chairs/chair_{i:04d}/points", rrd_points_with_alpha(pts, np.tile(np.array([[255, 235, 0]], dtype=np.uint8), (pts.shape[0], 1)), 90))
        log_obb_as_lines(
            f"scene/feasible_chairs/chair_{i:04d}/obb",
            cand.center,
            chair_size,
            cand.rot_mat,
            np.array([255, 220, 0], dtype=np.uint8),
        )
    rr.save(str(out_rrd))


def export_rrd_3_best_only(
    scene_xyz: np.ndarray,
    scene_rgb: np.ndarray,
    local_render_pts: np.ndarray,
    best: PoseCandidate | None,
    chair_size: np.ndarray,
    out_rrd: Path,
) -> None:
    rr.init("Pose Search Best", spawn=False)
    rr.log("scene/points", rr.Points3D(scene_xyz, colors=scene_rgb))
    if best is not None:
        pts = transform_template(local_render_pts, best.center, best.rot_mat)
        rr.log("scene/best_chair/points", rrd_points_with_alpha(pts, np.tile(np.array([[70, 255, 120]], dtype=np.uint8), (pts.shape[0], 1)), 100))
        log_obb_as_lines("scene/best_chair/obb", best.center, chair_size, best.rot_mat, np.array([40, 255, 100], dtype=np.uint8))
    rr.save(str(out_rrd))


def write_markdown(md_path: Path, summary_name: str, rrd1_name: str, rrd2_name: str, rrd3_name: str) -> None:
    content = f"""# Pose Search RRD Guide

## File 1: {rrd1_name}
- Content: Full scene point cloud + full 2D search grid + original chair OBB.
- Meaning: Shows the complete candidate sampling manifold before filtering.

## File 2: {rrd2_name}
- Content: Full scene + all feasible candidate chair poses after filtering.
- Rendering: each feasible chair is shown as semi-transparent yellow point cloud with its OBB.
- Meaning: Visualizes feasible free-space anchors that satisfy support, collision, and rule constraints.

## File 3: {rrd3_name}
- Content: Full scene + only the final selected compliant chair pose.
- Rendering: selected chair is semi-transparent green with OBB.
- Meaning: Shows the minimum-move feasible solution from the original violating pose.

## Supplementary Summary
- See `{summary_name}` for detailed hyperparameters, candidate counts at each stage, and final selected pose metadata.
"""
    md_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline 2D chair pose search with multi-constraint filtering and RRD checkpoints")
    # 输入与输出路径：改动这些参数只会改变读写位置，不影响搜索逻辑。
    parser.add_argument("--scene_ply", default="data/SampleLabel/pointclouds/SamplePointMapWithRightPose_scaled_x2.75.ply")
    parser.add_argument("--label_json", default="outputs/sample_label_demo/SamplePointMapWithRightPose_labels_scaled_x2.75.json")
    parser.add_argument("--chair_template_ply", default="outputs/sample_label_demo/SamplePointMapWithRightPose_scaled_x2.75_chair_only_binary_le.ply")
    parser.add_argument("--out_dir", default="outputs/sample_label_demo/pose_search_experiment")

    # 网格搜索参数：控制候选点覆盖范围、稠密程度和可视化清晰度。
    # grid_k 越大，网格步长越大，候选更少、更快，但更容易漏掉可行位姿。
    parser.add_argument("--grid_k", type=float, default=0.2, help="grid step = grid_k * chair bottom diagonal")
    # grid_step_min 越大，网格不会过密；越小则更细，但计算量显著上升。
    parser.add_argument("--grid_step_min", type=float, default=0.10)
    # full 会覆盖整个场景外接范围；percentile 会裁掉极端边缘，更聚焦但可能少候选。
    parser.add_argument("--grid_extent_mode", choices=["full", "percentile"], default="full")
    # 只在 percentile 模式下生效。p_low 越大、p_high 越小，网格范围越窄。
    parser.add_argument("--grid_p_low", type=float, default=1.0)
    parser.add_argument("--grid_p_high", type=float, default=99.0)
    # grid_margin 越大，网格范围在四周留白越多，便于容错但会增加搜索面积。
    parser.add_argument("--grid_margin", type=float, default=0.0)
    # grid_point_radius 越大，RRD 里网格点越醒目，但画面更容易变乱。
    parser.add_argument("--grid_point_radius", type=float, default=0.008)
    # grid_post_height 越大，网格竖柱越高，更容易截图观察位置，但会遮挡更多场景。
    parser.add_argument("--grid_post_height", type=float, default=0.10)
    # yaw_num 越大，朝向采样越细，能找到更接近原始朝向的解，但运行更慢。
    parser.add_argument("--yaw_num", type=int, default=16)

    # 地面与支撑参数：控制“椅子是否站稳”的筛选方式。
    # floor_percentile 越大，估计地面越靠上；越小则越接近真实最低点但更怕噪声。
    parser.add_argument("--floor_percentile", type=float, default=1.0)
    # floor_band 越大，地面带更宽，能容纳更多低矮支撑点，但也更容易把杂点算进去。
    parser.add_argument("--floor_band", type=float, default=0.0)
    # support_probe_depth 越大，向下探测的厚度越大，支撑判定更宽松，但更可能混入非支撑点。
    parser.add_argument("--support_probe_depth", type=float, default=5)
    # support_probe_start_height_fraction 越大，探测窗口起点越高，更能避开贴底穿模；过大则可能错过真正支撑。
    parser.add_argument("--support_probe_start_height_fraction", type=float, default=0.3)
    # support_count_ratio_threshold 越大，支撑门槛越严格；越小则更容易保留候选。
    parser.add_argument("--support_count_ratio_threshold", type=float, default=0.1)
    # footprint_expand 越大，支撑底面矩形在 x/y 方向外扩越多，判定更宽松但更容易把侧向点算作支撑。
    parser.add_argument("--footprint_expand", type=float, default=0.0)
    # 打开后每个 yaw 单独做支撑判定（更严格，且支撑通过数会受 yaw 影响）。
    parser.add_argument("--support_check_per_yaw", action="store_true")

    # 碰撞参数：控制椅子和环境点云的相交判定。
    # collision_inflate 越大，椅子包围盒越膨胀，碰撞判断越保守；越小则更宽松。
    parser.add_argument("--collision_inflate", type=float, default=0.03)
    # collision_max_points 越小，越严格；只要碰撞点稍多就会被拒绝。
    parser.add_argument("--collision_max_points", type=int, default=1000)
    # 对全部候选统一忽略椅子底部这部分比例的碰撞；数值越大，候选越容易通过碰撞筛选。
    parser.add_argument("--collision_ignore_bottom_fraction_when_x_negative", type=float, default=0.15)
    # 环境碰撞采样上限越大，判定越稳定，但速度更慢。
    parser.add_argument("--env_collision_sample", type=int, default=160000)

    # 规则区域参数：控制灭火器前方禁入圆柱的大小和采样精度。
    # rule_depth 越大，禁入区沿前向延伸越远，约束越强。
    parser.add_argument("--rule_depth", type=float, default=1.0)
    # rule_radius 越大，禁入区横向越宽，越不容易把椅子放到灭火器前面。
    parser.add_argument("--rule_radius", type=float, default=0.8)
    # rule_template_sample 越大，规则检测用的椅子点越多，判断更稳，但速度更慢。
    parser.add_argument("--rule_template_sample", type=int, default=2500)

    # RRD 渲染采样：控制可视化里椅子点云的显示密度。
    # render_template_sample 越大，候选椅子越完整，但 RRD 体积更大、渲染更慢。
    parser.add_argument("--render_template_sample", type=int, default=3500)
    args = parser.parse_args()

    scene_ply = Path(args.scene_ply)
    label_json = Path(args.label_json)
    chair_template_ply = Path(args.chair_template_ply)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_xyz, scene_rgb = load_binary_ply_xyzrgba(scene_ply)
    objects = parse_objects(label_json)
    chair_obj = next((o for o in objects if str(o.get("name", "")).lower() == "chair"), None)
    fire_obj = next((o for o in objects if "fire" in str(o.get("name", "")).lower()), None)
    if chair_obj is None:
        raise ValueError("Chair object not found in label JSON")
    if fire_obj is None:
        raise ValueError("Fire extinguisher object not found in label JSON")

    chair_center0, chair_size, chair_rot0, chair_yaw0 = object_to_obb(chair_obj)
    fire_center, _fire_size, fire_rot, _ = object_to_obb(fire_obj)

    # Chair template points for candidate rendering.
    if chair_template_ply.exists():
        chair_xyz_raw, _ = load_binary_ply_xyzrgba(chair_template_ply)
    else:
        chair_mask_scene = points_in_obb(scene_xyz, chair_center0, chair_size, chair_rot0)
        chair_xyz_raw = scene_xyz[chair_mask_scene]
    if chair_xyz_raw.shape[0] < 100:
        raise ValueError("Chair template has too few points")

    chair_local = (chair_xyz_raw - chair_center0[None, :]) @ chair_rot0
    render_idx = sample_indices(chair_local.shape[0], int(args.render_template_sample))
    rule_idx = sample_indices(chair_local.shape[0], int(args.rule_template_sample))
    chair_local_render = chair_local[render_idx]
    chair_local_rule = chair_local[rule_idx]

    # Rule forbidden zone from fire extinguisher.
    fire_forward = choose_forward_normal_from_plus_y(fire_rot)
    rule_cylinder_base = fire_center.copy()

    # Scene precomputation for filtering.
    chair_mask_scene = points_in_obb(scene_xyz, chair_center0, chair_size, chair_rot0)
    non_chair_scene = scene_xyz[~chair_mask_scene]
    # Exclude likely floor/support slab from collision candidates.
    floor_z = infer_floor_height(scene_xyz, q=float(args.floor_percentile))
    floor_band = float(args.floor_band)
    non_floor_scene = non_chair_scene[np.abs(non_chair_scene[:, 2] - floor_z) > (1.5 * floor_band)]
    if non_floor_scene.shape[0] == 0:
        non_floor_scene = non_chair_scene
    env_idx = sample_indices(non_floor_scene.shape[0], int(args.env_collision_sample))
    env_collision = non_floor_scene[env_idx]

    support_tree = cKDTree(non_chair_scene)
    support_rot_ref = R.from_euler("xyz", [0.0, 0.0, float(chair_yaw0)], degrees=True).as_matrix().astype(np.float32)

    ref_support_count = support_probe_count_for_grid_point(
        support_points=non_chair_scene,
        support_tree=support_tree,
        center=chair_center0,
        support_rot_mat=support_rot_ref,
        size=chair_size,
        footprint_expand=float(args.footprint_expand),
        probe_start_height_fraction=float(args.support_probe_start_height_fraction),
        probe_depth=float(args.support_probe_depth),
    )
    support_count_threshold = float(ref_support_count * float(args.support_count_ratio_threshold))

    # Search grid and yaw list.
    bottom_diag = float(np.linalg.norm(chair_size[:2]))
    grid_step = max(float(args.grid_step_min), float(args.grid_k) * bottom_diag)
    xs, ys, grid_bounds = build_grid(
        scene_xyz,
        step=grid_step,
        extent_mode=str(args.grid_extent_mode),
        p_low=float(args.grid_p_low),
        p_high=float(args.grid_p_high),
        margin=float(args.grid_margin),
    )
    grid_xy = np.array([[x, y] for x in xs for y in ys], dtype=np.float32)
    yaws = np.linspace(-180.0, 180.0, int(args.yaw_num), endpoint=False)

    # Original bottom-center reference.
    bottom_center0 = chair_center0.copy()
    bottom_center0[2] = float(chair_center0[2] - chair_size[2] / 2.0)

    total_candidates = int(grid_xy.shape[0] * yaws.shape[0])
    support_grid_pass = 0
    support_pass = 0
    collision_pass = 0
    rule_pass = 0
    feasible_raw: list[PoseCandidate] = []
    x_axis0 = chair_rot0[:, 0].astype(np.float32)

    side_total = {"positive": 0, "negative": 0, "near_zero": 0}
    side_support = {"positive": 0, "negative": 0, "near_zero": 0}
    side_collision = {"positive": 0, "negative": 0, "near_zero": 0}
    side_rule = {"positive": 0, "negative": 0, "near_zero": 0}

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            center = np.array([x, y, chair_center0[2]], dtype=np.float32)
            support_count_grid = support_probe_count_for_grid_point(
                support_points=non_chair_scene,
                support_tree=support_tree,
                center=center,
                support_rot_mat=support_rot_ref,
                size=chair_size,
                footprint_expand=float(args.footprint_expand),
                probe_start_height_fraction=float(args.support_probe_start_height_fraction),
                probe_depth=float(args.support_probe_depth),
            )
            grid_supported = float(support_count_grid) >= support_count_threshold
            if not bool(args.support_check_per_yaw) and grid_supported:
                support_grid_pass += 1

            grid_has_any_supported_yaw = False

            for yaw in yaws:
                proj = float(np.dot((center - chair_center0), x_axis0))
                if proj > 1e-6:
                    bucket = "positive"
                elif proj < -1e-6:
                    bucket = "negative"
                else:
                    bucket = "near_zero"
                side_total[bucket] += 1

                rot = R.from_euler("xyz", [0.0, 0.0, float(yaw)], degrees=True).as_matrix().astype(np.float32)
                if bool(args.support_check_per_yaw):
                    support_count = support_probe_count_for_grid_point(
                        support_points=non_chair_scene,
                        support_tree=support_tree,
                        center=center,
                        support_rot_mat=rot,
                        size=chair_size,
                        footprint_expand=float(args.footprint_expand),
                        probe_start_height_fraction=float(args.support_probe_start_height_fraction),
                        probe_depth=float(args.support_probe_depth),
                    )
                    if float(support_count) < support_count_threshold:
                        continue
                    grid_has_any_supported_yaw = True
                else:
                    if not grid_supported:
                        continue
                    support_count = support_count_grid
                    grid_has_any_supported_yaw = True

                support_pass += 1
                side_support[bucket] += 1

                c_points = collision_points_for_pose(
                    env_points=env_collision,
                    center=center,
                    rot_mat=rot,
                    size=chair_size,
                    inflate=float(args.collision_inflate),
                    ignore_bottom_fraction=float(args.collision_ignore_bottom_fraction_when_x_negative),
                )
                if c_points > int(args.collision_max_points):
                    continue
                collision_pass += 1
                side_collision[bucket] += 1

                chair_pts_rule = transform_template(chair_local_rule, center, rot)
                rule_hit = points_in_forward_cylinder(
                    chair_pts_rule,
                    cylinder_base=rule_cylinder_base,
                    axis_unit=fire_forward,
                    depth_limit=float(args.rule_depth),
                    radius_limit=float(args.rule_radius),
                )
                rule_hits = int(np.sum(rule_hit))
                if rule_hits > 0:
                    continue
                rule_pass += 1
                side_rule[bucket] += 1

                bottom_center = center.copy()
                bottom_center[2] = float(center[2] - chair_size[2] / 2.0)
                dist_cost = float(np.linalg.norm(bottom_center - bottom_center0))

                feasible_raw.append(
                    PoseCandidate(
                        idx_xy=(ix, iy),
                        center=center,
                        rot_mat=rot,
                        yaw_deg=float(yaw),
                        yaw_delta_deg=angular_diff_deg(float(yaw), float(chair_yaw0)),
                        support_score=support_count,
                        collision_points=c_points,
                        rule_hit_points=rule_hits,
                        distance_cost=dist_cost,
                    )
                )

            if bool(args.support_check_per_yaw) and grid_has_any_supported_yaw:
                support_grid_pass += 1

    # Dedup: same x-y keeps only the yaw closest to original yaw.
    dedup_map: dict[tuple[int, int], PoseCandidate] = {}
    for cand in feasible_raw:
        key = cand.idx_xy
        if key not in dedup_map:
            dedup_map[key] = cand
            continue
        prev = dedup_map[key]
        if cand.yaw_delta_deg < prev.yaw_delta_deg:
            dedup_map[key] = cand
        elif cand.yaw_delta_deg == prev.yaw_delta_deg and cand.distance_cost < prev.distance_cost:
            dedup_map[key] = cand

    feasible = list(dedup_map.values())
    feasible.sort(key=lambda c: (c.distance_cost, c.yaw_delta_deg))
    best = feasible[0] if feasible else None

    def side_counts(cands: list[PoseCandidate]) -> dict[str, int]:
        pos = 0
        neg = 0
        zero = 0
        for c in cands:
            proj = float(np.dot((c.center - chair_center0), x_axis0))
            if proj > 1e-6:
                pos += 1
            elif proj < -1e-6:
                neg += 1
            else:
                zero += 1
        return {"positive": pos, "negative": neg, "near_zero": zero}

    side_stats_raw = side_counts(feasible_raw)
    side_stats_dedup = side_counts(feasible)

    # Export RRD 1: scene + full search grid.
    rrd1 = out_dir / "pose_search_grid_full.rrd"
    export_rrd_1_grid(
        scene_xyz=scene_xyz,
        scene_rgb=scene_rgb,
        grid_xy=grid_xy,
        floor_z=floor_z,
        grid_point_radius=float(args.grid_point_radius),
        grid_post_height=float(args.grid_post_height),
        chair_center=chair_center0,
        chair_size=chair_size,
        chair_rot=chair_rot0,
        out_rrd=rrd1,
    )

    # Export RRD 2: all feasible candidates (deduped by x-y).
    rrd2 = out_dir / "pose_search_all_feasible.rrd"
    export_rrd_2_all_feasible(
        scene_xyz=scene_xyz,
        scene_rgb=scene_rgb,
        local_render_pts=chair_local_render,
        candidates=feasible,
        chair_size=chair_size,
        out_rrd=rrd2,
    )

    # Export RRD 3: only best candidate.
    rrd3 = out_dir / "pose_search_best_only.rrd"
    export_rrd_3_best_only(
        scene_xyz=scene_xyz,
        scene_rgb=scene_rgb,
        local_render_pts=chair_local_render,
        best=best,
        chair_size=chair_size,
        out_rrd=rrd3,
    )

    summary = {
        "inputs": {
            "scene_ply": str(scene_ply),
            "label_json": str(label_json),
            "chair_template_ply": str(chair_template_ply),
        },
        "hyperparams": {
            "grid_k": float(args.grid_k),
            "grid_step_min": float(args.grid_step_min),
            "grid_step_final": float(grid_step),
            "grid_extent_mode": str(args.grid_extent_mode),
            "grid_p_low": float(args.grid_p_low),
            "grid_p_high": float(args.grid_p_high),
            "grid_margin": float(args.grid_margin),
            "grid_point_radius": float(args.grid_point_radius),
            "grid_post_height": float(args.grid_post_height),
            "yaw_num": int(args.yaw_num),
            "floor_percentile": float(args.floor_percentile),
            "floor_band": float(args.floor_band),
            "support_probe_depth": float(args.support_probe_depth),
            "support_probe_start_height_fraction": float(args.support_probe_start_height_fraction),
            "support_count_ratio_threshold": float(args.support_count_ratio_threshold),
            "footprint_expand": float(args.footprint_expand),
            "support_check_per_yaw": bool(args.support_check_per_yaw),
            "collision_inflate": float(args.collision_inflate),
            "collision_max_points": int(args.collision_max_points),
            "collision_ignore_bottom_fraction_when_x_negative": float(args.collision_ignore_bottom_fraction_when_x_negative),
            "rule_depth": float(args.rule_depth),
            "rule_radius": float(args.rule_radius),
        },
        "support_reference": {
            "reference_support_count": int(ref_support_count),
            "support_count_threshold": float(support_count_threshold),
        },
        "counts": {
            "grid_points": int(grid_xy.shape[0]),
            "total_candidates": int(total_candidates),
            "support_grid_pass": int(support_grid_pass),
            "support_pass": int(support_pass),
            "collision_pass": int(collision_pass),
            "rule_pass": int(rule_pass),
            "feasible_before_dedup": int(len(feasible_raw)),
            "feasible_after_dedup": int(len(feasible)),
        },
        "grid_diagnostics": grid_bounds,
        "directional_diagnostics": {
            "reference_axis": "chair_original_local_x",
            "all_candidates": side_total,
            "support_pass": side_support,
            "collision_pass": side_collision,
            "rule_pass": side_rule,
            "feasible_before_dedup": side_stats_raw,
            "feasible_after_dedup": side_stats_dedup,
        },
        "best_pose": None,
        "outputs": {
            "rrd_grid": str(rrd1),
            "rrd_all_feasible": str(rrd2),
            "rrd_best": str(rrd3),
        },
    }
    if best is not None:
        summary["best_pose"] = {
            "center": best.center.tolist(),
            "yaw_deg": float(best.yaw_deg),
            "yaw_delta_deg": float(best.yaw_delta_deg),
            "distance_cost": float(best.distance_cost),
        }

    summary_path = out_dir / "pose_search_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = out_dir / "rrd_guide.md"
    write_markdown(
        md_path=md_path,
        summary_name=summary_path.name,
        rrd1_name=rrd1.name,
        rrd2_name=rrd2.name,
        rrd3_name=rrd3.name,
    )

    print("完成")
    print(f"- 网格点数: {grid_xy.shape[0]}")
    print(f"- 总候选数: {total_candidates}")
    print(f"- 支撑参考计数: {ref_support_count}")
    print(f"- 支撑计数阈值: {support_count_threshold:.3f}")
    print(f"- 网格支撑通过: {support_grid_pass}")
    print(f"- 支撑通过: {support_pass}")
    print(f"- 碰撞通过: {collision_pass}")
    print(f"- 规则通过: {rule_pass}")
    print(f"- 去重前可行解: {len(feasible_raw)}")
    print(f"- 去重后可行解: {len(feasible)}")
    if best is None:
        print("- Best pose: None")
    else:
        print(f"- Best center: {best.center}")
        print(f"- Best yaw: {best.yaw_deg:.3f} deg (delta {best.yaw_delta_deg:.3f})")
        print(f"- Best distance: {best.distance_cost:.4f} m")
    print(f"- RRD1: {rrd1}")
    print(f"- RRD2: {rrd2}")
    print(f"- RRD3: {rrd3}")
    print(f"- Summary: {summary_path}")
    print(f"- Guide: {md_path}")
    print(f"- Grid extent mode: {args.grid_extent_mode}")
    print(f"- Grid bounds used (x0,x1,y0,y1): {grid_bounds['grid_xy_bounds_used']}")


if __name__ == "__main__":
    main()
