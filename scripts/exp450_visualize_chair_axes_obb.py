#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D


def setup_chinese_font() -> str:
    font_candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for font_path in font_candidates:
        p = Path(font_path)
        if p.exists():
            font_manager.fontManager.addfont(str(p))

    matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Serif CJK JP", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return matplotlib.rcParams["font.sans-serif"][0]


def load_binary_ply_xyz(ply_path: Path) -> np.ndarray:
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

    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    return xyz


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Zero-length vector cannot be normalized")
    return v / n


def filter_bottom_z_percent(points_xyz: np.ndarray, trim_bottom_percent: float) -> tuple[np.ndarray, dict]:
    if trim_bottom_percent < 0.0 or trim_bottom_percent >= 100.0:
        raise ValueError("trim_bottom_percent must be in [0, 100)")

    if trim_bottom_percent == 0.0:
        return points_xyz, {
            "trim_bottom_percent": 0.0,
            "z_threshold": float(np.min(points_xyz[:, 2])),
            "total_points": int(points_xyz.shape[0]),
            "kept_points": int(points_xyz.shape[0]),
            "removed_points": 0,
        }

    z_threshold = float(np.percentile(points_xyz[:, 2], trim_bottom_percent))
    keep_mask = points_xyz[:, 2] > z_threshold
    filtered = points_xyz[keep_mask]
    if filtered.shape[0] < 3:
        raise ValueError("Too few points left after trimming; reduce trim_bottom_percent")

    return filtered, {
        "trim_bottom_percent": float(trim_bottom_percent),
        "z_threshold": z_threshold,
        "total_points": int(points_xyz.shape[0]),
        "kept_points": int(filtered.shape[0]),
        "removed_points": int(points_xyz.shape[0] - filtered.shape[0]),
    }


def compute_axes_and_obb(points_xyz: np.ndarray) -> dict:
    m = points_xyz.shape[0]
    if m < 3:
        raise ValueError("Point cloud must contain at least 3 points")

    centroid = np.mean(points_xyz, axis=0)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    pts_xy = points_xyz[:, :2]
    centroid_xy = np.mean(pts_xy, axis=0)
    centered_xy = pts_xy - centroid_xy
    cov_xy = (centered_xy.T @ centered_xy) / float(m)

    eigvals, eigvecs = np.linalg.eigh(cov_xy)
    front_xy = eigvecs[:, int(np.argmax(eigvals))]
    front = normalize(np.array([front_xy[0], front_xy[1], 0.0], dtype=np.float64))

    right = normalize(np.cross(up, front))
    front = normalize(np.cross(right, up))

    centered = points_xyz - centroid[None, :]
    proj_front = centered @ front
    proj_right = centered @ right
    proj_up = centered @ up

    min_front, max_front = float(np.min(proj_front)), float(np.max(proj_front))
    min_right, max_right = float(np.min(proj_right)), float(np.max(proj_right))
    min_up, max_up = float(np.min(proj_up)), float(np.max(proj_up))

    len_front = max_front - min_front
    len_right = max_right - min_right
    len_up = max_up - min_up

    local_center = np.array(
        [
            0.5 * (min_front + max_front),
            0.5 * (min_right + max_right),
            0.5 * (min_up + max_up),
        ],
        dtype=np.float64,
    )
    obb_center = centroid + local_center[0] * front + local_center[1] * right + local_center[2] * up

    local_corners = np.array(
        [
            [min_front, min_right, min_up],
            [max_front, min_right, min_up],
            [max_front, max_right, min_up],
            [min_front, max_right, min_up],
            [min_front, min_right, max_up],
            [max_front, min_right, max_up],
            [max_front, max_right, max_up],
            [min_front, max_right, max_up],
        ],
        dtype=np.float64,
    )

    basis = np.stack([front, right, up], axis=1)
    corners = centroid[None, :] + local_corners @ basis.T

    return {
        "centroid": centroid,
        "cov_xy": cov_xy,
        "u_front": front,
        "u_right": right,
        "u_up": up,
        "length_front": len_front,
        "length_right": len_right,
        "length_up": len_up,
        "obb_center": obb_center,
        "corners": corners,
        "z_min": float(np.min(points_xyz[:, 2])),
    }


def set_axes_equal(ax: plt.Axes, points_xyz: np.ndarray, corners: np.ndarray) -> None:
    all_pts = np.vstack([points_xyz, corners])
    x_min, y_min, z_min = np.min(all_pts, axis=0)
    x_max, y_max, z_max = np.max(all_pts, axis=0)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    cx = 0.5 * (x_max + x_min)
    cy = 0.5 * (y_max + y_min)
    cz = 0.5 * (z_max + z_min)
    half = 0.5 * max_range

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)


def draw_obb_edges(ax: plt.Axes, corners: np.ndarray, color: str = "tab:blue") -> None:
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        seg = np.vstack([corners[i], corners[j]])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=1.8)


def visualize(
    points_xyz: np.ndarray,
    geom: dict,
    out_figure: Path,
    axis_scale: float | None,
    point_size: float,
) -> None:
    setup_chinese_font()
    fig = plt.figure(figsize=(12, 10), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        points_xyz[:, 0],
        points_xyz[:, 1],
        points_xyz[:, 2],
        c="dimgray",
        s=point_size,
        alpha=0.60,
        depthshade=False,
    )

    corners = geom["corners"]
    draw_obb_edges(ax, corners, color="tab:blue")

    lengths = np.array([geom["length_front"], geom["length_right"], geom["length_up"]], dtype=np.float64)
    scale = float(axis_scale) if axis_scale is not None else float(np.max(lengths) * 0.7)
    origin = geom["obb_center"]

    vf = geom["u_front"] * scale
    vr = geom["u_right"] * scale
    vu = geom["u_up"] * scale

    ax.quiver(origin[0], origin[1], origin[2], vf[0], vf[1], vf[2], color="tab:red", linewidth=2.4, arrow_length_ratio=0.10)
    ax.quiver(origin[0], origin[1], origin[2], vr[0], vr[1], vr[2], color="tab:green", linewidth=2.4, arrow_length_ratio=0.10)
    ax.quiver(origin[0], origin[1], origin[2], vu[0], vu[1], vu[2], color="tab:orange", linewidth=2.4, arrow_length_ratio=0.10)

    z_min = geom["z_min"]
    x_min, y_min = np.min(points_xyz[:, 0]), np.min(points_xyz[:, 1])
    x_max, y_max = np.max(points_xyz[:, 0]), np.max(points_xyz[:, 1])
    gx = np.array([[x_min, x_max], [x_min, x_max]], dtype=np.float64)
    gy = np.array([[y_min, y_min], [y_max, y_max]], dtype=np.float64)
    gz = np.full_like(gx, z_min)
    ax.plot_surface(gx, gy, gz, color="lightblue", alpha=0.15, linewidth=0, shade=False)

    ax.set_title("椅子点云三轴与有向包围盒可视化", fontsize=15, fontweight="bold", pad=18)
    ax.set_xlabel("X 轴", fontsize=12)
    ax.set_ylabel("Y 轴", fontsize=12)
    ax.set_zlabel("Z 轴", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.35)

    set_axes_equal(ax, points_xyz, corners)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="dimgray", markersize=8, label="椅子点云"),
        Line2D([0], [0], color="tab:red", linewidth=2.4, label="前向轴 u_front"),
        Line2D([0], [0], color="tab:green", linewidth=2.4, label="右向轴 u_right"),
        Line2D([0], [0], color="tab:orange", linewidth=2.4, label="竖直轴 u_up"),
        Line2D([0], [0], color="tab:blue", linewidth=2.0, label="有向包围盒"),
        Line2D([0], [0], color="lightblue", linewidth=6.0, alpha=0.5, label="地面参考平面 (z_min)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95, fontsize=10)

    out_figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_figure, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_result_json(geom: dict, filter_meta: dict, out_json: Path) -> None:
    payload = {
        "filter": filter_meta,
        "u_front": geom["u_front"].tolist(),
        "u_right": geom["u_right"].tolist(),
        "u_up": geom["u_up"].tolist(),
        "length_front": float(geom["length_front"]),
        "length_right": float(geom["length_right"]),
        "length_up": float(geom["length_up"]),
        "centroid": geom["centroid"].tolist(),
        "obb_center": geom["obb_center"].tolist(),
        "corners": geom["corners"].tolist(),
        "cov_xy": geom["cov_xy"].tolist(),
        "z_min_ground": float(geom["z_min"]),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and visualize chair axes + oriented bounding box")
    parser.add_argument(
        "--ply",
        default="outputs/sample_label_demo/SamplePointMapWithRightPose_scaled_x2.75_chair_only_binary_le.ply",
        help="Input binary-little-endian PLY file",
    )
    parser.add_argument(
        "--out_figure",
        default="outputs/sample_label_demo/chair_axes_obb_visualization.png",
        help="Output figure path",
    )
    parser.add_argument(
        "--out_json",
        default="outputs/sample_label_demo/chair_axes_obb_result.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--axis_scale",
        type=float,
        default=None,
        help="Visualized axis vector length; default is 0.7 * max(box lengths)",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=1.0,
        help="Point size for 3D scatter",
    )
    parser.add_argument(
        "--trim_bottom_percent",
        type=float,
        default=0.0,
        help="Remove bottom Z percentile points before computing axes/OBB, e.g. 5 means remove lowest 5%%",
    )
    args = parser.parse_args()

    ply_path = Path(args.ply)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    points_xyz = load_binary_ply_xyz(ply_path)
    points_used, filter_meta = filter_bottom_z_percent(points_xyz, args.trim_bottom_percent)
    geom = compute_axes_and_obb(points_used)

    visualize(
        points_xyz=points_used,
        geom=geom,
        out_figure=Path(args.out_figure),
        axis_scale=args.axis_scale,
        point_size=args.point_size,
    )
    export_result_json(geom=geom, filter_meta=filter_meta, out_json=Path(args.out_json))

    print("完成：")
    print(f"- 输入点云: {ply_path}")
    print(
        f"- 底部裁剪: {filter_meta['trim_bottom_percent']}% "
        f"(阈值 z <= {filter_meta['z_threshold']:.6f} 被移除)"
    )
    print(
        f"- 点数变化: {filter_meta['total_points']} -> {filter_meta['kept_points']} "
        f"(移除 {filter_meta['removed_points']})"
    )
    print(f"- 前向轴 u_front: {geom['u_front']}")
    print(f"- 右向轴 u_right: {geom['u_right']}")
    print(f"- 竖直轴 u_up: {geom['u_up']}")
    print(
        f"- 包围盒尺寸 (front/right/up): "
        f"{geom['length_front']:.6f}, {geom['length_right']:.6f}, {geom['length_up']:.6f}"
    )
    print(f"- 可视化图: {args.out_figure}")
    print(f"- 结果JSON: {args.out_json}")


if __name__ == "__main__":
    main()