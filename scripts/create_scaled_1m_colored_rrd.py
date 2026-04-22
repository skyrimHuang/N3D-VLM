#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rerun as rr


def load_binary_ply_xyzrgb(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def find_fire_center(label_json: Path) -> np.ndarray:
    payload = json.loads(label_json.read_text(encoding="utf-8"))
    objects = payload.get("objects", [])
    for obj in objects:
        name = str(obj.get("name", "")).lower()
        if "fire" in name:
            return np.array(
                [
                    float(obj["centroid"]["x"]),
                    float(obj["centroid"]["y"]),
                    float(obj["centroid"]["z"]),
                ],
                dtype=np.float32,
            )
    raise ValueError("Cannot find fire-extinguisher-like object in labels")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create RRD: 1m inside yellow, outside green, from scaled point cloud")
    parser.add_argument(
        "--ply",
        default="data/SampleLabel/pointclouds/SamplePointMapWithRightPose_scaled_x2.75.ply",
        help="Scaled PLY file path",
    )
    parser.add_argument(
        "--label_json",
        default="outputs/sample_label_demo/SamplePointMapWithRightPose_labels_scaled_x2.75.json",
        help="Scaled label JSON path",
    )
    parser.add_argument("--radius", type=float, default=1.0, help="Sphere radius in meters")
    parser.add_argument(
        "--out_rrd",
        default="outputs/sample_label_demo/sample_scaled_x2.75_1m_inside_yellow_outside_green.rrd",
        help="Output RRD path",
    )
    args = parser.parse_args()

    ply_path = Path(args.ply)
    label_path = Path(args.label_json)
    out_rrd = Path(args.out_rrd)
    out_rrd.parent.mkdir(parents=True, exist_ok=True)

    points_xyz, _ = load_binary_ply_xyzrgb(ply_path)
    fire_center = find_fire_center(label_path)

    dists = np.linalg.norm(points_xyz - fire_center[None, :], axis=1)
    inside = dists <= float(args.radius)

    colors = np.empty((points_xyz.shape[0], 3), dtype=np.uint8)
    colors[inside] = np.array([255, 255, 0], dtype=np.uint8)  # yellow
    colors[~inside] = np.array([0, 255, 0], dtype=np.uint8)   # green

    rr.init("Scaled 1m Distance Coloring", spawn=False)
    rr.log("scene/points", rr.Points3D(points_xyz, colors=colors))
    rr.log(
        "scene/fire_center",
        rr.Points3D(
            fire_center[None, :],
            colors=np.array([[255, 0, 0]], dtype=np.uint8),
            labels=["fire_center"],
        ),
    )
    rr.save(str(out_rrd))

    print(f"Saved RRD: {out_rrd}")
    print(f"Total points: {points_xyz.shape[0]}")
    print(f"Inside {args.radius:.2f}m (yellow): {int(np.sum(inside))}")
    print(f"Outside {args.radius:.2f}m (green): {int(np.sum(~inside))}")


if __name__ == "__main__":
    main()
