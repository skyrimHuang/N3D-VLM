#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def load_binary_ply_vertices(ply_path: Path) -> tuple[list[bytes], np.ndarray]:
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

    return header_lines, verts


def rewrite_header_vertex_count(header_lines: list[bytes], new_count: int) -> list[bytes]:
    updated: list[bytes] = []
    for line in header_lines:
        try:
            txt = line.decode("ascii")
        except Exception:
            updated.append(line)
            continue
        if txt.startswith("element vertex "):
            updated.append(f"element vertex {new_count}\n".encode("ascii"))
        else:
            updated.append(line)
    return updated


def save_binary_ply(header_lines: list[bytes], verts: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for line in header_lines:
            f.write(line)
        verts.tofile(f)


def load_chair_obb(label_json_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(label_json_path.read_text(encoding="utf-8"))
    objects = payload.get("objects", [])
    chair_obj = None
    for obj in objects:
        if str(obj.get("name", "")).lower() == "chair":
            chair_obj = obj
            break
    if chair_obj is None:
        raise ValueError("Cannot find Chair object in labels")

    center = np.array(
        [
            float(chair_obj["centroid"]["x"]),
            float(chair_obj["centroid"]["y"]),
            float(chair_obj["centroid"]["z"]),
        ],
        dtype=np.float32,
    )
    size = np.array(
        [
            float(chair_obj["dimensions"]["length"]),
            float(chair_obj["dimensions"]["width"]),
            float(chair_obj["dimensions"]["height"]),
        ],
        dtype=np.float32,
    )
    rot = R.from_euler(
        "xyz",
        [
            float(chair_obj["rotations"].get("x", 0.0)),
            float(chair_obj["rotations"].get("y", 0.0)),
            float(chair_obj["rotations"].get("z", 0.0)),
        ],
        degrees=True,
    ).as_matrix().astype(np.float32)
    return center, size, rot


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract chair points from scaled point cloud using scaled labels")
    parser.add_argument(
        "--ply",
        default="data/SampleLabel/pointclouds/SamplePointMapWithRightPose_scaled_x2.75.ply",
        help="Input scaled point cloud PLY",
    )
    parser.add_argument(
        "--label_json",
        default="outputs/sample_label_demo/SamplePointMapWithRightPose_labels_scaled_x2.75.json",
        help="Scaled labels JSON",
    )
    parser.add_argument(
        "--out_ply",
        default="outputs/sample_label_demo/SamplePointMapWithRightPose_scaled_x2.75_chair_only.ply",
        help="Output chair-only PLY",
    )
    args = parser.parse_args()

    in_ply = Path(args.ply)
    label_json = Path(args.label_json)
    out_ply = Path(args.out_ply)

    header_lines, verts = load_binary_ply_vertices(in_ply)
    center, size, rot = load_chair_obb(label_json)

    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
    local = (xyz - center[None, :]) @ rot
    half = size / 2.0
    inside = np.all(np.abs(local) <= half[None, :], axis=1)

    chair_verts = verts[inside]
    new_header = rewrite_header_vertex_count(header_lines, int(chair_verts.shape[0]))
    save_binary_ply(new_header, chair_verts, out_ply)

    print(f"Input PLY: {in_ply}")
    print(f"Label JSON: {label_json}")
    print(f"Output chair-only PLY: {out_ply}")
    print(f"Extracted points: {chair_verts.shape[0]}")


if __name__ == "__main__":
    main()
