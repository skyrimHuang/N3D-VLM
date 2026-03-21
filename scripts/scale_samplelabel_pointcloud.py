#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_binary_ply_xyzrgba(ply_path: Path) -> tuple[list[bytes], np.ndarray]:
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


def save_binary_ply(header_lines: list[bytes], verts: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for line in header_lines:
            f.write(line)
        verts.tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Uniformly scale SampleLabel point cloud to realistic metric size")
    parser.add_argument(
        "--label_json",
        default="data/SampleLabel/SamplePointMapWithRightPose.json",
        help="Path to SampleLabel JSON",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.75,
        help="Uniform XYZ scale factor",
    )
    parser.add_argument(
        "--out_ply",
        default="data/SampleLabel/pointclouds/SamplePointMapWithRightPose_scaled_x2.75.ply",
        help="Output PLY path",
    )
    args = parser.parse_args()

    label_path = Path(args.label_json)
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    ply_rel = payload.get("path") or payload.get("filename")
    if not ply_rel:
        raise ValueError("Cannot find point cloud path in label json")

    in_ply = (label_path.parent / ply_rel).resolve()
    if not in_ply.exists():
        in_ply = (label_path.parent / "pointclouds" / Path(ply_rel).name).resolve()
    if not in_ply.exists():
        raise FileNotFoundError(f"Input ply not found: {in_ply}")

    header_lines, verts = load_binary_ply_xyzrgba(in_ply)

    scale = float(args.scale)
    verts = verts.copy()
    verts["x"] *= scale
    verts["y"] *= scale
    verts["z"] *= scale

    out_ply = Path(args.out_ply)
    save_binary_ply(header_lines, verts, out_ply)

    pts = np.stack([verts["x"], verts["y"], verts["z"]], axis=1)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    print(f"Input PLY : {in_ply}")
    print(f"Output PLY: {out_ply}")
    print(f"Uniform scale factor: {scale:.4f}")
    print(f"Scaled scene extent (m): {(maxs - mins).tolist()}")

    # Report scaled labeled object dimensions for sanity check
    print("\nScaled object dimensions (m):")
    for obj in payload.get("objects", []):
        name = obj.get("name", "unknown")
        length = float(obj["dimensions"]["length"]) * scale
        width = float(obj["dimensions"]["width"]) * scale
        height = float(obj["dimensions"]["height"]) * scale
        print(f"- {name}: L={length:.3f}, W={width:.3f}, H={height:.3f}")


if __name__ == "__main__":
    main()
