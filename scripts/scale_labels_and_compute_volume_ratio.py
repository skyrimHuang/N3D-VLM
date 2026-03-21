#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def find_object(objects: list[dict[str, Any]], keyword: str) -> dict[str, Any]:
    keyword_lower = keyword.lower()
    for obj in objects:
        if keyword_lower in str(obj.get("name", "")).lower():
            return obj
    raise ValueError(f"Cannot find object with keyword: {keyword}")


def scale_label_payload(payload: dict[str, Any], scale: float, scaled_ply_path: str | None) -> dict[str, Any]:
    out = json.loads(json.dumps(payload, ensure_ascii=False))
    for obj in out.get("objects", []):
        obj["centroid"]["x"] = float(obj["centroid"]["x"]) * scale
        obj["centroid"]["y"] = float(obj["centroid"]["y"]) * scale
        obj["centroid"]["z"] = float(obj["centroid"]["z"]) * scale

        obj["dimensions"]["length"] = float(obj["dimensions"]["length"]) * scale
        obj["dimensions"]["width"] = float(obj["dimensions"]["width"]) * scale
        obj["dimensions"]["height"] = float(obj["dimensions"]["height"]) * scale

    if scaled_ply_path is not None:
        out["path"] = scaled_ply_path
        out["filename"] = Path(scaled_ply_path).name

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale labels uniformly and compute chair volume ratio inside 1m sphere around extinguisher")
    parser.add_argument(
        "--label_json",
        default="data/SampleLabel/SamplePointMapWithRightPose.json",
        help="Original label JSON path",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=2.75,
        help="Uniform xyz scale factor",
    )
    parser.add_argument(
        "--scaled_ply",
        default="data/SampleLabel/pointclouds/SamplePointMapWithRightPose_scaled_x2.75.ply",
        help="Scaled PLY path to be referenced in scaled labels",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Sphere radius around extinguisher center (m)",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/sample_label_demo",
        help="Output directory",
    )
    args = parser.parse_args()

    label_path = Path(args.label_json)
    payload = json.loads(label_path.read_text(encoding="utf-8"))

    scaled_payload = scale_label_payload(payload, args.scale, args.scaled_ply)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_scaled_label = out_dir / f"SamplePointMapWithRightPose_labels_scaled_x{args.scale:.2f}.json"
    out_scaled_label.write_text(json.dumps(scaled_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    objects = scaled_payload.get("objects", [])
    chair = find_object(objects, "chair")
    extinguisher = find_object(objects, "fire")

    chair_l = float(chair["dimensions"]["length"])
    chair_w = float(chair["dimensions"]["width"])
    chair_h = float(chair["dimensions"]["height"])
    chair_volume = chair_l * chair_w * chair_h

    sphere_volume = (4.0 / 3.0) * math.pi * (args.radius ** 3)
    ratio = chair_volume / sphere_volume

    ext_center = {
        "x": float(extinguisher["centroid"]["x"]),
        "y": float(extinguisher["centroid"]["y"]),
        "z": float(extinguisher["centroid"]["z"]),
    }

    result = {
        "scale_factor": args.scale,
        "radius_m": args.radius,
        "scaled_label_json": str(out_scaled_label),
        "scaled_ply": args.scaled_ply,
        "extinguisher_center_m": ext_center,
        "chair_dimensions_m": {
            "length": chair_l,
            "width": chair_w,
            "height": chair_h,
        },
        "chair_volume_m3": chair_volume,
        "sphere_volume_m3": sphere_volume,
        "chair_volume_ratio_of_1m_sphere": ratio,
        "chair_volume_ratio_percent": ratio * 100.0,
    }

    out_result = out_dir / "scaled_label_volume_ratio.json"
    out_result.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved scaled labels: {out_scaled_label}")
    print(f"Saved volume result: {out_result}")
    print(f"Chair volume (m^3): {chair_volume:.6f}")
    print(f"1m sphere volume (m^3): {sphere_volume:.6f}")
    print(f"Chair/Sphere ratio: {ratio:.6f} ({ratio*100.0:.3f}%)")


if __name__ == "__main__":
    main()
