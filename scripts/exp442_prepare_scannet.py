from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import Box3D
from src.geometry import fit_pca_obb


def parse_scene_meta(meta_path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for line in meta_path.read_text().splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        meta[k.strip()] = v.strip()
    return meta


def parse_axis_alignment(meta: Dict[str, str]) -> np.ndarray:
    raw = meta.get("axisAlignment", None)
    if raw is None:
        return np.eye(4, dtype=np.float64)
    values = [float(v) for v in raw.split()]
    if len(values) != 16:
        return np.eye(4, dtype=np.float64)
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def read_intrinsic_matrix(path: Path) -> np.ndarray:
    mat = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        mat.append([float(x) for x in line.strip().split()])
    matrix = np.asarray(mat, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Invalid intrinsic matrix shape {matrix.shape} from {path}")
    return matrix[:3, :3]


def read_scannet_mesh_vertices(ply_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with ply_path.open("rb") as f:
        vertex_count = None
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"Cannot parse vertex count from {ply_path}")
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
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)

    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float64)
    rgb = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def apply_axis_alignment(points_xyz: np.ndarray, axis_alignment: np.ndarray) -> np.ndarray:
    homo = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=np.float64)], axis=1)
    aligned = (axis_alignment @ homo.T).T[:, :3]
    return aligned


def build_seg_to_vertices(seg_indices: List[int]) -> Dict[int, np.ndarray]:
    seg_to_verts: Dict[int, List[int]] = {}
    for vid, seg_id in enumerate(seg_indices):
        seg_to_verts.setdefault(seg_id, []).append(vid)
    return {k: np.asarray(v, dtype=np.int64) for k, v in seg_to_verts.items()}


def compute_gt_boxes(
    points_xyz: np.ndarray,
    aggregation: dict,
    seg_to_verts: Dict[int, np.ndarray],
    min_points_per_object: int,
) -> List[dict]:
    gt_items: List[dict] = []
    for seg_group in aggregation.get("segGroups", []):
        obj_id = str(seg_group.get("objectId", seg_group.get("id", "")))
        label = str(seg_group.get("label", "unknown"))
        segs = seg_group.get("segments", [])
        vert_ids: List[np.ndarray] = [seg_to_verts[s] for s in segs if s in seg_to_verts]
        if len(vert_ids) == 0:
            continue
        indices = np.unique(np.concatenate(vert_ids, axis=0))
        if indices.size < min_points_per_object:
            continue
        obj_points = points_xyz[indices]

        min_xyz = obj_points.min(axis=0)
        max_xyz = obj_points.max(axis=0)
        aabb = Box3D(
            center=(min_xyz + max_xyz) / 2.0,
            size=np.maximum(max_xyz - min_xyz, 1e-3),
            yaw=0.0,
            label=label,
            instance_id=obj_id,
        )

        obb = fit_pca_obb(obj_points).to_box3d(label=label, instance_id=obj_id)

        gt_items.append(
            {
                "instance_id": obj_id,
                "label": label,
                "num_points": int(indices.size),
                "aabb": aabb.to_dict(),
                "obb": obb.to_dict(),
            }
        )
    return gt_items


def depth_to_camera_points(depth_m: np.ndarray, intr: np.ndarray) -> np.ndarray:
    h, w = depth_m.shape
    ys, xs = np.indices((h, w), dtype=np.float32)
    z = depth_m
    fx, fy = intr[0, 0], intr[1, 1]
    cx, cy = intr[0, 2], intr[1, 2]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([-x, -y, z], axis=-1)
    return points


def export_frame_npz(scene_dir: Path, output_scene_dir: Path, frame_stride: int, max_frames: int) -> List[dict]:
    try:
        import cv2  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenCV is required for --export_frame_npz. Please install opencv-python-headless."
        ) from exc

    color_dir = scene_dir / "color"
    depth_dir = scene_dir / "depth"
    pose_dir = scene_dir / "pose"
    intr = read_intrinsic_matrix(scene_dir / "intrinsic" / "intrinsic_depth.txt")

    frame_ids = sorted(int(p.stem) for p in depth_dir.glob("*.png"))
    if frame_stride > 1:
        frame_ids = frame_ids[::frame_stride]
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]

    exports: List[dict] = []
    for fid in tqdm(frame_ids, desc=f"Frames {scene_dir.name}", leave=False):
        depth_path = depth_dir / f"{fid}.png"
        color_path = color_dir / f"{fid}.jpg"
        pose_path = pose_dir / f"{fid}.txt"
        if not depth_path.exists() or not color_path.exists() or not pose_path.exists():
            continue

        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            continue
        depth_m = depth_raw.astype(np.float32) / 1000.0
        mask = depth_m > 0
        pcd = np.concatenate([depth_to_camera_points(depth_m, intr), mask[..., None]], axis=-1).astype(np.float16)

        save_dir = output_scene_dir / "frames"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{fid:06d}.npz"
        np.savez_compressed(
            save_path,
            pcd=pcd,
            depth=depth_m.astype(np.float32),
            mask=mask,
            intr=intr.astype(np.float32),
        )

        exports.append(
            {
                "frame_id": fid,
                "image_path": str(color_path),
                "depth_path": str(depth_path),
                "pose_path": str(pose_path),
                "point_path": str(save_path),
            }
        )
    return exports


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ScanNet real-point-cloud data for exp 4.4.2")
    parser.add_argument("--scans_root", type=str, default="/home/hba/Documents/Dataset/ScanNet/scans")
    parser.add_argument("--output_root", type=str, default="outputs/exp442_scannet")
    parser.add_argument("--scene_prefix", type=str, default="scene")
    parser.add_argument("--max_scenes", type=int, default=0)
    parser.add_argument("--frame_stride", type=int, default=60)
    parser.add_argument("--max_frames_per_scene", type=int, default=20)
    parser.add_argument("--min_points_per_object", type=int, default=80)
    parser.add_argument("--export_frame_npz", action="store_true")
    args = parser.parse_args()

    scans_root = Path(args.scans_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    scenes = sorted([p for p in scans_root.iterdir() if p.is_dir() and p.name.startswith(args.scene_prefix)])
    if args.max_scenes > 0:
        scenes = scenes[: args.max_scenes]

    print(f"[Stage] Preparing ScanNet scenes from: {scans_root}")
    manifest: List[dict] = []
    for scene_dir in tqdm(scenes, desc="Preparing ScanNet scenes"):
        scene_id = scene_dir.name
        mesh_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
        seg_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
        if not seg_path.exists():
            seg_path = scene_dir / f"{scene_id}_vh_clean.segs.json"
        aggr_path = scene_dir / f"{scene_id}.aggregation.json"
        meta_path = scene_dir / f"{scene_id}.txt"

        if not (mesh_path.exists() and seg_path.exists() and aggr_path.exists() and meta_path.exists()):
            continue

        meta = parse_scene_meta(meta_path)
        axis = parse_axis_alignment(meta)
        points_xyz, points_rgb = read_scannet_mesh_vertices(mesh_path)
        points_xyz = apply_axis_alignment(points_xyz, axis)

        seg_data = json.loads(seg_path.read_text())
        aggregation = json.loads(aggr_path.read_text())
        seg_to_verts = build_seg_to_vertices(seg_data["segIndices"])
        gt_boxes = compute_gt_boxes(
            points_xyz=points_xyz,
            aggregation=aggregation,
            seg_to_verts=seg_to_verts,
            min_points_per_object=args.min_points_per_object,
        )

        scene_out = output_root / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)
        scene_point_path = scene_out / "scene_points.npz"
        np.savez_compressed(
            scene_point_path,
            points=points_xyz.astype(np.float32),
            colors=points_rgb.astype(np.uint8),
        )

        gt_path = scene_out / "gt_boxes.json"
        gt_payload = {"scene_id": scene_id, "num_objects": len(gt_boxes), "objects": gt_boxes}
        gt_path.write_text(json.dumps(gt_payload, indent=2, ensure_ascii=False))

        frame_entries: List[dict] = []
        if args.export_frame_npz:
            frame_entries = export_frame_npz(
                scene_dir=scene_dir,
                output_scene_dir=scene_out,
                frame_stride=args.frame_stride,
                max_frames=args.max_frames_per_scene,
            )

        manifest.append(
            {
                "scene_id": scene_id,
                "scene_point_cloud": str(scene_point_path),
                "gt_boxes": str(gt_path),
                "num_gt_objects": len(gt_boxes),
                "frames": frame_entries,
            }
        )

    print("[Stage] Writing manifest...")
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Saved manifest to: {manifest_path}")
    print(f"Prepared scenes: {len(manifest)}")


if __name__ == "__main__":
    main()