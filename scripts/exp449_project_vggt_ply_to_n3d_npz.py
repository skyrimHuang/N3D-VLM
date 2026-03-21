from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


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
            ("x", "<f8"),
            ("y", "<f8"),
            ("z", "<f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )

    with ply_path.open("rb") as f:
        f.seek(data_start)
        verts = np.fromfile(f, dtype=dtype, count=vertex_count)

    xyz_world = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
    rgb = np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.float32) / 255.0
    return xyz_world, rgb


def project_to_image(
    xyz_world: np.ndarray,
    rgb_world: np.ndarray,
    w2c_3x4: np.ndarray,
    k_3x3: np.ndarray,
    out_w: int,
    out_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = xyz_world.shape[0]
    ones = np.ones((n, 1), dtype=np.float32)
    world_h = np.concatenate([xyz_world, ones], axis=1)

    cam = (w2c_3x4 @ world_h.T).T.astype(np.float32)
    x_cam = cam[:, 0]
    y_cam = cam[:, 1]
    z_cam = cam[:, 2]

    valid_z = z_cam > 1e-6
    x_n = x_cam[valid_z] / z_cam[valid_z]
    y_n = y_cam[valid_z] / z_cam[valid_z]

    fx = k_3x3[0, 0]
    fy = k_3x3[1, 1]
    cx = k_3x3[0, 2]
    cy = k_3x3[1, 2]

    u = fx * x_n + cx
    v = fy * y_n + cy

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    in_bounds = (ui >= 0) & (ui < out_w) & (vi >= 0) & (vi < out_h)

    idx_valid = np.where(valid_z)[0][in_bounds]
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    depths = z_cam[idx_valid]

    depth_map = np.full((out_h, out_w), np.inf, dtype=np.float32)
    xyz_map = np.zeros((out_h, out_w, 3), dtype=np.float32)
    rgb_map = np.zeros((out_h, out_w, 3), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=np.float32)

    order = np.argsort(depths)
    for idx in order:
        u_px = ui[idx]
        v_px = vi[idx]
        d = depths[idx]
        if d < depth_map[v_px, u_px]:
            depth_map[v_px, u_px] = d
            world_idx = idx_valid[idx]
            x_left = -x_cam[world_idx]
            y_up = -y_cam[world_idx]
            z_fwd = z_cam[world_idx]
            xyz_map[v_px, u_px, 0] = x_left
            xyz_map[v_px, u_px, 1] = y_up
            xyz_map[v_px, u_px, 2] = z_fwd
            rgb_map[v_px, u_px] = rgb_world[world_idx]
            mask[v_px, u_px] = 1.0

    return xyz_map, rgb_map, mask


def choose_best_frame(
    xyz_world: np.ndarray,
    intr_all: np.ndarray,
    w2c_all: np.ndarray,
    input_w: int,
    input_h: int,
    out_w: int,
    out_h: int,
    stride: int = 10,
) -> int:
    best_idx = 0
    best_score = -1
    sample_idx = np.arange(0, len(intr_all), stride)

    xyz_h = np.concatenate([xyz_world, np.ones((xyz_world.shape[0], 1), dtype=np.float32)], axis=1)
    for i in sample_idx:
        w2c = w2c_all[i]
        k = intr_all[i].copy()
        sx = out_w / float(input_w)
        sy = out_h / float(input_h)
        k[0, 0] *= sx
        k[0, 2] *= sx
        k[1, 1] *= sy
        k[1, 2] *= sy

        cam = (w2c @ xyz_h.T).T
        z = cam[:, 2]
        valid = z > 1e-6
        if not np.any(valid):
            continue

        x = cam[valid, 0] / z[valid]
        y = cam[valid, 1] / z[valid]
        u = k[0, 0] * x + k[0, 2]
        v = k[1, 1] * y + k[1, 2]
        in_bounds = (u >= 0) & (u < out_w) & (v >= 0) & (v < out_h)
        score = int(np.sum(in_bounds))

        if score > best_score:
            best_score = score
            best_idx = int(i)

    return best_idx


def main() -> None:
    root = Path("/home/hba/Documents/N3D-VLM/data/消防违规示例_original_vggt_dense_with_camera")
    cam_dir = root / "camera_params"
    ply_path = root / "reconstructed_points_original_vggt_dense.ply"

    out_npz = Path("/home/hba/Documents/N3D-VLM/data/fire_violation_projected_n3d.npz")
    out_img = Path("/home/hba/Documents/N3D-VLM/data/fire_violation_projected_n3d.jpg")
    out_meta = Path("/home/hba/Documents/N3D-VLM/data/fire_violation_projected_n3d_meta.json")

    xyz_world, rgb_world = load_binary_ply_xyzrgb(ply_path)

    intr_input = np.load(cam_dir / "intrinsics_input_res.npy").astype(np.float32)
    w2c = np.load(cam_dir / "extrinsics_w2c.npy").astype(np.float32)

    input_w, input_h = 518, 294
    out_w = 640
    out_h = int(round(input_h * out_w / input_w))

    frame_idx = choose_best_frame(
        xyz_world=xyz_world,
        intr_all=intr_input,
        w2c_all=w2c,
        input_w=input_w,
        input_h=input_h,
        out_w=out_w,
        out_h=out_h,
        stride=10,
    )

    k = intr_input[frame_idx].copy()
    sx = out_w / float(input_w)
    sy = out_h / float(input_h)
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy

    xyz_map, rgb_map, mask = project_to_image(
        xyz_world=xyz_world,
        rgb_world=rgb_world,
        w2c_3x4=w2c[frame_idx],
        k_3x3=k,
        out_w=out_w,
        out_h=out_h,
    )

    pcd = np.concatenate([xyz_map, mask[..., None]], axis=-1).astype(np.float16)
    np.savez_compressed(out_npz, pcd=pcd, intr=k.astype(np.float32), mask=mask.astype(np.float32), rgb=rgb_map.astype(np.float16))

    rgb_img = np.clip(rgb_map * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_img).save(out_img)

    meta = {
        "source_ply": str(ply_path),
        "camera_dir": str(cam_dir),
        "selected_frame_idx": frame_idx,
        "output_resolution": [out_w, out_h],
        "valid_ratio": float(mask.mean()),
        "n_valid": int(mask.sum()),
        "n_pixels": int(mask.size),
    }
    with out_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Saved:", out_npz)
    print("Saved:", out_img)
    print("Saved:", out_meta)
    print("Frame:", frame_idx, "valid ratio:", meta["valid_ratio"])


if __name__ == "__main__":
    main()
