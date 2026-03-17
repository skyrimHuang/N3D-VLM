from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.eval.box_metrics_3d import Box3D


EPS = 1e-9


@dataclass
class OBBResult:
    center: np.ndarray
    size: np.ndarray
    rotation: np.ndarray
    yaw: float

    def to_box3d(self, label: str = "", score: float = 1.0, instance_id: str = "") -> Box3D:
        return Box3D(
            center=self.center,
            size=np.maximum(self.size, EPS),
            yaw=float(self.yaw),
            label=label,
            score=score,
            instance_id=instance_id,
        )


def _ensure_right_handed(rotation: np.ndarray) -> np.ndarray:
    rot = rotation.copy()
    if np.linalg.det(rot) < 0:
        rot[:, -1] *= -1.0
    return rot


def fit_pca_obb(points_xyz: np.ndarray, up_axis: int = 2) -> OBBResult:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be shape (N, 3)")
    if points_xyz.shape[0] < 3:
        center = points_xyz.mean(axis=0)
        size = np.maximum(points_xyz.max(axis=0) - points_xyz.min(axis=0), 1e-3)
        rotation = np.eye(3, dtype=np.float64)
        yaw = 0.0
        return OBBResult(center=center, size=size, rotation=rotation, yaw=yaw)

    center = points_xyz.mean(axis=0)
    centered = points_xyz - center
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    rotation = _ensure_right_handed(eigvecs)

    local = centered @ rotation
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    size = np.maximum(local_max - local_min, 1e-3)
    local_center = 0.5 * (local_min + local_max)
    world_center = center + local_center @ rotation.T

    horizontal_axes = [axis for axis in [0, 1, 2] if axis != up_axis]
    principal = rotation[:, 0]
    yaw = float(np.arctan2(principal[horizontal_axes[1]], principal[horizontal_axes[0]]))

    return OBBResult(center=world_center, size=size, rotation=rotation, yaw=yaw)


def refine_with_pca(
    initial_box: Optional[Box3D],
    support_points_xyz: np.ndarray,
    blend: float = 1.0,
    up_axis: int = 2,
) -> Box3D:
    if not (0.0 <= blend <= 1.0):
        raise ValueError("blend must be in [0, 1]")

    refined_obb = fit_pca_obb(support_points_xyz, up_axis=up_axis)
    refined_box = refined_obb.to_box3d(
        label=(initial_box.label if initial_box is not None else ""),
        score=(initial_box.score if initial_box is not None else 1.0),
        instance_id=(initial_box.instance_id if initial_box is not None else ""),
    )

    if initial_box is None or blend >= 1.0:
        return refined_box

    center = (1.0 - blend) * initial_box.center + blend * refined_box.center
    size = (1.0 - blend) * initial_box.size + blend * refined_box.size
    yaw = float(np.arctan2(
        (1.0 - blend) * np.sin(initial_box.yaw) + blend * np.sin(refined_box.yaw),
        (1.0 - blend) * np.cos(initial_box.yaw) + blend * np.cos(refined_box.yaw),
    ))

    return Box3D(
        center=center,
        size=np.maximum(size, 1e-3),
        yaw=yaw,
        label=initial_box.label,
        score=initial_box.score,
        instance_id=initial_box.instance_id,
    )