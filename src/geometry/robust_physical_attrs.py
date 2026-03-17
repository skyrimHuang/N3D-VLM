from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhysicalAttrs:
    centroid: np.ndarray
    dimensions: np.ndarray
    rotation: np.ndarray
    normal: np.ndarray
    yaw: float
    inlier_ratio: float


def _mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median))) + 1e-9


def robust_inlier_mask(points_xyz: np.ndarray, keep_ratio: float = 0.9) -> np.ndarray:
    center = np.median(points_xyz, axis=0)
    d = np.linalg.norm(points_xyz - center, axis=1)
    threshold = np.quantile(d, keep_ratio)
    return d <= threshold


def robust_pca_physical_attrs(points_xyz: np.ndarray, keep_ratio: float = 0.9) -> PhysicalAttrs:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be shape (N, 3)")
    if points_xyz.shape[0] < 6:
        centroid = points_xyz.mean(axis=0)
        dimensions = np.maximum(points_xyz.max(axis=0) - points_xyz.min(axis=0), 1e-3)
        rotation = np.eye(3, dtype=np.float64)
        normal = rotation[:, 2]
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        return PhysicalAttrs(
            centroid=centroid,
            dimensions=dimensions,
            rotation=rotation,
            normal=normal,
            yaw=yaw,
            inlier_ratio=1.0,
        )

    inlier_mask = robust_inlier_mask(points_xyz, keep_ratio=keep_ratio)
    inliers = points_xyz[inlier_mask]
    if inliers.shape[0] < 6:
        inliers = points_xyz
        inlier_mask = np.ones(points_xyz.shape[0], dtype=bool)

    centroid = inliers.mean(axis=0)
    centered = inliers - centroid

    cov = (centered.T @ centered) / max(1, inliers.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1.0

    local = centered @ eigvecs
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    dimensions = np.maximum(local_max - local_min, 1e-3)
    local_center = 0.5 * (local_min + local_max)
    obb_centroid = centroid + local_center @ eigvecs.T

    normal = eigvecs[:, 2]
    yaw = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    return PhysicalAttrs(
        centroid=obb_centroid,
        dimensions=dimensions,
        rotation=eigvecs,
        normal=normal,
        yaw=yaw,
        inlier_ratio=float(inlier_mask.mean()),
    )
