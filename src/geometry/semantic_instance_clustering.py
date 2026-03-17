from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class ClusterResult:
    indices: np.ndarray
    centroid: np.ndarray
    purity: float
    score: float


def euclidean_dbscan_like(points_xyz: np.ndarray, eps: float, min_points: int) -> List[np.ndarray]:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be shape (N, 3)")
    if points_xyz.shape[0] < min_points:
        return []

    tree = cKDTree(points_xyz)
    visited = np.zeros(points_xyz.shape[0], dtype=bool)
    clusters: List[np.ndarray] = []

    for seed in range(points_xyz.shape[0]):
        if visited[seed]:
            continue
        neighbors = tree.query_ball_point(points_xyz[seed], r=eps)
        if len(neighbors) < min_points:
            visited[seed] = True
            continue

        queue = list(neighbors)
        members: List[int] = []
        while queue:
            idx = queue.pop()
            if visited[idx]:
                continue
            visited[idx] = True
            members.append(idx)
            local_neighbors = tree.query_ball_point(points_xyz[idx], r=eps)
            if len(local_neighbors) >= min_points:
                queue.extend(local_neighbors)

        if len(members) >= min_points:
            clusters.append(np.asarray(members, dtype=np.int64))

    return clusters


def select_instance_cluster(
    points_xyz: np.ndarray,
    clusters: Iterable[np.ndarray],
    reference_center: np.ndarray,
    target_label: Optional[str] = None,
    point_labels: Optional[List[str]] = None,
) -> Optional[ClusterResult]:
    best: Optional[ClusterResult] = None
    ref = np.asarray(reference_center, dtype=np.float64)

    for ids in clusters:
        if ids.size == 0:
            continue
        pts = points_xyz[ids]
        centroid = pts.mean(axis=0)
        dist = float(np.linalg.norm(centroid - ref))

        purity = 1.0
        if target_label is not None and point_labels is not None and len(point_labels) == points_xyz.shape[0]:
            labels = [point_labels[i] for i in ids.tolist()]
            hit = sum(1 for label in labels if label == target_label)
            purity = float(hit / max(1, len(labels)))

        score = float(ids.size) + 80.0 * purity - 8.0 * dist
        candidate = ClusterResult(indices=ids, centroid=centroid, purity=purity, score=score)

        if best is None or candidate.score > best.score:
            best = candidate

    return best


def cluster_with_semantic_constraint(
    points_xyz: np.ndarray,
    reference_center: np.ndarray,
    target_label: Optional[str],
    point_labels: Optional[List[str]],
    eps: float,
    min_points: int,
) -> Optional[ClusterResult]:
    clusters = euclidean_dbscan_like(points_xyz=points_xyz, eps=eps, min_points=min_points)
    if not clusters:
        return None
    return select_instance_cluster(
        points_xyz=points_xyz,
        clusters=clusters,
        reference_center=reference_center,
        target_label=target_label,
        point_labels=point_labels,
    )
