from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

import numpy as np


EPS = 1e-9


@dataclass
class SemanticObservation:
    point_id: int
    label: str
    confidence: float
    z_cam: float
    depth_ref: float
    normal: np.ndarray
    view_vector: np.ndarray
    z_opt: float
    sigma_d: float
    depth_tolerance: float


@dataclass
class PointSemanticResult:
    point_id: int
    best_label: str
    score: float
    label_scores: Dict[str, float]


def visibility_indicator(z_cam: float, depth_ref: float, depth_tolerance: float) -> int:
    if z_cam <= 0.0:
        return 0
    return int(abs(z_cam - depth_ref) < depth_tolerance)


def distance_weight(z_cam: float, z_opt: float, sigma_d: float) -> float:
    sigma = max(float(sigma_d), EPS)
    gap = max(0.0, z_cam - z_opt)
    return float(np.exp(-(gap * gap) / (2.0 * sigma * sigma)))


def angle_weight(normal: np.ndarray, view_vector: np.ndarray) -> float:
    normal = np.asarray(normal, dtype=np.float64)
    view_vector = np.asarray(view_vector, dtype=np.float64)
    n_norm = np.linalg.norm(normal)
    v_norm = np.linalg.norm(view_vector)
    if n_norm < EPS or v_norm < EPS:
        return 0.0
    value = float(np.dot(normal, view_vector) / (n_norm * v_norm))
    return max(0.0, value)


def observation_weight(obs: SemanticObservation) -> float:
    w_dist = distance_weight(obs.z_cam, obs.z_opt, obs.sigma_d)
    w_angle = angle_weight(obs.normal, obs.view_vector)
    return w_dist * w_angle


def fuse_point_semantics(observations: Iterable[SemanticObservation]) -> Dict[int, PointSemanticResult]:
    grouped: Dict[int, List[SemanticObservation]] = defaultdict(list)
    for obs in observations:
        grouped[obs.point_id].append(obs)

    results: Dict[int, PointSemanticResult] = {}
    for point_id, obs_list in grouped.items():
        label_scores: Dict[str, float] = defaultdict(float)
        for obs in obs_list:
            visible = visibility_indicator(obs.z_cam, obs.depth_ref, obs.depth_tolerance)
            if visible == 0:
                continue
            w = observation_weight(obs)
            label_scores[obs.label] += float(obs.confidence) * w

        if len(label_scores) == 0:
            results[point_id] = PointSemanticResult(
                point_id=point_id,
                best_label="unknown",
                score=0.0,
                label_scores={},
            )
            continue

        best_label, best_score = max(label_scores.items(), key=lambda x: x[1])
        results[point_id] = PointSemanticResult(
            point_id=point_id,
            best_label=best_label,
            score=float(best_score),
            label_scores=dict(label_scores),
        )
    return results


def voxelize_semantics(
    points_xyz: np.ndarray,
    semantic_results: Mapping[int, PointSemanticResult],
    voxel_size: float,
) -> List[dict]:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be shape (N, 3)")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    voxels: Dict[tuple[int, int, int], List[int]] = defaultdict(list)
    idx = np.floor(points_xyz / voxel_size).astype(np.int64)
    for i, key in enumerate(map(tuple, idx.tolist())):
        voxels[key].append(i)

    nodes: List[dict] = []
    for key, ids in voxels.items():
        xyz = points_xyz[ids]
        center = xyz.mean(axis=0)

        label_acc: Dict[str, float] = defaultdict(float)
        label_count: Dict[str, int] = defaultdict(int)
        for pid in ids:
            result = semantic_results.get(pid)
            if result is None:
                continue
            label_acc[result.best_label] += result.score
            label_count[result.best_label] += 1

        if len(label_acc) == 0:
            label = "unknown"
            conf = 0.0
        else:
            label = max(label_acc.items(), key=lambda x: x[1])[0]
            count = max(label_count[label], 1)
            conf = float(label_acc[label] / count)

        node = {
            "voxel_index": key,
            "position": center.tolist(),
            "semantic_label": label,
            "confidence": conf,
            "occupancy": True,
            "num_points": len(ids),
        }
        nodes.append(node)
    return nodes