from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


EPS = 1e-9


@dataclass
class Box3D:
    center: np.ndarray
    size: np.ndarray
    yaw: float = 0.0
    label: str = ""
    score: float = 1.0
    instance_id: str = ""

    @staticmethod
    def from_dict(data: dict) -> "Box3D":
        center = np.asarray(data["center"], dtype=np.float64)
        size = np.asarray(data["size"], dtype=np.float64)
        return Box3D(
            center=center,
            size=np.maximum(size, EPS),
            yaw=float(data.get("yaw", 0.0)),
            label=str(data.get("label", "")),
            score=float(data.get("score", 1.0)),
            instance_id=str(data.get("instance_id", "")),
        )

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "yaw": float(self.yaw),
            "label": self.label,
            "score": float(self.score),
            "instance_id": self.instance_id,
        }


def _yaw_rotation(yaw: float) -> np.ndarray:
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _rect_xy_corners(box: Box3D) -> np.ndarray:
    half = box.size[:2] / 2.0
    local = np.array(
        [
            [-half[0], -half[1]],
            [half[0], -half[1]],
            [half[0], half[1]],
            [-half[0], half[1]],
        ],
        dtype=np.float64,
    )
    rot = _yaw_rotation(box.yaw)
    return local @ rot.T + box.center[:2]


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= -EPS


def _line_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < EPS:
        return p2.copy()
    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denominator
    py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denominator
    return np.array([px, py], dtype=np.float64)


def _convex_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    output = subject.copy()
    for i in range(len(clipper)):
        edge_start = clipper[i]
        edge_end = clipper[(i + 1) % len(clipper)]
        input_list = output
        if len(input_list) == 0:
            return np.zeros((0, 2), dtype=np.float64)
        output_points: List[np.ndarray] = []
        s = input_list[-1]
        for e in input_list:
            if _inside(e, edge_start, edge_end):
                if not _inside(s, edge_start, edge_end):
                    output_points.append(_line_intersection(s, e, edge_start, edge_end))
                output_points.append(e)
            elif _inside(s, edge_start, edge_end):
                output_points.append(_line_intersection(s, e, edge_start, edge_end))
            s = e
        output = np.asarray(output_points, dtype=np.float64)
    return output


def iou_3d_aabb(box_a: Box3D, box_b: Box3D) -> float:
    min_a = box_a.center - box_a.size / 2.0
    max_a = box_a.center + box_a.size / 2.0
    min_b = box_b.center - box_b.size / 2.0
    max_b = box_b.center + box_b.size / 2.0

    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter = np.maximum(0.0, inter_max - inter_min)
    inter_vol = float(inter[0] * inter[1] * inter[2])

    vol_a = float(np.prod(np.maximum(box_a.size, 0.0)))
    vol_b = float(np.prod(np.maximum(box_b.size, 0.0)))
    denom = vol_a + vol_b - inter_vol
    if denom <= EPS:
        return 0.0
    return inter_vol / denom


def iou_3d_obb_yaw(box_a: Box3D, box_b: Box3D) -> float:
    poly_a = _rect_xy_corners(box_a)
    poly_b = _rect_xy_corners(box_b)
    inter_poly = _convex_clip(poly_a, poly_b)
    inter_area = _polygon_area(inter_poly)

    zmin_a = box_a.center[2] - box_a.size[2] / 2.0
    zmax_a = box_a.center[2] + box_a.size[2] / 2.0
    zmin_b = box_b.center[2] - box_b.size[2] / 2.0
    zmax_b = box_b.center[2] + box_b.size[2] / 2.0
    inter_h = max(0.0, min(zmax_a, zmax_b) - max(zmin_a, zmin_b))

    inter_vol = inter_area * inter_h
    vol_a = _polygon_area(poly_a) * max(0.0, box_a.size[2])
    vol_b = _polygon_area(poly_b) * max(0.0, box_b.size[2])
    denom = vol_a + vol_b - inter_vol
    if denom <= EPS:
        return 0.0
    return inter_vol / denom


def centroid_error_cm(box_a: Box3D, box_b: Box3D) -> float:
    return float(np.linalg.norm(box_a.center - box_b.center) * 100.0)


def as_boxes(data: Iterable[dict | Box3D]) -> List[Box3D]:
    boxes: List[Box3D] = []
    for item in data:
        if isinstance(item, Box3D):
            boxes.append(item)
        else:
            boxes.append(Box3D.from_dict(item))
    return boxes


def summarize(values: Sequence[float]) -> dict:
    if len(values) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }