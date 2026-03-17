from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from scipy.optimize import linear_sum_assignment

from .box_metrics_3d import Box3D


@dataclass
class MatchRecord:
    pred_idx: int
    gt_idx: int
    iou: float
    centroid_error_cm: float
    label: str


def _build_iou_matrix(
    preds: List[Box3D],
    gts: List[Box3D],
    iou_fn: Callable[[Box3D, Box3D], float],
    class_aware: bool,
) -> np.ndarray:
    iou_mat = np.zeros((len(preds), len(gts)), dtype=np.float64)
    for i, pred in enumerate(preds):
        for j, gt in enumerate(gts):
            if class_aware and pred.label != gt.label:
                continue
            iou_mat[i, j] = iou_fn(pred, gt)
    return iou_mat


def hungarian_match(
    preds: List[Box3D],
    gts: List[Box3D],
    iou_fn: Callable[[Box3D, Box3D], float],
    centroid_fn: Callable[[Box3D, Box3D], float],
    iou_threshold: float = 0.25,
    class_aware: bool = True,
) -> List[MatchRecord]:
    if len(preds) == 0 or len(gts) == 0:
        return []

    iou_mat = _build_iou_matrix(preds, gts, iou_fn=iou_fn, class_aware=class_aware)
    if np.all(iou_mat <= 0.0):
        return []

    cost = 1.0 - iou_mat
    row_idx, col_idx = linear_sum_assignment(cost)

    matches: List[MatchRecord] = []
    for pi, gi in zip(row_idx, col_idx):
        iou_val = float(iou_mat[pi, gi])
        if iou_val < iou_threshold:
            continue
        cen_err = float(centroid_fn(preds[pi], gts[gi]))
        matches.append(
            MatchRecord(
                pred_idx=int(pi),
                gt_idx=int(gi),
                iou=iou_val,
                centroid_error_cm=cen_err,
                label=gts[gi].label,
            )
        )
    return matches


def recall_at_threshold(
    preds: List[Box3D],
    gts: List[Box3D],
    iou_fn: Callable[[Box3D, Box3D], float],
    centroid_fn: Callable[[Box3D, Box3D], float],
    iou_threshold: float,
    class_aware: bool = True,
) -> float:
    if len(gts) == 0:
        return 0.0
    matches = hungarian_match(
        preds=preds,
        gts=gts,
        iou_fn=iou_fn,
        centroid_fn=centroid_fn,
        iou_threshold=iou_threshold,
        class_aware=class_aware,
    )
    return len(matches) / len(gts)