from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.eval import Box3D


@dataclass
class SupportConfig:
    contact_tolerance: float = 0.04
    max_lift: float = 0.25


def estimate_ground_z(points_xyz: np.ndarray, quantile: float = 0.03) -> float:
    return float(np.quantile(points_xyz[:, 2], quantile))


def apply_support_constraints(boxes: List[Box3D], points_xyz: np.ndarray, cfg: SupportConfig) -> List[Box3D]:
    if len(boxes) == 0:
        return []

    ground_z = estimate_ground_z(points_xyz)
    corrected: List[Box3D] = []

    for box in boxes:
        center = box.center.copy()
        bottom = center[2] - box.size[2] / 2.0
        if bottom < ground_z - cfg.contact_tolerance:
            center[2] += (ground_z - bottom)
        elif bottom > ground_z + cfg.max_lift:
            center[2] -= min(bottom - ground_z, cfg.max_lift)

        corrected.append(
            Box3D(
                center=center,
                size=box.size,
                yaw=box.yaw,
                label=box.label,
                score=box.score,
                instance_id=box.instance_id,
            )
        )

    return corrected
