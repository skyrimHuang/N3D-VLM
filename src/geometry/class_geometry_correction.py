from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.eval import Box3D


PLANE_LIKE = {"table", "desk", "counter", "shelf", "bed", "floor", "wall"}
TALL_LIKE = {"lamp", "curtain", "door", "cabinet", "bookshelf", "refrigerator"}
SMALL_LIKE = {"pillow", "clock", "backpack", "trash can", "scale"}


@dataclass
class CorrectionConfig:
    min_dim: float = 0.04
    max_dim: float = 4.0
    upright_blend: float = 0.35


def _canonical_label(label: str) -> str:
    return " ".join(label.lower().split())


def _clip_dims(size: np.ndarray, cfg: CorrectionConfig) -> np.ndarray:
    return np.clip(size, cfg.min_dim, cfg.max_dim)


def correct_box_by_class(box: Box3D, cfg: CorrectionConfig) -> Box3D:
    label = _canonical_label(box.label)
    size = _clip_dims(box.size.copy(), cfg)
    yaw = float(box.yaw)

    if any(name in label for name in PLANE_LIKE):
        size[2] = max(cfg.min_dim, min(size[2], 0.35))
    elif any(name in label for name in TALL_LIKE):
        size[2] = max(size[2], max(size[0], size[1]))
    elif any(name in label for name in SMALL_LIKE):
        size = np.minimum(size, np.array([1.2, 1.2, 1.2], dtype=np.float64))

    # Stabilize near-axis objects
    if abs(np.sin(yaw)) < 0.12:
        yaw = float(np.round(yaw / (np.pi / 2.0)) * (np.pi / 2.0))

    return Box3D(
        center=box.center,
        size=size,
        yaw=yaw,
        label=box.label,
        score=box.score,
        instance_id=box.instance_id,
    )
