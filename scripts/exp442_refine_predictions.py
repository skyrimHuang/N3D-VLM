from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import Box3D
from src.geometry import (
    CorrectionConfig,
    SupportConfig,
    apply_support_constraints,
    cluster_with_semantic_constraint,
    correct_box_by_class,
    refine_with_pca,
    robust_pca_physical_attrs,
)


def _boxes_from_json(path: Path) -> Dict[str, List[Box3D]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return {k: [Box3D.from_dict(x) for x in v] for k, v in payload.items()}
    if isinstance(payload, list):
        return {row["scene_id"]: [Box3D.from_dict(x) for x in row.get("boxes", [])] for row in payload}
    raise ValueError(f"Unsupported prediction format: {path}")


def _boxes_to_json(scene_to_boxes: Dict[str, List[Box3D]], path: Path) -> None:
    payload = {scene_id: [box.to_dict() for box in boxes] for scene_id, boxes in scene_to_boxes.items()}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _select_support_points(points_xyz: np.ndarray, box: Box3D, expand_ratio: float) -> np.ndarray:
    half = 0.5 * box.size * expand_ratio
    lower = box.center - half
    upper = box.center + half
    mask = np.all((points_xyz >= lower) & (points_xyz <= upper), axis=1)
    return points_xyz[mask]


def refine_predictions(
    manifest: List[dict],
    predictions: Dict[str, List[Box3D]],
    expand_ratio: float,
    blend: float,
    min_support: int,
    cluster_eps: float,
    min_cluster_size: int,
    support_cap: int,
) -> tuple[Dict[str, List[Box3D]], List[dict]]:
    rng = np.random.default_rng(1234)
    correction_cfg = CorrectionConfig()
    support_cfg = SupportConfig()

    output: Dict[str, List[Box3D]] = {}
    diagnostics: List[dict] = []

    for scene in tqdm(manifest, desc="Refining predictions"):
        scene_id = scene["scene_id"]
        points = np.load(scene["scene_point_cloud"])["points"].astype(np.float64)
        boxes = predictions.get(scene_id, [])

        refined_scene: List[Box3D] = []
        for box in boxes:
            support = _select_support_points(points, box, expand_ratio=expand_ratio)
            if support.shape[0] > support_cap:
                idx = rng.choice(support.shape[0], size=support_cap, replace=False)
                support = support[idx]

            if support.shape[0] < min_support:
                refined_scene.append(box)
                diagnostics.append(
                    {
                        "scene_id": scene_id,
                        "instance_id": box.instance_id,
                        "label": box.label,
                        "status": "insufficient_support",
                        "support_points": int(support.shape[0]),
                    }
                )
                continue

            cluster = cluster_with_semantic_constraint(
                points_xyz=support,
                reference_center=box.center,
                target_label=box.label,
                point_labels=None,
                eps=cluster_eps,
                min_points=min_cluster_size,
            )
            if cluster is None:
                refined_scene.append(box)
                diagnostics.append(
                    {
                        "scene_id": scene_id,
                        "instance_id": box.instance_id,
                        "label": box.label,
                        "status": "no_cluster",
                        "support_points": int(support.shape[0]),
                    }
                )
                continue

            instance_points = support[cluster.indices]
            attrs = robust_pca_physical_attrs(instance_points)
            from_instance = Box3D(
                center=attrs.centroid,
                size=attrs.dimensions,
                yaw=attrs.yaw,
                label=box.label,
                score=box.score,
                instance_id=box.instance_id,
            )
            refined = refine_with_pca(initial_box=box, support_points_xyz=instance_points, blend=blend)
            refined.center = 0.5 * refined.center + 0.5 * from_instance.center
            refined.size = np.maximum(0.5 * refined.size + 0.5 * from_instance.size, 1e-3)
            refined.yaw = float(
                np.arctan2(
                    np.sin(refined.yaw) + np.sin(from_instance.yaw),
                    np.cos(refined.yaw) + np.cos(from_instance.yaw),
                )
            )
            refined = correct_box_by_class(refined, correction_cfg)
            refined_scene.append(refined)

            diagnostics.append(
                {
                    "scene_id": scene_id,
                    "instance_id": box.instance_id,
                    "label": box.label,
                    "status": "refined",
                    "support_points": int(support.shape[0]),
                    "cluster_points": int(instance_points.shape[0]),
                    "cluster_purity": float(cluster.purity),
                    "inlier_ratio": float(attrs.inlier_ratio),
                }
            )

        refined_scene = apply_support_constraints(refined_scene, points_xyz=points, cfg=support_cfg)
        output[scene_id] = refined_scene

    return output, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-wise geometric refinement for exp4.4.2")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--input_pred", type=str, required=True)
    parser.add_argument("--output_pred", type=str, required=True)
    parser.add_argument("--output_diag", type=str, default="")
    parser.add_argument("--expand_ratio", type=float, default=1.4)
    parser.add_argument("--blend", type=float, default=0.8)
    parser.add_argument("--min_support", type=int, default=60)
    parser.add_argument("--cluster_eps", type=float, default=0.15)
    parser.add_argument("--min_cluster_size", type=int, default=35)
    parser.add_argument("--support_cap", type=int, default=15000)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    predictions = _boxes_from_json(Path(args.input_pred))
    refined, diagnostics = refine_predictions(
        manifest=manifest,
        predictions=predictions,
        expand_ratio=args.expand_ratio,
        blend=args.blend,
        min_support=args.min_support,
        cluster_eps=args.cluster_eps,
        min_cluster_size=args.min_cluster_size,
        support_cap=args.support_cap,
    )

    out_pred = Path(args.output_pred)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    _boxes_to_json(refined, out_pred)

    diag_path = Path(args.output_diag) if args.output_diag else out_pred.with_suffix(".diagnostics.json")
    diag_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False))
    print(f"Saved refined predictions: {out_pred}")
    print(f"Saved diagnostics: {diag_path}")


if __name__ == "__main__":
    main()
