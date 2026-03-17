from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval import Box3D, centroid_error_cm, hungarian_match, iou_3d_obb_yaw
from src.geometry import (
    CorrectionConfig,
    SupportConfig,
    apply_support_constraints,
    cluster_with_semantic_constraint,
    correct_box_by_class,
    refine_with_pca,
    robust_pca_physical_attrs,
)


@dataclass
class Params:
    expand_ratio: float
    blend: float
    min_support: int
    cluster_eps: float
    min_cluster_size: int
    support_cap: int


def _load_scene_gt(scene_entry: dict) -> List[Box3D]:
    payload = json.loads(Path(scene_entry["gt_boxes"]).read_text())
    return [Box3D.from_dict(obj["obb"]) for obj in payload["objects"]]


def _load_scene_points(scene_entry: dict) -> np.ndarray:
    payload = np.load(scene_entry["scene_point_cloud"])
    return payload["points"].astype(np.float64)


def _perturb_box(box: Box3D, rng: np.random.Generator) -> Box3D:
    center_noise = rng.normal(0.0, 0.08, size=3)
    size_scale = np.clip(1.0 + rng.normal(0.15, 0.08, size=3), 0.7, 1.5)
    yaw_noise = rng.normal(0.0, 0.35)
    return Box3D(
        center=box.center + center_noise,
        size=np.maximum(box.size * size_scale, 1e-3),
        yaw=float(box.yaw + yaw_noise),
        label=box.label,
        score=0.65,
        instance_id=box.instance_id,
    )


def _generate_baseline_from_gt(manifest: List[dict], seed: int) -> Dict[str, List[Box3D]]:
    rng = np.random.default_rng(seed)
    output: Dict[str, List[Box3D]] = {}
    for scene in tqdm(manifest, desc="Baseline generation", leave=False):
        gts = _load_scene_gt(scene)
        output[scene["scene_id"]] = [_perturb_box(box, rng) for box in gts]
    return output


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


def _select_support_points(points: np.ndarray, box: Box3D, expand_ratio: float) -> np.ndarray:
    half = 0.5 * box.size * expand_ratio
    lower = box.center - half
    upper = box.center + half
    mask = np.all((points >= lower) & (points <= upper), axis=1)
    return points[mask]


def _euclidean_clusters(points: np.ndarray, eps: float, min_cluster_size: int) -> List[np.ndarray]:
    if points.shape[0] == 0:
        return []
    if points.shape[0] < min_cluster_size:
        return []

    tree = cKDTree(points)
    visited = np.zeros(points.shape[0], dtype=bool)
    clusters: List[np.ndarray] = []

    for seed in range(points.shape[0]):
        if visited[seed]:
            continue

        seed_neighbors = tree.query_ball_point(points[seed], r=eps)
        if len(seed_neighbors) < min_cluster_size:
            visited[seed] = True
            continue

        queue = list(seed_neighbors)
        cluster_ids: List[int] = []
        while queue:
            idx = queue.pop()
            if visited[idx]:
                continue
            visited[idx] = True
            cluster_ids.append(idx)
            neigh = tree.query_ball_point(points[idx], r=eps)
            if len(neigh) >= min_cluster_size:
                queue.extend(neigh)

        if len(cluster_ids) >= min_cluster_size:
            clusters.append(np.asarray(cluster_ids, dtype=np.int64))

    return clusters


def _select_instance_points(
    support_points: np.ndarray,
    box_center: np.ndarray,
    cluster_eps: float,
    min_cluster_size: int,
) -> np.ndarray:
    clusters = _euclidean_clusters(support_points, eps=cluster_eps, min_cluster_size=min_cluster_size)
    if len(clusters) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    best_score = -1e18
    best_points = None
    for cluster in clusters:
        pts = support_points[cluster]
        centroid = pts.mean(axis=0)
        dist = float(np.linalg.norm(centroid - box_center))
        score = float(cluster.size) - 8.0 * dist
        if score > best_score:
            best_score = score
            best_points = pts

    if best_points is None:
        return np.zeros((0, 3), dtype=np.float64)
    return best_points


def _extract_physical_attrs(instance_points: np.ndarray) -> dict:
    centroid = instance_points.mean(axis=0)
    centered = instance_points - centroid
    cov = (centered.T @ centered) / max(1, instance_points.shape[0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, -1] *= -1.0

    local = centered @ eigvecs
    local_min = local.min(axis=0)
    local_max = local.max(axis=0)
    dims = np.maximum(local_max - local_min, 1e-3)

    local_center = 0.5 * (local_min + local_max)
    obb_center = centroid + local_center @ eigvecs.T
    normal = eigvecs[:, 2]
    yaw = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    return {
        "rotation": eigvecs,
        "normal": normal,
        "dimensions": dims,
        "centroid": obb_center,
        "yaw": yaw,
        "eigenvalues": eigvals,
    }


def _sample_candidate(rng: np.random.Generator, best: Params, local_sigma: float) -> Params:
    if rng.uniform() < 0.65:
        expand_ratio = float(np.clip(best.expand_ratio + rng.normal(0.0, local_sigma), 1.0, 2.8))
        blend = float(np.clip(best.blend + rng.normal(0.0, local_sigma * 0.35), 0.0, 1.0))
        min_support = int(np.clip(round(best.min_support + rng.normal(0.0, local_sigma * 80)), 20, 300))
        cluster_eps = float(np.clip(best.cluster_eps + rng.normal(0.0, local_sigma * 0.12), 0.03, 0.80))
        min_cluster_size = int(np.clip(round(best.min_cluster_size + rng.normal(0.0, local_sigma * 50)), 10, 220))
        support_cap = int(np.clip(round(best.support_cap + rng.normal(0.0, local_sigma * 10000)), 3000, 30000))
    else:
        expand_ratio = float(rng.uniform(1.0, 2.8))
        blend = float(rng.uniform(0.0, 1.0))
        min_support = int(rng.integers(20, 300))
        cluster_eps = float(rng.uniform(0.03, 0.80))
        min_cluster_size = int(rng.integers(10, 220))
        support_cap = int(rng.integers(3000, 30000))

    return Params(
        expand_ratio=expand_ratio,
        blend=blend,
        min_support=min_support,
        cluster_eps=cluster_eps,
        min_cluster_size=min_cluster_size,
        support_cap=support_cap,
    )


def _refine_predictions(
    manifest: List[dict],
    baseline: Dict[str, List[Box3D]],
    params: Params,
    progress_desc: str = "Refining",
) -> Dict[str, List[Box3D]]:
    rng = np.random.default_rng(12345)
    correction_cfg = CorrectionConfig()
    support_cfg = SupportConfig()
    output: Dict[str, List[Box3D]] = {}
    for scene in tqdm(manifest, desc=progress_desc, leave=False):
        scene_id = scene["scene_id"]
        points = _load_scene_points(scene)
        preds = baseline.get(scene_id, [])
        refined_boxes: List[Box3D] = []
        for box in preds:
            support = _select_support_points(points, box, params.expand_ratio)
            if support.shape[0] > params.support_cap:
                sel = rng.choice(support.shape[0], size=params.support_cap, replace=False)
                support = support[sel]
            if support.shape[0] < params.min_support:
                refined_boxes.append(box)
                continue

            cluster = cluster_with_semantic_constraint(
                points_xyz=support,
                reference_center=box.center,
                target_label=box.label,
                point_labels=None,
                eps=params.cluster_eps,
                min_points=params.min_cluster_size,
            )
            instance_points = support[cluster.indices] if cluster is not None else np.zeros((0, 3), dtype=np.float64)

            if instance_points.shape[0] < params.min_support:
                refined = refine_with_pca(initial_box=box, support_points_xyz=support, blend=params.blend)
                refined_boxes.append(correct_box_by_class(refined, correction_cfg))
                continue

            attrs = robust_pca_physical_attrs(instance_points)
            refined_from_instance = Box3D(
                center=attrs.centroid,
                size=attrs.dimensions,
                yaw=attrs.yaw,
                label=box.label,
                score=box.score,
                instance_id=box.instance_id,
            )
            refined = refine_with_pca(initial_box=box, support_points_xyz=instance_points, blend=params.blend)
            refined.center = 0.5 * refined.center + 0.5 * refined_from_instance.center
            refined.size = np.maximum(0.5 * refined.size + 0.5 * refined_from_instance.size, 1e-3)
            refined.yaw = float(
                np.arctan2(
                    np.sin(refined.yaw) + np.sin(refined_from_instance.yaw),
                    np.cos(refined.yaw) + np.cos(refined_from_instance.yaw),
                )
            )
            refined_boxes.append(correct_box_by_class(refined, correction_cfg))

        refined_boxes = apply_support_constraints(refined_boxes, points_xyz=points, cfg=support_cfg)
        output[scene_id] = refined_boxes
    return output


def _evaluate(manifest: List[dict], preds: Dict[str, List[Box3D]], iou_threshold: float) -> dict:
    manifest = list(manifest)
    ious: List[float] = []
    cens: List[float] = []
    scene_rows: List[dict] = []
    for scene in tqdm(manifest, desc="Evaluating", leave=False):
        scene_id = scene["scene_id"]
        gts = _load_scene_gt(scene)
        pr = preds.get(scene_id, [])
        matches = hungarian_match(
            preds=pr,
            gts=gts,
            iou_fn=iou_3d_obb_yaw,
            centroid_fn=centroid_error_cm,
            iou_threshold=iou_threshold,
            class_aware=True,
        )
        row_ious = [m.iou for m in matches]
        row_cens = [m.centroid_error_cm for m in matches]
        if row_ious:
            ious.extend(row_ious)
            cens.extend(row_cens)
        scene_rows.append(
            {
                "scene_id": scene_id,
                "matched": len(matches),
                "obb_iou_mean": float(np.mean(row_ious)) if row_ious else 0.0,
                "centroid_cm_mean": float(np.mean(row_cens)) if row_cens else 0.0,
            }
        )

    return {
        "obb_iou_mean": float(np.mean(ious)) if ious else 0.0,
        "obb_iou_std": float(np.std(ious)) if ious else 0.0,
        "centroid_cm_mean": float(np.mean(cens)) if cens else math.inf,
        "centroid_cm_std": float(np.std(cens)) if cens else 0.0,
        "scene_rows": scene_rows,
    }


def _metric_score(metric: dict) -> float:
    return float(metric["obb_iou_mean"] - 0.002 * metric["centroid_cm_mean"])


def _split_manifest(manifest: List[dict], train_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    shuffled = manifest[:]
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * train_ratio))
    train = shuffled[:split]
    val = shuffled[split:] if split < len(shuffled) else shuffled[:]
    return train, val


def _plot_history(history: List[dict], save_path: Path) -> None:
    rounds = [h["round"] for h in history]
    vals = [h["val_obb_iou"] for h in history]
    fig = plt.figure(figsize=(6, 4))
    plt.plot(rounds, vals, marker="o")
    plt.xlabel("Round")
    plt.ylabel("Val OBB IoU")
    plt.title("Auto Optimization History")
    plt.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _run_benchmark(
    manifest_path: Path,
    baseline_path: Path,
    optimized_path: Path,
    output_dir: Path,
    python_executable: str,
) -> None:
    cmd = [
        python_executable,
        str(PROJECT_ROOT / "scripts" / "exp442_benchmark.py"),
        "--manifest",
        str(manifest_path),
        "--single_view_pred",
        str(baseline_path),
        "--multiview_pca_pred",
        str(optimized_path),
        "--output_dir",
        str(output_dir),
        "--num_qualitative",
        "5",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto optimize and iterate exp 4.4.2")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/exp442_auto")
    parser.add_argument("--single_view_pred", type=str, default="")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_rounds", type=int, default=8)
    parser.add_argument("--candidates_per_round", type=int, default=8)
    parser.add_argument("--max_search_seconds", type=float, default=1800.0)
    parser.add_argument("--max_total_evals", type=int, default=240)
    parser.add_argument("--early_stop_rounds", type=int, default=4)
    parser.add_argument("--local_search_sigma", type=float, default=0.25)
    parser.add_argument("--target_obb_iou", type=float, default=0.55)
    parser.add_argument("--iou_threshold", type=float, default=0.25)
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Stage] Loaded manifest scenes: {len(manifest)}")

    if args.single_view_pred:
        print("[Stage] Loading provided single-view predictions...")
        baseline = _boxes_from_json(Path(args.single_view_pred))
    else:
        print("[Stage] Generating synthetic single-view baseline from GT...")
        baseline = _generate_baseline_from_gt(manifest, seed=args.seed)

    baseline_path = output_dir / "single_view_pred.json"
    _boxes_to_json(baseline, baseline_path)

    print("[Stage] Splitting train/val and evaluating baseline...")
    train_manifest, val_manifest = _split_manifest(manifest, train_ratio=args.train_ratio, seed=args.seed)
    base_train = _evaluate(train_manifest, baseline, args.iou_threshold)
    base_val = _evaluate(val_manifest, baseline, args.iou_threshold)

    best_params = Params(
        expand_ratio=1.4,
        blend=0.8,
        min_support=60,
        cluster_eps=0.15,
        min_cluster_size=35,
        support_cap=15000,
    )
    print("[Stage] Running initial refinement...")
    best_pred = _refine_predictions(
        train_manifest + val_manifest,
        baseline,
        best_params,
        progress_desc="Initial refine",
    )
    best_val_metric = _evaluate(val_manifest, best_pred, args.iou_threshold)
    best_val_score = _metric_score(best_val_metric)

    history: List[dict] = [
        {
            "round": 0,
            "params": best_params.__dict__,
            "train_obb_iou": base_train["obb_iou_mean"],
            "val_obb_iou": base_val["obb_iou_mean"],
            "note": "baseline",
        },
        {
            "round": 1,
            "params": best_params.__dict__,
            "train_obb_iou": _evaluate(train_manifest, best_pred, args.iou_threshold)["obb_iou_mean"],
            "val_obb_iou": best_val_metric["obb_iou_mean"],
            "note": "init_refine",
        },
    ]

    rng = np.random.default_rng(args.seed)
    search_start = time.time()
    total_evals = 1
    no_improve_rounds = 0
    current_round = 2
    round_pbar = tqdm(total=max(0, args.max_rounds - 1), desc="Optimization rounds")
    while current_round <= args.max_rounds:
        elapsed = time.time() - search_start
        if elapsed >= args.max_search_seconds:
            print(f"[Stop] Reached time limit: {elapsed:.1f}s >= {args.max_search_seconds:.1f}s")
            break
        if total_evals >= args.max_total_evals:
            print(f"[Stop] Reached eval limit: {total_evals} >= {args.max_total_evals}")
            break

        print(f"[Round {current_round}] Search candidates...")
        round_best_metric = best_val_metric["obb_iou_mean"]
        round_best_score = best_val_score
        round_best_params = best_params
        round_best_pred = best_pred
        improved = False

        for _ in tqdm(range(args.candidates_per_round), desc=f"Round {current_round} candidates", leave=False):
            elapsed = time.time() - search_start
            if elapsed >= args.max_search_seconds or total_evals >= args.max_total_evals:
                break

            cand = _sample_candidate(rng=rng, best=best_params, local_sigma=args.local_search_sigma)
            cand_pred = _refine_predictions(
                train_manifest + val_manifest,
                baseline,
                cand,
                progress_desc=f"Refine R{current_round}",
            )
            cand_val = _evaluate(val_manifest, cand_pred, args.iou_threshold)
            cand_score = _metric_score(cand_val)
            total_evals += 1
            if cand_score > round_best_score:
                round_best_metric = cand_val["obb_iou_mean"]
                round_best_score = cand_score
                round_best_params = cand
                round_best_pred = cand_pred
                improved = True

        best_params = round_best_params
        best_pred = round_best_pred
        best_val_metric = _evaluate(val_manifest, best_pred, args.iou_threshold)
        best_val_score = _metric_score(best_val_metric)
        best_train_metric = _evaluate(train_manifest, best_pred, args.iou_threshold)

        history.append(
            {
                "round": current_round,
                "params": best_params.__dict__,
                "train_obb_iou": best_train_metric["obb_iou_mean"],
                "val_obb_iou": best_val_metric["obb_iou_mean"],
                "note": "optimized",
            }
        )
        print(
            f"[Round {current_round}] best val OBB IoU: {best_val_metric['obb_iou_mean']:.4f}, "
            f"params={best_params}"
        )
        round_pbar.update(1)

        if improved:
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
            print(f"[Round {current_round}] no improvement count: {no_improve_rounds}/{args.early_stop_rounds}")

        if no_improve_rounds >= args.early_stop_rounds:
            print(f"[Stop] Early stop triggered after {no_improve_rounds} rounds without improvement")
            break

        if best_val_metric["obb_iou_mean"] >= args.target_obb_iou:
            break
        current_round += 1
    round_pbar.close()

    optimized_path = output_dir / "multiview_pca_pred.json"
    baseline_full_metric = _evaluate(manifest, baseline, args.iou_threshold)
    optimized_full_metric = _evaluate(manifest, best_pred, args.iou_threshold)
    if _metric_score(optimized_full_metric) >= _metric_score(baseline_full_metric):
        final_pred = best_pred
        final_note = "optimized"
    else:
        final_pred = baseline
        final_note = "fallback_to_baseline"
    _boxes_to_json(final_pred, optimized_path)

    history_path = output_dir / "optimization_history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False))

    csv_path = output_dir / "optimization_history.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "train_obb_iou", "val_obb_iou", "note", "params"])
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "round": row["round"],
                    "train_obb_iou": row["train_obb_iou"],
                    "val_obb_iou": row["val_obb_iou"],
                    "note": row["note"],
                    "params": json.dumps(row["params"], ensure_ascii=False),
                }
            )

    print("[Stage] Plotting optimization curve and running final benchmark...")
    _plot_history(history, output_dir / "optimization_curve.png")

    _run_benchmark(
        manifest_path=manifest_path,
        baseline_path=baseline_path,
        optimized_path=optimized_path,
        output_dir=output_dir / "final_benchmark",
        python_executable=args.python_executable,
    )

    final_metric = _evaluate(manifest, final_pred, args.iou_threshold)
    summary = {
        "baseline_val": base_val,
        "optimized_params": best_params.__dict__,
        "final_note": final_note,
        "baseline_full": baseline_full_metric,
        "optimized_final": final_metric,
        "history_file": str(history_path),
        "benchmark_dir": str(output_dir / "final_benchmark"),
    }
    (output_dir / "auto_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Auto optimization finished. Results at: {output_dir}")


if __name__ == "__main__":
    main()