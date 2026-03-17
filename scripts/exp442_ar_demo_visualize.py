#!/usr/bin/env python3
"""
Generate visualization figures for AR demo results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator

# 配置中文字体 - 优先使用 Noto Sans CJK
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Sans CJK TC', 'SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.size'] = 10


def load_metrics_csv(csv_path: Path) -> tuple[list[dict], dict]:
    """Load and aggregate metrics."""
    samples = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                "scene_id": row["scene_id"],
                "aabb_iou": float(row["aabb_iou_mean"]),
                "obb_iou": float(row["obb_iou_mean"]),
                "centroid_cm": float(row["centroid_cm_mean"]),
                "recall_025": float(row["recall_025"]),
                "recall_050": float(row["recall_050"]),
            })
    
    aabb_ious = [s["aabb_iou"] for s in samples]
    obb_ious = [s["obb_iou"] for s in samples]
    centroids = [s["centroid_cm"] for s in samples]
    
    stats = {
        "aabb_mean": np.mean(aabb_ious),
        "aabb_std": np.std(aabb_ious),
        "obb_mean": np.mean(obb_ious),
        "obb_std": np.std(obb_ious),
        "centroid_mean": np.mean(centroids),
        "centroid_std": np.std(centroids),
    }
    
    return samples, stats


def plot_metric_comparison(single_stats: dict, multi_stats: dict, out_path: Path) -> None:
    """Generate comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("N3D-VLM: 单视图 vs 多视图性能对比", fontsize=14, fontweight='bold')
    
    methods = ["单视图\nVLM", "多视图\n+PCA/OBB"]
    
    # AABB IoU
    ax = axes[0]
    aabb_vals = [single_stats["aabb_mean"], multi_stats["aabb_mean"]]
    aabb_errs = [single_stats["aabb_std"], multi_stats["aabb_std"]]
    colors = ["#FF7F0E", "#2CA02C"]
    bars = ax.bar(methods, aabb_vals, yerr=aabb_errs, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel("AABB IoU", fontsize=11, fontweight='bold')
    ax.set_ylim([0, 0.7])
    ax.axhline(y=0.45, color='red', linestyle='--', linewidth=2, label='AR 最小需求 (0.45)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val, err) in enumerate(zip(bars, aabb_vals, aabb_errs)):
        ax.text(bar.get_x() + bar.get_width()/2, val + err + 0.02, f"{val:.4f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # OBB IoU
    ax = axes[1]
    obb_vals = [single_stats["obb_mean"], multi_stats["obb_mean"]]
    obb_errs = [single_stats["obb_std"], multi_stats["obb_std"]]
    bars = ax.bar(methods, obb_vals, yerr=obb_errs, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel("OBB IoU（带朝向）", fontsize=11, fontweight='bold')
    ax.set_ylim([0, 0.7])
    ax.axhline(y=0.50, color='red', linestyle='--', linewidth=2, label='AR 最小需求 (0.50)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val, err) in enumerate(zip(bars, obb_vals, obb_errs)):
        ax.text(bar.get_x() + bar.get_width()/2, val + err + 0.02, f"{val:.4f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Centroid Error
    ax = axes[2]
    cent_vals = [single_stats["centroid_mean"], multi_stats["centroid_mean"]]
    cent_errs = [single_stats["centroid_std"], multi_stats["centroid_std"]]
    bars = ax.bar(methods, cent_vals, yerr=cent_errs, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel("质心误差 (cm)", fontsize=11, fontweight='bold')
    ax.set_ylim([0, 20])
    ax.axhline(y=15, color='red', linestyle='--', linewidth=2, label='AR 最大允许 (15cm)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val, err) in enumerate(zip(bars, cent_vals, cent_errs)):
        ax.text(bar.get_x() + bar.get_width()/2, val + err + 0.5, f"{val:.2f}", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Viz] Saved metric comparison: {out_path}")
    plt.close()


def plot_performance_distribution(single_samples: list[dict], multi_samples: list[dict], out_path: Path) -> None:
    """Generate violin plot showing distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("N3D-VLM: 场景级性能分布", fontsize=14, fontweight='bold')
    
    # AABB IoU Distribution
    ax = axes[0]
    single_aabb = [s["aabb_iou"] for s in single_samples]
    multi_aabb = [s["aabb_iou"] for s in multi_samples]
    
    bp = ax.boxplot([single_aabb, multi_aabb], tick_labels=["单视图", "多视图"], patch_artist=True,
                     widths=0.6, meanline=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#FF7F0E', '#2CA02C']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("AABB IoU", fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.45, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # OBB IoU Distribution
    ax = axes[1]
    single_obb = [s["obb_iou"] for s in single_samples]
    multi_obb = [s["obb_iou"] for s in multi_samples]
    
    bp = ax.boxplot([single_obb, multi_obb], tick_labels=["单视图", "多视图"], patch_artist=True,
                     widths=0.6, meanline=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#FF7F0E', '#2CA02C']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("OBB IoU", fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.50, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Centroid Distribution
    ax = axes[2]
    single_cent = [s["centroid_cm"] for s in single_samples]
    multi_cent = [s["centroid_cm"] for s in multi_samples]
    
    bp = ax.boxplot([single_cent, multi_cent], tick_labels=["单视图", "多视图"], patch_artist=True,
                     widths=0.6, meanline=True, showmeans=True)
    for patch, color in zip(bp['boxes'], ['#FF7F0E', '#2CA02C']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("质心误差 (cm)", fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=15, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Viz] Saved distribution plot: {out_path}")
    plt.close()


def plot_improvement_analysis(single_samples: list[dict], multi_samples: list[dict], out_path: Path) -> None:
    """Generate improvement heatmap-style visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("N3D-VLM: 多视图融合的改进分析", fontsize=14, fontweight='bold')
    
    # OBB IoU Improvement per Scene
    ax = axes[0, 0]
    scene_ids = [s["scene_id"] for s in multi_samples]
    improvements = [
        (multi_samples[i]["obb_iou"] - single_samples[i]["obb_iou"]) * 100
        for i in range(len(multi_samples))
    ]
    colors_imp = ['#2CA02C' if x > 0 else '#FF7F0E' for x in improvements]
    
    x_pos = np.arange(len(scene_ids))
    bars = ax.bar(x_pos, improvements, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel("场景", fontsize=10, fontweight='bold')
    ax.set_ylabel("OBB IoU 改进 (%)", fontsize=10, fontweight='bold')
    ax.set_title("单个场景的改进幅度", fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos[::5])
    ax.set_xticklabels(scene_ids[::5], rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Cumulative improvement
    ax = axes[0, 1]
    improvement_obb = np.array([multi_samples[i]["obb_iou"] - single_samples[i]["obb_iou"] 
                                 for i in range(len(multi_samples))])
    sorted_imp = np.sort(improvement_obb)
    cumsum = np.cumsum(sorted_imp)
    
    ax.plot(cumsum, linewidth=2.5, marker='o', markersize=4, color='#2CA02C')
    ax.fill_between(range(len(cumsum)), cumsum, alpha=0.3, color='#2CA02C')
    ax.set_xlabel("场景（按改进排序）", fontsize=10, fontweight='bold')
    ax.set_ylabel("累计 OBB IoU 改进", fontsize=10, fontweight='bold')
    ax.set_title("改进的累积效果", fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Scatter: Single-view vs Multi-view
    ax = axes[1, 0]
    ax.scatter([s["obb_iou"] for s in single_samples], 
               [m["obb_iou"] for m in multi_samples],
               s=80, alpha=0.6, color='#1f77b4', edgecolor='black', linewidth=0.5)
    
    # Add diagonal line (no improvement)
    min_val = min([s["obb_iou"] for s in single_samples] + [m["obb_iou"] for m in multi_samples])
    max_val = max([s["obb_iou"] for s in single_samples] + [m["obb_iou"] for m in multi_samples])
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='无改进线')
    
    ax.set_xlabel("单视图 OBB IoU", fontsize=10, fontweight='bold')
    ax.set_ylabel("多视图 OBB IoU", fontsize=10, fontweight='bold')
    ax.set_title("单视图 vs 多视图对比", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # AR Feasibility per metric
    ax = axes[1, 1]
    metrics = ["AABB IoU\n(>0.45)", "OBB IoU\n(>0.50)", "质心误差\n(<15cm)"]
    
    single_passes = [
        sum(1 for s in single_samples if s["aabb_iou"] > 0.45) / len(single_samples) * 100,
        sum(1 for s in single_samples if s["obb_iou"] > 0.50) / len(single_samples) * 100,
        sum(1 for s in single_samples if s["centroid_cm"] < 15) / len(single_samples) * 100,
    ]
    
    multi_passes = [
        sum(1 for m in multi_samples if m["aabb_iou"] > 0.45) / len(multi_samples) * 100,
        sum(1 for m in multi_samples if m["obb_iou"] > 0.50) / len(multi_samples) * 100,
        sum(1 for m in multi_samples if m["centroid_cm"] < 15) / len(multi_samples) * 100,
    ]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, single_passes, width, label='单视图', 
                   color='#FF7F0E', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x_pos + width/2, multi_passes, width, label='多视图',
                   color='#2CA02C', alpha=0.7, edgecolor='black')
    
    ax.set_ylabel("满足 AR 需求的场景比例 (%)", fontsize=10, fontweight='bold')
    ax.set_title("AR 可行性评估", fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim([0, 110])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[Viz] Saved improvement analysis: {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AR demo visualizations")
    parser.add_argument("--single_view_csv", type=str, required=True)
    parser.add_argument("--multiview_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/exp442_ar_demo/figures")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    single_samples, single_stats = load_metrics_csv(Path(args.single_view_csv))
    multi_samples, multi_stats = load_metrics_csv(Path(args.multiview_csv))
    
    print("[Stage] Loaded metrics for visualization")
    
    # Generate plots
    plot_metric_comparison(single_stats, multi_stats, out_dir / "figure_metric_comparison.png")
    plot_performance_distribution(single_samples, multi_samples, out_dir / "figure_distribution.png")
    plot_improvement_analysis(single_samples, multi_samples, out_dir / "figure_improvement.png")
    
    print(f"\n[Stage] All visualizations saved to: {out_dir}")
    print(f"  - Metric Comparison: figure_metric_comparison.png")
    print(f"  - Distribution: figure_distribution.png")
    print(f"  - Improvement Analysis: figure_improvement.png")


if __name__ == "__main__":
    main()
