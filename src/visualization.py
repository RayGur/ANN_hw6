"""
視覺化模組
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # 使用非互動式後端
import seaborn as sns
from typing import Dict, List
import pandas as pd

# 設定中文字型
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_training_process(history: List[Dict], save_path: str = None):
    """
    繪製訓練過程圖

    Parameters:
    -----------
    history : list
        訓練歷程
    save_path : str
        儲存路徑
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    situations = [h["situation_id"] for h in history]
    clusters = [h["winner"] for h in history]
    similarities = [h["R_value"] for h in history]
    actions = [h["action"] for h in history]

    # 圖1: Situation 分配到的 Cluster
    colors = ["green" if a == "update" else "red" for a in actions]
    ax1.scatter(situations, clusters, c=colors, s=100, alpha=0.6)
    ax1.set_xlabel("Situation ID", fontsize=12)
    ax1.set_ylabel("Cluster ID", fontsize=12)
    ax1.set_title("Situation Assignment to Clusters", fontsize=14)
    ax1.grid(True, alpha=0.3)

    # 圖例
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", alpha=0.6, label="Update"),
        Patch(facecolor="red", alpha=0.6, label="Add"),
    ]
    ax1.legend(handles=legend_elements, loc="best")

    # 圖2: 相似度 R 值
    ax2.plot(situations, similarities, marker="o", linestyle="-", linewidth=2)
    ax2.axhline(
        y=history[0]["R_value"] if history else 0,
        color="r",
        linestyle="--",
        label="Vigilance ρ",
        linewidth=1.5,
    )
    ax2.set_xlabel("Situation ID", fontsize=12)
    ax2.set_ylabel("Similarity (R)", fontsize=12)
    ax2.set_title("Similarity Values During Training", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 訓練過程圖已儲存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_weight_matrix(W_b: np.ndarray, neuron_idx: int, save_path: str = None):
    """
    繪製權重矩陣熱圖

    Parameters:
    -----------
    W_b : ndarray
        反饋權重矩陣
    neuron_idx : int
        神經元索引
    save_path : str
        儲存路徑
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 只顯示指定神經元的權重
    weights = W_b[:, neuron_idx].reshape(1, -1)

    sns.heatmap(
        weights,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Weight Value"},
        xticklabels=[f"x{i+1}" for i in range(weights.shape[1])],
        yticklabels=[f"Neuron {neuron_idx}"],
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_title(f"Weight Vector W^(b) for Neuron {neuron_idx}", fontsize=14)
    ax.set_xlabel("Input Features", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 權重矩陣圖已儲存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_clustering_distribution(
    clusters: Dict[int, List[int]], rho: float, save_path: str = None
):
    """
    繪製聚類分佈圖

    Parameters:
    -----------
    clusters : dict
        聚類結果
    rho : float
        警戒參數
    save_path : str
        儲存路徑
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    cluster_ids = sorted(clusters.keys())
    cluster_sizes = [len(clusters[k]) for k in cluster_ids]

    bars = ax.bar(cluster_ids, cluster_sizes, color="steelblue", alpha=0.7)

    # 在柱狀圖上標註數值
    for bar, size in zip(bars, cluster_sizes):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Cluster ID", fontsize=12)
    ax.set_ylabel("Number of Situations", fontsize=12)
    ax.set_title(
        f"Clustering Distribution (ρ = {rho:.2f}, Total Clusters = {len(cluster_ids)})",
        fontsize=14,
    )
    ax.set_xticks(cluster_ids)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 聚類分佈圖已儲存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_parameter_impact(
    results: Dict[str, Dict], param_name: str, save_path: str = None
):
    """
    繪製參數影響圖

    Parameters:
    -----------
    results : dict
        實驗結果 {param_value: {'n_clusters': ..., 'clusters': {...}}}
    param_name : str
        參數名稱 ('rho' 或 'init_F2')
    save_path : str
        儲存路徑
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    params = []
    n_clusters_list = []

    for key, result in results.items():
        params.append(key)
        n_clusters_list.append(result["n_clusters"])

    # 排序
    sorted_indices = np.argsort(params)
    params = [params[i] for i in sorted_indices]
    n_clusters_list = [n_clusters_list[i] for i in sorted_indices]

    ax.plot(
        params,
        n_clusters_list,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color="steelblue",
    )

    # 標註數值
    for x, y in zip(params, n_clusters_list):
        ax.text(x, y, f"{int(y)}", ha="center", va="bottom", fontsize=10)

    if param_name == "rho":
        ax.set_xlabel("Vigilance Parameter (ρ)", fontsize=12)
        ax.set_title("Impact of Vigilance Parameter on Number of Clusters", fontsize=14)
    else:
        ax.set_xlabel("Initial F2 Neurons", fontsize=12)
        ax.set_title("Impact of Initial F2 Neurons on Number of Clusters", fontsize=14)

    ax.set_ylabel("Number of Clusters", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 參數影響圖已儲存: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison_heatmap(results: Dict, save_path: str = None):
    """
    繪製實驗結果比較熱圖

    Parameters:
    -----------
    results : dict
        實驗結果 {(init_F2, rho): n_clusters}
    save_path : str
        儲存路徑
    """
    # 提取參數和結果
    init_F2_values = sorted(set([k[0] for k in results.keys()]))
    rho_values = sorted(set([k[1] for k in results.keys()]))

    # 建立矩陣
    matrix = np.zeros((len(init_F2_values), len(rho_values)))

    for i, init_F2 in enumerate(init_F2_values):
        for j, rho in enumerate(rho_values):
            if (init_F2, rho) in results:
                matrix[i, j] = results[(init_F2, rho)]

    # 繪圖
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        xticklabels=[f"{r:.2f}" for r in rho_values],
        yticklabels=[f"{f}" for f in init_F2_values],
        cbar_kws={"label": "Number of Clusters"},
        ax=ax,
    )

    ax.set_xlabel("Vigilance Parameter (ρ)", fontsize=12)
    ax.set_ylabel("Initial F2 Neurons", fontsize=12)
    ax.set_title("Number of Clusters: Impact of Parameters", fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ 比較熱圖已儲存: {save_path}")
    else:
        plt.show()

    plt.close()
