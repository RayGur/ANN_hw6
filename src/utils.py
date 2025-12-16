"""
輔助函數模組
"""

import numpy as np
import pandas as pd
from typing import Dict, List


def create_training_log_table(history: List[Dict]) -> pd.DataFrame:
    """
    將訓練歷程轉換為表格

    Parameters:
    -----------
    history : list
        訓練歷程記錄

    Returns:
    --------
    df : DataFrame
        訓練日誌表格
    """
    df = pd.DataFrame(history)
    df = df.rename(
        columns={
            "situation_id": "Situation",
            "winner": "Cluster",
            "R_value": "Similarity (R)",
            "action": "Action",
            "n_active": "Active Clusters",
        }
    )
    return df


def format_clustering_result(clusters: Dict[int, List[int]]) -> str:
    """
    格式化聚類結果

    Parameters:
    -----------
    clusters : dict
        聚類結果 {cluster_id: [situation_ids]}

    Returns:
    --------
    result : str
        格式化的字串
    """
    result = []
    for cluster_id in sorted(clusters.keys()):
        situations = clusters[cluster_id]
        result.append(f"Cluster {cluster_id}: {situations}")
    return "\n".join(result)


def calculate_cluster_statistics(clusters: Dict[int, List[int]]) -> Dict:
    """
    計算聚類統計資訊

    Parameters:
    -----------
    clusters : dict
        聚類結果

    Returns:
    --------
    stats : dict
        統計資訊
    """
    n_clusters = len(clusters)
    cluster_sizes = {k: len(v) for k, v in clusters.items()}

    stats = {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "min_size": min(cluster_sizes.values()) if cluster_sizes else 0,
        "max_size": max(cluster_sizes.values()) if cluster_sizes else 0,
        "avg_size": np.mean(list(cluster_sizes.values())) if cluster_sizes else 0,
    }

    return stats


def compare_clustering_results(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    比較不同參數設定的聚類結果

    Parameters:
    -----------
    results : dict
        {config_name: {'clusters': {...}, 'n_clusters': ..., ...}}

    Returns:
    --------
    df : DataFrame
        比較表格
    """
    comparison = []

    for config_name, result in results.items():
        stats = calculate_cluster_statistics(result["clusters"])
        comparison.append(
            {
                "Configuration": config_name,
                "Number of Clusters": stats["n_clusters"],
                "Min Cluster Size": stats["min_size"],
                "Max Cluster Size": stats["max_size"],
                "Avg Cluster Size": f"{stats['avg_size']:.2f}",
            }
        )

    df = pd.DataFrame(comparison)
    return df


def save_results_to_file(filepath: str, content: str):
    """
    儲存結果到檔案

    Parameters:
    -----------
    filepath : str
        檔案路徑
    content : str
        內容
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✓ 結果已儲存至: {filepath}")
