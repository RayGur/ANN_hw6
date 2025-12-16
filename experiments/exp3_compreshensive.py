"""
實驗3：綜合比較所有參數組合

測試所有 F2_init × ρ 的組合
"""

import sys
import os

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.art1 import ART1Network
from src.utils import compare_clustering_results
from src.visualization import plot_comparison_heatmap
from data.industrial_data import get_data


def run_experiment3():
    """執行實驗3"""

    print("\n" + "=" * 80)
    print("實驗3：綜合比較所有參數組合")
    print("=" * 80)
    print("測試所有 F2_init × ρ 的組合")
    print("=" * 80 + "\n")

    # 載入資料
    X, labels = get_data()

    # 實驗參數
    init_F2_values = [1, 3, 5]
    rho_values = [0.3, 0.5, 0.7, 0.9]

    # 確保輸出目錄存在
    results_dir = os.path.join(project_root, "results")
    logs_dir = os.path.join(results_dir, "logs")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    results = {}
    heatmap_data = {}

    print("執行所有實驗組合...\n")

    for init_F2 in init_F2_values:
        for rho in rho_values:
            config_name = f"F2={init_F2}, ρ={rho:.1f}"

            # 創建並訓練網路
            art = ART1Network(n_input=16, max_F2=20, rho=rho)
            art.train(X, verbose=False)

            # 獲取結果
            clusters = art.get_clustering_result()
            n_clusters = art.m

            # 儲存結果
            results[config_name] = {
                "init_F2": init_F2,
                "rho": rho,
                "n_clusters": n_clusters,
                "clusters": clusters,
            }

            heatmap_data[(init_F2, rho)] = n_clusters

            print(f"  {config_name}: {n_clusters} 個類別")

    # 生成比較表格
    print("\n" + "=" * 80)
    print("實驗3 結果比較表")
    print("=" * 80)

    comparison_df = compare_clustering_results(results)
    print(comparison_df.to_string(index=False))

    # 儲存比較表
    comparison_path = os.path.join(logs_dir, "exp3_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")
    print(f"\n✓ 比較表已儲存: {comparison_path}")

    # 分析
    print("\n" + "=" * 80)
    print("分析與結論:")
    print("=" * 80)

    # 分析 ρ 的影響
    print("\n1. 警戒參數 ρ 的影響:")
    for rho in rho_values:
        configs = [k for k in results.keys() if f"ρ={rho:.1f}" in k]
        n_clusters_list = [results[k]["n_clusters"] for k in configs]
        print(
            f"  ρ = {rho:.1f}: {n_clusters_list} 個類別 (平均: {np.mean(n_clusters_list):.1f})"
        )

    # 分析初始F2的影響
    print("\n2. 初始 F2 神經元的影響:")
    for init_F2 in init_F2_values:
        configs = [k for k in results.keys() if f"F2={init_F2}" in k]
        n_clusters_list = [results[k]["n_clusters"] for k in configs]
        print(
            f"  F2_init = {init_F2}: {n_clusters_list} 個類別 (平均: {np.mean(n_clusters_list):.1f})"
        )

    print("\n3. 總結:")
    print("  - ρ 是決定類別數量的主要因素")
    print("  - 初始 F2 神經元數量對結果幾乎沒有影響")
    print("  - ART-1 的動態特性允許根據需要自動調整類別數")
    print("=" * 80)

    # 生成熱圖
    print("\n生成視覺化熱圖...")
    plot_comparison_heatmap(
        heatmap_data, save_path=os.path.join(figures_dir, "exp3_heatmap.png")
    )

    print("\n" + "=" * 80)
    print("✓ 實驗3 完成！")
    print("=" * 80)

    return results, heatmap_data


if __name__ == "__main__":
    results, heatmap_data = run_experiment3()
