"""
實驗2：不同警戒參數 ρ 的影響

固定 F2_init = 1，測試 ρ = [0.3, 0.5, 0.7, 0.9]
"""

import sys
import os

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.art1 import ART1Network
from src.utils import (
    create_training_log_table,
    format_clustering_result,
    save_results_to_file,
)
from src.visualization import (
    plot_training_process,
    plot_clustering_distribution,
    plot_parameter_impact,
)
from data.industrial_data import get_data


def run_experiment2():
    """執行實驗2"""

    print("\n" + "=" * 80)
    print("實驗2：不同警戒參數 ρ 的影響")
    print("=" * 80)
    print("固定參數: 初始 F2 神經元 = 1")
    print("變動參數: 警戒參數 ρ = [0.3, 0.5, 0.7, 0.9]")
    print("=" * 80 + "\n")

    # 載入資料
    X, labels = get_data()

    # 實驗參數
    rho_values = [0.3, 0.5, 0.7, 0.9]

    # 確保輸出目錄存在
    results_dir = os.path.join(project_root, "results")
    logs_dir = os.path.join(results_dir, "logs")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    results = {}

    for rho in rho_values:
        print(f"\n{'='*80}")
        print(f"測試配置: ρ = {rho}, 初始 F2 神經元 = 1")
        print(f"{'='*80}\n")

        # 創建並訓練網路
        art = ART1Network(n_input=16, max_F2=20, rho=rho)
        art.train(X, verbose=False)

        # 獲取結果
        clusters = art.get_clustering_result()
        n_clusters = art.m

        # 儲存結果
        results[rho] = {
            "n_clusters": n_clusters,
            "clusters": clusters,
            "history": art.history,
            "art": art,
        }

        print(f"訓練完成:")
        print(f"  警戒參數 ρ: {rho}")
        print(f"  最終類別數: {n_clusters}")
        print(f"\n聚類結果:")
        print(format_clustering_result(clusters))
        print("=" * 80)

    # 生成報告
    print("\n" + "=" * 80)
    print("實驗2 結果比較")
    print("=" * 80)

    for rho in rho_values:
        result = results[rho]
        print(f"\nρ = {rho}:")
        print(f"  最終類別數: {result['n_clusters']}")
        print(f"  聚類分佈:")
        for cluster_id in sorted(result["clusters"].keys()):
            situations = result["clusters"][cluster_id]
            print(f"    Cluster {cluster_id}: {situations}")

    # 觀察與結論
    print("\n" + "=" * 80)
    print("觀察與結論:")
    print("=" * 80)
    print("警戒參數 ρ 對聚類結果有顯著影響：")
    print(f"  ρ = 0.3 (低):  {results[0.3]['n_clusters']} 個類別 → 粗分類，較寬鬆")
    print(f"  ρ = 0.5 (中):  {results[0.5]['n_clusters']} 個類別 → 平衡分類")
    print(f"  ρ = 0.7 (高):  {results[0.7]['n_clusters']} 個類別 → 細分類")
    print(
        f"  ρ = 0.9 (極高): {results[0.9]['n_clusters']} 個類別 → 極細分類，趨近每個樣本一類"
    )
    print("\n警戒參數越高，對相似度要求越嚴格，產生的類別越多。")
    print("=" * 80)

    # 儲存訓練日誌
    for rho in rho_values:
        result = results[rho]
        log_table = create_training_log_table(result["history"])
        log_path = os.path.join(logs_dir, f"exp2_rho{rho:.1f}_log.csv")
        log_table.to_csv(log_path, index=False, encoding="utf-8")
        print(f"✓ 訓練日誌已儲存: {log_path}")

    # 生成視覺化
    print("\n生成視覺化圖表...")

    # 為每個配置生成訓練過程圖
    for rho in rho_values:
        result = results[rho]
        plot_training_process(
            result["history"],
            save_path=os.path.join(figures_dir, f"exp2_rho{rho:.1f}_training.png"),
        )
        plot_clustering_distribution(
            result["clusters"],
            rho=rho,
            save_path=os.path.join(figures_dir, f"exp2_rho{rho:.1f}_distribution.png"),
        )

    # 參數影響圖
    impact_results = {
        k: {"n_clusters": v["n_clusters"], "clusters": v["clusters"]}
        for k, v in results.items()
    }
    plot_parameter_impact(
        impact_results,
        param_name="rho",
        save_path=os.path.join(figures_dir, "exp2_parameter_impact.png"),
    )

    print("\n" + "=" * 80)
    print("✓ 實驗2 完成！")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_experiment2()
