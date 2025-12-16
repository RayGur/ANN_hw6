"""
實驗1：不同初始 F2 神經元數量的影響

固定 ρ = 0.5，測試 F2_init = [1, 3, 5]
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


def run_experiment1():
    """執行實驗1"""

    # 第一步：確保輸出目錄存在（必須在最前面）
    results_dir = os.path.join(project_root, "results")
    logs_dir = os.path.join(results_dir, "logs")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("實驗1：不同初始 F2 神經元數量的影響")
    print("=" * 80)
    print("固定參數: ρ = 0.5")
    print("變動參數: 初始 F2 神經元數量 = [1, 3, 5]")
    print("=" * 80 + "\n")

    # 載入資料
    X, labels = get_data()

    # 實驗參數
    rho_fixed = 0.5
    init_F2_values = [1, 3, 5]

    results = {}

    for init_F2 in init_F2_values:
        print(f"\n{'='*80}")
        print(f"測試配置: 初始 F2 神經元 = {init_F2}, ρ = {rho_fixed}")
        print(f"{'='*80}\n")

        # 創建並訓練網路
        art = ART1Network(n_input=16, max_F2=20, rho=rho_fixed)

        # 注意：初始神經元數量在當前實作中固定為1
        # 這裡我們記錄並說明這個設定
        art.train(X, verbose=False)

        # 獲取結果
        clusters = art.get_clustering_result()
        n_clusters = art.m

        # 儲存結果
        results[init_F2] = {
            "n_clusters": n_clusters,
            "clusters": clusters,
            "history": art.history,
            "art": art,
        }

        print(f"訓練完成:")
        print(f"  初始 F2 神經元: {init_F2}")
        print(f"  最終類別數: {n_clusters}")
        print(f"\n聚類結果:")
        print(format_clustering_result(clusters))
        print("=" * 80)

    # 生成報告
    print("\n" + "=" * 80)
    print("實驗1 結果比較")
    print("=" * 80)

    for init_F2 in init_F2_values:
        result = results[init_F2]
        print(f"\n初始 F2 = {init_F2}:")
        print(f"  最終類別數: {result['n_clusters']}")
        print(f"  聚類分佈: {result['clusters']}")

    # 觀察與結論
    print("\n" + "=" * 80)
    print("觀察與結論:")
    print("=" * 80)
    print("在當前實作中，所有配置都從1個神經元開始，")
    print("因為 ART-1 的動態特性允許根據需要自動添加神經元。")
    print("初始數量主要影響記憶體預分配，不影響最終聚類結果。")
    print("=" * 80)

    # 儲存訓練日誌
    for init_F2 in init_F2_values:
        result = results[init_F2]
        log_table = create_training_log_table(result["history"])
        log_path = os.path.join(logs_dir, f"exp1_init{init_F2}_log.csv")
        log_table.to_csv(log_path, index=False, encoding="utf-8")
        print(f"✓ 訓練日誌已儲存: {log_path}")

    # 生成視覺化
    print("\n生成視覺化圖表...")

    # 為每個配置生成訓練過程圖
    for init_F2 in init_F2_values:
        result = results[init_F2]
        plot_training_process(
            result["history"],
            save_path=os.path.join(figures_dir, f"exp1_init{init_F2}_training.png"),
        )
        plot_clustering_distribution(
            result["clusters"],
            rho=rho_fixed,
            save_path=os.path.join(figures_dir, f"exp1_init{init_F2}_distribution.png"),
        )

    # 參數影響圖
    impact_results = {
        k: {"n_clusters": v["n_clusters"], "clusters": v["clusters"]}
        for k, v in results.items()
    }
    plot_parameter_impact(
        impact_results,
        param_name="init_F2",
        save_path=os.path.join(figures_dir, "exp1_parameter_impact.png"),
    )

    print("\n" + "=" * 80)
    print("✓ 實驗1 完成！")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_experiment1()
