"""
完整測試：訓練全部 10 個 situations
驗證與簡報結果的一致性
"""

import sys
import os

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.art1 import ART1Network
from data.industrial_data import get_data


def compare_with_slides():
    """與簡報結果比對"""

    # 簡報的參考結果
    # ρ = 0.4: 6 個類別, 分類為 [1,2,3,8], [4,9], [5], [6], [7], [10]
    # ρ = 0.7: 10 個類別 (每個 situation 一個類別)

    slides_results = {
        0.4: {
            "n_clusters": 6,
            "expected_groups": "Cluster-1: [1,2,3,8], Cluster-2: [4,9], ...",
        },
        0.7: {"n_clusters": 10, "expected_groups": "每個 situation 獨立一個類別"},
    }

    # 載入資料
    X, labels = get_data()

    print("\n" + "=" * 80)
    print("完整測試：全部 10 個 Situations")
    print("與簡報結果比對")
    print("=" * 80)

    for rho in [0.4, 0.7]:
        print(f"\n{'='*80}")
        print(f"測試 ρ = {rho}")
        print(f"{'='*80}\n")

        # 創建並訓練網路
        art = ART1Network(n_input=16, max_F2=20, rho=rho)
        art.train(X, verbose=False)  # 不顯示詳細過程

        # 獲取結果
        clusters = art.get_clustering_result()
        n_clusters = art.m

        # 印出摘要
        print(f"訓練完成:")
        print(f"  警戒參數 ρ: {rho}")
        print(f"  創建的類別數: {n_clusters}")
        print(f"\n聚類結果:")
        for cluster_id in sorted(clusters.keys()):
            situations = clusters[cluster_id]
            print(f"  Cluster {cluster_id}: {situations}")

        # 與簡報比對
        print(f"\n與簡報比對:")
        expected = slides_results[rho]
        print(f"  簡報預期類別數: {expected['n_clusters']}")
        print(f"  實際類別數: {n_clusters}")

        if n_clusters == expected["n_clusters"]:
            print(f"  ✓ 類別數量一致！")
        else:
            print(f"  ⚠ 類別數量不同（可能因初始條件或實作細節差異）")

        print(f"\n  簡報預期分組: {expected['expected_groups']}")

        print("=" * 80)


def detailed_test_rho04():
    """詳細測試 ρ=0.4，顯示完整訓練過程"""

    X, labels = get_data()

    print("\n" + "=" * 80)
    print("詳細訓練過程: ρ = 0.4")
    print("=" * 80 + "\n")

    art = ART1Network(n_input=16, max_F2=20, rho=0.4)
    art.train(X, verbose=True)
    art.print_summary()


def detailed_test_rho07():
    """詳細測試 ρ=0.7，顯示完整訓練過程"""

    X, labels = get_data()

    print("\n" + "=" * 80)
    print("詳細訓練過程: ρ = 0.7")
    print("=" * 80 + "\n")

    art = ART1Network(n_input=16, max_F2=20, rho=0.7)
    art.train(X, verbose=True)
    art.print_summary()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["compare", "detail04", "detail07", "all"],
        default="compare",
        help="測試模式",
    )
    args = parser.parse_args()

    if args.mode == "compare":
        compare_with_slides()
    elif args.mode == "detail04":
        detailed_test_rho04()
    elif args.mode == "detail07":
        detailed_test_rho07()
    elif args.mode == "all":
        compare_with_slides()
        detailed_test_rho04()
        detailed_test_rho07()
