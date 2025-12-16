"""
小規模測試：訓練前 3 個 situations
驗證訓練流程和權重更新邏輯
"""

import sys
import os

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.art1 import ART1Network
from data.industrial_data import get_data


def test_three_situations():
    """測試前 3 個 situations"""

    # 載入資料
    X_all, labels = get_data()
    X_test = X_all[:3]  # 只取前 3 個

    print("\n" + "=" * 80)
    print("小規模測試：前 3 個 Situations")
    print("=" * 80)

    # 測試不同的 ρ 值
    rho_values = [0.5, 0.7]

    for rho in rho_values:
        print(f"\n{'='*80}")
        print(f"測試 ρ = {rho}")
        print(f"{'='*80}\n")

        # 創建網路
        art = ART1Network(n_input=16, max_F2=20, rho=rho)

        # 訓練
        art.train(X_test, verbose=True)

        # 印出摘要
        art.print_summary()

        # 驗證聚類結果
        clusters = art.get_clustering_result()
        print(f"\n驗證:")
        print(f"  總 situation 數: 3")
        print(f"  創建類別數: {art.m}")
        print(
            f"  所有 situation 是否都分類: {sum(len(v) for v in clusters.values()) == 3}"
        )

        print("\n" + "=" * 80)


if __name__ == "__main__":
    test_three_situations()
