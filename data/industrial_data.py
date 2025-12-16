"""
工業流程分類資料集
來源：Neural Network HW6 - Classification of industry process by ART-1

資料描述：
- 10 個不同的系統行為情境 (situations)
- 每個情境由 16 個二元狀態變數描述
- 目標：使用 ART-1 將相似情境分類成群組
"""

import numpy as np

# 10 situations × 16 status variables
X_train = np.array(
    [
        [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # Situation 1
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],  # Situation 2
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],  # Situation 3
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0],  # Situation 4
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],  # Situation 5
        [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # Situation 6
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0],  # Situation 7
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],  # Situation 8
        [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],  # Situation 9
        [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1],  # Situation 10
    ],
    dtype=np.int32,
)

# 情境標籤（用於記錄和可視化）
situation_labels = [f"Situation {i+1}" for i in range(10)]


def get_data():
    """
    返回訓練資料

    Returns:
        X_train: shape (10, 16) 的二元矩陣
        labels: 情境標籤列表
    """
    return X_train.copy(), situation_labels.copy()


def print_data_info():
    """印出資料集基本資訊"""
    print("=" * 60)
    print("工業流程分類資料集")
    print("=" * 60)
    print(f"樣本數量: {X_train.shape[0]}")
    print(f"特徵維度: {X_train.shape[1]}")
    print(f"資料類型: 二元向量 (0/1)")
    print(f"\n前 3 個樣本:")
    for i in range(3):
        print(f"{situation_labels[i]}: {X_train[i]}")
    print("=" * 60)


if __name__ == "__main__":
    print_data_info()

    # 驗證資料完整性
    assert X_train.shape == (10, 16), "資料維度錯誤"
    assert np.all((X_train == 0) | (X_train == 1)), "資料必須是二元的"
    print("\n✓ 資料驗證通過")
