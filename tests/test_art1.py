"""
ART-1 網路單元測試
"""

import sys
import os

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.art1 import ART1Network


def test_initialization():
    """測試權重初始化"""
    print("測試 1: 權重初始化")
    print("-" * 60)

    art = ART1Network(n_input=16, max_F2=20, rho=0.5)

    # 檢查 W^(f) 初始化
    expected_wf = 1.0 / (1.0 + 16)
    assert np.allclose(art.W_f[0, :], expected_wf), "W^(f) 初始化錯誤"
    print(f"✓ W^(f) 初始化正確: {expected_wf:.6f}")

    # 檢查 W^(b) 初始化
    assert np.all(art.W_b == 1.0), "W^(b) 初始化錯誤"
    print(f"✓ W^(b) 初始化正確: 全部為 1")

    # 檢查狀態
    assert art.status[0] == "active", "第一個神經元應為 active"
    assert art.status[1] == "inactive", "其他神經元應為 inactive"
    print(f"✓ 神經元狀態正確: active={art.m}, inactive={art.max_m - art.m}")

    print("=" * 60)
    print("✓ 測試 1 通過\n")


def test_activation_calculation():
    """測試激活值計算"""
    print("測試 2: 激活值計算")
    print("-" * 60)

    art = ART1Network(n_input=8, max_F2=10, rho=0.5)

    # 測試輸入
    x = np.array([1, 0, 1, 1, 1, 0, 1, 0])

    # 計算激活值
    u = art._calculate_activation(x)

    # 手動計算預期值：W^(f) = 1/9, x 有 5 個 1
    expected_u0 = 5 * (1.0 / 9.0)

    assert np.isclose(u[0], expected_u0), f"激活值計算錯誤: {u[0]} != {expected_u0}"
    print(f"✓ 激活值計算正確: u_0 = {u[0]:.6f} (預期: {expected_u0:.6f})")

    # 檢查 inactive 神經元
    assert u[1] == -np.inf, "inactive 神經元激活值應為 -inf"
    print(f"✓ inactive 神經元激活值正確: -inf")

    print("=" * 60)
    print("✓ 測試 2 通過\n")


def test_similarity_test():
    """測試相似度測試"""
    print("測試 3: 相似度測試")
    print("-" * 60)

    art = ART1Network(n_input=8, max_F2=10, rho=0.8)

    # 設定測試權重
    art.W_b[:, 0] = np.array([1, 1, 0, 0, 1, 1, 1, 0])

    # 測試輸入
    x = np.array([1, 0, 1, 1, 1, 0, 1, 0])

    # 執行相似度測試
    passed, R = art._similarity_test(x, k=0)

    # 手動計算: 共同的 1 有 3 個 (位置 0, 4, 6), x 有 5 個 1
    # R = 3 / 5 = 0.6
    expected_R = 3.0 / 5.0

    assert np.isclose(R, expected_R), f"相似度計算錯誤: {R} != {expected_R}"
    print(f"✓ 相似度計算正確: R = {R:.2f} (預期: {expected_R:.2f})")

    # 檢查判斷結果
    assert passed == (R >= 0.8), "相似度判斷錯誤"
    print(f"✓ 相似度判斷正確: R={R:.2f} >= ρ={0.8} → {passed}")

    print("=" * 60)
    print("✓ 測試 3 通過\n")


def test_weight_update():
    """測試權重更新"""
    print("測試 4: 權重更新")
    print("-" * 60)

    art = ART1Network(n_input=8, max_F2=10, rho=0.5)

    # 設定初始權重
    art.W_b[:, 0] = np.array([1, 0, 0, 1, 1, 1, 0, 1])

    # 測試輸入
    x = np.array([1, 0, 1, 1, 1, 0, 1, 1])

    # 執行權重更新
    art._update_weights(x, k=0)

    # 檢查 W^(b) 更新 (AND 操作)
    expected_wb = np.array([1, 0, 0, 1, 1, 0, 0, 1])
    assert np.array_equal(art.W_b[:, 0], expected_wb), "W^(b) 更新錯誤"
    print(f"✓ W^(b) 更新正確 (AND 操作)")
    print(f"  更新前: [1, 0, 0, 1, 1, 1, 0, 1]")
    print(f"  輸入 x: {x}")
    print(f"  更新後: {art.W_b[:, 0]}")

    # 檢查 W^(f) 更新
    numerator = expected_wb * x
    denominator = 0.5 + np.sum(expected_wb * x)
    expected_wf = numerator / denominator

    assert np.allclose(art.W_f[0, :], expected_wf), "W^(f) 更新錯誤"
    print(f"✓ W^(f) 更新正確")

    print("=" * 60)
    print("✓ 測試 4 通過\n")


def test_single_situation():
    """測試單一 situation 訓練"""
    print("測試 5: 單一 Situation 訓練")
    print("-" * 60)

    art = ART1Network(n_input=16, max_F2=20, rho=0.5)

    # 單一輸入
    X = np.array([[0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1]])

    # 訓練
    art.train(X, verbose=False)

    # 檢查結果
    assert len(art.history) == 1, "歷程記錄數量錯誤"
    assert art.history[0]["action"] in ["update", "add"], "動作類型錯誤"
    assert art.m >= 1, "至少應有 1 個神經元"

    print(f"✓ 訓練完成")
    print(f"  創建類別數: {art.m}")
    print(f"  動作類型: {art.history[0]['action']}")
    print(f"  分配到 Cluster: {art.history[0]['winner']}")

    print("=" * 60)
    print("✓ 測試 5 通過\n")


def run_all_tests():
    """執行所有測試"""
    print("\n")
    print("=" * 80)
    print("ART-1 單元測試")
    print("=" * 80)
    print("\n")

    try:
        test_initialization()
        test_activation_calculation()
        test_similarity_test()
        test_weight_update()
        test_single_situation()

        print("=" * 80)
        print("✓✓✓ 所有測試通過！ ✓✓✓")
        print("=" * 80)
        return True

    except AssertionError as e:
        print(f"\n✗ 測試失敗: {e}")
        return False
    except Exception as e:
        print(f"\n✗ 執行錯誤: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
