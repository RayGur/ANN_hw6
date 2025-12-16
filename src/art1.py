"""
ART-1 (Adaptive Resonance Theory 1) 神經網路實作

參考文獻：
- Da Silva et al. (2017), Artificial Neural Networks - A Practical Course
- Carpenter & Grossberg (1987), ART-1: Self-organization of stable category recognition codes
"""

import numpy as np
from typing import List, Dict, Tuple


class ART1Network:
    """
    ART-1 神經網路

    Parameters:
    -----------
    n_input : int
        輸入層神經元數量（等於輸入向量維度）
    max_F2 : int
        F2 層最大神經元數量（靜態資料結構）
    rho : float
        警戒參數 (vigilance parameter), 範圍 (0, 1)

    Attributes:
    -----------
    W_f : ndarray, shape (max_F2, n_input)
        前饋權重矩陣 (F1 → F2)
    W_b : ndarray, shape (n_input, max_F2)
        反饋權重矩陣 (F2 → F1)
    status : list
        每個 F2 神經元的狀態 ('active' 或 'inactive')
    m : int
        當前啟用的 F2 神經元數量
    """

    def __init__(self, n_input: int, max_F2: int, rho: float):
        self.n = n_input
        self.max_m = max_F2
        self.rho = rho
        self.m = 1  # 初始啟用 1 個神經元

        # 神經元狀態管理（靜態資料結構）
        self.status = ["active"] + ["inactive"] * (max_F2 - 1)

        # 初始化權重矩陣
        self._initialize_weights()

        # 訓練歷程記錄
        self.history = []

    def _initialize_weights(self):
        """
        初始化權重矩陣

        依據 Da Silva et al. (2017) 公式:
        - W^(f)_ji = 1/(1+n)  (公式 10.1)
        - W^(b)_ij = 1        (公式 10.2)
        """
        # 前饋權重：小值避免飽和
        self.W_f = np.full((self.max_m, self.n), 1.0 / (1.0 + self.n))

        # 反饋權重：全部初始化為 1
        self.W_b = np.ones((self.n, self.max_m))

    def _calculate_activation(self, x: np.ndarray) -> np.ndarray:
        """
        計算 F2 層神經元的激活值

        公式 (10.3): u_j = Σ W^(f)_ji * x_i

        Parameters:
        -----------
        x : ndarray, shape (n_input,)
            輸入向量

        Returns:
        --------
        u : ndarray, shape (max_F2,)
            所有神經元的激活值（inactive 神經元返回 -inf）
        """
        u = np.full(self.max_m, -np.inf)

        # 只計算 active 神經元的激活值
        for j in range(self.max_m):
            if self.status[j] == "active":
                u[j] = np.sum(self.W_f[j, :] * x)

        return u

    def _get_winner(self, u: np.ndarray) -> int:
        """
        獲取勝者神經元索引

        公式 (10.4): k = arg max{u_j}

        Parameters:
        -----------
        u : ndarray
            激活值向量

        Returns:
        --------
        k : int
            勝者神經元索引
        """
        return np.argmax(u)

    def _similarity_test(self, x: np.ndarray, k: int) -> Tuple[bool, float]:
        """
        執行相似度測試

        公式 (10.6): R = Σ(W^(b)_jk * x_j) / Σ(x_j)
        判斷: R >= ρ 則通過

        Parameters:
        -----------
        x : ndarray
            輸入向量
        k : int
            勝者神經元索引

        Returns:
        --------
        passed : bool
            是否通過相似度測試
        R : float
            相似度比值
        """
        # 計算分子：共同的 1 的數量
        numerator = np.sum(self.W_b[:, k] * x)

        # 計算分母：輸入向量中 1 的數量
        denominator = np.sum(x)

        # 避免除以零
        if denominator == 0:
            R = 0.0
        else:
            R = numerator / denominator

        # 判斷是否通過
        passed = R >= self.rho

        return passed, R

    def _update_weights(self, x: np.ndarray, k: int):
        """
        更新勝者神經元的權重

        公式 (10.9): W^(b)_jk(t+1) = W^(b)_jk(t) * x_j  (AND 操作)
        公式 (10.10): W^(f)_kj(t+1) = W^(b)_jk(t) * x_j / (0.5 + Σ(W^(b)_ik(t) * x_i))

        Parameters:
        -----------
        x : ndarray
            輸入向量
        k : int
            勝者神經元索引
        """
        # 更新反饋權重 W^(b) - AND 操作
        self.W_b[:, k] = self.W_b[:, k] * x

        # 更新前饋權重 W^(f)
        numerator = self.W_b[:, k] * x
        denominator = 0.5 + np.sum(self.W_b[:, k] * x)

        self.W_f[k, :] = numerator / denominator

    def _add_neuron(self) -> int:
        """
        啟用新的 F2 神經元

        Returns:
        --------
        new_k : int
            新啟用的神經元索引
        """
        if self.m >= self.max_m:
            raise RuntimeError(f"已達到最大神經元數量 {self.max_m}")

        # 啟用下一個神經元
        new_k = self.m
        self.status[new_k] = "active"
        self.m += 1

        return new_k

    def train(self, X: np.ndarray, verbose: bool = True):
        """
        訓練 ART-1 網路

        Parameters:
        -----------
        X : ndarray, shape (n_samples, n_input)
            訓練資料
        verbose : bool
            是否顯示訓練過程
        """
        n_samples = X.shape[0]

        if verbose:
            print("=" * 80)
            print(f"開始 ART-1 訓練")
            print(
                f"參數設定: ρ={self.rho:.2f}, 初始 F2 神經元={self.m}, 最大 F2={self.max_m}"
            )
            print("=" * 80)

        for idx, x in enumerate(X):
            situation_id = idx + 1

            if verbose:
                print(f"\n處理 Situation {situation_id}: {x}")

            # 迭代尋找合適的神經元
            iteration = 0
            winner_found = False
            tested_neurons = set()

            while not winner_found:
                iteration += 1

                # (II) Recognition Phase: 計算激活值並選擇勝者
                u = self._calculate_activation(x)
                k = self._get_winner(u)

                if verbose:
                    print(f"  Iteration {iteration}: 勝者神經元 k={k}, u_k={u[k]:.4f}")

                # (III) Comparison Phase: 相似度測試
                passed, R = self._similarity_test(x, k)

                if verbose:
                    print(
                        f"    相似度測試: R={R:.4f}, ρ={self.rho:.2f} → {'通過' if passed else '失敗'}"
                    )

                if passed:
                    # (V) Update Phase: 更新權重
                    self._update_weights(x, k)
                    action = "update"
                    winner_found = True

                    if verbose:
                        print(f"    動作: 更新神經元 {k} 的權重")

                else:
                    # (IV) Search Phase: 禁用當前勝者，尋找下一個
                    tested_neurons.add(k)
                    self.status[k] = "inactive"

                    if verbose:
                        print(f"    動作: 禁用神經元 {k}")

                    # 檢查是否還有可用神經元
                    active_neurons = [
                        j for j in range(self.m) if self.status[j] == "active"
                    ]

                    if len(active_neurons) == 0:
                        # 沒有可用神經元，新增一個
                        new_k = self._add_neuron()
                        self._update_weights(x, new_k)
                        k = new_k
                        action = "add"
                        winner_found = True

                        if verbose:
                            print(f"    動作: 新增神經元 {k}")

            # 恢復被測試過但未選中的神經元
            for j in tested_neurons:
                if j != k:
                    self.status[j] = "active"

            # (VI) Provision Phase: 記錄分類結果
            self.history.append(
                {
                    "situation_id": situation_id,
                    "winner": k,
                    "R_value": R,
                    "action": action,
                    "n_active": self.m,
                }
            )

            if verbose:
                print(
                    f"  最終: Situation {situation_id} → Cluster {k} (動作: {action})"
                )

        if verbose:
            print("\n" + "=" * 80)
            print(f"訓練完成! 總共創建 {self.m} 個類別")
            print("=" * 80)

    def predict(self, x: np.ndarray) -> int:
        """
        預測輸入向量的類別

        Parameters:
        -----------
        x : ndarray
            輸入向量

        Returns:
        --------
        cluster : int
            預測的類別索引
        """
        u = self._calculate_activation(x)
        k = self._get_winner(u)
        return k

    def get_clustering_result(self) -> Dict[int, List[int]]:
        """
        獲取聚類結果

        Returns:
        --------
        clusters : dict
            {cluster_id: [situation_ids]}
        """
        clusters = {}
        for record in self.history:
            cluster = record["winner"]
            situation = record["situation_id"]

            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(situation)

        return clusters

    def print_summary(self):
        """印出訓練摘要"""
        print("\n" + "=" * 80)
        print("訓練摘要")
        print("=" * 80)
        print(f"警戒參數 ρ: {self.rho:.2f}")
        print(f"創建的類別數: {self.m}")

        clusters = self.get_clustering_result()
        print(f"\n聚類結果:")
        for cluster_id in sorted(clusters.keys()):
            situations = clusters[cluster_id]
            print(f"  Cluster {cluster_id}: {situations}")

        print("=" * 80)
