# ANN HW6: 工業流程分類使用 ART-1

本專案實作 ART-1 (Adaptive Resonance Theory 1) 神經網路，用於分類 10 個工業流程情境。

## 專案結構

```
ANN_hw6/
├── data/
│   └── industrial_data.py         # 資料集
├── src/
│   ├── art1.py                    # ART-1 核心實作
│   ├── utils.py                   # 輔助函數
│   └── visualization.py           # 視覺化工具
├── experiments/
│   ├── exp1_init_neurons.py       # 實驗1
│   ├── exp2_vigilance.py          # 實驗2
│   └── exp3_comprehensive.py      # 實驗3
├── tests/
│   ├── test_art1.py              # 單元測試
│   ├── test_small_scale.py       # 小規模測試
│   └── test_full.py              # 完整測試
├── results/
│   ├── logs/                      # 訓練日誌 (CSV)
│   └── figures/                   # 視覺化圖表
├── report_chinese.md              # 中文完整報告
└── README.md                      # 本檔案
```

## 快速開始

### 1. 環境需求

```bash
Python 3.8+
numpy
pandas
matplotlib
seaborn
```

安裝依賴：
```bash
pip install numpy pandas matplotlib seaborn
```

### 2. 執行測試

**在 Windows 環境下**：
```bash
cd ANN_hw6\tests
python test_art1.py          # 單元測試
python test_small_scale.py   # 小規模測試
python test_full.py          # 完整測試
```

**在 Linux/Mac 環境下**：
```bash
cd ANN_hw6/tests
python test_art1.py          # 單元測試
python test_small_scale.py   # 小規模測試
python test_full.py          # 完整測試
```

### 3. 執行實驗

**在 Windows 環境下**：
```bash
cd ANN_hw6
python -m experiments.exp1_init_neurons    # 實驗1
python -m experiments.exp2_vigilance       # 實驗2
python -m experiments.exp3_comprehensive   # 實驗3
```

**在 Linux/Mac 環境下**：
```bash
cd ANN_hw6
python -m experiments.exp1_init_neurons    # 實驗1
python -m experiments.exp2_vigilance       # 實驗2
python -m experiments.exp3_comprehensive   # 實驗3
```

## 主要功能

### ART-1 神經網路類別

```python
from src.art1 import ART1Network

# 創建網路
art = ART1Network(n_input=16, max_F2=20, rho=0.5)

# 訓練
art.train(X, verbose=True)

# 預測
cluster = art.predict(x_new)

# 獲取聚類結果
clusters = art.get_clustering_result()
```

### 關鍵參數

- `n_input`: 輸入向量維度
- `max_F2`: F²層最大神經元數量
- `rho`: 警戒參數 (0 < ρ < 1)
  - 低 (0.3-0.4): 粗分類
  - 中 (0.5-0.6): 平衡
  - 高 (0.7-0.9): 細分類

## 實驗結果摘要

### 實驗1: 初始F²神經元影響
- 測試: F²_init = [1, 3, 5], ρ = 0.5
- 結果: 所有配置產生相同的 4 個類別
- 結論: 初始神經元數量對結果無影響

### 實驗2: 警戒參數影響
- 測試: ρ = [0.3, 0.5, 0.7, 0.9], F²_init = 1
- 結果: 類別數 [2, 4, 5, 7]
- 結論: ρ 是決定類別數的關鍵因素

### 實驗3: 綜合比較
- 測試: 12組參數組合
- 發現: ρ 完全決定結果，初始F²無影響

## 視覺化結果

所有圖表位於 `results/figures/`:

- `exp1_init*_training.png` - 訓練過程圖
- `exp1_init*_distribution.png` - 聚類分佈圖
- `exp2_rho*_training.png` - 不同ρ的訓練過程
- `exp2_rho*_distribution.png` - 不同ρ的聚類分佈
- `exp2_parameter_impact.png` - ρ對類別數的影響
- `exp3_heatmap.png` - 參數組合熱圖

## 核心算法

### 六階段訓練流程

1. **Initialization**: 權重初始化
   - W^(f) = 1/(1+n)
   - W^(b) = 1

2. **Recognition**: 計算激活值，選擇勝者
   - u_j = Σ W^(f)_ji * x_i
   - k = arg max{u_j}

3. **Comparison**: 相似度測試
   - R = Σ(W^(b)_jk * x_j) / Σ(x_j)
   - 判斷: R >= ρ

4. **Search**: 搜尋合適神經元
   - 若不通過，禁用當前勝者
   - 重複直到找到合適神經元

5. **Update**: 權重更新
   - W^(b)_jk(t+1) = W^(b)_jk(t) * x_j
   - W^(f)_kj(t+1) = ...

6. **Provision**: 提供分類結果

## 測試覆蓋

- ✅ 權重初始化測試
- ✅ 激活值計算測試
- ✅ 相似度測試驗證
- ✅ 權重更新測試
- ✅ 單一樣本訓練測試
- ✅ 小規模訓練測試 (3 situations)
- ✅ 完整訓練測試 (10 situations)

## 參考文獻

1. Da Silva et al. (2017). *Artificial Neural Networks - A Practical Course*
2. Carpenter & Grossberg (1987). *ART-1: Self-organization of stable category recognition codes*

## 作者

Ray - 2024年12月

## 授權

本專案僅供學術用途