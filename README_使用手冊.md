# BTC 深度學習預測模型 — 使用手冊

> 本專案改編自 Kaniel, Lin, Pelger & Van Nieuwerburgh (JFE 2023)
> "Machine-learning the skill of mutual fund managers"
> 將共同基金預測框架移植至比特幣（及主流加密資產）的週報酬預測。

---

## 一、專案結構

```
deep_learning_btc/
├── config_btc.json          # 超參數設定（網路結構、學習率、特徵選擇）
├── model_btc.py             # TF2 版 FFN 模型（BTCTrainer + FFNModel）
├── btc_data_layer.py        # 資料載入層（完全獨立，不依賴原始 TF1 目錄）
├── train_btc.py             # 主訓練腳本（使用 model_btc.BTCTrainer）
├── prepare_btc_data.py      # 資料下載、特徵工程與輸出 .npz
├── datasets/                # 輸出的 .npz Panel 資料（執行後自動建立）
│   └── btc_panel.npz
└── sampling_folds/          # 訓練/驗證/測試分割索引（執行後自動建立）
    └── btc_chronological_folds.npy

# 注意：本目錄完全獨立，不依賴 ../deep_learning/ 中的任何 TF1 模組
```

---

## 二、環境架設

### 2.1 系統需求

本版本程式碼（`model_btc.py`、`train_btc.py`、`btc_data_layer.py`）
**完全使用 TensorFlow 2.x 原生 API**，不再依賴任何 TF1 模組。

| 項目 | 建議版本 |
|------|---------|
| Python | 3.8 ~ 3.10 |
| TensorFlow | 2.6 以上（建議 2.12） |
| CUDA（GPU，可選） | 11.2 |
| cuDNN（GPU，可選）| 8.1 |

---

### 2.2 CPU 環境（無 GPU，最簡單）

```bash
conda create -n btc_dl python=3.9 -y
conda activate btc_dl

pip install tensorflow==2.12.0
pip install numpy pandas scipy scikit-learn
pip install yfinance requests
pip install coinmetrics-api-client          # 鏈上資料（可選）
pip install matplotlib seaborn              # 視覺化（可選）
```


# 建議使用 Python 3.10 以確保與 TensorFlow 2.12 及最新資料科學套件的相容性

```bash

conda create -n btc_dl python=3.10 -y
conda activate btc_dl

pip install tensorflow==2.12.0
pip install numpy pandas scipy scikit-learn
pip install --upgrade requests yfinance     # 確保抓取套件為最新版，避免 API 變更導致抓不到資料
pip install "urllib3<2"                     # 限制 urllib3 版本以避免 requests 報錯或相依性衝突
pip install coinmetrics-api-client          # 鏈上資料（可選）
pip install matplotlib seaborn              # 視覺化（可選）
pip install cloudscraper                    # 跳過爬蟲程式
```

---

### 2.3 GPU 環境（CUDA 11.2 + cuDNN 8.1）

```bash
conda create -n btc_dl_gpu python=3.9 -y
conda activate btc_dl_gpu

# CUDA 11.2 + cuDNN 8.1（conda 一鍵配對）
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 -y

pip install tensorflow==2.12.0
pip install numpy pandas scipy scikit-learn
pip install yfinance requests
pip install coinmetrics-api-client matplotlib seaborn

pip install cloudscraper

```

> TF 2.13+ 請改用 `pip install tensorflow[and-cuda]`（CUDA 版本需求不同）。

---

### 2.4 驗證安裝

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
# 應顯示：2.12.x

python -c "import numpy, pandas, yfinance; print('套件載入正常')"

# GPU 確認（有 GPU 時）
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 三、特徵說明（22 個特徵）

模型輸入為 14 個主流加密資產（BTC、ETH、SOL 等）的週頻 Panel 資料。

| 類別 | 索引 | 特徵名稱 | 說明 |
|------|------|---------|------|
| A. 價格動能 | 0~4 | r1w, r4w, r12w, r26w, r52w | 1/4/12/26/52 週報酬 |
| B. 技術指標 | 5~9 | rsi_14, bb_pct, vol_ratio, atr_pct, obv_change | RSI、布林通道、波動比、ATR、OBV 變化 |
| C. 鏈上指標 | 10~14 | active_addr, tx_count, nvt, exchange_net_flow, mvrv | 活躍地址數、交易量、NVT、交易所淨流入、MVRV |
| D. 總體/情緒 | 15~18 | fear_greed, spx_ret, dxy_ret, vix | 恐懼貪婪指數、S&P500、美元指數、VIX |
| E. ETF+預測市場 | 19~21 | etf_net_flow_norm, polymarket_btc, etf_net_flow_raw | BTC ETF 淨流入（標準化/原始）、Polymarket 看漲指數 |

**標準化方式：**
- 類別 A~C（個別資產特徵）：每週橫截面排名標準化，映射至 [-1, 1]
- 類別 D~E（總體特徵）：52 週滾動 z-score 標準化
- 所有特徵 clip 至 [-3, 3]

---

## 四、執行步驟

### 步驟 1：資料準備

切換至 `deep_learning_btc/` 目錄：

```bash
cd "Data and Code for Machine-Learning the Skill of Mutual Fund Managers/deep_learning_btc"
```

**完整資料集（含鏈上 + Polymarket）：**

```bash
python prepare_btc_data.py \
    --start 2020-01-01 \
    --end   2025-12-31 \
    --out   datasets/btc_panel.npz
```

**快速模式（跳過鏈上資料與 Polymarket，僅需 yfinance）：**

```bash
python prepare_btc_data.py \
    --start 2020-01-01 \
    --out   datasets/btc_panel.npz \
    --skip_onchain \
    --skip_polymarket
```

**參數說明：**

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `--start` | `2020-01-01` | 資料起始日 |
| `--end` | 今日 | 資料結束日 |
| `--out` | `datasets/btc_panel.npz` | 輸出 NPZ 路徑 |
| `--etf_csv` | 無 | ETF 流量 CSV 備援（可選） |
| `--skip_onchain` | False | 跳過 CoinMetrics 鏈上資料 |
| `--skip_polymarket` | False | 跳過 Polymarket 資料 |

執行成功後輸出格式：
```
data.shape = (T, 14, 23)   # T 週 × 14 資產 × (1 return + 22 features)
```

---

### 步驟 2：模型訓練

```bash
python train_btc.py \
    --config          config_btc.json \
    --logdir          ./checkpoints \
    --max_num_process 0
```

**參數說明：**

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `--config` | `config_btc.json` | 超參數設定檔路徑 |
| `--logdir` | `./checkpoints` | Checkpoint 儲存目錄 |
| `--max_num_process` | `0` | 平行進程數（0 = 單進程，>0 = 多進程） |
| `--printOnConsole` | `True` | 是否列印訓練過程 |
| `--saveLog` | `True` | 是否儲存訓練日誌 |
| `--printFreq` | `10` | 列印頻率（每 N 個 epoch） |

**多核心平行訓練（訓練 4 個模型同時執行）：**

```bash
python train_btc.py \
    --config          config_btc.json \
    --logdir          ./checkpoints \
    --max_num_process 4
```

---

## 五、超參數設定（config_btc.json）

```json
{
  "individual_feature_file": "datasets/btc_panel.npz",
  "individual_feature_dim": 22,
  "num_layers":    1,
  "hidden_dim":    [64],
  "dropout":       0.95,
  "num_epochs":    300,
  "optimizer":     "Adam",
  "learning_rate": 0.001,
  "reg_l1":        0.0,
  "reg_l2":        0.001,
  "weighted_loss": false
}
```

| 超參數 | 說明 | 建議範圍 |
|-------|------|---------|
| `num_layers` | 隱藏層數 | 1~3 |
| `hidden_dim` | 每層節點數（list）| [32], [64], [128] |
| `dropout` | Keep probability | 0.90~1.0 |
| `learning_rate` | 學習率 | 0.0001~0.01 |
| `reg_l2` | L2 正則化係數 | 0.0001~0.01 |
| `num_epochs` | 訓練輪數 | 200~500 |

---

## 六、模型架構說明

本模型使用與論文相同的 **前饋神經網路（Feedforward Neural Network, FFN）**：

```
輸入層（22 個特徵）
      ↓
隱藏層（64 節點，ReLU 激活）
      ↓ Dropout
輸出層（1 個節點，線性）→ 預測下週報酬
```

**訓練目標：** 最小化預測報酬與實際報酬的均方誤差（MSE），並同時懲罰模型的 Sharpe Ratio 方向（`Factor_sharpe` 模式）

**時序分割（避免 Look-ahead Bias）：**
- Train: 前 70% 週
- Validation: 中間 15% 週（超參數調整）
- Test: 後 15% 週（out-of-sample 評估）

**Ensemble 策略：**
訓練 8 個不同隨機種子的模型，取預測平均值，降低單一模型的變異性。

---

## 七、特徵子集（可自訂研究設計）

在 `train_btc.py` 的 `get_tuned_network()` 中修改 `subset`：

| 模型代號 | subset | 特徵類別 | 說明 |
|---------|--------|---------|------|
| (A) | `range(0, 10)` | 價格 + 技術 | 最輕量，無外部 API |
| (B) | `range(0, 15)` | 價格 + 技術 + 鏈上 | 需 CoinMetrics |
| (C) | `range(0, 22)` | 全部 22 個特徵 | 最完整 |
| (D) | `range(0,5) + range(15,22)` | 動能 + 總體 + ETF | 簡約模型 |

---

## 八、輸出結果解讀

訓練完成後，每個模型 checkpoint 存於 `checkpoints/btc/fold_{seed}/...`

分析腳本（參考 `../analysis_code/`）可計算：
- 各分位組（decile）的累積報酬
- 多空組合（Long-Short）的 Sharpe Ratio
- 特徵重要性（Interaction Effect）

---

## 九、常見問題

**Q: 如何安裝 TensorFlow？**

本版本只需 TF 2.x，直接用 pip：

```bash
pip install tensorflow==2.12.0        # CPU 或 GPU（CUDA 11.2）
# 或
pip install tensorflow[and-cuda]      # TF 2.13+，自動安裝 CUDA
```

**Q: 與原始論文的 TF1 程式碼有何差異？**

`model_btc.py` 以 TF2 Keras GradientTape 完整重寫了原版 `FeedForwardModelWithNA_Return`，
邏輯完全相同（Panel 資料遮罩、MSE 損失、Sharpe 選模），但**不需要 `tf.Session`**。
原始 `deep_learning/` 目錄的 TF1 程式碼完全未被修改。

**Q: `squeeze_data` import 失敗**

確認執行路徑在 `deep_learning_btc/` 目錄下，且上層 `deep_learning/` 目錄存在：

```bash
ls ../deep_learning/src/utils.py   # 應存在此檔案
```

**Q: yfinance 下載失敗**

```bash
pip install --upgrade yfinance
```

**Q: CoinMetrics API 需要金鑰**

免費社群版不需要金鑰，但資料覆蓋範圍有限。
若取得 API 金鑰：

```bash
export COINMETRICS_API_KEY="your_key_here"
```

---

## 十、引用

若使用本模型，請引用原始論文：

> Kaniel, R., Lin, Z., Pelger, M., & Van Nieuwerburgh, S. (2023).
> Machine-learning the skill of mutual fund managers.
> *Journal of Financial Economics*, 150, 94–138.
> https://doi.org/10.1016/j.jfineco.2023.07.004
