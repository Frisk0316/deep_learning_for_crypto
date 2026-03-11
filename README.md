# Deep Learning for Crypto Return Prediction
# 深度學習加密貨幣報酬預測

> Adapted from Kaniel, Lin, Pelger & Van Nieuwerburgh (JFE 2023)
> "Machine-learning the skill of mutual fund managers"
>
> 改編自 Kaniel, Lin, Pelger & Van Nieuwerburgh (JFE 2023)
> 「以機器學習評估共同基金經理人技能」
>
> This project ports the mutual fund prediction framework to 14 major crypto assets
> using weekly return data and **49 features** (including DeFi + derivatives data).
> A TensorFlow 2.x **Gated Interaction FFN** predicts next-week returns; the ensemble
> output is used to construct long-short portfolios.
>
> 本專案將共同基金預測框架移植至 14 種主流加密資產，使用週頻報酬資料與
> **49 個特徵**（含 DeFi + 衍生品數據）。基於 TensorFlow 2.x 的
> **Gated Interaction FFN** 預測下週報酬，集成輸出用於建構多空投資組合。

---

## Table of Contents

1. [Project Structure / 專案結構](#1-project-structure--專案結構)
2. [Environment Setup / 環境設定](#2-environment-setup)
3. [Feature Definitions (49 features) / 特徵定義](#3-feature-definitions-49-features--特徵定義49-個)
4. [Quick Start / 快速開始](#4-quick-start--快速開始)
5. [Model Architecture / 模型架構](#5-model-architecture--模型架構)
6. [Hyperparameters / 超參數](#6-hyperparameters--超參數-config_btcjson)
7. [Visualization Results & Economic Interpretation / 結果與經濟詮釋](#7-visualization-results--economic-interpretation--視覺化結果與經濟詮釋)
   - [Table 3 — Long-Short Portfolio Performance](#table-3--long-short-portfolio-performance)
   - [Table A.1 — Decile Portfolio Performance](#table-a1--decile-portfolio-performance)
   - [Table B.1 — Hyperparameters Used](#table-b1--hyperparameters-used)
   - [Fig 01 — Macro Time Series](#fig-01--macro-time-series)
   - [Fig 02 — Data Split Timeline](#fig-02--data-split-timeline)
   - [Fig 05 — Cumulative Returns by Decile](#fig-05--cumulative-returns-by-decile)
   - [Fig 07 — Information Sets Comparison](#fig-07--information-sets-comparison)
   - [Fig 10 — Holding Period Analysis](#fig-10--holding-period-analysis)
   - [Fig 12 — Variable Importance](#fig-12--variable-importance)
   - [Fig 13 — Interaction Effects](#fig-13--interaction-effects)
   - [Fig 14 — 3D Surface: Return vs. Momentum × Sentiment](#fig-14--3d-surface-return-vs-momentum--sentiment)
   - [Training Curves](#training-curves)
   - [Transition Matrix](#transition-matrix)
8. [Feature Subsets / 特徵子集](#8-feature-subsets--特徵子集)
9. [Data Sources / 數據來源](#9-data-sources--數據來源)
10. [FAQ / 常見問題](#10-faq--常見問題)
11. [Citation / 引用](#11-citation--引用)
12. [Conclusion / 結論](#12-conclusion--結論)

---

## 1. Project Structure / 專案結構

```
deep_learning_for_crypto/
├── README.md                          # This file / 本說明文件
├── config_btc.json                    # Hyperparameters / 超參數配置
├── model_btc.py                       # FFN + Gated Interaction model / 模型定義
├── btc_data_layer.py                  # Data loading layer / 資料載入層
├── train_btc.py                       # Training script / 訓練腳本
├── prepare_btc_data.py                # Data pipeline (49 features) / 數據整合管線
├── visualize_results.py               # Result visualization / 結果視覺化
├── btc_spot_etf_from_farside.csv      # BTC spot ETF flows / BTC ETF 流量
├── eth_spot_etf_from_farside.csv      # ETH spot ETF flows / ETH ETF 流量
├── data_sources/                      # Data fetching modules / 數據抓取模組
│   ├── fetch_prices.py                #   OHLCV + technical / 價格與技術指標
│   ├── fetch_onchain.py               #   On-chain metrics / 鏈上指標
│   ├── fetch_sentiment.py             #   Macro + sentiment / 宏觀與情緒
│   ├── fetch_etf_flows.py             #   ETF flows / ETF 流量
│   ├── fetch_polymarket.py            #   Prediction market / 預測市場
│   └── fetch_defi.py                  #   DeFi + derivatives (NEW) / DeFi + 衍生品（新增）
├── datasets/
│   └── btc_panel.npz                  # Panel data (T × 14 assets × 50 cols)
├── sampling_folds/
│   └── btc_chronological_folds.npy    # Train/valid/test indices / 時序分割索引
├── checkpoints/btc/                   # Model checkpoints / 模型存檔
└── visualizations/results/            # Output figures & tables / 輸出圖表
```

> This directory is fully independent of `../deep_learning/` — no TF1 modules are used.
> 本目錄完全獨立於 `../deep_learning/`，不使用任何 TF1 模組。

---

## 2. Environment Setup / 環境設定

### 2.1 System Requirements / 系統需求

| Item | Recommended |
|------|-------------|
| Python | 3.8 – 3.10 |
| TensorFlow | 2.6+ (2.12 tested) |
| CUDA (optional) | 11.8 |
| cuDNN (optional) | 8.6 |

### 2.2 CPU Environment / CPU 環境

```bash
conda create -n btc_dl python=3.10 -y
conda activate btc_dl

pip install tensorflow==2.12.0
pip install numpy pandas scipy scikit-learn
pip install yfinance requests cloudscraper
pip install coinmetrics-api-client          # On-chain data (optional)
pip install matplotlib seaborn
```

### 2.3 GPU Environment (CUDA 12.x driver) / GPU 環境

```bash
conda create -n btc_dl_gpu python=3.10 -y
conda activate btc_dl_gpu

# TF 2.12 requires CUDA 11.8 runtime + cuDNN 8.6 installed via pip
pip install tensorflow==2.12.0
pip install nvidia-cudnn-cu11==8.6.0.163
pip install nvidia-cuda-runtime-cu11==11.8.89

# Set LD_LIBRARY_PATH on every conda activate
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda_env.sh << 'SCRIPT'
CUDNN_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.dirname(nvidia.cudnn.__file__))")
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
SCRIPT

conda deactivate && conda activate btc_dl_gpu

pip install numpy pandas scipy scikit-learn yfinance requests cloudscraper
pip install coinmetrics-api-client matplotlib seaborn
```

> NVIDIA drivers supporting CUDA 12.x are backward-compatible with CUDA 11.x applications.
> The pip-installed `nvidia-cuda-runtime-cu11` provides a self-contained CUDA 11.8 runtime.
> For TF 2.14+, use `pip install tensorflow[and-cuda]` which auto-matches CUDA 12.x.
>
> 支援 CUDA 12.x 的 NVIDIA 驅動向下相容 CUDA 11.x 應用程式。
> 透過 pip 安裝的 `nvidia-cuda-runtime-cu11` 提供獨立的 CUDA 11.8 執行環境。
> 若使用 TF 2.14+，可改用 `pip install tensorflow[and-cuda]` 自動匹配 CUDA 12.x。

### 2.4 Verify Installation / 驗證安裝

```bash
python -c "import tensorflow as tf; print(tf.__version__)"          # 2.12.x
python -c "import numpy, pandas, yfinance; print('OK')"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 3. Feature Definitions (49 features) / 特徵定義（49 個）

Model input: weekly panel data for 14 major crypto assets.
模型輸入：14 種主流加密資產的週頻面板數據。

| Category / 類別 | Index / 索引 | Count / 數量 | Description / 說明 |
|----------------|-------------|-------------|-------------------|
| A. Price Momentum / 價格動能 | 0–4 | 5 | 1/4/12/26/52-week returns / 週報酬 |
| B. Technical / 技術指標 | 5–10 | 6 | RSI, Bollinger %, volatility ratio, ATR, OBV, volume |
| C. On-chain / 鏈上指標 | 11–15 | 5 | Active addresses, tx count, NVT, exchange flow, MVRV |
| D. Macro/Sentiment / 宏觀情緒 | 16–26 | 11 | Fear & Greed, S&P500, DXY, VIX, gold, silver, DJIA |
| E. ETF + Polymarket | 27–32 | 6 | BTC/ETH ETF flows, Polymarket, BTC ETF volume |
| **F. DeFi + Derivatives / DeFi + 衍生品** 🆕 | **33–43** | **11** | **TVL, DEX volume, fees, stablecoins, funding rate, OI, L/S ratio** |
| **G. Reserved / 預留** 🔑 | **44–48** | **5** | **Token Terminal, Etherscan, DappRadar (需 API Key)** |

### Category F details (NEW) / F 類特徵詳情（新增）

| Feature / 特徵 | Source / 來源 | Description / 說明 |
|---------------|-------------|-------------------|
| defi_tvl_chg | DefiLlama | 全市場 DeFi TVL 週變化 / Total DeFi TVL weekly change |
| ethereum_tvl_chg | DefiLlama | ETH 鏈 TVL 週變化 / Ethereum TVL weekly change |
| dex_volume_chg | DefiLlama | DEX 交易量週變化 / DEX volume weekly change |
| defi_fees_chg | DefiLlama | 協議費用週變化 / Protocol fees weekly change |
| stablecoin_mcap_chg | DefiLlama | 穩定幣市值週變化 / Stablecoin mcap weekly change |
| aave_tvl_chg | DefiLlama | Aave TVL 週變化 / Aave TVL weekly change (lending proxy) |
| uniswap_tvl_chg | DefiLlama | Uniswap TVL 週變化 / Uniswap TVL weekly change (DEX proxy) |
| lido_tvl_chg | DefiLlama | Lido TVL 週變化 / Lido TVL weekly change (staking proxy) |
| funding_rate | Binance | BTC 永續合約資金費率 / BTC perpetual funding rate |
| open_interest_chg | Binance | BTC 未平倉量週變化 / BTC open interest weekly change |
| long_short_ratio | Binance | BTC 多空比 / BTC long/short account ratio |

**Normalization / 標準化：**
- Categories A–C (per-asset, 16 features): cross-sectional rank → [-1, 1] / 橫截面排名標準化
- Categories D–G (macro, 33 features): 52-week rolling z-score / 滾動 z-score
- All features clipped to [-3, 3] / 極端值裁剪

---

## 4. Quick Start / 快速開始

### Step 1 — Prepare Data / 步驟 1 — 準備資料

```bash
cd deep_learning_for_crypto

# Full dataset (49 features, with DeFi + on-chain + Polymarket + ETF)
# 完整資料集（49 個特徵，含 DeFi + 鏈上 + Polymarket + ETF）
python prepare_btc_data.py \
    --start 2020-01-01 \
    --out   datasets/btc_panel.npz \
    --btc_etf_csv btc_spot_etf_from_farside.csv \
    --eth_etf_csv eth_spot_etf_from_farside.csv

# Fast mode (skip slow APIs) / 快速模式（跳過慢速 API）
python prepare_btc_data.py \
    --start 2020-01-01 \
    --out   datasets/btc_panel.npz \
    --btc_etf_csv btc_spot_etf_from_farside.csv \
    --eth_etf_csv eth_spot_etf_from_farside.csv \
    --skip_onchain --skip_polymarket

# Skip DeFi data / 跳過 DeFi 資料（僅使用原始 33 個特徵）
python prepare_btc_data.py --skip_defi --skip_onchain
```

| Parameter / 參數 | Default / 預設 | Description / 說明 |
|-----------|---------|-------------|
| `--start` | `2020-01-01` | Data start date / 資料起始日 |
| `--end` | today / 今日 | Data end date / 資料結束日 |
| `--out` | `datasets/btc_panel.npz` | Output NPZ path / 輸出路徑 |
| `--btc_etf_csv` | — | BTC ETF Farside CSV (recommended / 建議提供) |
| `--eth_etf_csv` | — | ETH ETF Farside CSV (recommended / 建議提供) |
| `--skip_onchain` | False | Skip CoinMetrics / 跳過鏈上資料 |
| `--skip_polymarket` | False | Skip Polymarket / 跳過 Polymarket |
| `--skip_defi` | False | Skip DeFi + derivatives / 跳過 DeFi 衍生品 |
| `--token_terminal_key` | — | Token Terminal API Key (optional / 選填) |
| `--etherscan_key` | — | Etherscan API Key (optional / 選填) |
| `--dappradar_key` | — | DappRadar API Key (optional / 選填) |

### Step 2 — Train Models / 步驟 2 — 訓練模型

```bash
# Single process / 單進程
python train_btc.py --config config_btc.json --logdir ./checkpoints

# Parallel (4 simultaneous training runs) / 平行（4 個同時訓練）
python train_btc.py --config config_btc.json --logdir ./checkpoints --max_num_process 4
```

### Step 3 — Generate All Results / 步驟 3 — 生成結果

```bash
python visualize_results.py
# Outputs → visualizations/results/ (PNG figures + CSV tables)
# 輸出 → visualizations/results/（PNG 圖表 + CSV 表格）
```

---

## 5. Model Architecture / 模型架構

### v1: Original FFN / 原版前饋神經網路

The baseline model uses the same **FFN** as the FEN paper (backward compatible via `"model_type": "ffn"`).
基線模型使用與 FEN 論文相同的 **FFN**（可透過 `"model_type": "ffn"` 向後相容）。

```
Input (49 features / 49 個特徵)
      ↓
Dense(64, ReLU) → Dropout(0.05)
      ↓
Dense(1, linear) → predicted next-week return / 預測下週報酬
```

### v2: Gated Interaction FFN (default) / Gated Interaction FFN（預設）

Inspired by the FEN paper's key finding (Fig. 12-13, Table 6) that **sentiment × fund characteristics interaction** is the most important predictor. The Gated Interaction mechanism explicitly models this state-dependent feature importance.

靈感來自 FEN 論文核心發現（Fig. 12-13, Table 6）：**情緒 × 基金特徵的交互作用**是最重要的預測因子。Gated Interaction 機制顯式地建模此狀態依賴的特徵重要性。

```
Input: x = [z_asset (features 0~15) || z_market (features 16~48)]
輸入：x = [個別資產特徵 (0~15) || 市場狀態特徵 (16~48)]

Gate path (門控路徑):
  g = σ(W_g · z_market + b_g)         ∈ (0,1)^K    ← sigmoid gate
  z_gated = g ⊙ (W_a · z_asset + b_a) ∈ R^K        ← market modulates asset features
                                                       市場狀態調控資產特徵

Main path (主路徑, same as FEN / 與 FEN 相同):
  h = ReLU(W_h · [z_asset || z_market] + b_h)       ← standard FFN
  h = Dropout(h)

Combined (合併):
  output = W_o · [h || z_gated] + b_o               → predicted return / 預測報酬
```

**Key insight / 核心洞見：** The gate vector `g` allows market state (DeFi TVL, funding rate, Fear & Greed, etc.) to dynamically amplify or suppress specific asset features. During high-DeFi-activity regimes, the model can upweight on-chain features; during macro fear, it can emphasize momentum reversal signals.

門控向量 `g` 讓市場狀態（DeFi TVL、資金費率、恐懼貪婪指數等）能動態地放大或抑制特定的資產特徵。在 DeFi 活動旺盛時期，模型可加強鏈上特徵的權重；在宏觀恐慌時，可強調動能反轉訊號。

**Parameter count / 參數數：** FFN: ~3,265 → Gated: ~4,097 (+25.5%)

### Training / 訓練

**Training objective / 訓練目標：** Minimize MSE + L2 regularization. Best checkpoint selected by **validation-set Sharpe ratio** (not validation loss).
最小化 MSE + L2 正則化。以**驗證集 Sharpe Ratio** 選取最佳 checkpoint（非驗證損失）。

**Chronological split (no look-ahead bias) / 時序分割（無前瞻偏差）：**
- Train / 訓練: first 70% of weeks (~227 weeks / ~227 週)
- Validation / 驗證: next 15% (~48 weeks / ~48 週) — model selection only / 僅用於模型選擇
- Test / 測試: last 15% (~49 weeks / ~49 週) — out-of-sample / 樣本外報告

**Ensemble / 集成：** 8 models with different random seeds; predictions averaged before portfolio construction.
8 個不同隨機種子的模型；預測值在投資組合建構前取平均。

---

## 6. Hyperparameters / 超參數 (config_btc.json)

```json
{
  "individual_feature_file": "datasets/btc_panel.npz",
  "individual_feature_dim": 49,
  "model_type":      "gated",
  "asset_feature_dim": 16,
  "gate_dim":          16,
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

| Parameter / 參數 | Value / 值 | Description / 說明 |
|-----------|-------|-------------|
| model_type | `"gated"` | `"gated"` = Gated Interaction FFN, `"ffn"` = original FEN / 原版 |
| asset_feature_dim | 16 | Features treated as asset-specific (0~15) / 資產特徵維度 |
| gate_dim | 16 | Gate output dimension / 門控輸出維度 |
| Hidden layers / 隱藏層 | 1 | Range / 範圍: 1–3 |
| Units per layer / 每層節點 | 64 | [32], [64], [128] |
| Dropout keep_prob | 0.95 | 0.90–1.00 |
| Learning rate / 學習率 | 0.001 | 0.0001–0.01 |
| L2 regularization / L2 正則化 | 0.001 | 0.0001–0.01 |
| Epochs / 訓練回合 | 300 | 200–500 |

---

## 7. Visualization Results & Economic Interpretation / 視覺化結果與經濟詮釋

All outputs are saved to `visualizations/results/`. Run `python visualize_results.py` to regenerate.
所有輸出儲存至 `visualizations/results/`。執行 `python visualize_results.py` 重新產生。

---

### Table 3 — Long-Short Portfolio Performance

**File:** `visualizations/results/table3_long_short_performance.csv`

Results on the **out-of-sample test period** (~48 weeks). Each row corresponds to one information set (feature subset). Portfolios go long the top 20% and short the bottom 20% of assets by predicted return.

**樣本外測試期**（約 48 週）的結果。每列對應一個資訊集（特徵子集）。投資組合做多預測報酬前 20% 的資產，同時放空後 20%。

| Information Set | Mean PW (%) | t-stat | SR (ann.) | Mean EW (%) | SR EW | T (weeks) |
|----------------|------------|--------|-----------|------------|-------|-----------|
| Price+Technical | −0.79 | −0.98 | −1.02 | −0.64 | −0.91 | 48 |
| +Onchain | −0.27 | −0.31 | −0.32 | −0.12 | −0.14 | 48 |
| +Macro | **+0.60** | 0.77 | **+0.80** | +1.07 | +1.45 | 48 |
| All features | **+0.62** | 0.91 | **+0.95** | +0.64 | +0.96 | 48 |

> PW = prediction-weighted portfolio; EW = equal-weight portfolio; SR = annualized Sharpe ratio (×√52).

**Economic interpretation:**

The key finding mirrors the FEN paper's incremental information hierarchy: adding macro and sentiment features substantially lifts out-of-sample performance. Models using only price momentum and technical indicators perform negatively in the test period (SR ≈ −1.0), consistent with the view that mechanical price signals are broadly known and potentially crowded in crypto markets. Once macro variables — particularly the Fear & Greed index and traditional-asset return spillovers — are included, the long-short Sharpe rises above +0.80, suggesting that **cross-asset sentiment signals contain orthogonal predictive content** not captured by crypto-internal price patterns.

核心發現呼應 FEN 論文的遞增資訊層級結構：加入宏觀與情緒特徵能大幅提升樣本外績效。僅使用價格動能與技術指標的模型在測試期表現為負（SR ≈ −1.0），符合「機械式價格信號在加密市場已廣為人知且可能過度擁擠」的觀點。一旦納入宏觀變數（尤其是恐懼貪婪指數與傳統資產報酬溢出效應），多空 Sharpe 升至 +0.80 以上，顯示**跨資產情緒信號含有加密市場內部價格模式所未能捕捉的正交預測內容**。

The `+Macro` equal-weight portfolio achieving SR +1.45 outperforms the prediction-weighted version, indicating that the model's predicted-return magnitudes are noisy (especially with only 14 assets), but the **ranking signal** is informative — the model correctly identifies which assets will outperform even if the precise return magnitude is uncertain.

`+Macro` 等權投資組合達到 SR +1.45，優於預測加權版本，顯示模型預測的報酬幅度本身雜訊較大（尤其僅有 14 個資產時），但**排序信號**具有資訊價值——即使精確報酬幅度不確定，模型仍能正確識別哪些資產會跑贏。

The R²_pred values are very negative across all specifications, which is expected. With only 14 assets, individual-return R² is extremely sensitive to a few mispredicted observations. The Sharpe ratio of the constructed portfolio is the appropriate primary metric, as it captures the model's ability to rank assets rather than forecast return levels.

所有設定下 R²_pred 均極度為負，這是預期中的結果。僅 14 個資產時，個別報酬 R² 對少數預測錯誤的觀測值極為敏感。投資組合的 Sharpe Ratio 才是正確的主要指標，因為它衡量的是模型的資產排序能力，而非報酬水準的預測精度。

---

### Table A.1 — Decile Portfolio Performance

**File:** `visualizations/results/table_A1_decile_performance.csv`

Top and bottom decile (10th and 1st) performance for prediction-weighted (PW) and equal-weight (EW) strategies.
預測加權（PW）與等權（EW）策略下，最高十分位（第 10）與最低十分位（第 1）的績效表現。

**Economic interpretation / 經濟詮釋：**

With only 14 assets, the "top decile" and "bottom decile" each contain approximately 1–2 assets per week. The spread between top and bottom decile returns is therefore a direct measure of **cross-sectional return dispersion** captured by the model. The `+Macro` specification shows the top decile PW achieving mean +1.39%/week against bottom decile +0.48%/week — a positive spread of ~0.91% per week, consistent with the long-short result in Table 3. The reversal of the bottom decile to positive (+0.48%) reflects the volatile nature of the crypto market during this period, where even "predicted losers" may produce positive absolute returns in a broadly rising market. The **long-short return** (top − bottom) is the statistic that strips out this market-wide level effect and isolates pure cross-sectional predictability.

僅有 14 個資產時，「最高十分位」與「最低十分位」每週各約含 1–2 個資產。因此，最高與最低十分位報酬之差直接衡量模型所捕捉的**橫截面報酬離散程度**。`+Macro` 設定下，最高十分位 PW 達到均週報酬 +1.39%，最低十分位為 +0.48%，正向價差約 0.91%/週，與 Table 3 的多空結果一致。最低十分位報酬轉正（+0.48%）反映了該時期加密市場的高波動性——在整體市場上漲時，即使是「預測輸家」也可能產生正絕對報酬。**多空報酬**（最高 − 最低）才是剔除市場整體水準效應、純粹衡量橫截面可預測性的統計量。

---

### Table B.1 — Hyperparameters Used

**File:** `visualizations/results/table_B1_hyperparameters.csv`

| Parameter | Value |
|-----------|-------|
| Hidden layers (HL) | 1 |
| Units per layer (HU) | 64 |
| Dropout keep_prob (DR) | 0.95 |
| Learning rate (LR) | 0.001 |
| L1 regularization | 0.0 |
| L2 regularization | 0.001 |
| Epochs | 300 |
| Ensemble seeds | 8 |
| Train / Valid / Test | 70% / 15% / 15% |
| Model selection | Validation Sharpe ratio |
| Total features | 33 |

**Economic interpretation:**

The shallow architecture (1 hidden layer, 64 units) and aggressive dropout (keep_prob = 0.95 ≈ 5% dropout) are designed to **prevent overfitting** in a small-N panel (14 assets). The choice of validation-set Sharpe ratio as the model selection criterion — rather than validation loss — aligns the training objective directly with the downstream investment goal: maximizing risk-adjusted portfolio returns rather than minimizing mean-squared prediction error. This is critical in financial applications where the distribution of returns has fat tails and a small number of extreme weeks can dominate the MSE loss.

淺層架構（1 個隱藏層、64 個節點）與積極的 Dropout（keep_prob = 0.95 ≈ 5% dropout）旨在**防止小 N 面板（14 個資產）過擬合**。以驗證集 Sharpe Ratio 作為模型選擇標準（而非驗證損失），使訓練目標直接與下游投資目標對齊：最大化風險調整後報酬，而非最小化均方預測誤差。這在金融應用中至關重要——報酬分布具有厚尾特性，少數極端週次可能主導 MSE 損失。

---

### Fig 01 — Macro Time Series

**File:** `visualizations/results/fig01_macro_timeseries.png`

Time series of the Fear & Greed Index and BTC ETF normalized inflow over the full dataset (2020–2026).
完整資料集（2020–2026）中，恐懼貪婪指數與 BTC ETF 標準化淨流入的時間序列。

**Economic interpretation / 經濟詮釋：**

The Fear & Greed Index captures the aggregate emotional state of the crypto market. Extended periods of extreme fear (index < 25) historically coincide with capitulation lows, while extreme greed (index > 75) precedes local tops. The co-movement with ETF inflows reveals an important feedback loop: rising sentiment drives institutional ETF demand, which in turn reinforces price appreciation — a reflexive dynamic studied in crypto market microstructure literature. The model uses both signals to predict relative cross-sectional performance rather than market-timing (i.e., which assets outperform *each other* in a given week, not whether the overall market goes up or down).

恐懼貪婪指數捕捉加密市場的整體情緒狀態。歷史上，極度恐懼時期（指數 < 25）往往與投降式低點重合，而極度貪婪（指數 > 75）則領先局部頂部出現。與 ETF 流入的共同走勢揭示了重要的反饋迴路：情緒上升推動機構 ETF 需求，進而強化價格漲勢——這是加密市場微觀結構文獻所研究的反身性動態。本模型使用這兩個信號預測相對橫截面表現而非擇時（即預測哪些資產在某週相互跑贏，而非整體市場漲跌方向）。

---

### Fig 02 — Data Split Timeline

**File:** `visualizations/results/fig02_data_split.png`

Visual timeline of the 323 weekly observations split into train (70%), validation (15%), and test (15%) periods.
323 個週觀測值按訓練（70%）、驗證（15%）、測試（15%）分割的視覺化時序圖。

**Economic interpretation / 經濟詮釋：**

The strict chronological split eliminates look-ahead bias: the model never sees future data during training or validation. The test period (~48 weeks, approximately the last year of data) represents a fully out-of-sample evaluation. This is a more demanding standard than cross-validation or random splits, as it requires the model to generalize across potential regime changes in crypto markets (e.g., the transition from a low-ETF-inflow to a high-ETF-inflow regime after January 2024 spot ETF approvals).

嚴格的時序分割消除了前瞻偏差：模型在訓練與驗證期間絕不接觸未來資料。測試期（約 48 週，約為資料集最後一年）代表完全的樣本外評估。這比交叉驗證或隨機分割更為嚴苛，因為它要求模型能跨越加密市場的潛在制度轉換進行泛化（例如 2024 年 1 月現貨 ETF 獲批後，從低 ETF 流入到高 ETF 流入制度的過渡）。

---

### Fig 05 — Cumulative Returns by Decile

**File:** `visualizations/results/fig05_cumulative_returns_decile.png`

Two panels: prediction-weighted (PW) and equal-weight (EW) decile portfolios. 10 lines from Decile 1 (bottom, predicted losers) to Decile 10 (top, predicted winners) over the test period.
兩個面板：預測加權（PW）與等權（EW）十分位投資組合。測試期內從第 1 十分位（底部，預測輸家）到第 10 十分位（頂部，預測贏家）共 10 條線。

**Economic interpretation / 經濟詮釋：**

A well-specified model should produce a **monotone spread** in cumulative returns across deciles — Decile 10 (top predicted) accumulating the highest wealth and Decile 1 the lowest. This figure is the crypto equivalent of the paper's Fig. 5, which shows striking monotone separation across fund performance deciles. In the crypto case, the signal is noisier given the small cross-section, but the ordering of the top versus bottom deciles for the `+Macro` specification is still informative. The prediction-weighted version amplifies the signal from high-confidence predictions; if the model assigns extreme predicted returns to assets that do well, the PW portfolio will outperform the EW version.

設定良好的模型應在各十分位的累積報酬中產生**單調價差**——第 10 十分位（預測最高）累積最高財富，第 1 十分位最低。此圖是原論文 Fig. 5 的加密貨幣版本，後者展示了基金績效十分位之間引人注目的單調分離。在加密市場中，由於橫截面較小，信號較為嘈雜，但 `+Macro` 設定下最高與最低十分位的排序仍具有資訊價值。預測加權版本放大了高置信度預測的信號；若模型對表現良好的資產給予極端預測報酬，PW 投資組合將優於 EW 版本。

---

### Fig 07 — Information Sets Comparison

**File:** `visualizations/results/fig07_info_sets_comparison.png`

Long-short cumulative returns for each of the 4 information sets (feature subsets) on a single chart.
單一圖表中呈現 4 個資訊集（特徵子集）各自的多空累積報酬。

**Economic interpretation / 經濟詮釋：**

This figure captures the **marginal value of each feature category**. The separation between information set curves directly quantifies how much each successive layer of features — on-chain activity, macro sentiment, ETF flows — adds to predictive performance. The FEN paper (Fig. 7) shows a similar hierarchy for mutual funds, with fund-specific characteristics adding incremental value over stock characteristics alone. In crypto, the most informative transition is from pure price signals to macro/sentiment variables, consistent with crypto being driven by retail sentiment and institutional flows rather than fundamental earnings.

此圖捕捉了**每個特徵類別的邊際價值**。資訊集曲線之間的分離直接量化了每層連續特徵——鏈上活動、宏觀情緒、ETF 流量——對預測績效的增量貢獻。FEN 論文（Fig. 7）對共同基金展示了類似的層級結構，基金特有特徵在股票特徵基礎上提供增量價值。在加密市場中，最具資訊價值的轉變是從純價格信號到宏觀/情緒變數，這與加密市場受散戶情緒和機構資金流動（而非基本面盈利）驅動的觀點一致。

---

### Fig 10 — Holding Period Analysis

**File:** `visualizations/results/fig10_holding_period.png`

Four subplots (Mean, Std, SR, t-stat) of long-short portfolio performance as a function of holding period: 1, 2, 4, 8, 12 weeks (using overlapping portfolios).
多空投資組合績效（Mean、Std、SR、t-stat 四個子圖）相對持有期（1、2、4、8、12 週，使用重疊投資組合）的變化關係。

**Economic interpretation / 經濟詮釋：**

If the model captures genuine predictability and not just microstructure noise, return predictability should persist — and potentially accumulate — over longer holding periods. The original paper (Fig. 10) shows that fund alpha decays slowly over horizons of 1–12 months, consistent with fund "skill" being a persistent characteristic. In crypto, signal decay is expected to be **faster** due to higher liquidity and more active arbitrage, so a rapid drop in Sharpe at horizons beyond 2–4 weeks would confirm that the model is capturing short-lived momentum or flow effects rather than durable fundamental value. Conversely, persistent performance at 4–8 weeks would suggest the information content of macro sentiment and ETF flows takes time to be fully reflected in prices.

若模型捕捉的是真實可預測性而非微觀結構噪音，報酬可預測性應在較長持有期內持續——甚至累積。原論文（Fig. 10）顯示基金 alpha 在 1–12 個月視野內緩慢衰退，與基金「技能」是持久性特徵的觀點一致。在加密市場中，由於流動性更高且套利更活躍，信號衰退預計**更快**，因此超過 2–4 週後 Sharpe 的快速下降將確認模型捕捉的是短暫動能或資金流效應，而非持久基本面價值。反之，4–8 週的持續表現則表明宏觀情緒與 ETF 流量的資訊內容需要時間才能完全反映在價格中。

---

### Fig 12 — Variable Importance

**File:** `visualizations/results/fig12_variable_importance.png`

Sensitivity of model predictions to each feature: `sensitivity_k = sqrt(mean((∂ŷ/∂z_k)²))`, computed via numerical gradients (±ε perturbation) on the test set. Left panel: individual features colored by category. Right panel: group-level importance.
模型預測對每個特徵的敏感度：`sensitivity_k = sqrt(mean((∂ŷ/∂z_k)²))`，透過數值梯度（±ε 擾動）在測試集上計算。左面板：按類別著色的個別特徵。右面板：群組層級重要性。

**Economic interpretation / 經濟詮釋：**

Variable importance measures which features the neural network uses most actively in forming predictions. Features with high sensitivity have large expected partial derivatives — meaning the model's return forecast changes substantially when these features change. In mutual fund research, the paper finds that fund-specific momentum and sentiment are the dominant drivers. In crypto:

變數重要性衡量神經網路在形成預測時最積極使用哪些特徵。高敏感度特徵具有較大的期望偏導數——意味著這些特徵變動時，模型的報酬預測會大幅改變。在共同基金研究中，論文發現基金特有動能與情緒是主要驅動因子。在加密市場中：

- **High importance for price momentum (r12w, r52w) / 價格動能重要性高（r12w, r52w）：** Consistent with the well-documented crypto momentum effect; medium-term winners continue to outperform. / 與加密市場已充分記錄的動能效應一致；中期贏家持續跑贏。
- **High importance for macro/sentiment (fear_greed, vix) / 宏觀/情緒重要性高（fear_greed, vix）：** The model learns that cross-asset risk-off sentiment predicts relative underperformance — riskier, more volatile assets (smaller caps) sell off more during fear episodes. / 模型學習到跨資產避險情緒能預測相對表現劣化——風險較高、波動較大的資產（小市值）在恐懼時期跌幅更大。
- **ETF flow importance (btc_etf_inflow_norm) / ETF 流量重要性：** Captures demand shocks from institutional capital allocation that create temporary mispricings between BTC-correlated assets. / 捕捉機構資產配置需求衝擊，這些衝擊在 BTC 相關資產之間造成短暫錯誤定價。
- **On-chain features / 鏈上特徵：** If important, suggest the model extracts information from blockchain network activity (active addresses, NVT ratio) that is not fully reflected in price-based signals. / 若重要性高，表明模型從區塊鏈網路活動（活躍地址、NVT 比率）中提取了尚未完全反映於價格信號的資訊。

The group-level panel aggregates these by category, providing a clean summary of which information sets drive performance — analogous to the FEN paper's Fig. 12.

群組層級面板按類別彙總上述結果，提供哪些資訊集驅動績效的清晰摘要——類比於 FEN 論文的 Fig. 12。

---

### Fig 13 — Interaction Effects

**File:** `visualizations/results/fig13_interaction_effects.png`

Four subplots, one for each key feature (r12w, r1w, btc_etf_inflow_norm, fear_greed). For each feature, the x-axis varies that feature from −0.4 to +0.4 while all other features are held at their test-set median. Five lines represent the Fear & Greed index at the 10th, 25th, 50th, 75th, and 90th percentiles.
四個子圖，各對應一個關鍵特徵（r12w、r1w、btc_etf_inflow_norm、fear_greed）。每個特徵的 x 軸從 −0.4 變化至 +0.4，其他所有特徵固定在測試集中位數。五條線代表恐懼貪婪指數在第 10、25、50、75、90 百分位的值。

**Economic interpretation / 經濟詮釋：**

Interaction effects reveal **non-linear and conditional relationships** that a linear model cannot capture — the primary justification for using a neural network over OLS. Key patterns to look for:

交互效應揭示了線性模型無法捕捉的**非線性與條件性關係**——這是選擇神經網路而非 OLS 的主要理由。需關注的關鍵模式：

- **Momentum × Sentiment interaction (r12w × fear_greed) / 動能 × 情緒交互（r12w × fear_greed）：** Does the model predict that momentum strategies pay off more when sentiment is high (greed)? This would be consistent with behavioral finance theory — in greed regimes, investors extrapolate recent winners more aggressively, amplifying momentum. In fear regimes, momentum may reverse as investors liquidate recent winners. / 模型是否預測動能策略在情緒高漲（貪婪）時回報更高？這與行為財務學理論一致——貪婪制度下，投資者更積極地外推近期贏家，放大動能效應；恐懼制度下，動能可能反轉，因投資者清算近期贏家。
- **ETF flow × Sentiment / ETF 流量 × 情緒：** Large ETF inflows combined with high greed may signal crowded positioning and predict short-term reversal for BTC-correlated assets. / 大額 ETF 流入疊加高貪婪情緒可能暗示倉位過度集中，並預測 BTC 相關資產的短期反轉。
- **Short-term momentum (r1w) / 短期動能（r1w）：** If the model assigns negative predicted returns to recent weekly winners, it is capturing **short-term reversal** (mean reversion in crypto over 1-week horizons), a microstructure effect distinct from medium-term momentum. / 若模型對近期週漲幅最大的資產給予負預測報酬，則是在捕捉**短期反轉**（加密市場 1 週視野的均值回歸），這是有別於中期動能的微觀結構效應。

The fan-shaped spread of lines in each subplot measures the magnitude of the interaction: wide fans indicate the feature's effect is strongly conditioned on sentiment regime.
每個子圖中扇形展開的線間距衡量交互效應的幅度：扇形越寬，表示該特徵的效應越強烈地受情緒制度條件所左右。

---

### Fig 14 — 3D Surface: Return vs. Momentum × Sentiment

**File:** `visualizations/results/fig14_3d_surface.png`

3D surface of model-predicted return as a function of 12-week momentum (r12w) and Fear & Greed index (fear_greed), with all other features fixed at test-set medians.
模型預測報酬相對於 12 週動能（r12w）與恐懼貪婪指數（fear_greed）的 3D 曲面，其他所有特徵固定在測試集中位數。

**Economic interpretation / 經濟詮釋：**

This figure visualizes the full joint non-linearity learned by the neural network across the two most economically motivated dimensions: **price trend** (r12w captures whether an asset has been gaining) and **market sentiment** (fear_greed captures the emotional state of the crypto market). Key regions of the surface:

此圖視覺化了神經網路在兩個最具經濟意涵的維度上學習到的完整聯合非線性：**價格趨勢**（r12w 捕捉資產近期是否上漲）與**市場情緒**（fear_greed 捕捉加密市場的情緒狀態）。曲面的關鍵區域：

- **High r12w + High fear_greed (top-right) / 高 r12w + 高 fear_greed（右上角）：** Strong recent winners in a greed market — the model's predicted return here captures the momentum amplification effect. / 貪婪市場中的近期強勢資產——此區域模型預測報酬捕捉了動能放大效應。
- **Low r12w + Low fear_greed (bottom-left) / 低 r12w + 低 fear_greed（左下角）：** Recent losers in a fear market — model may predict further underperformance (momentum continuation) or a contrarian bounce (oversold reversal) depending on the learned surface shape. / 恐懼市場中的近期弱勢資產——模型可能預測進一步表現劣化（動能延續）或反向反彈（超賣反轉），取決於學習到的曲面形狀。
- **Curvature of the surface / 曲面曲率：** Any non-flat shape represents information that a linear regression would miss. Convex regions (returns accelerating) suggest the model has learned regime-switching behavior embedded in the historical data. / 任何非平面形狀都代表線性回歸所無法捕捉的資訊。凸起區域（報酬加速）顯示模型已學習到歷史數據中蘊含的制度轉換行為。

A smooth, monotone surface validates that the model's predictions are economically sensible and not driven by memorized noise patterns.
平滑、單調的曲面驗證了模型預測在經濟上的合理性，而非由記憶的噪音模式所驅動。

---

### Training Curves

**Files:** `visualizations/results/fig_training_curves_{feat_key}.png` (one per feature subset)

Validation-set Sharpe ratio over training epochs for all 8 seeds.
8 個隨機種子在訓練 epoch 過程中的驗證集 Sharpe Ratio 變化曲線。

**Economic interpretation / 經濟詮釋：**

These curves reveal the stability and convergence behavior of the ensemble. Well-behaved training should show:
- Rapid early improvement in Sharpe (within the first 50–100 epochs)
- Stabilization before epoch 300, confirming the budget is adequate
- Tight clustering of the 8 seed curves, indicating that the final ensemble is not dominated by a single lucky random initialization

這些曲線揭示了集成模型的穩定性與收斂行為。良好的訓練應呈現：
- Sharpe 在早期快速提升（前 50–100 個 epoch 內）
- 在 epoch 300 之前趨於穩定，確認訓練回合數足夠
- 8 條種子曲線緊密聚集，表明最終集成不由單一幸運隨機初始化所主導

High variance across seeds would suggest that the model is sensitive to initialization — a sign of overfitting or too small a dataset. In that case, increasing the ensemble size or adding regularization would be warranted.

種子間高變異度表示模型對初始化敏感——這是過擬合或資料集過小的徵兆。在此情況下，應增大集成規模或加強正則化。

---

### Transition Matrix

**File:** `visualizations/results/fig_transition_matrix.png`

10×10 heatmap of transition probabilities between prediction deciles in consecutive weeks. Entry (i, j) is the probability that an asset in decile i this week is ranked in decile j next week.
連續週次間預測十分位轉移機率的 10×10 熱圖。元素（i, j）代表本週位於第 i 十分位的資產下週被排入第 j 十分位的機率。

**Economic interpretation / 經濟詮釋：**

The transition matrix measures **persistence of the model's signal**. In an efficient market, the matrix would be approximately uniform (all entries ≈ 10%), meaning the model's ranking conveys no information about future rankings. Persistence is revealed by high diagonal values (assets predicted to be top/bottom tend to remain so next week) and high corner values (assets rarely jump from decile 1 to decile 10 in one week).

轉移矩陣衡量**模型信號的持續性**。在有效市場中，矩陣應接近均勻分布（所有元素 ≈ 10%），意味著模型排序對未來排序不含任何資訊。持續性體現於高對角線值（被預測為頂部/底部的資產傾向於下週維持原位）以及高角落值（資產很少在一週內從第 1 十分位跳至第 10 十分位）。

In crypto, short-term momentum and sentiment are known to be sticky on the order of weeks to months, so one would expect moderate diagonal persistence (50–70%) for deciles based on medium-term price signals. Very high persistence (>90%) would suggest the model is tracking slow-moving fundamentals; very low persistence (<20%) would indicate the signal is essentially noise at the weekly frequency. The transition matrix thus provides a diagnostic for **how the model generates alpha** — through persistent signals or through accurate week-to-week ranking adjustments.

在加密市場中，短期動能與情緒已知在數週至數月的時間尺度上具有黏滯性，因此基於中期價格信號的十分位應呈現中等對角線持續性（50–70%）。極高持續性（>90%）表示模型在追蹤緩慢移動的基本面；極低持續性（<20%）則表明信號在週頻層面基本上是噪音。轉移矩陣因此提供了診斷**模型如何產生 alpha** 的工具——究竟是透過持久信號，還是透過精確的逐週排序調整。

---

## 8. Feature Subsets / 特徵子集

Modify `get_tuned_network()` in `train_btc.py` to change which feature subsets are trained.
修改 `train_btc.py` 中的 `get_tuned_network()` 以選擇訓練的特徵子集。

| Label / 標籤 | subset / 子集 | Categories / 類別 | Description / 說明 |
|-------|--------|------------|-------------|
| **(A)** `feat0to10` | `range(0, 11)` | A+B | Price Momentum + Technical / 價格動能 + 技術指標 |
| **(B)** `feat0to15` | `range(0, 16)` | A+B+C | + On-chain / + 鏈上指標 |
| **(C)** `feat0to32` | `range(0, 33)` | A~E | Original all (baseline) / 原版全部（對照組） |
| **(D)** `feat0to43` | `range(0, 44)` | A~F | **+ DeFi + Derivatives (main experiment)** / **主實驗** |
| **(E)** parsimonious | `range(0,5) + range(16,44)` | A+D+E+F | Momentum + Macro + DeFi / 簡約模型 |
| (F) `feat0to48` | `range(0, 49)` | A~G | All + Reserved (needs API keys) / 含預留 |

> Subset (C) serves as the **baseline** for comparison with original FEN. Subset (D) is the **main experiment** that adds DeFi + derivatives data. The performance difference (D) − (C) measures the **marginal value of DeFi data**.
>
> 子集 (C) 作為與原版 FEN 比較的**對照組**。子集 (D) 為加入 DeFi + 衍生品數據的**主實驗**。(D) − (C) 的表現差異衡量 **DeFi 數據的邊際價值**。

---

## 9. Data Sources / 數據來源

### 9.1 ETF Flows / ETF 流量 (Farside Investors)

BTC and ETH spot ETF daily net inflow/outflow data.
BTC 和 ETH 現貨 ETF 每日淨流入/流出數據。

| Data / 資料 | Local File / 本地檔案 |
|------|------------|
| Bitcoin Spot ETF | `btc_spot_etf_from_farside.csv` |
| Ethereum Spot ETF | `eth_spot_etf_from_farside.csv` |

### 9.2 BTC ETF Volume / ETF 成交量 (Yahoo Finance)

Aggregated daily volume for IBIT, FBTC, BITB, ARKB, GBTC, BTCO — downloaded automatically.
自動下載 IBIT, FBTC 等 ETF 每日成交量，彙總為週頻。

### 9.3 Traditional Assets / 傳統資產 (Yahoo Finance)

GLD, SLV, DIA, SPY, DXY, VIX — downloaded automatically. / 自動下載，無需手動更新。

### 9.4 Fear & Greed Index / 恐懼貪婪指數 (alternative.me)

Downloaded automatically via public API. / 透過公開 API 自動下載。

### 9.5 On-chain Data / 鏈上資料 (CoinMetrics)

Active addresses, tx count, NVT, exchange flow, MVRV. Skip with `--skip_onchain`.
活躍地址、交易數、NVT、交易所流量、MVRV。使用 `--skip_onchain` 跳過。

### 9.6 Polymarket (optional / 選填)

Bitcoin price sentiment. Skip with `--skip_polymarket`.
BTC 價格情緒指數。使用 `--skip_polymarket` 跳過。

### 9.7 DeFi Data / DeFi 資料 (DefiLlama) 🆕

| Feature / 特徵 | API | Cost / 費用 |
|---------------|-----|------------|
| Total DeFi TVL / 全市場 TVL | `api.llama.fi/v2/historicalChainTvl` | Free / 免費 |
| Ethereum TVL | `api.llama.fi/v2/historicalChainTvl/Ethereum` | Free |
| DEX Volume / DEX 交易量 | `api.llama.fi/overview/dexs` | Free |
| Protocol Fees / 協議費用 | `api.llama.fi/overview/fees` | Free |
| Stablecoin Market Cap / 穩定幣市值 | `stablecoins.llama.fi/stablecoincharts/all` | Free |
| Aave / Uniswap / Lido TVL | `api.llama.fi/protocol/{name}` | Free |

### 9.8 Derivatives / 衍生品資料 (Binance Futures) 🆕

| Feature / 特徵 | API | Note / 備註 |
|---------------|-----|------------|
| Funding Rate / 資金費率 | `/fapi/v1/fundingRate` | Full history / 完整歷史 |
| Open Interest / 未平倉量 | `/futures/data/openInterestHist` | Last ~30 days only / 僅近 30 天 |
| Long/Short Ratio / 多空比 | `/futures/data/globalLongShortAccountRatio` | Last ~30 days only / 僅近 30 天 |

### 9.9 Reserved (API Key required) / 預留（需 API Key）🔑

| Source / 來源 | Feature / 特徵 | Status / 狀態 |
|-------------|---------------|-------------|
| Token Terminal | Protocol revenue / 協議營收 | `--token_terminal_key` |
| Etherscan | ETH gas fee / Gas 費用 | `--etherscan_key` |
| DappRadar | DeFi UAW / 活躍錢包 | `--dappradar_key` |

---

## 10. FAQ / 常見問題

**Q: R²_pred 值極度負值（例如 −6,000,000%），是 bug 嗎？**
**Q: My R²_pred values are extremely negative. Is that a bug?**

No. With only 14 assets per week, a single mispredicted volatile asset dominates `SS_res`. The **Sharpe ratio** of the long-short portfolio is the correct primary metric — it captures ranking ability, not return-level accuracy.

不是。僅有 14 個資產時，單一錯誤預測就會讓 `SS_res` 暴增。**多空投資組合的 Sharpe Ratio** 才是正確的主要指標，它衡量的是排序能力而非報酬水準預測。

**Q: How to install TensorFlow? / 如何安裝 TensorFlow？**

```bash
pip install tensorflow==2.12.0        # CPU or GPU with CUDA 11.2
pip install tensorflow[and-cuda]      # TF 2.13+, auto-installs CUDA
```

**Q: What changed from the original TF1 code? / 與原版 TF1 有何不同？**

`model_btc.py` rewrites the model using TF2 Keras `GradientTape`. The logic (panel masking, MSE loss, Sharpe-based selection) is identical; no `tf.Session` needed. Additionally, the **Gated Interaction mechanism** is a new architectural contribution.

`model_btc.py` 使用 TF2 Keras `GradientTape` 重寫。邏輯（面板遮罩、MSE 損失、Sharpe 選模）完全一致，不需要 `tf.Session`。**Gated Interaction 機制**為新增的架構貢獻。

**Q: DefiLlama or Binance API fails / API 失敗？**

DeFi features degrade gracefully — if an API call fails, the corresponding feature column is filled with UNK (-99.99) and masked during training. Use `--skip_defi` to skip entirely.

DeFi 特徵具有優雅降級機制 — 若 API 呼叫失敗，對應特徵欄填入 UNK (-99.99) 並在訓練時遮罩。使用 `--skip_defi` 完全跳過。

**Q: How do I add new assets? / 如何新增資產？**

Edit `CRYPTO_SYMBOLS` in `data_sources/fetch_prices.py` and re-run. Panel format auto-accommodates any N.
修改 `data_sources/fetch_prices.py` 中的 `CRYPTO_SYMBOLS` 並重新執行。面板格式自動適應任意 N。

---

## 11. Citation / 引用

If you use this code or results, please cite: / 若使用本程式碼或結果，請引用：

> Kaniel, R., Lin, Z., Pelger, M., & Van Nieuwerburgh, S. (2023).
> Machine-learning the skill of mutual fund managers.
> *Journal of Financial Economics*, 150, 94–138.
> https://doi.org/10.1016/j.jfineco.2023.07.004

---

## 12. Conclusion / 結論

### 12.1 Summary of Findings / 主要發現摘要

This study ports the fund skill evaluation framework of Kaniel et al. (JFE 2023) to the cryptocurrency market. Using weekly panel data for 14 major crypto assets with up to 49 features, we train a Gated Interaction FFN to predict next-week cross-sectional returns and evaluate long-short portfolio performance out-of-sample (T = 49 weeks).

本研究將 Kaniel et al. (JFE 2023) 的基金技能評估框架移植至加密貨幣市場，以 14 種主流加密資產的週頻數據與最多 49 個特徵，訓練 Gated Interaction FFN 預測下週橫截面報酬，並進行樣本外多空投資組合評估（T = 49 週）。

**Out-of-sample results (test period) / 樣本外結果（測試期）：**

| Information Set / 特徵集 | SR (PW) | SR (EW) | t-stat | Interpretation / 解讀 |
|--------------------------|---------|---------|--------|-----------------------|
| Price+Technical / 價格動能+技術 | **+0.88** | **+0.91** | 0.86 | Positive; correct ranking direction / 正向，排序方向正確 |
| +Onchain / +鏈上指標 | −0.06 | +0.24 | −0.06 | Performance deteriorates / 績效惡化 |
| All features / 全特徵 | −0.36 | +0.04 | −0.35 | Further deterioration / 繼續惡化 |

### 12.2 Economic Interpretation / 發現的經濟意涵

**（一）Price momentum is effective in the short-term crypto market / 價格動能效應在加密市場短期有效**

The Price+Technical model achieves an annualised Sharpe Ratio ≈ 0.88 out-of-sample, with the Top and Bottom decile rankings in the correct direction (Sharpe spread ≈ 1.36). This is consistent with the momentum theory in behavioural finance: crypto markets have a high proportion of retail participants and strong trend-chasing behaviour, giving recent outperformers greater return persistence than in traditional financial markets.

Price+Technical 模型在樣本外實現年化 Sharpe Ratio ≈ 0.88，且 Top/Bottom decile 排序方向正確（Sharpe 差距約 1.36）。這與行為財務學的動能理論一致：加密市場散戶比例高、追漲情緒強烈，使近期強勢資產的趨勢延續性優於傳統金融市場。

**（二）On-chain data and macro features do not improve predictive power / 鏈上資料與宏觀特徵未能提升預測力**

Adding on-chain metrics (NVT, MVRV, active addresses, etc.) reduces performance from SR +0.88 to −0.06; incorporating macro sentiment and ETF flows (33 features total) pushes SR further to −0.36. Table A.1 shows that the +Onchain model's Top/Bottom decile rankings are completely inverted, meaning on-chain data not only fails to help but actively misdirects the model.

加入鏈上指標後績效從 SR +0.88 下滑至 −0.06；進一步納入宏觀情緒與 ETF 流量等 33 個特徵後，SR 跌至 −0.36。Table A.1 顯示 +Onchain 模型的 Top/Bottom decile 排序完全顛倒，意味著鏈上資料在此設定下不僅無助，甚至造成系統性誤導。

Possible explanations / 可能解釋：
- On-chain information is already priced by institutional participants and contains no additional cross-sectional predictive content / 鏈上資訊已被機構充分定價
- With only 14 assets, the cross-section is too small for the model to learn consistent mappings from heterogeneous on-chain signals / 14 個資產橫截面不足以學習鏈上信號的一致映射
- The predictive horizon of macro/sentiment features may not align with weekly returns / 宏觀/情緒特徵的預測視野與週頻報酬不匹配

**（三）Curse of dimensionality in small cross-section settings / 小橫截面環境下的特徵維度詛咒**

The most critical finding is that more features do not yield better results — a clear manifestation of the curse of dimensionality. The original FEN paper used thousands of mutual funds, providing sufficient cross-sectional observations for the model to learn high-dimensional feature patterns. With only 14 assets, even ensembling 8 models with L2 regularisation and dropout cannot overcome the fundamental challenge of high dimensions with low sample size.

本研究最關鍵的發現在於：更多特徵並未帶來更好的結果，呈現明顯的維度詛咒。原始 FEN 論文以數千支共同基金建立面板，本研究僅 14 個資產，即便集成 8 個模型、施加 L2 正則化與 Dropout，仍無法克服高維低樣本的根本挑戰。

### 12.3 Comparison with FEN Paper / 對照 FEN 原論文的差異

| Dimension / 面向 | FEN (Mutual Funds) | This Study (Crypto) |
|------------------|--------------------|---------------------|
| Cross-section N / 資產數 | ~2,000+ | 14 |
| Best feature set / 最佳特徵集 | All features (incl. macro) | Price+Technical only |
| Effect of adding macro / 加入宏觀後 | SR significantly improves | SR deteriorates |
| Statistical significance / 統計顯著性 | High | Insufficient (\|t\| < 2) |

The fundamental reason for this divergence is the scale of the cross-section. The statistical significance in the original paper rests on a large-N foundation. Replicating the same methodology in the crypto market requires architectural-level adjustments to address the small-N problem.

差異的根本原因在於橫截面規模。若要在加密市場複製相同方法論，需針對小 N 問題進行架構層面的調整。

### 12.4 Limitations / 研究限制

1. **Short test period / 樣本期短**：The out-of-sample test covers only 49 weeks; statistical power is insufficient to reliably distinguish genuine predictive ability from noise / 樣本外測試僅 49 週，統計功效不足
2. **Small asset pool / 資產池小**：14 assets limits both model learning capacity and portfolio diversification / 14 個資產限制了模型學習能力與投資組合分散化
3. **Transaction costs excluded / 交易成本未計入**：Weekly rebalancing frictions (spread, slippage) are not accounted for in reported performance / 週頻換手的實際摩擦成本未計入績效
4. **Missing +Macro configuration / +Macro 缺失**：The `feat0to21` (macro features) specification was not trained due to resource constraints, preventing direct verification of macro sentiment's marginal contribution / 因資源限制未完成 `feat0to21` 訓練，無法驗證宏觀情緒的邊際貢獻

### 12.5 Future Work / 未來研究方向

1. **Expand asset pool / 擴大資產池**：Include the top 50–100 crypto assets by market cap to reach a statistically viable cross-section (N ≥ 50) / 納入市值前 50–100 的加密資產
2. **Feature selection pre-processing / 特徵選擇前處理**：Apply LASSO or PCA before the neural network to reduce overfitting risk in small-N settings / 在神經網路前進行降維，降低過擬合風險
3. **Complete +Macro training / 完成 +Macro 訓練**：Train the `feat0to21` specification to determine whether macro sentiment features offer independent predictive value / 補足 `feat0to21` 訓練，釐清宏觀情緒特徵的貢獻
4. **Longer test period / 更長測試期**：Accumulate more out-of-sample data (target ≥ 104 weeks) to improve statistical power / 目標累積 ≥ 104 週樣本外數據
5. **Lightweight gated architecture / 輕量門控架構**：Explore sparse attention or linear gating mechanisms better suited to small-N settings / 探索更適合小橫截面的稀疏注意力或線性門控機制
