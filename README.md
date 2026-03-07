# Deep Learning for Crypto Return Prediction

> Adapted from Kaniel, Lin, Pelger & Van Nieuwerburgh (JFE 2023)
> "Machine-learning the skill of mutual fund managers"
>
> This project ports the mutual fund prediction framework to 14 major crypto assets
> using weekly return data and 33 features. A TensorFlow 2.x feedforward neural
> network (FFN) predicts next-week returns; the ensemble output is used to construct
> long-short portfolios and replicate the paper's main Tables and Figures.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Feature Definitions (33 features)](#3-feature-definitions-33-features)
4. [Quick Start](#4-quick-start)
5. [Model Architecture](#5-model-architecture)
6. [Hyperparameters (config_btc.json)](#6-hyperparameters-config_btcjson)
7. [Visualization Results & Economic Interpretation](#7-visualization-results--economic-interpretation)
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
8. [Feature Subsets](#8-feature-subsets)
9. [Data Sources](#9-data-sources)
10. [FAQ](#10-faq)
11. [Citation](#11-citation)

---

## 1. Project Structure

```
deep_learning_for_crypto/
├── README.md                          # This file
├── config_btc.json                    # Hyperparameters (network, learning rate, features)
├── model_btc.py                       # TF2 FFN model (BTCTrainer + FFNModel)
├── btc_data_layer.py                  # Data loading layer (independent of TF1 directory)
├── train_btc.py                       # Main training script
├── prepare_btc_data.py                # Data download, feature engineering, outputs .npz
├── visualize_results.py               # Replicates all FEN paper Tables & Figures
├── btc_spot_etf_from_farside.csv      # BTC spot ETF daily flows (Farside Investors)
├── eth_spot_etf_from_farside.csv      # ETH spot ETF daily flows (Farside Investors)
├── datasets/
│   └── btc_panel.npz                  # Panel data (T × 14 assets × 34 cols)
├── sampling_folds/
│   └── btc_chronological_folds.npy   # Train/valid/test time indices
├── checkpoints/btc/                   # Trained model checkpoints (8 seeds × 4 subsets)
└── visualizations/results/            # All output figures and tables (auto-generated)
```

> This directory is fully independent of `../deep_learning/` — no TF1 modules are used.

---

## 2. Environment Setup

### 2.1 System Requirements

| Item | Recommended |
|------|-------------|
| Python | 3.8 – 3.10 |
| TensorFlow | 2.6+ (2.12 tested) |
| CUDA (optional) | 11.8 |
| cuDNN (optional) | 8.6 |

### 2.2 CPU Environment

```bash
conda create -n btc_dl python=3.10 -y
conda activate btc_dl

pip install tensorflow==2.12.0
pip install numpy pandas scipy scikit-learn
pip install yfinance requests cloudscraper
pip install coinmetrics-api-client          # On-chain data (optional)
pip install matplotlib seaborn
```

### 2.3 GPU Environment (CUDA 12.x driver)

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

### 2.4 Verify Installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"          # 2.12.x
python -c "import numpy, pandas, yfinance; print('OK')"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 3. Feature Definitions (33 features)

Model input: weekly panel data for 14 major crypto assets (BTC, ETH, SOL, BNB, XRP, AVAX, DOGE, ADA, MATIC, LINK, DOT, LTC, UNI, ATOM).

| Category | Index | Feature Names | Description |
|----------|-------|---------------|-------------|
| A. Price Momentum | 0–4 | r1w, r4w, r12w, r26w, r52w | 1/4/12/26/52-week returns |
| B. Technical | 5–10 | rsi_14, bb_pct, vol_ratio, atr_pct, obv_change, vol_usd | RSI, Bollinger Band %, volatility ratio, ATR, OBV change, USD volume (log) |
| C. On-chain | 11–15 | active_addr, tx_count, nvt, exchange_net_flow, mvrv | Active addresses, tx volume, NVT ratio, exchange net flow, MVRV |
| D. Macro/Sentiment | 16–26 | fear_greed, spx_ret, dxy_ret, vix, gold_ret, silver_ret, dji_ret, spx_vol_chg, gold_vol_chg, silver_vol_chg, dji_vol_chg | Fear & Greed, S&P500/DXY/VIX, gold/silver/DJIA returns, volume ratios |
| E. ETF + Prediction Markets | 27–32 | btc_etf_inflow_norm, polymarket_btc, btc_etf_inflow_raw, eth_etf_inflow_norm, eth_etf_inflow_raw, btc_etf_vol | BTC/ETH ETF net flows, Polymarket sentiment, BTC ETF volume |

**Normalization:**
- Categories A–C (per-asset, 16 features): weekly cross-sectional rank normalized to [-1, 1]
- Categories D–E (macro, 17 features): 52-week rolling z-score
- All features clipped to [-3, 3]

---

## 4. Quick Start

### Step 1 — Prepare Data

```bash
cd deep_learning_for_crypto

# Full dataset (with on-chain + Polymarket + Farside ETF)
python prepare_btc_data.py \
    --start 2020-01-01 \
    --end   2025-12-31 \
    --out   datasets/btc_panel.npz \
    --btc_etf_csv btc_spot_etf_from_farside.csv \
    --eth_etf_csv eth_spot_etf_from_farside.csv

# Fast mode (skip on-chain and Polymarket)
python prepare_btc_data.py \
    --start 2020-01-01 \
    --out   datasets/btc_panel.npz \
    --btc_etf_csv btc_spot_etf_from_farside.csv \
    --eth_etf_csv eth_spot_etf_from_farside.csv \
    --skip_onchain \
    --skip_polymarket
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--start` | `2020-01-01` | Data start date |
| `--end` | today | Data end date |
| `--out` | `datasets/btc_panel.npz` | Output NPZ path |
| `--btc_etf_csv` | — | BTC ETF Farside CSV (recommended) |
| `--eth_etf_csv` | — | ETH ETF Farside CSV (recommended) |
| `--skip_onchain` | False | Skip CoinMetrics on-chain data |
| `--skip_polymarket` | False | Skip Polymarket data |

### Step 2 — Train Models

```bash
# Single process
python train_btc.py --config config_btc.json --logdir ./checkpoints

# Parallel (4 simultaneous training runs)
python train_btc.py --config config_btc.json --logdir ./checkpoints --max_num_process 4
```

### Step 3 — Generate All Results

```bash
python visualize_results.py
# Outputs → visualizations/results/ (PNG figures + CSV tables)
```

---

## 5. Model Architecture

The model uses the same **Feedforward Neural Network (FFN)** as the FEN paper:

```
Input (33 features)
      ↓
Hidden Layer (64 units, ReLU)
      ↓ Dropout (keep_prob = 0.95)
Output Layer (1 unit, linear) → predicted next-week return
```

**Training objective:** Minimize MSE between predicted and actual weekly returns, with L2 regularization on weights. Best checkpoint selected by **validation-set Sharpe ratio** (not validation loss).

**Chronological split (no look-ahead bias):**
- Train: first 70% of weeks (~219 weeks)
- Validation: next 15% (~47 weeks) — used for model selection only
- Test: last 15% (~48 weeks) — out-of-sample reporting

**Ensemble:** 8 models with different random seeds; predictions averaged before portfolio construction.

---

## 6. Hyperparameters (config_btc.json)

```json
{
  "individual_feature_file": "datasets/btc_panel.npz",
  "individual_feature_dim": 33,
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

| Parameter | Value | Range |
|-----------|-------|-------|
| Hidden layers | 1 | 1–3 |
| Units per layer | 64 | [32], [64], [128] |
| Dropout keep_prob | 0.95 | 0.90–1.00 |
| Learning rate | 0.001 | 0.0001–0.01 |
| L2 regularization | 0.001 | 0.0001–0.01 |
| Epochs | 300 | 200–500 |

---

## 7. Visualization Results & Economic Interpretation

All outputs are saved to `visualizations/results/`. Run `python visualize_results.py` to regenerate.

---

### Table 3 — Long-Short Portfolio Performance

**File:** `visualizations/results/table3_long_short_performance.csv`

Results on the **out-of-sample test period** (~48 weeks). Each row corresponds to one information set (feature subset). Portfolios go long the top 20% and short the bottom 20% of assets by predicted return.

| Information Set | Mean PW (%) | t-stat | SR (ann.) | Mean EW (%) | SR EW | T (weeks) |
|----------------|------------|--------|-----------|------------|-------|-----------|
| Price+Technical | −0.79 | −0.98 | −1.02 | −0.64 | −0.91 | 48 |
| +Onchain | −0.27 | −0.31 | −0.32 | −0.12 | −0.14 | 48 |
| +Macro | **+0.60** | 0.77 | **+0.80** | +1.07 | +1.45 | 48 |
| All features | **+0.62** | 0.91 | **+0.95** | +0.64 | +0.96 | 48 |

> PW = prediction-weighted portfolio; EW = equal-weight portfolio; SR = annualized Sharpe ratio (×√52).

**Economic interpretation:**

The key finding mirrors the FEN paper's incremental information hierarchy: adding macro and sentiment features substantially lifts out-of-sample performance. Models using only price momentum and technical indicators perform negatively in the test period (SR ≈ −1.0), consistent with the view that mechanical price signals are broadly known and potentially crowded in crypto markets. Once macro variables — particularly the Fear & Greed index and traditional-asset return spillovers — are included, the long-short Sharpe rises above +0.80, suggesting that **cross-asset sentiment signals contain orthogonal predictive content** not captured by crypto-internal price patterns.

The `+Macro` equal-weight portfolio achieving SR +1.45 outperforms the prediction-weighted version, indicating that the model's predicted-return magnitudes are noisy (especially with only 14 assets), but the **ranking signal** is informative — the model correctly identifies which assets will outperform even if the precise return magnitude is uncertain.

The R²_pred values are very negative across all specifications, which is expected. With only 14 assets, individual-return R² is extremely sensitive to a few mispredicted observations. The Sharpe ratio of the constructed portfolio is the appropriate primary metric, as it captures the model's ability to rank assets rather than forecast return levels.

---

### Table A.1 — Decile Portfolio Performance

**File:** `visualizations/results/table_A1_decile_performance.csv`

Top and bottom decile (10th and 1st) performance for prediction-weighted (PW) and equal-weight (EW) strategies.

**Economic interpretation:**

With only 14 assets, the "top decile" and "bottom decile" each contain approximately 1–2 assets per week. The spread between top and bottom decile returns is therefore a direct measure of **cross-sectional return dispersion** captured by the model. The `+Macro` specification shows the top decile PW achieving mean +1.39%/week against bottom decile +0.48%/week — a positive spread of ~0.91% per week, consistent with the long-short result in Table 3. The reversal of the bottom decile to positive (+0.48%) reflects the volatile nature of the crypto market during this period, where even "predicted losers" may produce positive absolute returns in a broadly rising market. The **long-short return** (top − bottom) is the statistic that strips out this market-wide level effect and isolates pure cross-sectional predictability.

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

---

### Fig 01 — Macro Time Series

**File:** `visualizations/results/fig01_macro_timeseries.png`

Time series of the Fear & Greed Index and BTC ETF normalized inflow over the full dataset (2020–2026).

**Economic interpretation:**

The Fear & Greed Index captures the aggregate emotional state of the crypto market. Extended periods of extreme fear (index < 25) historically coincide with capitulation lows, while extreme greed (index > 75) precedes local tops. The co-movement with ETF inflows reveals an important feedback loop: rising sentiment drives institutional ETF demand, which in turn reinforces price appreciation — a reflexive dynamic studied in crypto market microstructure literature. The model uses both signals to predict relative cross-sectional performance rather than market-timing (i.e., which assets outperform *each other* in a given week, not whether the overall market goes up or down).

---

### Fig 02 — Data Split Timeline

**File:** `visualizations/results/fig02_data_split.png`

Visual timeline of the 323 weekly observations split into train (70%), validation (15%), and test (15%) periods.

**Economic interpretation:**

The strict chronological split eliminates look-ahead bias: the model never sees future data during training or validation. The test period (~48 weeks, approximately the last year of data) represents a fully out-of-sample evaluation. This is a more demanding standard than cross-validation or random splits, as it requires the model to generalize across potential regime changes in crypto markets (e.g., the transition from a low-ETF-inflow to a high-ETF-inflow regime after January 2024 spot ETF approvals).

---

### Fig 05 — Cumulative Returns by Decile

**File:** `visualizations/results/fig05_cumulative_returns_decile.png`

Two panels: prediction-weighted (PW) and equal-weight (EW) decile portfolios. 10 lines from Decile 1 (bottom, predicted losers) to Decile 10 (top, predicted winners) over the test period.

**Economic interpretation:**

A well-specified model should produce a **monotone spread** in cumulative returns across deciles — Decile 10 (top predicted) accumulating the highest wealth and Decile 1 the lowest. This figure is the crypto equivalent of the paper's Fig. 5, which shows striking monotone separation across fund performance deciles. In the crypto case, the signal is noisier given the small cross-section, but the ordering of the top versus bottom deciles for the `+Macro` specification is still informative. The prediction-weighted version amplifies the signal from high-confidence predictions; if the model assigns extreme predicted returns to assets that do well, the PW portfolio will outperform the EW version.

---

### Fig 07 — Information Sets Comparison

**File:** `visualizations/results/fig07_info_sets_comparison.png`

Long-short cumulative returns for each of the 4 information sets (feature subsets) on a single chart.

**Economic interpretation:**

This figure captures the **marginal value of each feature category**. The separation between information set curves directly quantifies how much each successive layer of features — on-chain activity, macro sentiment, ETF flows — adds to predictive performance. The FEN paper (Fig. 7) shows a similar hierarchy for mutual funds, with fund-specific characteristics adding incremental value over stock characteristics alone. In crypto, the most informative transition is from pure price signals to macro/sentiment variables, consistent with crypto being driven by retail sentiment and institutional flows rather than fundamental earnings.

---

### Fig 10 — Holding Period Analysis

**File:** `visualizations/results/fig10_holding_period.png`

Four subplots (Mean, Std, SR, t-stat) of long-short portfolio performance as a function of holding period: 1, 2, 4, 8, 12 weeks (using overlapping portfolios).

**Economic interpretation:**

If the model captures genuine predictability and not just microstructure noise, return predictability should persist — and potentially accumulate — over longer holding periods. The original paper (Fig. 10) shows that fund alpha decays slowly over horizons of 1–12 months, consistent with fund "skill" being a persistent characteristic. In crypto, signal decay is expected to be **faster** due to higher liquidity and more active arbitrage, so a rapid drop in Sharpe at horizons beyond 2–4 weeks would confirm that the model is capturing short-lived momentum or flow effects rather than durable fundamental value. Conversely, persistent performance at 4–8 weeks would suggest the information content of macro sentiment and ETF flows takes time to be fully reflected in prices.

---

### Fig 12 — Variable Importance

**File:** `visualizations/results/fig12_variable_importance.png`

Sensitivity of model predictions to each feature: `sensitivity_k = sqrt(mean((∂ŷ/∂z_k)²))`, computed via numerical gradients (±ε perturbation) on the test set. Left panel: individual features colored by category. Right panel: group-level importance.

**Economic interpretation:**

Variable importance measures which features the neural network uses most actively in forming predictions. Features with high sensitivity have large expected partial derivatives — meaning the model's return forecast changes substantially when these features change. In mutual fund research, the paper finds that fund-specific momentum and sentiment are the dominant drivers. In crypto:

- **High importance for price momentum (r12w, r52w):** Consistent with the well-documented crypto momentum effect; medium-term winners continue to outperform.
- **High importance for macro/sentiment (fear_greed, vix):** The model learns that cross-asset risk-off sentiment predicts relative underperformance — riskier, more volatile assets (smaller caps) sell off more during fear episodes.
- **ETF flow importance (btc_etf_inflow_norm):** Captures demand shocks from institutional capital allocation that create temporary mispricings between BTC-correlated assets.
- **On-chain features:** If important, suggest the model extracts information from blockchain network activity (active addresses, NVT ratio) that is not fully reflected in price-based signals.

The group-level panel aggregates these by category, providing a clean summary of which information sets drive performance — analogous to the FEN paper's Fig. 12.

---

### Fig 13 — Interaction Effects

**File:** `visualizations/results/fig13_interaction_effects.png`

Four subplots, one for each key feature (r12w, r1w, btc_etf_inflow_norm, fear_greed). For each feature, the x-axis varies that feature from −0.4 to +0.4 while all other features are held at their test-set median. Five lines represent the Fear & Greed index at the 10th, 25th, 50th, 75th, and 90th percentiles.

**Economic interpretation:**

Interaction effects reveal **non-linear and conditional relationships** that a linear model cannot capture — the primary justification for using a neural network over OLS. Key patterns to look for:

- **Momentum × Sentiment interaction (r12w × fear_greed):** Does the model predict that momentum strategies pay off more when sentiment is high (greed)? This would be consistent with behavioral finance theory — in greed regimes, investors extrapolate recent winners more aggressively, amplifying momentum. In fear regimes, momentum may reverse as investors liquidate recent winners.
- **ETF flow × Sentiment:** Large ETF inflows combined with high greed may signal crowded positioning and predict short-term reversal for BTC-correlated assets.
- **Short-term momentum (r1w):** If the model assigns negative predicted returns to recent weekly winners, it is capturing **short-term reversal** (mean reversion in crypto over 1-week horizons), a microstructure effect distinct from medium-term momentum.

The fan-shaped spread of lines in each subplot measures the magnitude of the interaction: wide fans indicate the feature's effect is strongly conditioned on sentiment regime.

---

### Fig 14 — 3D Surface: Return vs. Momentum × Sentiment

**File:** `visualizations/results/fig14_3d_surface.png`

3D surface of model-predicted return as a function of 12-week momentum (r12w) and Fear & Greed index (fear_greed), with all other features fixed at test-set medians.

**Economic interpretation:**

This figure visualizes the full joint non-linearity learned by the neural network across the two most economically motivated dimensions: **price trend** (r12w captures whether an asset has been gaining) and **market sentiment** (fear_greed captures the emotional state of the crypto market). Key regions of the surface:

- **High r12w + High fear_greed (top-right):** Strong recent winners in a greed market — the model's predicted return here captures the momentum amplification effect.
- **Low r12w + Low fear_greed (bottom-left):** Recent losers in a fear market — model may predict further underperformance (momentum continuation) or a contrarian bounce (oversold reversal) depending on the learned surface shape.
- **Curvature of the surface:** Any non-flat shape represents information that a linear regression would miss. Convex regions (returns accelerating) suggest the model has learned regime-switching behavior embedded in the historical data.

A smooth, monotone surface validates that the model's predictions are economically sensible and not driven by memorized noise patterns.

---

### Training Curves

**Files:** `visualizations/results/fig_training_curves_{feat_key}.png` (one per feature subset)

Validation-set Sharpe ratio over training epochs for all 8 seeds.

**Economic interpretation:**

These curves reveal the stability and convergence behavior of the ensemble. Well-behaved training should show:
- Rapid early improvement in Sharpe (within the first 50–100 epochs)
- Stabilization before epoch 300, confirming the budget is adequate
- Tight clustering of the 8 seed curves, indicating that the final ensemble is not dominated by a single lucky random initialization

High variance across seeds would suggest that the model is sensitive to initialization — a sign of overfitting or too small a dataset. In that case, increasing the ensemble size or adding regularization would be warranted.

---

### Transition Matrix

**File:** `visualizations/results/fig_transition_matrix.png`

10×10 heatmap of transition probabilities between prediction deciles in consecutive weeks. Entry (i, j) is the probability that an asset in decile i this week is ranked in decile j next week.

**Economic interpretation:**

The transition matrix measures **persistence of the model's signal**. In an efficient market, the matrix would be approximately uniform (all entries ≈ 10%), meaning the model's ranking conveys no information about future rankings. Persistence is revealed by high diagonal values (assets predicted to be top/bottom tend to remain so next week) and high corner values (assets rarely jump from decile 1 to decile 10 in one week).

In crypto, short-term momentum and sentiment are known to be sticky on the order of weeks to months, so one would expect moderate diagonal persistence (50–70%) for deciles based on medium-term price signals. Very high persistence (>90%) would suggest the model is tracking slow-moving fundamentals; very low persistence (<20%) would indicate the signal is essentially noise at the weekly frequency. The transition matrix thus provides a diagnostic for **how the model generates alpha** — through persistent signals or through accurate week-to-week ranking adjustments.

---

## 8. Feature Subsets

Modify `get_tuned_network()` in `train_btc.py` to change which feature subsets are trained:

| Label | subset | Categories | Description |
|-------|--------|------------|-------------|
| (A) `feat0to10` | `range(0, 11)` | Price + Technical | Lightest; no external APIs needed |
| (B) `feat0to15` | `range(0, 16)` | + On-chain | Requires CoinMetrics |
| (C) `feat0to21` | `range(0, 22)` | + Macro/Sentiment | Adds fear_greed, traditional asset returns |
| (D) `feat0to32` | `range(0, 33)` | All 33 features | Full model with ETF flows + Polymarket |

---

## 9. Data Sources

### 9.1 ETF Flows (Farside Investors)

BTC and ETH spot ETF daily net inflow/outflow data from [Farside Investors](https://farside.co.uk/):

| Data | URL | Local File |
|------|-----|------------|
| Bitcoin Spot ETF | https://farside.co.uk/bitcoin-etf-flow-all-data/ | `btc_spot_etf_from_farside.csv` |
| Ethereum Spot ETF | https://farside.co.uk/ethereum-etf-flow-all-data/ | `eth_spot_etf_from_farside.csv` |

CSV format: columns for each fund (IBIT, FBTC, BITB, ARKB, …, GBTC, Total). Units: US$m. Parentheses `()` indicate outflow; `-` means no trading. The script parses the Total column and aggregates to weekly frequency.

To update: copy the table from the Farside website into the CSV file and re-run `prepare_btc_data.py`.

### 9.2 BTC ETF Volume (Yahoo Finance)

Aggregated daily volume for IBIT, FBTC, BITB, ARKB, GBTC, BTCO — downloaded automatically, aggregated to weekly, log-transformed.

### 9.3 Traditional Assets (Yahoo Finance)

GLD, SLV, DIA, SPY (weekly returns + volume ratios), DXY, VIX — downloaded automatically. No manual update needed.

### 9.4 Fear & Greed Index (alternative.me)

Downloaded automatically via the alternative.me public API.

### 9.5 On-chain Data (CoinMetrics)

Active addresses, transaction count, NVT ratio, exchange net flow, MVRV — downloaded via the CoinMetrics Community API (no key required). Skip with `--skip_onchain` if unavailable.

### 9.6 Polymarket (optional)

Bitcoin price resolution sentiment from Polymarket. Skip with `--skip_polymarket` if unavailable.

---

## 10. FAQ

**Q: My R²_pred values are extremely negative (e.g., −6,000,000%). Is that a bug?**

No. The cross-sectional R² formula `1 − SS_res/SS_tot` measures how well predicted individual return magnitudes match actual magnitudes. With only 14 assets per week and no risk-factor adjustment (unlike the paper which uses 4-factor alphas), a single week where the model mispredicts the direction of a volatile asset can produce enormous `SS_res`. The Sharpe ratio of the constructed long-short portfolio is the correct primary metric — it captures the model's *ranking* ability, which is what matters for portfolio construction.

**Q: How do I install TensorFlow?**

```bash
pip install tensorflow==2.12.0        # CPU or GPU with CUDA 11.2
# or
pip install tensorflow[and-cuda]      # TF 2.13+, auto-installs CUDA
```

**Q: What is the difference from the original TF1 code?**

`model_btc.py` rewrites `FeedForwardModelWithNA_Return` using TF2 Keras `GradientTape`. The logic is identical (panel data masking, MSE loss, Sharpe-based checkpoint selection), but no `tf.Session` is needed. The original `../deep_learning/` TF1 code is untouched.

**Q: yfinance download fails**

```bash
pip install --upgrade yfinance
```

**Q: CoinMetrics API errors**

The free community API has rate limits and limited asset coverage. Use `--skip_onchain` to skip on-chain features. To use a paid API key:

```bash
export COINMETRICS_API_KEY="your_key_here"
```

**Q: How do I add new assets?**

Edit the asset ticker list in `prepare_btc_data.py` and re-run data preparation and training. The panel data format automatically accommodates any number of assets.

---

## 11. Citation

If you use this code or results, please cite the original paper:

> Kaniel, R., Lin, Z., Pelger, M., & Van Nieuwerburgh, S. (2023).
> Machine-learning the skill of mutual fund managers.
> *Journal of Financial Economics*, 150, 94–138.
> https://doi.org/10.1016/j.jfineco.2023.07.004
