# Research Gap Analysis Report

> Auto-generated from clustering analysis of academic literature (Cryptocurrency × Machine Learning / Deep Learning)

## 1. Literature Landscape Overview

A total of **743 papers** were analyzed, spanning the period **2017–2026**. The corpus was clustered into **8 thematic groups** using KMEANS (silhouette score: 0.0440).

The dominant methodology is **Not specified**, reflecting the field's strong preference for sequence-based deep learning. The most common data source is **Bitcoin price**, indicating that Bitcoin remains the primary experimental asset.

### Publication Trend by Year

| Year | Papers |
|------|--------|
| 2017 | 3 |
| 2018 | 4 |
| 2019 | 11 |
| 2020 | 21 |
| 2021 | 22 |
| 2022 | 49 |
| 2023 | 104 |
| 2024 | 160 |
| 2025 | 244 |
| 2026 | 124 |

## 2. Methodology Distribution

| Rank | Methodology | Papers |
|------|-------------|--------|
| 1 | Not specified | 212 |
| 2 | LSTM | 183 |
| 3 | Reinforcement Learning | 134 |
| 4 | Random Forest | 70 |
| 5 | GRU | 63 |
| 6 | Ensemble | 61 |
| 7 | RNN | 57 |
| 8 | CNN | 54 |
| 9 | Transformer | 53 |
| 10 | XGBoost | 49 |

## 3. Data Source Distribution

| Rank | Data Source | Papers |
|------|-------------|--------|
| 1 | Bitcoin price | 384 |
| 2 | Not specified | 206 |
| 3 | Blockchain data | 103 |
| 4 | Cryptocurrency market | 90 |
| 5 | Ethereum | 83 |
| 6 | Transaction data | 79 |
| 7 | Sentiment data | 77 |
| 8 | Stock market data | 60 |
| 9 | Mining/Hash rate data | 33 |
| 10 | Binance | 27 |

## 4. Cluster Details

### Cluster 0 (194 papers, 2019–2026)

- **Top methods**: LSTM, GRU, RNN
- **Top data sources**: Bitcoin price, Ethereum, Sentiment data
- **Sample titles**:
  - Financial time series augmentation using transformer based GAN architecture
  - TFT-ACB-XML: Decision-Level Integration of Customized Temporal Fusion Transforme
  - Smart Timing for Mining: A Deep Learning Framework for Bitcoin Hardware ROI Pred

### Cluster 1 (57 papers, 2021–2026)

- **Top methods**: Reinforcement Learning, Not specified, DQN
- **Top data sources**: Not specified
- **Sample titles**:
  - A Quantum Neural Network Transfer-Learning Model for Forecasting Problems with C
  - Dynamic AI-Enhanced Therapeutic Framework for Precision Medicine Using Multi-Mod
  - Digital Predistortion Model Extraction Using Reinforcement Learning Without Time

### Cluster 2 (91 papers, 2016–2026)

- **Top methods**: Not specified, Reinforcement Learning, Transformer
- **Top data sources**: Not specified, Mining/Hash rate data, DeFi protocol data
- **Sample titles**:
  - A Comprehensive Survey on Architectural Advances in Deep CNNs: Challenges, Appli
  - Deep Learning and Machine Learning with GPGPU and CUDA: Unlocking the Power of P
  - Emergent Specialization in Learner Populations: Competition as the Source of Div

### Cluster 3 (122 papers, 2020–2026)

- **Top methods**: Random Forest, LSTM, Not specified
- **Top data sources**: Bitcoin price, Sentiment data, Ethereum
- **Sample titles**:
  - Analytics of Business Time Series Using Machine Learning and Bayesian Inference
  - What drives bitcoin? An approach from continuous local transfer entropy and deep
  - Efficient Bitcoin Price Forecasting Using Deep Learning and Ensemble Methods

### Cluster 4 (89 papers, 2017–2026)

- **Top methods**: Not specified, Graph Neural Network, Random Forest
- **Top data sources**: Bitcoin price, Transaction data, Blockchain data
- **Sample titles**:
  - DynBERG: Dynamic BERT-based Graph neural network for financial fraud detection
  - From Asset Flow to Status, Action and Intention Discovery: Early Malice Detectio
  - The Broad Impact of Feature Imitation: Neural Enhancements Across Financial, Spe

### Cluster 5 (40 papers, 2017–2026)

- **Top methods**: Not specified, Reinforcement Learning, Transformer
- **Top data sources**: Bitcoin price, Blockchain data, Mining/Hash rate data
- **Sample titles**:
  - Mastering AI: Big Data, Deep Learning, and the Evolution of Large Language Model
  - Optical Proof of Work
  - Coin.AI: A Proof-of-Useful-Work Scheme for Blockchain-based Distributed Deep Lea

### Cluster 6 (75 papers, 2018–2026)

- **Top methods**: Not specified, Transformer, GARCH
- **Top data sources**: Bitcoin price, Not specified, Sentiment data
- **Sample titles**:
  - Expert System for Bitcoin Forecasting: Integrating Global Liquidity via TimeXer 
  - From Patterns to Predictions: A Shapelet-Based Framework for Directional Forecas
  - Bitcoin Price Forecasting Based on Hybrid Variational Mode Decomposition and Lon

### Cluster 7 (75 papers, 2020–2026)

- **Top methods**: Reinforcement Learning, DQN, PPO
- **Top data sources**: Cryptocurrency market, Bitcoin price, Not specified
- **Sample titles**:
  - Feature-Rich Long-term Bitcoin Trading Assistant
  - Bitcoin Transaction Strategy Construction Based on Deep Reinforcement Learning
  - A comparative study of Bitcoin and Ripple cryptocurrencies trading using Deep Re

## 5. Overcrowded Areas

The following areas show high paper density with diminishing marginal novelty:

- **Cluster 0** (194 papers): LSTM, GRU applied to crypto price prediction. This combination is heavily studied; incremental improvements offer low novelty.
- **Cluster 3** (122 papers): Random Forest, LSTM applied to crypto price prediction. This combination is heavily studied; incremental improvements offer low novelty.

## 6. Research Gaps & Opportunities

Based on the distribution of methods and data sources, the following directions appear underexplored:

### Gap 1: On-chain Data × Deep Learning

**Why it matters**: Most papers focus on price OHLCV data. On-chain metrics (e.g., UTXO age, miner flow, exchange reserves) encode fundamentally different market dynamics and remain underutilized.

**Suggested methodology**: Graph Neural Networks or Transformer on transaction graph snapshots

**Expected data sources**: Glassnode, CryptoQuant, Dune Analytics

### Gap 2: Cross-asset & Macro Integration

**Why it matters**: Literature rarely combines crypto with macro indicators (e.g., DXY, Fed rate, VIX). Cross-market spillover effects are understudied.

**Suggested methodology**: Multi-variate Transformer / Attention with macro time series

**Expected data sources**: FRED, Yahoo Finance, Binance OHLCV

### Gap 3: DeFi Protocol Risk Modeling

**Why it matters**: DeFi-specific risks (liquidation cascades, impermanent loss, smart-contract exploits) are not well-modeled by existing price-prediction ML pipelines.

**Suggested methodology**: Reinforcement Learning or Survival Analysis on protocol event logs

**Expected data sources**: The Graph, Dune Analytics, DeFi Llama

## 7. Emerging Trends

Publication volume has concentrated in the most recent years (2024, 2025, 2026), with **528 papers** — indicating rapid growth in the field.

**Trending methods in recent clusters**: Not specified, LSTM, Reinforcement Learning, GRU, Random Forest

---
*Report generated automatically from cluster statistics.*