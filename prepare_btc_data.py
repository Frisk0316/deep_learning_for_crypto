"""
prepare_btc_data.py
-------------------
BTC 多資產 Panel 資料整合主腳本。

執行流程：
  1. 下載 14 個主流加密資產的週 OHLCV → 計算價格動能 + 技術指標 (10 個特徵)
  2. 從 CoinMetrics Community API 取得鏈上指標 (5 個特徵)
  3. 從 alternative.me + yfinance 取得情緒/總體特徵 (4 個特徵)
  4. 從 Coinglass/SoSoValue 取得 BTC ETF 淨流入 (2 個特徵)
  5. 從 Polymarket 取得 BTC 看漲情緒指數 (1 個特徵)
  6. 橫截面排名標準化 → 儲存為 .npz

輸出資料集格式（與原始 mutual fund 程式相容）：
  data     : np.ndarray, shape (T, N, M+1)
             data[:, :, 0]  = 下週報酬（target），缺失為 -99.99
             data[:, :, 1:] = 22 個特徵（詳見 VARIABLE_NAMES）
  date     : np.ndarray, shape (T,)  週結束日（字串）
  wficn    : np.ndarray, shape (N,)  資產代碼（等同原始的基金 ID）
  variable : np.ndarray, shape (22,) 特徵名稱

特徵索引對應（22 個特徵）：
  Category A - Price Momentum  [0~4]  : r1w, r4w, r12w, r26w, r52w
  Category B - Technical       [5~9]  : rsi_14, bb_pct, vol_ratio, atr_pct, obv_change
  Category C - On-chain        [10~14]: active_addr, tx_count, nvt, exchange_net_flow, mvrv
  Category D - Market/Macro    [15~18]: fear_greed, spx_ret, dxy_ret, vix
  Category E - ETF + Polymarket[19~21]: etf_net_flow_norm, polymarket_btc (+ etf_net_flow raw)

訓練腳本中 subset = range(0, 22) 代表全部特徵；
可依研究設計選取子集，例如 range(0, 10) 僅使用價格/技術特徵。

使用方式：
  python prepare_btc_data.py [--start 2020-01-01] [--end 2025-12-31]
                             [--out datasets/btc_panel.npz]
                             [--etf_csv datasets/btc_etf_flows.csv]
                             [--skip_onchain] [--skip_polymarket]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# 將 data_sources 模組加入搜尋路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_sources.fetch_prices    import build_price_feature_panel, CRYPTO_SYMBOLS
from data_sources.fetch_onchain   import build_onchain_panel
from data_sources.fetch_sentiment import build_sentiment_panel
from data_sources.fetch_etf_flows import build_etf_panel
from data_sources.fetch_polymarket import build_polymarket_panel

UNK = -99.99

# 特徵名稱（共 22 個，index 對應 data[:, :, 1:23]）
VARIABLE_NAMES = [
    # Category A: Price Momentum
    "r1w", "r4w", "r12w", "r26w", "r52w",
    # Category B: Technical Indicators
    "rsi_14", "bb_pct", "vol_ratio", "atr_pct", "obv_change",
    # Category C: On-chain Metrics
    "active_addr", "tx_count", "nvt", "exchange_net_flow", "mvrv",
    # Category D: Macro / Sentiment（對所有資產相同）
    "fear_greed", "spx_ret", "dxy_ret", "vix",
    # Category E: BTC ETF Flows + Polymarket（對所有資產相同）
    "etf_net_flow_norm", "polymarket_btc", "etf_net_flow_raw",
]

N_FEATURES = len(VARIABLE_NAMES)  # 22


def _cross_sectional_rank_normalize(
    panel: np.ndarray,
    feature_slice: slice,
    unk: float = UNK,
) -> np.ndarray:
    """
    對個別資產特徵（Category A~C）做橫截面排名標準化：
      每個時間點 t，對所有有效資產的特徵值做百分位排名，
      映射至 [-1, 1]。

    Parameters
    ----------
    panel         : shape (T, N, F)  僅傳入特徵子陣列
    feature_slice : panel 中要標準化的特徵切片
    """
    T, N, F = panel.shape
    panel_out = panel.copy()

    for t in range(T):
        for f in range(F):
            col = panel[t, :, f]
            valid_mask = col != unk
            if valid_mask.sum() < 2:
                continue
            ranks = rankdata(col[valid_mask]).astype(np.float32)
            n_valid = valid_mask.sum()
            normalized = (ranks - 1) / (n_valid - 1) * 2 - 1  # → [-1, 1]
            panel_out[t, valid_mask, f] = normalized

    return panel_out


def _time_series_normalize(
    macro_panel: np.ndarray,
    unk: float = UNK,
    window: int = 52,
) -> np.ndarray:
    """
    對總體特徵（Category D~E，對所有資產相同）做時間序列滾動 z-score 標準化。

    Parameters
    ----------
    macro_panel : shape (T, F)  總體特徵矩陣
    """
    T, F = macro_panel.shape
    out = macro_panel.copy()

    for f in range(F):
        col = macro_panel[:, f].copy()
        valid = col != unk

        # 建立 rolling mean/std（只使用有效值）
        ser = pd.Series(np.where(valid, col, np.nan))
        roll_mean = ser.rolling(window, min_periods=4).mean()
        roll_std  = ser.rolling(window, min_periods=4).std()

        for t in range(T):
            if not valid[t]:
                continue
            mu  = roll_mean.iloc[t]
            std = roll_std.iloc[t]
            if pd.isna(mu) or pd.isna(std) or std < 1e-10:
                continue
            out[t, f] = (col[t] - mu) / std

    return out


def _clip_outliers(data: np.ndarray, unk: float = UNK, clip: float = 3.0) -> np.ndarray:
    """將非缺失值 clip 至 [-clip, clip]，避免極端值干擾訓練。"""
    out = data.copy()
    valid = out != unk
    out[valid] = np.clip(out[valid], -clip, clip)
    return out


def build_dataset(
    start: str = "2020-01-01",
    end:   str | None = None,
    skip_onchain:    bool = False,
    skip_polymarket: bool = False,
    etf_csv:         str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    整合所有資料來源，回傳 (data, dates, assets, variable_names)。

    data shape : (T, N, N_FEATURES + 1)
                 data[:, :, 0] = target return
                 data[:, :, 1:] = features
    """
    print("=" * 60)
    print("Step 1: 下載 OHLCV 資料 & 計算價格/技術特徵 (features 0~9)")
    print("=" * 60)
    dates, assets, price_panel = build_price_feature_panel(start=start, end=end)
    T, N = len(dates), len(assets)

    print(f"\n資料維度：T={T} 週, N={N} 個資產")
    print(f"日期範圍：{dates[0].date()} ~ {dates[-1].date()}")
    print(f"資產列表：{assets}")

    # 最終 data 陣列：[T, N, 1+N_FEATURES]
    # col 0 = return, col 1~22 = features
    data = np.full((T, N, 1 + N_FEATURES), UNK, dtype=np.float32)

    # price_panel shape: (T, N, 11) = [target_return, 10 features]
    data[:, :, 0]    = price_panel[:, :, 0]   # target
    data[:, :, 1:11] = price_panel[:, :, 1:]  # features 0~9

    # ── Step 2: 鏈上指標 ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: 取得鏈上指標 (features 10~14)")
    print("=" * 60)
    if not skip_onchain:
        onchain_panel = build_onchain_panel(assets=assets, dates=dates, start=start, end=end)
        data[:, :, 11:16] = onchain_panel  # features 10~14
    else:
        print("  [SKIP] 跳過鏈上資料")

    # ── Step 3: 情緒/總體特徵 ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: 取得情緒/總體特徵 (features 15~18)")
    print("=" * 60)
    sentiment_panel = build_sentiment_panel(dates=dates, start=start)  # (T, 4)
    # 廣播至所有資產（這些特徵對所有資產相同）
    for n in range(N):
        data[:, n, 16:20] = sentiment_panel  # features 15~18

    # ── Step 4: BTC ETF 淨流入 ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: 取得 BTC ETF 流量 (features 19, 21)")
    print("=" * 60)
    etf_panel = build_etf_panel(dates=dates, csv_backup=etf_csv)  # (T, 2)
    for n in range(N):
        data[:, n, 20] = etf_panel[:, 1]  # etf_net_flow_norm → feature 19
        data[:, n, 22] = etf_panel[:, 0]  # etf_net_flow_raw  → feature 21

    # ── Step 5: Polymarket ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 5: 取得 Polymarket BTC 看漲指數 (feature 20)")
    print("=" * 60)
    if not skip_polymarket:
        poly_panel = build_polymarket_panel(dates=dates, start=start)  # (T, 1)
        for n in range(N):
            data[:, n, 21] = poly_panel[:, 0]  # polymarket_btc → feature 20
    else:
        print("  [SKIP] 跳過 Polymarket 資料")

    # ── Step 6: 標準化 ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 6: 標準化")
    print("=" * 60)

    # 個別資產特徵（Category A~C, features 0~14）：橫截面排名標準化
    print("  橫截面排名標準化 (features 0~14)...")
    feat_cross = data[:, :, 1:16].copy()  # (T, N, 15)
    feat_cross = _cross_sectional_rank_normalize(feat_cross, slice(0, 15))
    data[:, :, 1:16] = feat_cross

    # 總體特徵（Category D~E, features 15~21）：時間序列滾動 z-score
    print("  時間序列 z-score 標準化 (features 15~21)...")
    feat_macro = data[:, 0, 16:23].copy()  # (T, 7)，取第 0 個資產代表（所有資產相同）
    feat_macro = _time_series_normalize(feat_macro)
    for n in range(N):
        data[:, n, 16:23] = feat_macro

    # clip 所有非缺失值至 [-3, 3]
    print("  Clip 極端值至 [-3, 3]...")
    data = _clip_outliers(data)

    print(f"\n資料集建立完成！")
    valid_return_pct = (data[:, :, 0] != UNK).mean() * 100
    print(f"目標報酬有效比例：{valid_return_pct:.1f}%")

    dates_arr  = np.array([d.strftime("%Y-%m-%d") for d in dates])
    assets_arr = np.array(assets)
    var_arr    = np.array(VARIABLE_NAMES)

    return data, dates_arr, assets_arr, var_arr


def main():
    parser = argparse.ArgumentParser(description="準備 BTC 多資產 Panel 資料集")
    parser.add_argument("--start",          default="2020-01-01",
                        help="資料起始日（預設 2020-01-01）")
    parser.add_argument("--end",            default=None,
                        help="資料結束日（預設今日）")
    parser.add_argument("--out",            default="datasets/btc_panel.npz",
                        help="輸出 NPZ 路徑")
    parser.add_argument("--etf_csv",        default=None,
                        help="ETF 流量 CSV 備援路徑（可選）")
    parser.add_argument("--skip_onchain",   action="store_true",
                        help="跳過 CoinMetrics 鏈上資料（加快速度）")
    parser.add_argument("--skip_polymarket",action="store_true",
                        help="跳過 Polymarket 資料（加快速度）")
    args = parser.parse_args()

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    data, dates, assets, variables = build_dataset(
        start           = args.start,
        end             = args.end,
        skip_onchain    = args.skip_onchain,
        skip_polymarket = args.skip_polymarket,
        etf_csv         = args.etf_csv,
    )

    np.savez(
        args.out,
        data     = data,
        date     = dates,
        wficn    = assets,   # 對應原始 mutual fund 的基金 WFICN
        variable = variables,
    )
    print(f"\n[DONE] 資料集儲存至：{args.out}")
    print(f"       data.shape = {data.shape}")
    print(f"       {len(dates)} 週 × {len(assets)} 資產 × {data.shape[2]} 欄（1 return + {len(variables)} features）")


if __name__ == "__main__":
    main()
