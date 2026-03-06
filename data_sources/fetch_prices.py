"""
fetch_prices.py
---------------
從 yfinance 下載多個加密貨幣的 OHLCV 週資料，
並計算價格動能（Momentum）與技術指標（Technical）特徵。

資料來源：Yahoo Finance (免費，無需 API key)
涵蓋資產：BTC, ETH, SOL, BNB, XRP, AVAX, DOGE, ADA, MATIC, LINK, DOT, LTC, UNI, ATOM
"""

import numpy as np
import pandas as pd
import yfinance as yf

# ── 資產列表（名稱 → Yahoo Finance ticker）──────────────────────────────────
CRYPTO_SYMBOLS = {
    "BTC":   "BTC-USD",
    "ETH":   "ETH-USD",
    "SOL":   "SOL-USD",
    "BNB":   "BNB-USD",
    "XRP":   "XRP-USD",
    "AVAX":  "AVAX-USD",
    "DOGE":  "DOGE-USD",
    "ADA":   "ADA-USD",
    "MATIC": "MATIC-USD",
    "LINK":  "LINK-USD",
    "DOT":   "DOT-USD",
    "LTC":   "LTC-USD",
    "UNI":   "UNI-USD",
    "ATOM":  "ATOM-USD",
}

UNK = -99.99  # 與原始 mutual fund 程式一致的缺失值標記


def fetch_ohlcv(start="2020-01-01", end=None, interval="1wk"):
    """
    下載所有資產的週 OHLCV 資料。

    Returns
    -------
    dict[str, pd.DataFrame]
        {asset_name: DataFrame(Open, High, Low, Close, Volume)}
    """
    data = {}
    for name, symbol in CRYPTO_SYMBOLS.items():
        try:
            df = yf.download(
                symbol, start=start, end=end,
                interval=interval, progress=False, auto_adjust=True
            )
            if df.empty:
                print(f"  [WARN] {name}: 空資料")
                continue
            
            # yfinance 多層欄位處理
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            
            # 統一週結束日為週日 (Pandas 2.2+ 相容寫法)
            # 利用 weekday 屬性 (週一=0, 週日=6)，計算距離週日的天數差並加上去
            df.index = df.index + pd.to_timedelta(6 - df.index.weekday, unit="D")
            
            data[name] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            print(f"  [OK] {name}: {len(df)} 週")
        except Exception as e:
            print(f"  [ERR] {name}: {e}")
    return data


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    從 OHLCV DataFrame 計算 10 個特徵：
      Price Momentum (5) + Technical Indicators (5)

    Parameters
    ----------
    df : pd.DataFrame
        欄位需包含 Open, High, Low, Close, Volume

    Returns
    -------
    pd.DataFrame
        index = 週結束日，欄位 = 10 個特徵
    """
    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    feat = pd.DataFrame(index=df.index)

    # ── Category 0：Price Momentum ──────────────────────────────────────────
    feat["r1w"]  = close.pct_change(1)   # 1 週報酬
    feat["r4w"]  = close.pct_change(4)   # 4 週（約 1 個月）
    feat["r12w"] = close.pct_change(12)  # 12 週（約 3 個月）
    feat["r26w"] = close.pct_change(26)  # 26 週（約 6 個月）
    feat["r52w"] = close.pct_change(52)  # 52 週（約 1 年）

    # ── Category 1：Technical Indicators ───────────────────────────────────
    # RSI-14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    feat["rsi_14"] = (100 - 100 / (1 + rs)) / 100  # 正規化至 [0,1]

    # Bollinger Band %B（相對位置）
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    feat["bb_pct"] = (close - lower) / (upper - lower + 1e-10)

    # Volume ratio（當週成交量 / 4 週均量）
    feat["vol_ratio"] = volume / (volume.rolling(4).mean() + 1e-10)

    # ATR%（ATR / Close，衡量波動率）
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    feat["atr_pct"] = tr.rolling(14).mean() / (close + 1e-10)

    # OBV 4 週變化率
    obv = (np.sign(close.diff()) * volume).cumsum()
    feat["obv_change"] = obv.pct_change(4)

    return feat


def build_price_feature_panel(
    start="2020-01-01", end=None
) -> tuple[pd.DatetimeIndex, list[str], np.ndarray]:
    """
    建立 (T, N, 11) 的 panel：第 0 欄為「下週報酬（target）」，1~10 欄為特徵。

    Returns
    -------
    dates    : pd.DatetimeIndex (長度 T)
    assets   : list[str]       (長度 N)
    panel    : np.ndarray      shape (T, N, 11)
               panel[:, :, 0]   = next-week return (target, -99.99 if missing)
               panel[:, :, 1:6] = momentum features
               panel[:, :, 6:]  = technical features
    """
    raw = fetch_ohlcv(start=start, end=end)
    if not raw:
        raise RuntimeError("未能下載任何資產資料")

    # 建立共同日期索引
    all_dates = sorted(set.union(*[set(df.index) for df in raw.values()]))
    dates  = pd.DatetimeIndex(all_dates)
    assets = list(raw.keys())
    T, N   = len(dates), len(assets)

    # panel[t, n, 0] = 下週報酬；panel[t, n, 1:] = 特徵
    panel = np.full((T, N, 11), UNK, dtype=np.float32)

    for n, name in enumerate(assets):
        df = raw[name].reindex(dates)
        feat = compute_features(df)

        for t, date in enumerate(dates):
            if date not in feat.index or pd.isna(df.loc[date, "Close"]):
                continue
            row = feat.loc[date]
            if row.isna().all():
                continue
            # target = 下一期的 r1w
            if t + 1 < T and feat.index[t + 1] == dates[t + 1]:
                next_r1w = feat.iloc[t + 1]["r1w"]
                panel[t, n, 0] = next_r1w if not np.isnan(next_r1w) else UNK
            # features
            vals = row.values.astype(np.float32)
            for f, v in enumerate(vals):
                panel[t, n, f + 1] = v if not np.isnan(v) else UNK

    return dates, assets, panel