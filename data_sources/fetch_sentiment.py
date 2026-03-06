"""
fetch_sentiment.py
------------------
取得巨觀情緒與市場情境指標（對所有加密資產為共同特徵）：

  fear_greed  : Bitcoin Fear & Greed Index（alternative.me，免費）
  spx_ret     : S&P 500 週報酬（yfinance，免費）
  dxy_ret     : 美元指數週報酬（yfinance，免費）
  vix         : VIX 恐慌指數週均值（yfinance，免費）

這 4 個特徵在每個時間點對所有資產相同，
模型可從中學習「市場情境 × 資產特徵」的交互作用。
"""

import numpy as np
import pandas as pd
import requests
import yfinance as yf

UNK = -99.99


# ── Fear & Greed Index ─────────────────────────────────────────────────────────

def fetch_fear_greed(start: str = "2020-01-01") -> pd.Series:
    """
    從 alternative.me 取得 Bitcoin Fear & Greed Index（0~100）。
    正規化至 [-1, 1]：0 = 極度恐懼、1 = 極度貪婪。
    聚合至週頻（週均值）。
    """
    url = "https://api.alternative.me/fng/"
    params = {"limit": 3000, "format": "json", "date_format": "us"}

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"]
    except Exception as e:
        print(f"  [ERR] Fear & Greed: {e}")
        return pd.Series(dtype=float, name="fear_greed")

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # 正規化至 [-1, 1]
    series = ((df["value"] - 50) / 50).rename("fear_greed")

    # 週頻
    series_weekly = series.resample("W").mean()
    return series_weekly[series_weekly.index >= pd.to_datetime(start)]


# ── 總體指標（S&P 500, DXY, VIX）──────────────────────────────────────────────

def fetch_macro(start: str = "2020-01-01") -> pd.DataFrame:
    """
    從 yfinance 下載週頻總體指標並計算特徵：
      spx_ret : S&P 500 週報酬（SPY ETF 代理）
      dxy_ret : DXY 週報酬（DX-Y.NYB）
      vix     : VIX 週均值，正規化至 [-1, 1] 以 80 為上界

    Returns
    -------
    pd.DataFrame, index = 週結束日, columns = [spx_ret, dxy_ret, vix]
    """
    tickers = {
        "SPY":      "spx_ret",
        "DX-Y.NYB": "dxy_ret",
        "^VIX":     "vix_raw",
    }

    dfs = {}
    for ticker, col_name in tickers.items():
        try:
            df = yf.download(
                ticker, start=start, interval="1wk",
                progress=False, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index = df.index + pd.to_timedelta(6 - df.index.weekday, unit="D")
            # 舊版寫法，discard df.index = df.index.to_period("W").to_timestamp("Sun")
            dfs[col_name] = df["Close"]
            print(f"  [OK] {ticker}: {len(df)} 週")
        except Exception as e:
            print(f"  [ERR] {ticker}: {e}")

    macro = pd.DataFrame(dfs)

    if "spx_ret" in macro.columns:
        macro["spx_ret"] = macro["spx_ret"].pct_change()
    if "dxy_ret" in macro.columns:
        macro["dxy_ret"] = macro["dxy_ret"].pct_change()
    if "vix_raw" in macro.columns:
        # 正規化：VIX 典型範圍 10~80，映射至 [-1, 1]
        macro["vix"] = (macro["vix_raw"].clip(10, 80) - 10) / 35 - 1
        macro = macro.drop(columns=["vix_raw"])

    return macro


# ── 整合成 (T, 4) 矩陣 ────────────────────────────────────────────────────────

def build_sentiment_panel(
    dates: pd.DatetimeIndex,
    start: str = "2020-01-01",
) -> np.ndarray:
    """
    建立情緒/總體特徵矩陣，shape = (T, 4)。
    這 4 個特徵對所有資產相同，在 prepare_btc_data.py 中
    會廣播至 (T, N, 4)。

    Returns
    -------
    np.ndarray, shape (T, 4)
      columns: fear_greed, spx_ret, dxy_ret, vix
    """
    print("  Fetching Fear & Greed Index...")
    fg = fetch_fear_greed(start=start)

    print("  Fetching macro data (SPY, DXY, VIX)...")
    macro = fetch_macro(start=start)

    T = len(dates)
    panel = np.full((T, 4), UNK, dtype=np.float32)
    feat_sources = [
        (0, fg,               "fear_greed"),
        (1, macro.get("spx_ret") if macro is not None else None, "spx_ret"),
        (2, macro.get("dxy_ret") if macro is not None else None, "dxy_ret"),
        (3, macro.get("vix")     if macro is not None else None, "vix"),
    ]

    for col_idx, series, name in feat_sources:
        if series is None or series.empty:
            print(f"  [WARN] {name}: 無資料，跳過")
            continue
        for t, date in enumerate(dates):
            if date in series.index:
                v = series.loc[date]
                panel[t, col_idx] = float(v) if not np.isnan(v) else UNK

    return panel
