"""
fetch_sentiment.py
------------------
取得巨觀情緒與市場情境指標（對所有加密資產為共同特徵）：

  fear_greed     : Bitcoin Fear & Greed Index（alternative.me，免費）
  spx_ret        : S&P 500 週報酬（yfinance，免費）
  dxy_ret        : 美元指數週報酬（yfinance，免費）
  vix            : VIX 恐慌指數週均值（yfinance，免費）
  gold_ret       : 黃金（GLD）週報酬
  silver_ret     : 白銀（SLV）週報酬
  dji_ret        : 道瓊工業指數（DIA）週報酬
  spx_vol_chg    : S&P 500（SPY）成交量比（vs 4 週均量）
  gold_vol_chg   : 黃金（GLD）成交量比
  silver_vol_chg : 白銀（SLV）成交量比
  dji_vol_chg    : 道瓊（DIA）成交量比

這 11 個特徵在每個時間點對所有資產相同，
模型可從中學習「市場情境 x 資產特徵」的交互作用。
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


# ── 總體指標（S&P 500, DXY, VIX, Gold, Silver, DJI）──────────────────────────

def fetch_macro(start: str = "2020-01-01") -> pd.DataFrame:
    """
    從 yfinance 下載週頻總體指標並計算特徵：
      spx_ret        : S&P 500 週報酬（SPY ETF 代理）
      dxy_ret        : DXY 週報酬（DX-Y.NYB）
      vix            : VIX 週均值，正規化至 [-1, 1] 以 80 為上界
      gold_ret       : 黃金 GLD 週報酬
      silver_ret     : 白銀 SLV 週報酬
      dji_ret        : 道瓊 DIA 週報酬
      spx_vol_chg    : SPY 成交量 / 4 週均量
      gold_vol_chg   : GLD 成交量 / 4 週均量
      silver_vol_chg : SLV 成交量 / 4 週均量
      dji_vol_chg    : DIA 成交量 / 4 週均量

    Returns
    -------
    pd.DataFrame, index = 週結束日
    """
    # 需要 Close 做報酬的 tickers
    tickers_close = {
        "SPY":      "spx",
        "DX-Y.NYB": "dxy",
        "^VIX":     "vix_raw",
        "GLD":      "gold",
        "SLV":      "silver",
        "DIA":      "dji",
    }
    # 需要 Volume 做量比的 tickers（DXY 和 VIX 無意義成交量）
    tickers_volume = {"SPY": "spx", "GLD": "gold", "SLV": "silver", "DIA": "dji"}

    closes = {}
    volumes = {}

    for ticker, label in tickers_close.items():
        try:
            df = yf.download(
                ticker, start=start, interval="1wk",
                progress=False, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index = df.index + pd.to_timedelta(6 - df.index.weekday, unit="D")
            closes[label] = df["Close"]
            if ticker in tickers_volume:
                volumes[label] = df["Volume"]
            print(f"  [OK] {ticker}: {len(df)} 週")
        except Exception as e:
            print(f"  [ERR] {ticker}: {e}")

    macro = pd.DataFrame(closes)

    # 計算報酬
    for label, col_name in [("spx", "spx_ret"), ("dxy", "dxy_ret"),
                            ("gold", "gold_ret"), ("silver", "silver_ret"),
                            ("dji", "dji_ret")]:
        if label in macro.columns:
            macro[col_name] = macro[label].pct_change()

    # VIX 正規化：典型範圍 10~80，映射至 [-1, 1]
    if "vix_raw" in macro.columns:
        macro["vix"] = (macro["vix_raw"].clip(10, 80) - 10) / 35 - 1

    # 計算成交量比（vs 4 週均量）
    vol_df = pd.DataFrame(volumes)
    for label in tickers_volume.values():
        col_name = f"{label}_vol_chg"
        if label in vol_df.columns:
            vol = vol_df[label].astype(float)
            macro[col_name] = vol / (vol.rolling(4).mean() + 1e-10)

    # 只保留最終特徵欄
    output_cols = [
        "spx_ret", "dxy_ret", "vix",
        "gold_ret", "silver_ret", "dji_ret",
        "spx_vol_chg", "gold_vol_chg", "silver_vol_chg", "dji_vol_chg",
    ]
    result = pd.DataFrame(index=macro.index)
    for col in output_cols:
        if col in macro.columns:
            result[col] = macro[col]
        else:
            result[col] = np.nan

    return result


# ── 整合成 (T, 11) 矩陣 ───────────────────────────────────────────────────────

def build_sentiment_panel(
    dates: pd.DatetimeIndex,
    start: str = "2020-01-01",
) -> np.ndarray:
    """
    建立情緒/總體特徵矩陣，shape = (T, 11)。
    這 11 個特徵對所有資產相同，在 prepare_btc_data.py 中
    會廣播至 (T, N, 11)。

    Returns
    -------
    np.ndarray, shape (T, 11)
      columns: fear_greed, spx_ret, dxy_ret, vix,
               gold_ret, silver_ret, dji_ret,
               spx_vol_chg, gold_vol_chg, silver_vol_chg, dji_vol_chg
    """
    print("  Fetching Fear & Greed Index...")
    fg = fetch_fear_greed(start=start)

    print("  Fetching macro data (SPY, DXY, VIX, GLD, SLV, DIA)...")
    macro = fetch_macro(start=start)

    T = len(dates)
    panel = np.full((T, 11), UNK, dtype=np.float32)

    # 特徵順序
    feat_sources = [
        (0,  fg, "fear_greed"),
        (1,  macro.get("spx_ret"),        "spx_ret"),
        (2,  macro.get("dxy_ret"),        "dxy_ret"),
        (3,  macro.get("vix"),            "vix"),
        (4,  macro.get("gold_ret"),       "gold_ret"),
        (5,  macro.get("silver_ret"),     "silver_ret"),
        (6,  macro.get("dji_ret"),        "dji_ret"),
        (7,  macro.get("spx_vol_chg"),    "spx_vol_chg"),
        (8,  macro.get("gold_vol_chg"),   "gold_vol_chg"),
        (9,  macro.get("silver_vol_chg"), "silver_vol_chg"),
        (10, macro.get("dji_vol_chg"),    "dji_vol_chg"),
    ]

    for col_idx, series, name in feat_sources:
        if series is None or (hasattr(series, 'empty') and series.empty):
            print(f"  [WARN] {name}: 無資料，跳過")
            continue
        for t, date in enumerate(dates):
            if date in series.index:
                v = series.loc[date]
                panel[t, col_idx] = float(v) if not np.isnan(v) else UNK

    return panel
