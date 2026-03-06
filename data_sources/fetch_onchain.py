"""
fetch_onchain.py
----------------
從 CoinMetrics Community API 取得主要加密貨幣的鏈上指標。

資料來源：CoinMetrics Community Data（免費，無需 API key）
API endpoint：https://community-api.coinmetrics.io/v4/timeseries/asset-metrics

涵蓋指標：
  - AdrActCnt   : 活躍地址數（Active Addresses）
  - TxCnt       : 交易筆數（Transaction Count）
  - NVTAdj      : 網路價值對交易量比（NVT Ratio，adjusted）
  - CapMVRVFF   : 市值 / 已實現市值（MVRV Ratio，free float）
  - FlowInExNtv : 交易所流入量（Exchange Inflow）
  - FlowOutExNtv: 交易所流出量（Exchange Outflow）
  - HashRate    : 雜湊率（僅 BTC/LTC 等 PoW 鏈可用）
"""

import time
import numpy as np
import pandas as pd
import requests

COINMETRICS_BASE = "https://community-api.coinmetrics.io/v4"

# CoinMetrics 資產代碼映射
ASSET_MAP = {
    "BTC":   "btc",
    "ETH":   "eth",
    "SOL":   "sol",
    "BNB":   "bnb",
    "XRP":   "xrp",
    "AVAX":  "avax",
    "DOGE":  "doge",
    "ADA":   "ada",
    "MATIC": "matic",
    "LINK":  "link",
    "DOT":   "dot",
    "LTC":   "ltc",
    "UNI":   "uni",
    "ATOM":  "atom",
}

# 嘗試抓取的鏈上指標（community tier 可能不支援全部）
METRICS = [
    "AdrActCnt",
    "TxCnt",
    "NVTAdj",
    "CapMVRVFF",
    "FlowInExNtv",
    "FlowOutExNtv",
    "HashRate",
]

UNK = -99.99


def _fetch_asset_metrics(cm_asset: str, start: str, end: str | None) -> pd.DataFrame:
    """
    呼叫 CoinMetrics Community API，取得單一資產的日頻鏈上指標。
    自動過濾掉該資產不支援的指標。
    """
    params = {
        "assets":      cm_asset,
        "metrics":     ",".join(METRICS),
        "start_time":  start,
        "frequency":   "1d",
        "page_size":   10000,
    }
    if end:
        params["end_time"] = end

    try:
        resp = requests.get(
            f"{COINMETRICS_BASE}/timeseries/asset-metrics",
            params=params, timeout=60
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        print(f"    [ERR] CoinMetrics {cm_asset}: {e}")
        return pd.DataFrame()

    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
    df = df.set_index("time").drop(columns=["asset"], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def _compute_onchain_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    將日頻鏈上指標轉換為週頻特徵（共 5 個）：
      active_addr      : 活躍地址數 4 週成長率
      tx_count         : 交易筆數 4 週成長率
      nvt              : NVT ratio（高 = 鏈上估值偏貴）
      exchange_net_flow: 交易所淨流入（正 = 賣壓增加）
      mvrv             : MVRV ratio（>1 表示市場整體獲利）
    缺少的欄位設為 NaN。
    """
    # 聚合至週（週日收盤）
    df = df_daily.resample("W").mean()
    feat = pd.DataFrame(index=df.index)

    if "AdrActCnt" in df.columns:
        feat["active_addr"] = df["AdrActCnt"].pct_change(4)

    if "TxCnt" in df.columns:
        feat["tx_count"] = df["TxCnt"].pct_change(4)

    if "NVTAdj" in df.columns:
        # log 化後做 4 週標準化
        nvt_log = np.log(df["NVTAdj"].clip(lower=1e-6))
        feat["nvt"] = (nvt_log - nvt_log.rolling(52, min_periods=4).mean()) / (
            nvt_log.rolling(52, min_periods=4).std() + 1e-10
        )

    if "FlowInExNtv" in df.columns and "FlowOutExNtv" in df.columns:
        total_flow = df["FlowInExNtv"] + df["FlowOutExNtv"] + 1e-10
        feat["exchange_net_flow"] = (
            (df["FlowInExNtv"] - df["FlowOutExNtv"]) / total_flow
        ).rolling(4).mean()
    else:
        feat["exchange_net_flow"] = np.nan

    if "CapMVRVFF" in df.columns:
        feat["mvrv"] = df["CapMVRVFF"]
    else:
        feat["mvrv"] = np.nan

    return feat


def build_onchain_panel(
    assets: list[str],
    dates: pd.DatetimeIndex,
    start: str = "2020-01-01",
    end: str | None = None,
) -> np.ndarray:
    """
    建立鏈上特徵 panel，shape = (T, N, 5)。
    缺失值填 UNK (-99.99)。

    Parameters
    ----------
    assets : list[str]  資產名稱列表（與 fetch_prices 回傳一致）
    dates  : pd.DatetimeIndex  目標日期索引（T）

    Returns
    -------
    np.ndarray, shape (T, N, 5)
    """
    T, N = len(dates), len(assets)
    panel = np.full((T, N, 5), UNK, dtype=np.float32)
    feature_names = ["active_addr", "tx_count", "nvt", "exchange_net_flow", "mvrv"]

    for n, name in enumerate(assets):
        cm_asset = ASSET_MAP.get(name)
        if cm_asset is None:
            print(f"  [SKIP] {name}: 無 CoinMetrics 對應代碼")
            continue

        print(f"  Fetching on-chain: {name} ({cm_asset})...")
        df_daily = _fetch_asset_metrics(cm_asset, start, end)
        time.sleep(0.5)  # 避免觸發 rate limit

        if df_daily.empty:
            print(f"    [WARN] {name}: 無鏈上資料")
            continue

        feat = _compute_onchain_features(df_daily)

        for f_idx, f_name in enumerate(feature_names):
            if f_name not in feat.columns:
                continue
            for t, date in enumerate(dates):
                if date in feat.index:
                    v = feat.loc[date, f_name]
                    panel[t, n, f_idx] = float(v) if not np.isnan(v) else UNK

    return panel
