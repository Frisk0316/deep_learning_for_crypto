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
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import requests

try:
    import cloudscraper
    _scraper = cloudscraper.create_scraper()
except ImportError:
    _scraper = None

COINMETRICS_BASE = "https://community-api.coinmetrics.io/v4"

# CoinMetrics 資產代碼映射
# 注意：CoinMetrics Community API 並非支援所有代幣，
# 不支援的資產會在 build_onchain_panel() 中自動跳過並填入 UNK。
ASSET_MAP = {
    # Tier 1
    "BTC":    "btc",
    "ETH":    "eth",
    "XRP":    "xrp",
    "BNB":    "bnb",
    "SOL":    "sol",
    "DOGE":   "doge",
    "ADA":    "ada",
    "TRX":    "trx",
    "AVAX":   "avax",
    "LINK":   "link",
    # Tier 2
    "SHIB":   "shib",
    "XLM":    "xlm",
    "DOT":    "dot",
    "HBAR":   "hbar",
    "BCH":    "bch",
    "LTC":    "ltc",
    "UNI":    "uni",
    "NEAR":   "near",
    "ICP":    "icp",
    "AAVE":   "aave",
    "ETC":    "etc",
    "FET":    "fet",
    "CRO":    "cro",
    "POL":    "matic",   # CoinMetrics 仍使用 matic
    "ATOM":   "atom",
    # Tier 3
    "VET":    "vet",
    "FIL":    "fil",
    "ALGO":   "algo",
    "GRT":    "grt",
    "THETA":  "theta",
    "FTM":    "ftm",
    "MKR":    "mkr",
    "LDO":    "ldo",
    # Tier 4
    "ENS":    "ens",
    "AXS":    "axs",
    "FLOW":   "flow",
    "NEO":    "neo",
    "GALA":   "gala",
    "KAVA":   "kava",
    "XTZ":    "xtz",
    "EOS":    "eos",
    "SAND":   "sand",
    "MANA":   "mana",
    "CHZ":    "chz",
    # Tier 5
    "SNX":    "snx",
    "CRV":    "crv",
    "COMP":   "comp",
    "ZEC":    "zec",
    "IOTA":   "iota",
    "ZIL":    "zil",
    "CELO":   "celo",
    "ANKR":   "ankr",
    "SKL":    "skl",
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

# 每個資產可用指標的快取（避免重複查詢 catalog）
_catalog_cache: dict[str, list[str]] = {}


def _get_available_metrics(cm_asset: str) -> list[str]:
    """查詢 CoinMetrics catalog，回傳該資產實際支援的指標列表。"""
    if cm_asset in _catalog_cache:
        return _catalog_cache[cm_asset]

    try:
        resp = requests.get(
            f"{COINMETRICS_BASE}/catalog/assets",
            params={"assets": cm_asset}, timeout=30
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data:
            all_metrics = [
                m.get("metric", m) if isinstance(m, dict) else m
                for m in data[0].get("metrics", [])
            ]
        else:
            all_metrics = []
    except Exception as e:
        print(f"    [WARN] catalog query failed for {cm_asset}: {e}")
        all_metrics = []

    _catalog_cache[cm_asset] = all_metrics
    return all_metrics


def _request_with_retry(url: str, params: dict, max_retries: int = 3) -> requests.Response | None:
    """帶重試與退避的 HTTP GET，403 時嘗試 cloudscraper。"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 403 and _scraper is not None:
                print(f"    [RETRY] 403 Forbidden, trying cloudscraper...")
                resp = _scraper.get(url, params=params, timeout=60)
                if resp.status_code == 200:
                    return resp
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = 2 ** (attempt + 1)
                print(f"    [RETRY] HTTP {resp.status_code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            # 400 等客戶端錯誤不重試
            print(f"    [ERR] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        except requests.exceptions.RequestException as e:
            wait = 2 ** (attempt + 1)
            print(f"    [RETRY] {e}, waiting {wait}s...")
            time.sleep(wait)
    return None


def _fetch_asset_metrics(cm_asset: str, start: str, end: str | None) -> pd.DataFrame:
    """
    呼叫 CoinMetrics Community API，取得單一資產的日頻鏈上指標。
    先查詢 catalog 確認可用指標，只請求實際支援的。
    """
    available = _get_available_metrics(cm_asset)
    supported = [m for m in METRICS if m in available]

    if not supported:
        print(f"    [WARN] {cm_asset}: 無可用鏈上指標 (catalog: {len(available)} metrics)")
        return pd.DataFrame()

    print(f"    支援指標: {supported}")

    params = {
        "assets":      cm_asset,
        "metrics":     ",".join(supported),
        "start_time":  start,
        "frequency":   "1d",
        "page_size":   10000,
    }
    if end:
        params["end_time"] = end

    resp = _request_with_retry(f"{COINMETRICS_BASE}/timeseries/asset-metrics", params)
    if resp is None:
        return pd.DataFrame()

    try:
        payload = resp.json()
    except Exception as e:
        print(f"    [ERR] JSON parse failed for {cm_asset}: {e}")
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
