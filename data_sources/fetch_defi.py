"""
fetch_defi.py
-------------
DeFi + 衍生品數據抓取模組。

資料來源：
  A) DefiLlama (免費, 無 Key)  ─ TVL, DEX 交易量, 協議費用, 穩定幣市值
  B) Binance Futures (免費)    ─ 資金費率, 未平倉量, 多空比
  C) [預留] Token Terminal      ─ 協議 Revenue, P/S ratio (需 API Key)
  D) [預留] Etherscan           ─ ETH Gas 費用 (需 API Key)
  E) [預留] DappRadar           ─ UAW, dApp 交易量 (需 API Key)
  F) [預留] Flipside / Bitquery ─ 自定義鏈上查詢 (需 API Key)

輸出特徵（週頻）：
  免費特徵 (11 個):
    defi_tvl_chg           全市場 DeFi TVL 週變化
    ethereum_tvl_chg       Ethereum 鏈 TVL 週變化
    dex_volume_chg         DEX 交易量週變化
    defi_fees_chg          DeFi 協議總費用週變化
    stablecoin_mcap_chg    穩定幣市值週變化
    aave_tvl_chg           Aave TVL 週變化 (借貸代表)
    uniswap_tvl_chg        Uniswap TVL 週變化 (DEX 代表)
    lido_tvl_chg           Lido TVL 週變化 (質押代表)
    funding_rate           BTC 永續合約平均資金費率
    open_interest_chg      BTC 未平倉量週變化
    long_short_ratio       BTC 帳戶多空比

  預留特徵 (最多 5 個, 需 API Key):
    protocol_revenue_chg   協議營收週變化 (Token Terminal)
    eth_gas_fee            ETH 平均 Gas 費 (Etherscan)
    defi_uaw_chg           DeFi 活躍錢包週變化 (DappRadar)
    reserved_4             預留欄位 4
    reserved_5             預留欄位 5
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests


# ── DefiLlama 數據 ────────────────────────────────────────────────────────────

def _fetch_defillama_total_tvl() -> pd.DataFrame:
    """全市場歷史 TVL"""
    resp = requests.get("https://api.llama.fi/v2/historicalChainTvl", timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df["date"] = pd.to_datetime(df["date"], unit="s")
    return df.set_index("date").rename(columns={"tvl": "total_tvl"})


def _fetch_defillama_chain_tvl(chain: str = "Ethereum") -> pd.DataFrame:
    """特定鏈的歷史 TVL"""
    resp = requests.get(f"https://api.llama.fi/v2/historicalChainTvl/{chain}", timeout=30)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df["date"] = pd.to_datetime(df["date"], unit="s")
    col = f"{chain.lower()}_tvl"
    return df.set_index("date").rename(columns={"tvl": col})[[col]]


def _fetch_defillama_protocol_tvl(protocol: str, col_name: str) -> pd.DataFrame:
    """單一協議的歷史 TVL"""
    resp = requests.get(f"https://api.llama.fi/protocol/{protocol}", timeout=30)
    resp.raise_for_status()
    tvl_data = resp.json().get("tvl", [])
    if not tvl_data:
        return pd.DataFrame()
    df = pd.DataFrame(tvl_data)
    df["date"] = pd.to_datetime(df["date"], unit="s")
    return df.set_index("date").rename(columns={"totalLiquidityUSD": col_name})[[col_name]]


def _fetch_defillama_dex_volume() -> pd.DataFrame:
    """DEX 每日交易量"""
    url = "https://api.llama.fi/overview/dexs"
    params = {"excludeTotalDataChart": "false",
              "excludeTotalDataChartBreakdown": "true",
              "dataType": "dailyVolume"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    chart = resp.json().get("totalDataChart", [])
    df = pd.DataFrame(chart, columns=["date", "dex_volume"])
    df["date"] = pd.to_datetime(df["date"], unit="s")
    return df.set_index("date")


def _fetch_defillama_fees() -> pd.DataFrame:
    """DeFi 協議總費用"""
    url = "https://api.llama.fi/overview/fees"
    params = {"excludeTotalDataChart": "false",
              "excludeTotalDataChartBreakdown": "true",
              "dataType": "dailyFees"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    chart = resp.json().get("totalDataChart", [])
    df = pd.DataFrame(chart, columns=["date", "total_fees"])
    df["date"] = pd.to_datetime(df["date"], unit="s")
    return df.set_index("date")


def _fetch_defillama_stablecoins() -> pd.DataFrame:
    """穩定幣歷史總市值 (USDT 為代表)"""
    resp = requests.get(
        "https://stablecoins.llama.fi/stablecoincharts/all",
        params={"stablecoin": 1}, timeout=30
    )
    resp.raise_for_status()
    records = []
    for item in resp.json():
        dt = pd.to_datetime(int(item["date"]), unit="s")
        mcap = item.get("totalCirculating", {}).get("peggedUSD", 0)
        records.append({"date": dt, "stablecoin_mcap": mcap})
    return pd.DataFrame(records).set_index("date")


# ── Binance Futures 數據 ──────────────────────────────────────────────────────

def _fetch_binance_funding_rate(symbol: str = "BTCUSDT",
                                 start: str = "2020-01-01") -> pd.DataFrame:
    """BTC 永續合約資金費率 (8h 一筆)"""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    all_records = []
    start_ts = int(pd.Timestamp(start).timestamp() * 1000)
    end_ts = int(datetime.now().timestamp() * 1000)

    while start_ts < end_ts:
        params = {"symbol": symbol, "startTime": start_ts, "limit": 1000}
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_records.extend(data)
        start_ts = data[-1]["fundingTime"] + 1
        time.sleep(0.15)

    if not all_records:
        return pd.DataFrame(columns=["fundingRate"])

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df.set_index("date")[["fundingRate"]]


def _fetch_binance_open_interest(symbol: str = "BTCUSDT",
                                  start: str = "2020-01-01") -> pd.DataFrame:
    """BTC 未平倉量歷史 (日頻)。
    注意：Binance API 已限制此端點僅返回最近 ~30 天資料。
    """
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    params = {"symbol": symbol, "period": "1d", "limit": 500}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"      [WARN] OI fetch failed: {e}")
        return pd.DataFrame(columns=["open_interest_usd"])

    if not data or not isinstance(data, list):
        return pd.DataFrame(columns=["open_interest_usd"])

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open_interest_usd"] = df["sumOpenInterestValue"].astype(float)
    print(f"      [INFO] OI data: {len(df)} days ({df['date'].min().date()} ~ {df['date'].max().date()})")
    return df.set_index("date")[["open_interest_usd"]]


def _fetch_binance_long_short_ratio(symbol: str = "BTCUSDT",
                                     start: str = "2020-01-01") -> pd.DataFrame:
    """BTC 多空比歷史 (日頻)。
    注意：Binance API 已限制此端點僅返回最近 ~30 天資料。
    """
    url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
    params = {"symbol": symbol, "period": "1d", "limit": 500}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"      [WARN] LSR fetch failed: {e}")
        return pd.DataFrame(columns=["long_short_ratio"])

    if not data or not isinstance(data, list):
        return pd.DataFrame(columns=["long_short_ratio"])

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["long_short_ratio"] = df["longShortRatio"].astype(float)
    print(f"      [INFO] LSR data: {len(df)} days ({df['date'].min().date()} ~ {df['date'].max().date()})")
    return df.set_index("date")[["long_short_ratio"]]


# ── 預留 API Key 來源 ─────────────────────────────────────────────────────────

def _fetch_token_terminal(api_key: str | None = None) -> pd.DataFrame:
    """[預留] Token Terminal 協議營收 (需 API Key)"""
    if not api_key:
        return pd.DataFrame()
    # TODO: 當取得 API Key 時實作
    # url = "https://api.tokenterminal.com/v2/projects/aave/metrics"
    # headers = {"Authorization": f"Bearer {api_key}"}
    return pd.DataFrame()


def _fetch_etherscan_gas(api_key: str | None = None) -> pd.DataFrame:
    """[預留] Etherscan ETH Gas 費用 (需免費 API Key)"""
    if not api_key:
        return pd.DataFrame()
    # TODO: 當取得 API Key 時實作
    # url = f"https://api.etherscan.io/api?module=stats&action=dailyavggasprice&apikey={api_key}"
    return pd.DataFrame()


def _fetch_dappradar_uaw(api_key: str | None = None) -> pd.DataFrame:
    """[預留] DappRadar DeFi 活躍錢包 (需 API Key)"""
    if not api_key:
        return pd.DataFrame()
    # TODO: 當取得 API Key 時實作
    return pd.DataFrame()


# ── 主函數：建構 DeFi 週頻面板 ─────────────────────────────────────────────────

# 免費特徵名稱 (11 個)
DEFI_FREE_FEATURES = [
    "defi_tvl_chg",
    "ethereum_tvl_chg",
    "dex_volume_chg",
    "defi_fees_chg",
    "stablecoin_mcap_chg",
    "aave_tvl_chg",
    "uniswap_tvl_chg",
    "lido_tvl_chg",
    "funding_rate",
    "open_interest_chg",
    "long_short_ratio",
]

# 預留特徵名稱 (最多 5 個)
DEFI_RESERVED_FEATURES = [
    "protocol_revenue_chg",
    "eth_gas_fee",
    "defi_uaw_chg",
    "reserved_4",
    "reserved_5",
]

DEFI_ALL_FEATURES = DEFI_FREE_FEATURES + DEFI_RESERVED_FEATURES
N_DEFI_FEATURES = len(DEFI_ALL_FEATURES)  # 16


def build_defi_panel(
    dates: list,
    start: str = "2020-01-01",
    token_terminal_key: str | None = None,
    etherscan_key: str | None = None,
    dappradar_key: str | None = None,
) -> np.ndarray:
    """
    建構 DeFi + 衍生品週頻面板。

    Parameters
    ----------
    dates : list of pd.Timestamp
        對齊目標日期（來自 build_price_feature_panel）
    start : str
        資料起始日
    token_terminal_key, etherscan_key, dappradar_key : str | None
        可選 API Keys

    Returns
    -------
    panel : np.ndarray, shape (T, N_DEFI_FEATURES)
        T = len(dates), N_DEFI_FEATURES = 16 (11 免費 + 5 預留)
        缺失值填入 -99.99
    """
    UNK = -99.99
    T = len(dates)
    panel = np.full((T, N_DEFI_FEATURES), UNK, dtype=np.float32)

    date_index = pd.DatetimeIndex(dates)

    # ── A) DefiLlama 數據 ──────────────────────────────────────────────────
    print("    [DefiLlama] 下載 TVL/DEX/Fees/穩定幣數據...")
    dfs_raw = {}

    try:
        dfs_raw["total_tvl"] = _fetch_defillama_total_tvl()
        time.sleep(0.3)
    except Exception as e:
        print(f"      ✗ total_tvl: {e}")

    try:
        dfs_raw["ethereum_tvl"] = _fetch_defillama_chain_tvl("Ethereum")
        time.sleep(0.3)
    except Exception as e:
        print(f"      ✗ ethereum_tvl: {e}")

    try:
        dfs_raw["dex_volume"] = _fetch_defillama_dex_volume()
        time.sleep(0.3)
    except Exception as e:
        print(f"      ✗ dex_volume: {e}")

    try:
        dfs_raw["total_fees"] = _fetch_defillama_fees()
        time.sleep(0.3)
    except Exception as e:
        print(f"      ✗ total_fees: {e}")

    try:
        dfs_raw["stablecoin_mcap"] = _fetch_defillama_stablecoins()
        time.sleep(0.3)
    except Exception as e:
        print(f"      ✗ stablecoin_mcap: {e}")

    try:
        df_aave = _fetch_defillama_protocol_tvl("aave", "aave_tvl")
        if not df_aave.empty:
            dfs_raw["aave_tvl"] = df_aave
        time.sleep(0.3)
    except Exception as e:
        print(f"      ✗ aave_tvl: {e}")

    try:
        df_uni = _fetch_defillama_protocol_tvl("uniswap", "uniswap_tvl")
        if not df_uni.empty:
            dfs_raw["uniswap_tvl"] = df_uni
        time.sleep(0.3)
    except Exception as e:
        print(f"      ✗ uniswap_tvl: {e}")

    try:
        df_lido = _fetch_defillama_protocol_tvl("lido", "lido_tvl")
        if not df_lido.empty:
            dfs_raw["lido_tvl"] = df_lido
        time.sleep(0.3)
    except Exception as e:
        print(f"      ✗ lido_tvl: {e}")

    # 合併 DefiLlama → 週頻變化率
    defi_names_map = {
        "total_tvl": 0,        # defi_tvl_chg
        "ethereum_tvl": 1,     # ethereum_tvl_chg
        "dex_volume": 2,       # dex_volume_chg
        "total_fees": 3,       # defi_fees_chg
        "stablecoin_mcap": 4,  # stablecoin_mcap_chg
        "aave_tvl": 5,         # aave_tvl_chg
        "uniswap_tvl": 6,      # uniswap_tvl_chg
        "lido_tvl": 7,         # lido_tvl_chg
    }

    for name, col_idx in defi_names_map.items():
        if name not in dfs_raw:
            continue
        df = dfs_raw[name]
        col = df.columns[0]
        # 日頻 → 週頻 (取最後一個有效值)
        df_w = df[col].resample("W").last().ffill()
        # 週變化率
        pct = df_w.pct_change()
        # 對齊到目標日期
        for t, d in enumerate(date_index):
            # 找最近的可用日期
            mask = pct.index <= d
            if mask.any():
                val = pct.loc[mask].iloc[-1]
                if pd.notna(val) and np.isfinite(val):
                    panel[t, col_idx] = float(val)

    n_defi_ok = sum(1 for i in range(8) if np.any(panel[:, i] != UNK))
    print(f"      DefiLlama: {n_defi_ok}/8 個特徵有數據")

    # ── B) Binance Futures 數據 ──────────────────────────────────────────────
    print("    [Binance] 下載資金費率/未平倉量/多空比...")

    try:
        funding = _fetch_binance_funding_rate(start=start)
        funding_w = funding["fundingRate"].resample("W").mean()
        for t, d in enumerate(date_index):
            mask = funding_w.index <= d
            if mask.any():
                val = funding_w.loc[mask].iloc[-1]
                if pd.notna(val):
                    panel[t, 8] = float(val)  # funding_rate
        print(f"      funding_rate: OK")
    except Exception as e:
        print(f"      ✗ funding_rate: {e}")

    try:
        oi = _fetch_binance_open_interest(start=start)
        if not oi.empty:
            oi_w = oi["open_interest_usd"].resample("W").last().ffill()
            oi_pct = oi_w.pct_change()
            for t, d in enumerate(date_index):
                mask = oi_pct.index <= d
                if mask.any():
                    val = oi_pct.loc[mask].iloc[-1]
                    if pd.notna(val) and np.isfinite(val):
                        panel[t, 9] = float(val)  # open_interest_chg
            print(f"      open_interest_chg: OK")
        else:
            print(f"      ✗ open_interest_chg: 無數據")
    except Exception as e:
        print(f"      ✗ open_interest_chg: {e}")

    try:
        lsr = _fetch_binance_long_short_ratio(start=start)
        if not lsr.empty:
            lsr_w = lsr["long_short_ratio"].resample("W").mean()
            for t, d in enumerate(date_index):
                mask = lsr_w.index <= d
                if mask.any():
                    val = lsr_w.loc[mask].iloc[-1]
                    if pd.notna(val):
                        panel[t, 10] = float(val)  # long_short_ratio
            print(f"      long_short_ratio: OK")
        else:
            print(f"      ✗ long_short_ratio: 無數據")
    except Exception as e:
        print(f"      ✗ long_short_ratio: {e}")

    # ── C~F) 預留 API Key 來源 ───────────────────────────────────────────────
    reserved_keys = {
        "token_terminal": (token_terminal_key, _fetch_token_terminal, 11),
        "etherscan":      (etherscan_key,      _fetch_etherscan_gas,  12),
        "dappradar":      (dappradar_key,      _fetch_dappradar_uaw, 13),
    }

    for source_name, (key, fetch_fn, col_idx) in reserved_keys.items():
        if key:
            try:
                df = fetch_fn(key)
                if not df.empty:
                    print(f"      {source_name}: OK (API Key 提供)")
                    # TODO: 實際對齊邏輯
                else:
                    print(f"      {source_name}: API Key 提供但無數據 (待實作)")
            except Exception as e:
                print(f"      ✗ {source_name}: {e}")
        else:
            print(f"      {source_name}: 跳過 (無 API Key)")

    # 統計
    n_valid = sum(1 for i in range(N_DEFI_FEATURES) if np.any(panel[:, i] != UNK))
    print(f"    DeFi 面板完成：{n_valid}/{N_DEFI_FEATURES} 個特徵有數據")

    return panel
