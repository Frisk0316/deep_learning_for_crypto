"""
fetch_etf_flows.py
------------------
取得比特幣/以太坊現貨 ETF 每日淨流入/流出資料及 BTC ETF 成交量。

資料來源：
  1. Farside Investors CSV（主要，手動下載）
     - BTC: https://farside.co.uk/bitcoin-etf-flow-all-data/
     - ETH: https://farside.co.uk/ethereum-etf-flow-all-data/
     CSV 欄位：Date, IBIT, FBTC, ..., Total（單位 US$m，() 為流出）
  2. Yahoo Finance（BTC ETF 成交量）
     - 聚合 IBIT, FBTC, BITB, ARKB, GBTC, BTCO 等主要 ETF 日成交量
  3. SoSoValue / Coinglass API（備用，可能被擋）
  4. 本地 CSV 手動備份（最後備援）

ETF 交易起始日：
  BTC 現貨 ETF: 2024-01-11
  ETH 現貨 ETF: 2024-07-23

輸出：
  build_etf_panel() → (T, 5) 矩陣
    col 0: btc_etf_inflow_norm  (BTC ETF 淨流入 rolling z-score)
    col 1: btc_etf_inflow_raw   (BTC ETF 淨流入原始值 US$m)
    col 2: eth_etf_inflow_norm  (ETH ETF 淨流入 rolling z-score)
    col 3: eth_etf_inflow_raw   (ETH ETF 淨流入原始值 US$m)
    col 4: btc_etf_vol          (BTC ETF 聚合成交量 log scale)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import yfinance as yf

try:
    import cloudscraper
    scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
except ImportError:
    import requests
    scraper = requests.Session()

UNK = -99.99
BTC_ETF_LAUNCH = pd.Timestamp("2024-01-11")
ETH_ETF_LAUNCH = pd.Timestamp("2024-07-23")

# BTC 現貨 ETF Yahoo Finance tickers（用於成交量）
BTC_ETF_TICKERS = ["IBIT", "FBTC", "BITB", "ARKB", "GBTC", "BTCO"]


# ── Farside CSV 解析 ──────────────────────────────────────────────────────────

def _parse_farside_value(val) -> float:
    """解析 Farside CSV 數值：(123.4) → -123.4, '-' → NaN, '1,234.5' → 1234.5"""
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s in ('-', '', 'nan', '-\xa0', '\xa0', '-\xa0-'):
        return np.nan
    # 移除千分位逗號
    s = s.replace(',', '')
    # 處理括號（流出）
    if s.startswith('(') and s.endswith(')'):
        try:
            return -float(s[1:-1])
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_farside_csv(csv_path: str) -> pd.DataFrame:
    """
    解析 Farside Investors ETF flow CSV（BTC 或 ETH 格式）。
    回傳以 date 為 index 的 DataFrame，包含 etf_net_flow 欄（Total, US$m）。
    """
    if not os.path.exists(csv_path):
        print(f"  [WARN] Farside CSV 不存在：{csv_path}")
        return pd.DataFrame(columns=["etf_net_flow"])

    try:
        df_raw = pd.read_csv(csv_path, header=0)
    except Exception as e:
        print(f"  [ERR] 讀取 Farside CSV 失敗: {e}")
        return pd.DataFrame(columns=["etf_net_flow"])

    # 找 Total 欄
    total_col = None
    for col in df_raw.columns:
        if 'total' in str(col).lower().strip():
            total_col = col
            break
    if total_col is None:
        total_col = df_raw.columns[-1]
        print(f"  [WARN] 未找到 'Total' 欄，使用最後一欄：{total_col}")

    # 第一欄為日期（可能是 unnamed）
    date_col = df_raw.columns[0]

    # 需要跳過的非資料列
    skip_labels = {'fee', 'date', 'seed', 'total', 'average', 'maximum', 'minimum', ''}

    records = []
    for _, row in df_raw.iterrows():
        date_str = str(row[date_col]).strip()
        # 跳過非日期列
        if date_str.lower() in skip_labels or pd.isna(row[date_col]):
            continue
        # 嘗試解析日期
        try:
            date = pd.to_datetime(date_str, dayfirst=True)
        except (ValueError, TypeError):
            continue
        # 解析 Total 值
        total_val = _parse_farside_value(row[total_col])
        records.append({'date': date, 'etf_net_flow': total_val})

    if not records:
        print(f"  [WARN] Farside CSV 解析結果為空")
        return pd.DataFrame(columns=["etf_net_flow"])

    result = pd.DataFrame(records)
    result = result.set_index('date').sort_index()
    result['etf_net_flow'] = pd.to_numeric(result['etf_net_flow'], errors='coerce')
    print(f"  [OK] Farside CSV: {len(result)} 筆, {result.index[0].date()} ~ {result.index[-1].date()}")
    return result


def _aggregate_to_weekly(df: pd.DataFrame, col: str = "etf_net_flow") -> pd.DataFrame:
    """將日頻 ETF 流量聚合至週頻（週合計），並計算 rolling z-score。"""
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[f"{col}", f"{col}_norm"])

    df_weekly = df[[col]].resample("W").sum()

    # 滾動 z-score（窗口 52 週 = 1 年）
    roll_mean = df_weekly[col].rolling(52, min_periods=4).mean()
    roll_std  = df_weekly[col].rolling(52, min_periods=4).std()
    df_weekly[f"{col}_norm"] = (df_weekly[col] - roll_mean) / (roll_std + 1e-10)

    return df_weekly


# ── Yahoo Finance ETF 成交量 ─────────────────────────────────────────────────

def fetch_etf_volume_yahoo(start: str = "2024-01-01") -> pd.DataFrame:
    """
    從 Yahoo Finance 下載 BTC 現貨 ETF 成交量（聚合 IBIT+FBTC+...），
    聚合至週頻，取 log 尺度。
    """
    print("  Fetching BTC ETF volume from Yahoo Finance...")
    try:
        df = yf.download(
            BTC_ETF_TICKERS, start=start, interval="1d",
            progress=False, auto_adjust=True, group_by="ticker"
        )
        if df.empty:
            print("  [WARN] Yahoo Finance ETF volume: 空資料")
            return pd.DataFrame(columns=["btc_etf_vol"])

        # 提取各 ticker 的 Volume 並加總
        total_vol = pd.Series(0.0, index=df.index)
        for ticker in BTC_ETF_TICKERS:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    vol = df[(ticker, "Volume")]
                else:
                    vol = df["Volume"]
                total_vol = total_vol.add(vol.fillna(0), fill_value=0)
            except (KeyError, TypeError):
                continue

        total_vol.index = pd.to_datetime(total_vol.index).tz_localize(None)
        # 週合計
        weekly = total_vol.resample("W").sum()
        # log 尺度
        result = pd.DataFrame({"btc_etf_vol": np.log1p(weekly)})
        print(f"  [OK] BTC ETF volume: {len(result)} 週")
        return result

    except Exception as e:
        print(f"  [WARN] Yahoo Finance ETF volume 失敗: {e}")
        return pd.DataFrame(columns=["btc_etf_vol"])


# ── SoSoValue / Coinglass API（備用） ────────────────────────────────────────

def _fetch_sosovalue() -> pd.DataFrame:
    """嘗試從 SoSoValue 取得 BTC 現貨 ETF 每日總淨流入（備用）。"""
    url = "https://sosovalue.com/api/etf/btc-total-net-flow"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://sosovalue.com/"
    }
    try:
        resp = scraper.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        date_col = next((c for c in df.columns if any(k in c.lower() for k in ["date", "time"])), None)
        flow_col = next((c for c in df.columns if any(k in c.lower() for k in ["net", "flow"])), None)
        if date_col is None or flow_col is None:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df[date_col])
        df = df.set_index("date").sort_index()
        df["etf_net_flow"] = pd.to_numeric(df[flow_col], errors="coerce")
        print(f"  [OK] SoSoValue: {len(df)} 筆 ETF 流量資料")
        return df[["etf_net_flow"]]
    except Exception as e:
        print(f"  [WARN] SoSoValue 失敗: {e}")
        return pd.DataFrame()


# ── 主函數 ────────────────────────────────────────────────────────────────────

def fetch_btc_etf_flows(
    btc_farside_csv: str | None = None,
    csv_backup: str | None = None,
) -> pd.DataFrame:
    """
    取得 BTC ETF 日頻淨流入資料。優先使用 Farside CSV。
    """
    print("  Fetching BTC ETF flows...")
    df = pd.DataFrame()

    # 優先：Farside CSV
    if btc_farside_csv:
        df = parse_farside_csv(btc_farside_csv)

    # 備用：SoSoValue API
    if df.empty:
        df = _fetch_sosovalue()

    # 最後備援：csv_backup
    if df.empty and csv_backup and os.path.exists(csv_backup):
        try:
            df = pd.read_csv(csv_backup, parse_dates=["date"]).set_index("date").sort_index()
            if "etf_net_flow" not in df.columns:
                flow_col = next((c for c in df.columns if "flow" in c.lower()), None)
                if flow_col:
                    df["etf_net_flow"] = pd.to_numeric(df[flow_col], errors="coerce")
                else:
                    df = pd.DataFrame()
            else:
                print(f"  [OK] 從備援 CSV 載入: {len(df)} 筆")
        except Exception:
            df = pd.DataFrame()

    if df.empty:
        print("  [WARN] 所有 BTC ETF 資料來源均失敗")
        return pd.DataFrame(columns=["etf_net_flow", "etf_net_flow_norm"])

    return _aggregate_to_weekly(df)


def fetch_eth_etf_flows(
    eth_farside_csv: str | None = None,
) -> pd.DataFrame:
    """
    取得 ETH ETF 日頻淨流入資料。來源：Farside CSV。
    """
    print("  Fetching ETH ETF flows...")
    if not eth_farside_csv:
        print("  [WARN] 未提供 ETH Farside CSV 路徑")
        return pd.DataFrame(columns=["etf_net_flow", "etf_net_flow_norm"])

    df = parse_farside_csv(eth_farside_csv)
    if df.empty:
        return pd.DataFrame(columns=["etf_net_flow", "etf_net_flow_norm"])

    return _aggregate_to_weekly(df)


# ── Panel 建構 ────────────────────────────────────────────────────────────────

def build_etf_panel(
    dates: pd.DatetimeIndex,
    btc_farside_csv: str | None = None,
    eth_farside_csv: str | None = None,
    csv_backup: str | None = None,
) -> np.ndarray:
    """
    建立 ETF 特徵矩陣，shape = (T, 5)：
      col 0: btc_etf_inflow_norm
      col 1: btc_etf_inflow_raw
      col 2: eth_etf_inflow_norm
      col 3: eth_etf_inflow_raw
      col 4: btc_etf_vol (log scale)
    """
    T = len(dates)
    panel = np.full((T, 5), UNK, dtype=np.float32)

    # ── BTC ETF 流量 ──
    btc_df = fetch_btc_etf_flows(
        btc_farside_csv=btc_farside_csv,
        csv_backup=csv_backup,
    )
    if not btc_df.empty:
        for t, date in enumerate(dates):
            if date < BTC_ETF_LAUNCH:
                continue
            if date in btc_df.index:
                raw = btc_df.loc[date, "etf_net_flow"]
                norm = btc_df.loc[date, "etf_net_flow_norm"] if "etf_net_flow_norm" in btc_df.columns else UNK
                panel[t, 0] = float(norm) if not (pd.isna(norm) or np.isinf(norm)) else UNK
                panel[t, 1] = float(raw)  if not (pd.isna(raw)  or np.isinf(raw))  else UNK

    # ── ETH ETF 流量 ──
    eth_df = fetch_eth_etf_flows(eth_farside_csv=eth_farside_csv)
    if not eth_df.empty:
        for t, date in enumerate(dates):
            if date < ETH_ETF_LAUNCH:
                continue
            if date in eth_df.index:
                raw = eth_df.loc[date, "etf_net_flow"]
                norm = eth_df.loc[date, "etf_net_flow_norm"] if "etf_net_flow_norm" in eth_df.columns else UNK
                panel[t, 2] = float(norm) if not (pd.isna(norm) or np.isinf(norm)) else UNK
                panel[t, 3] = float(raw)  if not (pd.isna(raw)  or np.isinf(raw))  else UNK

    # ── BTC ETF 成交量（Yahoo Finance）──
    etf_vol = fetch_etf_volume_yahoo(start="2024-01-01")
    if not etf_vol.empty:
        for t, date in enumerate(dates):
            if date < BTC_ETF_LAUNCH:
                continue
            if date in etf_vol.index:
                v = etf_vol.loc[date, "btc_etf_vol"]
                panel[t, 4] = float(v) if not (pd.isna(v) or np.isinf(v)) else UNK

    return panel
