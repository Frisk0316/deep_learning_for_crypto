"""
fetch_etf_volumes_yfinance.py
------------------
取得比特幣/以太坊現貨 ETF 每日淨流入/流出資料及 BTC ETF 美元成交量。
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

# 更新為 12 檔主要 Bitcoin Spot ETF 代碼
BTC_ETF_TICKERS = [
    "IBIT", "FBTC", "GBTC", "BITB", "ARKB", 
    "HODL", "EZBC", "BRRR", "BTCO", "BTCW", 
    "DEFI", "BTC"
]

# ── Farside CSV 解析 ──────────────────────────────────────────────────────────

def _parse_farside_value(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s in ('-', '', 'nan', '-\xa0', '\xa0', '-\xa0-'):
        return np.nan
    s = s.replace(',', '')
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
    if not os.path.exists(csv_path):
        print(f"  [WARN] Farside CSV 不存在：{csv_path}")
        return pd.DataFrame(columns=["etf_net_flow"])

    try:
        df_raw = pd.read_csv(csv_path, header=0)
    except Exception as e:
        print(f"  [ERR] 讀取 Farside CSV 失敗: {e}")
        return pd.DataFrame(columns=["etf_net_flow"])

    total_col = None
    for col in df_raw.columns:
        if 'total' in str(col).lower().strip():
            total_col = col
            break
    if total_col is None:
        total_col = df_raw.columns[-1]

    date_col = df_raw.columns[0]
    skip_labels = {'fee', 'date', 'seed', 'total', 'average', 'maximum', 'minimum', ''}

    records = []
    for _, row in df_raw.iterrows():
        date_str = str(row[date_col]).strip()
        if date_str.lower() in skip_labels or pd.isna(row[date_col]):
            continue
        try:
            date = pd.to_datetime(date_str, dayfirst=True)
        except (ValueError, TypeError):
            continue
        total_val = _parse_farside_value(row[total_col])
        records.append({'date': date, 'etf_net_flow': total_val})

    if not records:
        return pd.DataFrame(columns=["etf_net_flow"])

    result = pd.DataFrame(records).set_index('date').sort_index()
    result['etf_net_flow'] = pd.to_numeric(result['etf_net_flow'], errors='coerce')
    print(f"  [OK] Farside CSV: {len(result)} 筆, {result.index[0].date()} ~ {result.index[-1].date()}")
    return result

def _aggregate_to_weekly(df: pd.DataFrame, col: str = "etf_net_flow") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[f"{col}", f"{col}_norm"])

    df_weekly = df[[col]].resample("W").sum()
    roll_mean = df_weekly[col].rolling(52, min_periods=4).mean()
    roll_std  = df_weekly[col].rolling(52, min_periods=4).std()
    df_weekly[f"{col}_norm"] = (df_weekly[col] - roll_mean) / (roll_std + 1e-10)
    return df_weekly

# ── Yahoo Finance ETF 成交量 ─────────────────────────────────────────────────

def fetch_etf_volume_yahoo(start: str = "2024-01-01") -> pd.DataFrame:
    print("  Fetching BTC ETF dollar volume from Yahoo Finance...")
    try:
        df = yf.download(BTC_ETF_TICKERS, start=start, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame(columns=["btc_etf_vol"])

        total_dollar_vol = pd.Series(0.0, index=df.index)

        for ticker in BTC_ETF_TICKERS:
            try:
                if isinstance(df.columns, pd.MultiIndex):
                    close_px = df[("Close", ticker)]
                    vol_shares = df[("Volume", ticker)]
                else:
                    close_px = df["Close"]
                    vol_shares = df["Volume"]
                
                dollar_vol = close_px * vol_shares
                total_dollar_vol = total_dollar_vol.add(dollar_vol.fillna(0), fill_value=0)
            except (KeyError, TypeError):
                continue

        total_dollar_vol.index = pd.to_datetime(total_dollar_vol.index).tz_localize(None)
        weekly = total_dollar_vol.resample("W").sum()
        result = pd.DataFrame({"btc_etf_vol": np.log1p(weekly)})
        print(f"  [OK] BTC ETF dollar volume: {len(result)} 週")
        return result
    except Exception as e:
        print(f"  [WARN] Yahoo Finance ETF volume 失敗: {e}")
        return pd.DataFrame(columns=["btc_etf_vol"])

def _fetch_sosovalue() -> pd.DataFrame:
    url = "https://sosovalue.com/api/etf/btc-total-net-flow"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    try:
        resp = scraper.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows)
        date_col = next((c for c in df.columns if any(k in c.lower() for k in ["date", "time"])), None)
        flow_col = next((c for c in df.columns if any(k in c.lower() for k in ["net", "flow"])), None)
        if not date_col or not flow_col: return pd.DataFrame()
        df["date"] = pd.to_datetime(df[date_col])
        df = df.set_index("date").sort_index()
        df["etf_net_flow"] = pd.to_numeric(df[flow_col], errors="coerce")
        return df[["etf_net_flow"]]
    except:
        return pd.DataFrame()

def fetch_btc_etf_flows(btc_farside_csv: str | None = None, csv_backup: str | None = None) -> pd.DataFrame:
    print("  Fetching BTC ETF flows...")
    df = pd.DataFrame()
    if btc_farside_csv: df = parse_farside_csv(btc_farside_csv)
    if df.empty: df = _fetch_sosovalue()
    if df.empty and csv_backup and os.path.exists(csv_backup):
        try:
            df = pd.read_csv(csv_backup, parse_dates=["date"]).set_index("date").sort_index()
            flow_col = next((c for c in df.columns if "flow" in c.lower()), None)
            df["etf_net_flow"] = pd.to_numeric(df[flow_col], errors="coerce") if flow_col else np.nan
        except: pass
    if df.empty: return pd.DataFrame(columns=["etf_net_flow", "etf_net_flow_norm"])
    return _aggregate_to_weekly(df)

def fetch_eth_etf_flows(eth_farside_csv: str | None = None) -> pd.DataFrame:
    print("  Fetching ETH ETF flows...")
    if not eth_farside_csv: return pd.DataFrame(columns=["etf_net_flow", "etf_net_flow_norm"])
    df = parse_farside_csv(eth_farside_csv)
    return _aggregate_to_weekly(df) if not df.empty else pd.DataFrame(columns=["etf_net_flow", "etf_net_flow_norm"])

def build_etf_panel(dates: pd.DatetimeIndex, btc_farside_csv: str | None = None, eth_farside_csv: str | None = None, csv_backup: str | None = None) -> np.ndarray:
    T = len(dates)
    panel = np.full((T, 5), UNK, dtype=np.float32)

    btc_df = fetch_btc_etf_flows(btc_farside_csv, csv_backup)
    if not btc_df.empty:
        for t, date in enumerate(dates):
            if date >= BTC_ETF_LAUNCH and date in btc_df.index:
                raw, norm = btc_df.loc[date, "etf_net_flow"], btc_df.loc[date, "etf_net_flow_norm"]
                panel[t, 0] = float(norm) if not pd.isna(norm) else UNK
                panel[t, 1] = float(raw)  if not pd.isna(raw)  else UNK

    eth_df = fetch_eth_etf_flows(eth_farside_csv)
    if not eth_df.empty:
        for t, date in enumerate(dates):
            if date >= ETH_ETF_LAUNCH and date in eth_df.index:
                raw, norm = eth_df.loc[date, "etf_net_flow"], eth_df.loc[date, "etf_net_flow_norm"]
                panel[t, 2] = float(norm) if not pd.isna(norm) else UNK
                panel[t, 3] = float(raw)  if not pd.isna(raw)  else UNK

    etf_vol = fetch_etf_volume_yahoo(start="2024-01-01")
    if not etf_vol.empty:
        for t, date in enumerate(dates):
            if date >= BTC_ETF_LAUNCH and date in etf_vol.index:
                v = etf_vol.loc[date, "btc_etf_vol"]
                panel[t, 4] = float(v) if not pd.isna(v) else UNK
    return panel