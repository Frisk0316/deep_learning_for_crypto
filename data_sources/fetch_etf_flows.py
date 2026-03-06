"""
fetch_etf_flows.py
------------------
取得比特幣現貨 ETF 每日淨流入/流出資料。

資料來源優先順序：
  1. SoSoValue API（主要，免費）
     https://sosovalue.com/
  2. Coinglass Public API（備用，部分需要 API key）
     https://www.coinglass.com/etf/bitcoin
  3. 本地 CSV 手動備份

ETF 自 2024-01-11 開始交易（IBIT, FBTC, BITB, ARKB, BTCO 等）。
此前日期的 etf_net_flow 設為缺失值 -99.99。

注意：此特徵對所有加密資產相同（BTC 現貨 ETF 的總體資金流），
      模型可學習「ETF 淨流入強度 × 個別資產動能」的交互效果。
"""

import os
import numpy as np
import pandas as pd
import requests
import cloudscraper  # 引入 cloudscraper 繞過 Cloudflare 防護

UNK = -99.99
ETF_LAUNCH_DATE = pd.Timestamp("2024-01-11")

# ── 建立繞過防爬蟲的 Scraper ──────────────────────────────────────────────────
# 模擬 Chrome 瀏覽器行為，提高 API 請求成功率
scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})

# ── SoSoValue API ──────────────────────────────────────────────────────────────

def _fetch_sosovalue() -> pd.DataFrame:
    """
    嘗試從 SoSoValue 取得 BTC 現貨 ETF 每日總淨流入（USD，單位：百萬）。
    """
    url = "https://sosovalue.com/api/etf/btc-total-net-flow"
    # 使用更真實的 Headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://sosovalue.com/"
    }
    try:
        # 改用 scraper 發送請求
        resp = scraper.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        # 兼容 list 或 {data: list} 兩種格式
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not rows:
            print("  [WARN] SoSoValue API 回傳空資料")
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)

        # 自動偵測日期欄和流量欄
        date_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["date", "time", "day"])),
            None
        )
        flow_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ["net", "flow", "inflow"])),
            None
        )
        if date_col is None or flow_col is None:
            print(f"  [WARN] SoSoValue 無法識別欄位。現有欄位：{df.columns.tolist()}")
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df[date_col])
        df = df.set_index("date").sort_index()
        df["etf_net_flow"] = pd.to_numeric(df[flow_col], errors="coerce")
        print(f"  [OK] SoSoValue: {len(df)} 筆 ETF 流量資料")
        return df[["etf_net_flow"]]

    except Exception as e:
        print(f"  [WARN] SoSoValue 失敗: {e}")
        return pd.DataFrame()


# ── Coinglass Public API ───────────────────────────────────────────────────────

def _fetch_coinglass(api_key: str = "") -> pd.DataFrame:
    """
    從 Coinglass Open API 取得 BTC ETF 資金流。
    """
    api_key = api_key or os.environ.get("COINGLASS_API_KEY", "")
    url = "https://open-api.coinglass.com/public/v2/bitcoin_etf/flow"
    headers = {
        "coinglassSecret": api_key,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    try:
        # 同樣改用 scraper
        resp = scraper.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if str(payload.get("code", "")) != "0":
            print(f"  [WARN] Coinglass API 錯誤碼：{payload.get('msg', 'unknown')}")
            return pd.DataFrame()

        rows = payload.get("data", [])
        if not rows:
            return pd.DataFrame()
            
        df = pd.DataFrame(rows)

        date_col = next(
            (c for c in df.columns if "date" in c.lower() or "time" in c.lower()),
            None
        )
        if date_col:
            df["date"] = pd.to_datetime(df[date_col])
            df = df.set_index("date").sort_index()

        # 加總所有 ETF 的淨流入
        flow_cols = [c for c in df.columns if "flow" in c.lower() or "Flow" in c]
        if "totalNetFlow" in df.columns:
            df["etf_net_flow"] = pd.to_numeric(df["totalNetFlow"], errors="coerce")
        elif flow_cols:
            df["etf_net_flow"] = df[flow_cols].apply(
                pd.to_numeric, errors="coerce"
            ).sum(axis=1)
        else:
            print("  [WARN] Coinglass 無法找到流量欄位")
            return pd.DataFrame()

        print(f"  [OK] Coinglass: {len(df)} 筆 ETF 流量資料")
        return df[["etf_net_flow"]]

    except Exception as e:
        print(f"  [WARN] Coinglass 失敗: {e}")
        return pd.DataFrame()


# ── 手動 CSV 備份讀取 ─────────────────────────────────────────────────────────

def load_from_csv(csv_path: str) -> pd.DataFrame:
    """
    從本地 CSV 讀取 ETF 流量資料（作為最後備援）。
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        if "etf_net_flow" not in df.columns:
            # 嘗試找替代欄
            flow_col = next(
                (c for c in df.columns if "net" in c.lower() or "flow" in c.lower()),
                None
            )
            if flow_col:
                df["etf_net_flow"] = pd.to_numeric(df[flow_col], errors="coerce")
            else:
                print("  [ERR] CSV 中找不到流量欄位")
                return pd.DataFrame()
        return df[["etf_net_flow"]]
    except Exception as e:
        print(f"  [ERR] 讀取 CSV 失敗: {e}")
        return pd.DataFrame()


# ── 主函數 ────────────────────────────────────────────────────────────────────

def fetch_etf_flows(
    start: str = "2024-01-01",
    csv_backup: str | None = None,
) -> pd.DataFrame:
    """
    依優先順序嘗試取得 BTC ETF 日頻淨流入資料。
    """
    print("  Fetching BTC ETF flows...")

    df = _fetch_sosovalue()
    if df.empty:
        df = _fetch_coinglass()
    if df.empty and csv_backup:
        df = load_from_csv(csv_backup)
        if not df.empty:
             print(f"  [OK] 從 CSV 載入: {len(df)} 筆 ETF 流量資料")
             
    if df.empty:
        print("  [WARN] 所有 ETF 資料來源均失敗。返回空資料框。")
        print("         請參閱 README：手動下載 CSV 後放置於 datasets/ 目錄。")
        # 建立一個包含必須欄位的空 DataFrame
        empty_df = pd.DataFrame(columns=["etf_net_flow", "etf_net_flow_norm"])
        empty_df.index = pd.DatetimeIndex([]) 
        empty_df.index.name = "date"
        return empty_df

    # 篩選 ETF 上市後的資料
    df = df[df.index >= pd.to_datetime(start)]

    if df.empty:
        return pd.DataFrame(columns=["etf_net_flow", "etf_net_flow_norm"])

    # 週頻：取週合計（資金流以合計更合理）
    # 使用 Pandas 2.2+ 相容寫法
    df_weekly = df.resample("W").sum()

    # 滾動 z-score（窗口 = 52 週，即 1 年）
    roll_mean = df_weekly["etf_net_flow"].rolling(52, min_periods=4).mean()
    roll_std  = df_weekly["etf_net_flow"].rolling(52, min_periods=4).std()
    df_weekly["etf_net_flow_norm"] = (
        (df_weekly["etf_net_flow"] - roll_mean) / (roll_std + 1e-10)
    )

    print(f"  [OK] ETF 流量：{len(df_weekly)} 週（{df_weekly.index[0].date()} ~ {df_weekly.index[-1].date()}）")
    return df_weekly


def build_etf_panel(
    dates: pd.DatetimeIndex,
    csv_backup: str | None = None,
) -> np.ndarray:
    """
    建立 ETF 流量矩陣，shape = (T, 2)：
    """
    df = fetch_etf_flows(csv_backup=csv_backup)
    T = len(dates)
    panel = np.full((T, 2), UNK, dtype=np.float32)

    if df.empty:
        return panel

    for t, date in enumerate(dates):
        if date < ETF_LAUNCH_DATE:
            continue  # ETF 上市前保持 UNK
        if date in df.index:
            raw  = df.loc[date, "etf_net_flow"]
            
            # 若 df 只有原始流量（例如只有一週資料導致無法計算 norm），給予預設值
            norm = df.loc[date, "etf_net_flow_norm"] if "etf_net_flow_norm" in df.columns else UNK
            
            panel[t, 0] = float(raw)  if not (pd.isna(raw)  or np.isinf(raw))  else UNK
            panel[t, 1] = float(norm) if not (pd.isna(norm) or np.isinf(norm)) else UNK

    return panel