"""
fetch_polymarket.py
-------------------
從 Polymarket 取得 BTC 相關預測市場的歷史機率資料。

資料來源：Polymarket Gamma API + CLOB API（免費，無需 API key）
  Gamma API : https://gamma-api.polymarket.com  （市場搜尋）
  CLOB API  : https://clob.polymarket.com        （歷史成交價）

策略（針對 Deep Learning 優化）：
  1. 搜尋所有包含 "bitcoin" 或 "BTC" 的預測市場
  2. 篩選「BTC 達到某價格目標」類型的市場（看漲情緒指標）
  3. 抓取每個市場 YES outcome 的日頻歷史成交機率
  4. 消除結算偏差：自動捨棄距離結算日 < 3 天的極端機率數據
  5. 聚合加權：以市場交易量為權重計算每日複合機率 p
  6. Logit 轉換：套用 ln(p / (1-p)) 將封閉的機率空間投射為連續平穩的時間序列
  7. 降頻至週頻，作為神經網路的輸入特徵
"""

import time
import numpy as np
import pandas as pd
import requests

GAMMA_API  = "https://gamma-api.polymarket.com"
CLOB_API   = "https://clob.polymarket.com"
UNK        = -99.99

BTC_KEYWORDS = ["bitcoin", "btc", "btcusdt", "btc price", "bitcoin price", "bitcoin reach"]


def _search_btc_markets(max_results: int = 200) -> list[dict]:
    """
    搜尋 Polymarket 上 BTC 相關的預測市場。
    """
    all_markets = []
    for offset in range(0, max_results, 100):
        params = {
            "tag_slug": "crypto",
            "limit":    100,
            "offset":   offset,
            "order":    "volume",
            "ascending": "false",
        }
        try:
            resp = requests.get(f"{GAMMA_API}/markets", params=params, timeout=30)
            resp.raise_for_status()
            batch = resp.json()
        except Exception as e:
            print(f"    [ERR] Gamma API: {e}")
            break

        if not batch:
            break

        for m in batch:
            question = (m.get("question") or "").lower()
            if any(kw in question for kw in BTC_KEYWORDS):
                all_markets.append({
                    "id":          m.get("id"),
                    "condition_id":m.get("conditionId"),
                    "question":    m.get("question", ""),
                    "volume":      float(m.get("volume") or 0),
                    "end_date":    m.get("endDateIso") or m.get("endDate"), # 取得結算日
                    "tokens":      m.get("tokens", []),
                })

        if len(batch) < 100:
            break
        time.sleep(0.3)

    # 依成交量排序，取前 30 個高流動性市場
    all_markets.sort(key=lambda x: x["volume"], reverse=True)
    print(f"  找到 {len(all_markets)} 個 BTC 市場，取前 30 個")
    return all_markets[:30]


def _fetch_yes_history(token_id: str, start: str = "2021-01-01", end_date_str: str = None) -> pd.Series:
    """
    從 CLOB API 取得日頻歷史機率。
    包含「消除結算偏差」邏輯：捨棄距離 end_date 不到 3 天的資料。
    """
    url = f"{CLOB_API}/prices-history"
    params = {
        "market":   token_id,
        "interval": "max",
        "fidelity": 1440,  # 每日
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        hist = resp.json().get("history", [])
    except Exception as e:
        print(f"      [ERR] CLOB history: {e}")
        return pd.Series(dtype=float)

    if not hist:
        return pd.Series(dtype=float)

    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["t"], unit="s").dt.normalize()
    df = df.set_index("date").sort_index()
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    series = df["p"].dropna()

    # 過濾至 start 以後
    series = series[series.index >= pd.to_datetime(start)]

    # 消除結算偏差 (Resolution Bias)：如果市場有結算日，提早 3 天截斷數據
    if end_date_str:
        try:
            # 轉換 Polymarket 的 ISO 日期格式，並去除時區資訊以利比較
            end_dt = pd.to_datetime(end_date_str).tz_localize(None).normalize()
            cutoff_dt = end_dt - pd.Timedelta(days=3)
            series = series[series.index <= cutoff_dt]
        except Exception:
            pass

    return series


def build_polymarket_panel(
    dates: pd.DatetimeIndex,
    start: str = "2021-01-01",
) -> np.ndarray:
    """
    建立 Polymarket BTC 滾動看漲情緒指數，shape = (T, 1)。
    使用 Logit 轉換 np.log(p / (1-p)) 將機率轉為連續特徵。
    """
    print("  Searching BTC markets on Polymarket...")
    markets = _search_btc_markets()

    T = len(dates)
    panel = np.full((T, 1), UNK, dtype=np.float32)

    if not markets:
        print("  [WARN] 無法找到 BTC 市場，Polymarket 欄位將全為 UNK")
        return panel

    weighted_series = []
    total_volume    = 0.0

    for mkt in markets:
        tokens = mkt.get("tokens", [])
        yes_token = next(
            (t for t in tokens if (t.get("outcome") or "").upper() == "YES"),
            None
        )
        if yes_token is None:
            continue

        token_id = yes_token.get("token_id") or yes_token.get("tokenId")
        if not token_id:
            continue

        # print(f"  Fetching: {mkt['question'][:55]}... (vol={mkt['volume']:.0f})")
        series = _fetch_yes_history(token_id, start=start, end_date_str=mkt.get("end_date"))
        time.sleep(0.2)  # rate limit 防檔

        # 如果市場過濾完結算偏差後，有效天數太少，則捨棄
        if series.empty or len(series) < 7:
            continue

        vol = mkt["volume"]
        weighted_series.append((series, vol))
        total_volume += vol

    if not weighted_series:
        print("  [WARN] 無有效 Polymarket 時間序列 (可能皆已過期或流動性不足)")
        return panel

    # 建立日頻共用時間軸
    daily_index = pd.date_range(
        start=min(s.index.min() for s, _ in weighted_series),
        end  =max(s.index.max() for s, _ in weighted_series),
        freq ="D"
    )
    combined = pd.Series(0.0, index=daily_index)
    weight_sum = pd.Series(0.0, index=daily_index)

    # 加權平均計算每日聚合機率 p
    for series, vol in weighted_series:
        s_aligned = series.reindex(daily_index).ffill(limit=7)
        mask = s_aligned.notna()
        combined[mask]    += s_aligned[mask] * vol
        weight_sum[mask]  += vol

    prob = (combined / weight_sum.replace(0, np.nan)).dropna()

    # Logit 轉換：限制在 [0.01, 0.99] 避免無限大，然後計算 ln(p / (1-p))
    prob_clipped = prob.clip(0.01, 0.99)
    logit_score = np.log(prob_clipped / (1.0 - prob_clipped))

    # 週頻：使用均值降頻 (Pandas 2.2+ 相容寫法)
    logit_weekly = logit_score.resample("W").mean()

    # 填入 panel 矩陣
    for t, date in enumerate(dates):
        if date in logit_weekly.index:
            v = logit_weekly.loc[date]
            panel[t, 0] = float(v) if not (pd.isna(v) or np.isinf(v)) else UNK

    valid_count = np.sum(panel[:, 0] != UNK)
    print(f"  [OK] Polymarket (Logit): {valid_count}/{T} 個時間點有效")
    return panel