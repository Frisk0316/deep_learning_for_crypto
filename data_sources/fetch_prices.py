"""
fetch_prices.py
---------------
從 yfinance 下載多個加密貨幣的 OHLCV 週資料，
並計算價格動能（Momentum）與技術指標（Technical）特徵。

資料來源：Yahoo Finance (免費，無需 API key)
涵蓋資產：前百大市值幣種（排除穩定幣 USDT, USDC, DAI, BUSD, TUSD, FDUSD, PYUSD 等）

§12.5「擴大資產池」改善：從 14 → 86 個資產，
解決橫截面 N 過小導致的維度詛咒與統計功效不足問題。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

# ── 資產列表（名稱 → Yahoo Finance ticker）──────────────────────────────────
# 前百大市值幣種（排除穩定幣），依約略市值排序
# 資料更新日：2026-03  |  Yahoo Finance ticker 格式：{SYMBOL}-USD
#
# 排除的穩定幣：USDT, USDC, DAI, BUSD, TUSD, FDUSD, PYUSD, FRAX, LUSD, USDP, GUSD
# 排除的 wrapped/pegged token：WBTC, WETH, stETH, cbETH（與原始資產高度共線性）
#
# 注意：部分較新或低流動性代幣在 yfinance 上可能無資料，
#       fetch_ohlcv() 會自動跳過下載失敗的資產。

CRYPTO_SYMBOLS = {
    # ── Tier 1：市值前 10（穩定幣除外）─────────────────────────────────────
    "BTC":    "BTC-USD",
    "ETH":    "ETH-USD",
    "XRP":    "XRP-USD",
    "BNB":    "BNB-USD",
    "SOL":    "SOL-USD",
    "DOGE":   "DOGE-USD",
    "ADA":    "ADA-USD",
    "TRX":    "TRX-USD",
    "AVAX":   "AVAX-USD",
    "LINK":   "LINK-USD",

    # ── Tier 2：市值 11~30 ──────────────────────────────────────────────────
    "TON":    "TON11419-USD",    # Toncoin
    "SHIB":   "SHIB-USD",
    "SUI":    "SUI20947-USD",
    "XLM":    "XLM-USD",        # Stellar
    "DOT":    "DOT-USD",
    "HBAR":   "HBAR-USD",       # Hedera
    "BCH":    "BCH-USD",
    "LTC":    "LTC-USD",
    "UNI":    "UNI7083-USD",
    "NEAR":   "NEAR-USD",       # NEAR Protocol
    "APT":    "APT21794-USD",   # Aptos
    "PEPE":   "PEPE24478-USD",
    "ICP":    "ICP-USD",        # Internet Computer
    "AAVE":   "AAVE-USD",
    "ETC":    "ETC-USD",        # Ethereum Classic
    "RENDER": "RNDR-USD",
    "FET":    "FET-USD",        # Fetch.ai / ASI Alliance
    "CRO":    "CRO-USD",       # Cronos
    "POL":    "POL-USD",       # Polygon (formerly MATIC)
    "ATOM":   "ATOM-USD",

    # ── Tier 3：市值 31~50 ──────────────────────────────────────────────────
    "VET":    "VET-USD",        # VeChain
    "FIL":    "FIL-USD",        # Filecoin
    "ARB":    "ARB11841-USD",   # Arbitrum
    "OP":     "OP-USD",         # Optimism
    "KAS":    "KAS-USD",        # Kaspa
    "STX":    "STX4847-USD",    # Stacks
    "MKR":    "MKR-USD",        # Maker
    "IMX":    "IMX10603-USD",   # Immutable X
    "INJ":    "INJ-USD",        # Injective
    "ALGO":   "ALGO-USD",       # Algorand
    "GRT":    "GRT6719-USD",    # The Graph
    "THETA":  "THETA-USD",
    "FTM":    "FTM-USD",        # Fantom / Sonic
    "SEI":    "SEI-USD",
    "RUNE":   "RUNE-USD",       # THORChain
    "LDO":    "LDO-USD",        # Lido DAO
    "BONK":   "BONK-USD",
    "FLOKI":  "FLOKI-USD",
    "WIF":    "WIF-USD",        # dogwifhat
    "ONDO":   "ONDO-USD",

    # ── Tier 4：市值 51~75 ──────────────────────────────────────────────────
    "JUP":    "JUP29210-USD",   # Jupiter
    "TIA":    "TIA22861-USD",   # Celestia
    "WLD":    "WLD-USD",        # Worldcoin
    "PYTH":   "PYTH-USD",
    "PENDLE": "PENDLE-USD",
    "ENS":    "ENS-USD",        # Ethereum Name Service
    "QNT":    "QNT-USD",        # Quant
    "AR":     "AR-USD",         # Arweave
    "EGLD":   "EGLD-USD",       # MultiversX
    "AXS":    "AXS-USD",        # Axie Infinity
    "FLOW":   "FLOW-USD",
    "NEO":    "NEO-USD",
    "GALA":   "GALA-USD",
    "KAVA":   "KAVA-USD",
    "XTZ":    "XTZ-USD",        # Tezos
    "EOS":    "EOS-USD",
    "SAND":   "SAND-USD",       # The Sandbox
    "MANA":   "MANA-USD",       # Decentraland
    "CHZ":    "CHZ-USD",        # Chiliz
    "JASMY":  "JASMY-USD",

    # ── Tier 5：市值 76~100 ─────────────────────────────────────────────────
    "SNX":    "SNX-USD",        # Synthetix
    "CRV":    "CRV-USD",        # Curve
    "DYDX":   "DYDX-USD",
    "COMP":   "COMP-USD",       # Compound
    "APE":    "APE-USD",        # ApeCoin
    "MINA":   "MINA-USD",
    "1INCH":  "1INCH-USD",
    "ZEC":    "ZEC-USD",        # Zcash
    "IOTA":   "IOTA-USD",
    "CAKE":   "CAKE-USD",       # PancakeSwap
    "CFX":    "CFX-USD",        # Conflux
    "ROSE":   "ROSE-USD",       # Oasis
    "ZIL":    "ZIL-USD",        # Zilliqa
    "CELO":   "CELO-USD",
    "ANKR":   "ANKR-USD",
    "SKL":    "SKL-USD",        # SKALE
}

# 向後相容：保留原始 14 資產的子集別名
CRYPTO_SYMBOLS_ORIGINAL_14 = {
    k: v for k, v in CRYPTO_SYMBOLS.items()
    if k in {"BTC", "ETH", "SOL", "BNB", "XRP", "AVAX", "DOGE",
             "ADA", "LINK", "DOT", "LTC", "UNI", "ATOM", "POL"}
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
    從 OHLCV DataFrame 計算 11 個特徵：
      Price Momentum (5) + Technical Indicators (6)

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

    # 週 USD 交易量（log 尺度）
    usd_vol = close * volume
    feat["vol_usd"] = np.log1p(usd_vol)

    return feat


def build_price_feature_panel(
    start="2020-01-01", end=None
) -> tuple[pd.DatetimeIndex, list[str], np.ndarray]:
    """
    建立 (T, N, 12) 的 panel：第 0 欄為「下週報酬（target）」，1~11 欄為特徵。

    Returns
    -------
    dates    : pd.DatetimeIndex (長度 T)
    assets   : list[str]       (長度 N)
    panel    : np.ndarray      shape (T, N, 12)
               panel[:, :, 0]    = next-week return (target, -99.99 if missing)
               panel[:, :, 1:6]  = momentum features
               panel[:, :, 6:12] = technical features (incl. vol_usd)
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
    panel = np.full((T, N, 12), UNK, dtype=np.float32)

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