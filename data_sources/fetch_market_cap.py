"""
fetch_market_cap.py — Fetch historical weekly market cap from CoinGecko.

CoinGecko free API provides historical market cap via:
  GET /coins/{id}/market_chart?vs_currency=usd&days=max

Usage:
    python fetch_market_cap.py [--out market_cap.npz]

Or import and use build_market_cap_panel() in prepare_btc_data.py.
"""
from __future__ import annotations

import json
import time as _time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

try:
    import requests
except ImportError:
    import urllib.request
    import urllib.error

    class _Requests:
        """Minimal fallback if requests is not installed."""
        class Response:
            def __init__(self, data, code):
                self._data = data
                self.status_code = code
            def json(self):
                return json.loads(self._data)

        def get(self, url, params=None, timeout=30):
            if params:
                qs = "&".join(f"{k}={v}" for k, v in params.items())
                url = f"{url}?{qs}"
            try:
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    return self.Response(resp.read().decode(), resp.getcode())
            except urllib.error.HTTPError as e:
                return self.Response(b"", e.code)

    requests = _Requests()


# ── CoinGecko ID mapping ─────────────────────────────────────────────
# Maps our internal asset names to CoinGecko API IDs.
COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "XRP": "ripple",
    "BNB": "binancecoin", "SOL": "solana", "DOGE": "dogecoin",
    "ADA": "cardano", "TRX": "tron", "AVAX": "avalanche-2",
    "LINK": "chainlink", "TON": "the-open-network", "SHIB": "shiba-inu",
    "SUI": "sui", "XLM": "stellar", "DOT": "polkadot",
    "HBAR": "hedera-hashgraph", "BCH": "bitcoin-cash", "LTC": "litecoin",
    "UNI": "uniswap", "NEAR": "near", "APT": "aptos",
    "PEPE": "pepe", "ICP": "internet-computer", "AAVE": "aave",
    "ETC": "ethereum-classic", "RENDER": "render-token", "FET": "fetch-ai",
    "CRO": "crypto-com-chain", "POL": "matic-network", "ATOM": "cosmos",
    "VET": "vechain", "FIL": "filecoin", "ARB": "arbitrum",
    "OP": "optimism", "KAS": "kaspa", "STX": "blockstack",
    "MKR": "maker", "IMX": "immutable-x", "INJ": "injective-protocol",
    "ALGO": "algorand", "GRT": "the-graph", "THETA": "theta-token",
    "FTM": "fantom", "SEI": "sei-network", "RUNE": "thorchain",
    "LDO": "lido-dao", "BONK": "bonk", "FLOKI": "floki",
    "WIF": "dogwifcoin", "ONDO": "ondo-finance",
    "JUP": "jupiter-exchange-solana", "TIA": "celestia",
    "WLD": "worldcoin-wld", "PYTH": "pyth-network",
    "PENDLE": "pendle", "ENS": "ethereum-name-service",
    "QNT": "quant-network", "AR": "arweave", "EGLD": "elrond-erd-2",
    "AXS": "axie-infinity", "FLOW": "flow", "NEO": "neo",
    "GALA": "gala", "KAVA": "kava", "XTZ": "tezos",
    "EOS": "eos", "SAND": "the-sandbox", "MANA": "decentraland",
    "CHZ": "chiliz", "JASMY": "jasmycoin",
    "SNX": "havven", "CRV": "curve-dao-token", "DYDX": "dydx",
    "COMP": "compound-governance-token", "APE": "apecoin",
    "MINA": "mina-protocol", "1INCH": "1inch",
    "ZEC": "zcash", "IOTA": "iota", "CAKE": "pancakeswap-token",
    "CFX": "conflux-token", "ROSE": "oasis-network",
    "ZIL": "zilliqa", "CELO": "celo", "ANKR": "ankr",
    "SKL": "skale",
}

UNK = -99.99


def _fetch_one(coin_id: str, days: int = 365,
               base_retry_wait: int = 60) -> pd.DataFrame | None:
    """Fetch historical market cap for one coin from CoinGecko.

    CoinGecko free tier: days ≤ 365. This covers the test period
    (~49 weeks / ~11.5 months), which is sufficient for VW portfolio
    construction. For longer history a paid API key is required.
    """
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days)}

    for attempt in range(5):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 429:
                wait = base_retry_wait * (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...")
                _time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code} for {coin_id}")
                return None
            data = resp.json()
            if "status" in data and "error_code" in data["status"]:
                # CoinGecko error response
                print(f"    API error for {coin_id}: {data['status']}")
                return None
            mc = data.get("market_caps", [])
            if not mc:
                return None
            df = pd.DataFrame(mc, columns=["timestamp", "market_cap"])
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
            df = df.groupby("date")["market_cap"].last().reset_index()
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception as e:
            print(f"    Error fetching {coin_id}: {e}")
            if attempt < 4:
                _time.sleep(10)
    return None


def build_market_cap_panel(
    dates: list | np.ndarray,
    assets: list | np.ndarray,
    start: str = "2020-01-01",
) -> np.ndarray:
    """
    Build (T, N) panel of weekly market cap data.

    Parameters
    ----------
    dates : week-ending dates (T,)
    assets : asset names (N,) — must match COINGECKO_IDS keys

    Returns
    -------
    np.ndarray : shape (T, N), market cap in USD. UNK for missing.
    """
    T = len(dates)
    N = len(assets)
    panel = np.full((T, N), UNK, dtype=np.float64)

    # Convert dates (handles numpy.str_ as well as regular str)
    week_dates = [pd.Timestamp(str(d)) for d in dates]

    print(f"  Fetching market cap for {N} assets from CoinGecko...")
    fetched = 0

    for n, asset in enumerate(assets):
        asset_str = str(asset)
        coin_id = COINGECKO_IDS.get(asset_str)
        if coin_id is None:
            print(f"    [SKIP] {asset_str}: no CoinGecko ID mapping")
            continue

        df = _fetch_one(coin_id)
        if df is None or df.empty:
            print(f"    [MISS] {asset_str}")
            continue

        # Map to weekly dates (use last available market cap before each week end)
        df = df.set_index("date").sort_index()
        for t, wd in enumerate(week_dates):
            # Find the most recent market cap on or before the week end date
            mask = df.index <= wd
            if mask.any():
                panel[t, n] = df.loc[mask, "market_cap"].iloc[-1]

        valid = (panel[:, n] != UNK).sum()
        print(f"    [OK] {asset_str} ({coin_id}): {valid}/{T} weeks")
        fetched += 1

        # Rate limiting: CoinGecko free tier allows ~5-10 req/min.
        # Sleep 15s between every request to stay well within limits.
        _time.sleep(15)

    valid_pct = (panel != UNK).mean() * 100
    print(f"  Market cap coverage: {valid_pct:.1f}%")
    return panel


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="../datasets/btc_panel.npz",
                        help="Path to btc_panel.npz (to get dates and assets)")
    parser.add_argument("--out", default="../datasets/market_cap.npz")
    args = parser.parse_args()

    npz = np.load(args.panel, allow_pickle=True)
    dates = npz["date"]
    assets = npz["wficn"]

    mcap = build_market_cap_panel(dates, assets)
    np.savez(args.out, market_cap=mcap, dates=dates, assets=assets)
    print(f"Saved to {args.out}")
