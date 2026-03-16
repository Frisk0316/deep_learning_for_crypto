"""
fetch_trump.py — Extract weekly Trump social media features for crypto prediction.

Data sources:
  - trump-code/data/own_archive.json : Truth Social posts (2022-02 → 2024-06)
  - trump-code/data/x_posts_full.json : X/Twitter posts (2025-01 → 2026-03)

Coverage gaps (filled with UNK):
  - 2020-01 → 2022-02 : Pre-Truth Social era
  - 2024-06 → 2025-01 : Archive gap

Output: 5 weekly macro features (same value for all crypto assets):
  [0] trump_post_count   — posts per week (silence = bullish signal)
  [1] trump_caps_ratio   — avg CAPS% per week (emotional intensity)
  [2] trump_tariff_score — tariff/trade-related post fraction
  [3] trump_crypto_score — crypto-related post fraction
  [4] trump_sentiment    — composite sentiment (exclamation density)
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta

import numpy as np

UNK = -99.99

# ── Keyword dictionaries ────────────────────────────────────────────

TARIFF_KEYWORDS = [
    'tariff', 'tariffs', 'trade war', 'trade deal', 'import tax',
    'duty', 'duties', 'china trade', 'reciprocal', 'trade deficit',
    'trade surplus', 'trade balance', 'trade agreement', 'trade policy',
    'customs', 'embargo', 'sanction', 'sanctions', 'protectionism',
]

CRYPTO_KEYWORDS = [
    'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'blockchain',
    'digital asset', 'digital assets', 'digital currency',
    'defi', 'stablecoin', 'stablecoins', 'strategic reserve',
    'bitcoin reserve', 'crypto reserve', 'ethereum', 'web3',
    'digital gold', 'cbdc',
]

N_TRUMP_FEATURES = 5

TRUMP_FEATURE_NAMES = [
    "trump_post_count",
    "trump_caps_ratio",
    "trump_tariff_score",
    "trump_crypto_score",
    "trump_sentiment",
]


def _load_posts(trump_code_path: str) -> list[dict]:
    """Load and merge posts from Truth Social archive + X/Twitter."""
    posts = []

    # Truth Social (own_archive.json)
    archive_path = os.path.join(trump_code_path, "data", "own_archive.json")
    if os.path.exists(archive_path):
        with open(archive_path, encoding="utf-8") as f:
            archive = json.load(f)
        raw = archive.get("posts", [])
        for p in raw:
            if p.get("is_retweet"):
                continue
            ts = p.get("created_at", "")
            content = p.get("content", "")
            if ts and content:
                posts.append({"created_at": ts, "content": content, "platform": "truth"})
        print(f"  Truth Social: {len(posts)} original posts loaded")

    # X/Twitter (x_posts_full.json)
    x_path = os.path.join(trump_code_path, "data", "x_posts_full.json")
    n_x = 0
    if os.path.exists(x_path):
        with open(x_path, encoding="utf-8") as f:
            x_data = json.load(f)
        tweets = x_data.get("tweets", [])
        for t in tweets:
            ts = t.get("created_at", "")
            content = t.get("text", "")
            if ts and content:
                posts.append({"created_at": ts, "content": content, "platform": "x"})
                n_x += 1
        print(f"  X/Twitter: {n_x} tweets loaded")

    if not posts:
        print("  WARNING: No posts found!")

    return posts


def _parse_timestamp(ts: str) -> datetime | None:
    """Parse ISO 8601 timestamp with various formats."""
    for fmt in ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f+00:00", "%Y-%m-%dT%H:%M:%S+00:00"]:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _compute_caps_ratio(text: str) -> float:
    """Compute fraction of alphabetic characters that are uppercase."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _keyword_match(text: str, keywords: list[str]) -> bool:
    """Check if text contains any keyword (case-insensitive)."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _compute_sentiment(text: str) -> float:
    """
    Simple sentiment proxy based on exclamation density.
    More exclamations = stronger emotional intensity.
    """
    n_chars = max(len(text), 1)
    n_excl = text.count("!")
    return n_excl / n_chars * 100  # exclamation percentage


def build_trump_panel(
    dates: list | np.ndarray,
    start: str = "2020-01-01",
    trump_code_path: str = "../trump-code",
) -> np.ndarray:
    """
    Build (T, 5) panel of weekly Trump social media features.

    Parameters
    ----------
    dates : array-like of datetime or date strings — week-ending dates
    start : str — start date (for logging)
    trump_code_path : str — path to cloned trump-code repository

    Returns
    -------
    np.ndarray : shape (T, 5) — 5 Trump features per week,
                 UNK (-99.99) for weeks with no data
    """
    print(f"  Loading Trump posts from: {trump_code_path}")
    posts = _load_posts(trump_code_path)

    T = len(dates)
    panel = np.full((T, N_TRUMP_FEATURES), UNK, dtype=np.float32)

    if not posts:
        return panel

    # Parse all post timestamps
    parsed = []
    for p in posts:
        dt = _parse_timestamp(p["created_at"])
        if dt:
            parsed.append({**p, "dt": dt})
    print(f"  Successfully parsed {len(parsed)} posts with timestamps")

    # Convert dates to datetime for comparison
    week_dates = []
    for d in dates:
        if isinstance(d, str):
            week_dates.append(datetime.strptime(d, "%Y-%m-%d"))
        elif hasattr(d, "strftime"):
            week_dates.append(datetime(d.year, d.month, d.day))
        else:
            week_dates.append(d)

    # Assign posts to weeks
    # Each week t covers (week_dates[t-1], week_dates[t]] (or start for t=0)
    for t in range(T):
        week_end = week_dates[t]
        if t == 0:
            week_start = week_end - timedelta(days=7)
        else:
            week_start = week_dates[t - 1]

        # Filter posts in this week
        week_posts = [p for p in parsed if week_start < p["dt"] <= week_end]

        if not week_posts:
            # No posts this week — keep UNK (model will mask it)
            # Unless we're in the covered date range (where silence IS the signal)
            min_date = min(p["dt"] for p in parsed)
            max_date = max(p["dt"] for p in parsed)
            if min_date <= week_end and week_start <= max_date:
                # We're within data coverage — zero posts is meaningful
                panel[t, 0] = 0.0   # post_count = 0
                panel[t, 1] = 0.0   # caps_ratio = 0
                panel[t, 2] = 0.0   # tariff_score = 0
                panel[t, 3] = 0.0   # crypto_score = 0
                panel[t, 4] = 0.0   # sentiment = 0
            continue

        n_posts = len(week_posts)
        contents = [p["content"] for p in week_posts]

        # Feature 0: Post count
        panel[t, 0] = float(n_posts)

        # Feature 1: Average CAPS ratio
        caps_ratios = [_compute_caps_ratio(c) for c in contents]
        panel[t, 1] = np.mean(caps_ratios)

        # Feature 2: Tariff-related post fraction
        tariff_count = sum(1 for c in contents if _keyword_match(c, TARIFF_KEYWORDS))
        panel[t, 2] = tariff_count / n_posts

        # Feature 3: Crypto-related post fraction
        crypto_count = sum(1 for c in contents if _keyword_match(c, CRYPTO_KEYWORDS))
        panel[t, 3] = crypto_count / n_posts

        # Feature 4: Composite sentiment (avg exclamation density)
        sentiments = [_compute_sentiment(c) for c in contents]
        panel[t, 4] = np.mean(sentiments)

    # Summary
    valid_weeks = (panel[:, 0] != UNK).sum()
    total_posts_assigned = panel[panel[:, 0] != UNK, 0].sum()
    print(f"  Coverage: {valid_weeks}/{T} weeks ({valid_weeks/T*100:.1f}%)")
    print(f"  Total posts assigned: {int(total_posts_assigned)}")
    if valid_weeks > 0:
        avg_posts = panel[panel[:, 0] != UNK, 0].mean()
        print(f"  Average posts/week: {avg_posts:.1f}")

    return panel


if __name__ == "__main__":
    # Quick test
    import sys
    trump_path = sys.argv[1] if len(sys.argv) > 1 else "../trump-code"

    # Generate weekly dates from 2020-01-05 to 2026-03-15
    start = datetime(2020, 1, 5)
    dates = []
    d = start
    while d <= datetime(2026, 3, 15):
        dates.append(d)
        d += timedelta(days=7)

    panel = build_trump_panel(dates, trump_code_path=trump_path)
    print(f"\nPanel shape: {panel.shape}")
    print(f"Feature names: {TRUMP_FEATURE_NAMES}")

    # Show sample
    for i, name in enumerate(TRUMP_FEATURE_NAMES):
        valid = panel[:, i] != UNK
        if valid.sum() > 0:
            vals = panel[valid, i]
            print(f"  {name}: valid={valid.sum()}, "
                  f"mean={vals.mean():.3f}, std={vals.std():.3f}, "
                  f"min={vals.min():.3f}, max={vals.max():.3f}")
