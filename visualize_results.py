"""
visualize_results.py
--------------------
Generates all Tables and Figures analogous to the FEN paper
(Kaniel, Lin, Pelger & Van Nieuwerburgh, JFE 2023) using the
trained BTC/crypto deep learning models.

Crypto adaptations vs. mutual fund paper:
  - "Abnormal returns"  → raw weekly crypto returns (no risk-factor model)
  - "Sentiment"         → Fear & Greed Index (feature 'fear_greed')
  - "Fund momentum"     → r12w / r52w (price momentum features)
  - "Fund flow"         → btc_etf_inflow_norm (ETF flow feature)
  - Information sets    → 4 feature subsets (Price+Tech / +Onchain / +Macro / All)

Outputs saved to: visualizations/results/

Usage:
    python visualize_results.py
"""

import os
import sys
import glob
import json
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 needed for 3D projection
from scipy import stats

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from btc_data_layer import DataInRamInputLayer
from model_btc import BTCTrainer, construct_long_short_portfolio, sharpe_ratio

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = os.path.join(_THIS_DIR, "checkpoints", "btc")
DATA_FILE      = os.path.join(_THIS_DIR, "datasets", "btc_panel.npz")
FOLD_FILE      = os.path.join(_THIS_DIR, "sampling_folds", "btc_chronological_folds.npy")
CONFIG_FILE    = os.path.join(_THIS_DIR, "config_btc.json")
OUTPUT_DIR     = os.path.join(_THIS_DIR, "visualizations", "results")

N_SEEDS   = 8
N_DECILES = 10

# Category colours (match btc_data_layer.CryptoChar)
CATEGORY_COLORS = {
    "Price Momentum":   "royalblue",
    "Technical":        "tomato",
    "On-chain":         "mediumseagreen",
    "Macro/Sentiment":  "darkviolet",
    "ETF & Polymarket": "darkorange",
}

# 33 global feature names in order
FEATURE_NAMES = [
    "r1w", "r4w", "r12w", "r26w", "r52w",             # 0-4   Price Momentum
    "rsi_14", "bb_pct", "vol_ratio", "atr_pct",        # 5-8   Technical
    "obv_change", "vol_usd",                            # 9-10
    "active_addr", "tx_count", "nvt",                  # 11-13 On-chain
    "exchange_net_flow", "mvrv",                        # 14-15
    "fear_greed", "spx_ret", "dxy_ret", "vix",         # 16-19 Macro/Sentiment
    "gold_ret", "silver_ret", "dji_ret",               # 20-22
    "spx_vol_chg", "gold_vol_chg", "silver_vol_chg",  # 23-25
    "dji_vol_chg",                                      # 26
    "btc_etf_inflow_norm", "polymarket_btc",            # 27-28 ETF & Polymarket
    "btc_etf_inflow_raw", "eth_etf_inflow_norm",       # 29-30
    "eth_etf_inflow_raw", "btc_etf_vol",               # 31-32
]

FEATURE_CATEGORIES = {
    "Price Momentum":   list(range(0, 5)),
    "Technical":        list(range(5, 11)),
    "On-chain":         list(range(11, 16)),
    "Macro/Sentiment":  list(range(16, 27)),
    "ETF & Polymarket": list(range(27, 33)),
}

# Feature subset candidates (from checkpoints in train_btc.py naming convention)
FEAT_CONFIGS_CANDIDATES = [
    {"name": "Price+Technical",  "subset": list(range(11)), "feat_key": "feat0to10"},
    {"name": "+Onchain",         "subset": list(range(16)), "feat_key": "feat0to15"},
    {"name": "+Macro",           "subset": list(range(22)), "feat_key": "feat0to21"},
    {"name": "All",              "subset": list(range(33)), "feat_key": "feat0to32"},
]

# Decile colours: bottom (pink/magenta) → top (black)
DECILE_COLORS = [
    "#8B008B", "#FF69B4", "#DC143C", "#FF6347", "#FFA500",
    "#DAA520", "#6B8E23", "#2E8B57", "#4169E1", "#000000",
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Data & Config Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    npz       = np.load(DATA_FILE, allow_pickle=True)
    data      = npz["data"]                        # (T, N, 34)
    dates     = np.array(npz["date"],     dtype=str)
    assets    = np.array(npz["wficn"],    dtype=str)
    variables = np.array(npz["variable"], dtype=str)
    folds     = np.load(FOLD_FILE, allow_pickle=True).tolist()
    train_idx, valid_idx, test_idx = folds[0]
    return data, dates, assets, variables, train_idx, valid_idx, test_idx


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Checkpoint Scanning
# ─────────────────────────────────────────────────────────────────────────────

def find_checkpoint_dir(seed_folder, feat_key):
    """Return path to training run directory (containing ckpt/) or None."""
    seed_path = os.path.join(CHECKPOINT_DIR, seed_folder)
    if not os.path.isdir(seed_path):
        return None
    for entry in os.scandir(seed_path):
        if entry.is_dir() and entry.name.startswith(feat_key + "_"):
            ckpt_file = os.path.join(entry.path, "ckpt", "checkpoint")
            if os.path.isfile(ckpt_file):
                return entry.path
    return None


def detect_available_configs():
    """Return list of feature configs that have at least one trained seed."""
    available = []
    for cfg in FEAT_CONFIGS_CANDIDATES:
        found = sum(
            1 for s in range(1, N_SEEDS + 1)
            if find_checkpoint_dir(f"folder_{s}", cfg["feat_key"]) is not None
        )
        if found > 0:
            c = cfg.copy()
            c["n_seeds_found"] = found
            available.append(c)
            print(f"  [OK] {cfg['name']:20s} ({cfg['feat_key']}): {found}/{N_SEEDS} seeds")
        else:
            print(f"  [--] {cfg['name']:20s} ({cfg['feat_key']}): no checkpoints")
    return available


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Ensemble Prediction Generation
# ─────────────────────────────────────────────────────────────────────────────

def _load_single_prediction(logdir, subset, test_idx):
    """Load one checkpoint, run inference on test set. Returns (pred, actual, mask)."""
    config = load_config()
    config["individual_feature_dim"] = len(subset)
    trainer = BTCTrainer(config)
    try:
        trainer.load_best(logdir)
    except Exception as e:
        print(f"    Warning: load_best failed for {logdir}: {e}")
        return None, None, None

    dl = DataInRamInputLayer(DATA_FILE, test_idx, subset)
    pred = trainer.get_prediction(dl)
    for I_macro, I, R, mask in dl.iterateOneEpoch():
        actual = R[mask]
        return pred, actual, mask
    return pred, None, None


def get_ensemble_predictions(feat_cfg, test_idx):
    """Average predictions across all available seeds."""
    subset   = feat_cfg["subset"]
    feat_key = feat_cfg["feat_key"]

    all_preds = []
    actual_ref = None
    mask_ref   = None

    for seed in range(1, N_SEEDS + 1):
        logdir = find_checkpoint_dir(f"folder_{seed}", feat_key)
        if logdir is None:
            continue
        pred, actual, mask = _load_single_prediction(logdir, subset, test_idx)
        if pred is None or len(pred) == 0:
            continue
        all_preds.append(pred)
        if actual_ref is None:
            actual_ref = actual
            mask_ref   = mask

    if not all_preds:
        return None, None, None

    ensemble = np.mean(np.stack(all_preds, axis=0), axis=0)
    print(f"    Ensemble from {len(all_preds)} seeds, pred shape: {ensemble.shape}")
    return ensemble, actual_ref, mask_ref


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Portfolio Construction
# ─────────────────────────────────────────────────────────────────────────────

def build_decile_portfolios(pred, actual, mask):
    """
    Build equal-weight and prediction-weighted decile portfolios.

    Returns
    -------
    ew_ret     : (T, N_DECILES) — equal-weight return per decile per week
    pw_ret     : (T, N_DECILES) — prediction-weighted return per decile per week
    valid_mask : (T,) bool      — weeks where decile assignment was possible
    """
    T, N = mask.shape
    ew_ret     = np.full((T, N_DECILES), np.nan)
    pw_ret     = np.full((T, N_DECILES), np.nan)
    valid_mask = np.zeros(T, dtype=bool)

    ptr = 0
    for t in range(T):
        n_t = int(mask[t].sum())
        if n_t == 0:
            continue
        pred_t   = pred[ptr: ptr + n_t]
        actual_t = actual[ptr: ptr + n_t]
        ptr += n_t

        if n_t < N_DECILES:      # need at least 10 assets for 10 deciles
            continue

        valid_mask[t] = True
        order   = np.argsort(pred_t)                     # ascending
        buckets = np.array_split(order, N_DECILES)

        for d, bucket in enumerate(buckets):
            p_b = pred_t[bucket]
            a_b = actual_t[bucket]

            ew_ret[t, d] = np.mean(a_b)

            # Prediction-weighted (paper eq. 4-6)
            if d == N_DECILES - 1:               # top decile
                shifted = p_b - p_b.min()
            elif d == 0:                          # bottom decile
                shifted = p_b.max() - p_b
            else:
                shifted = np.ones(len(bucket))
            total = shifted.sum()
            w = shifted / total if total > 1e-12 else np.ones(len(bucket)) / len(bucket)
            pw_ret[t, d] = np.dot(w, a_b)

    return ew_ret, pw_ret, valid_mask


def build_long_short(decile_ret, valid_mask, top_frac=0.2, bot_frac=0.2):
    """Long-short: average top X% deciles minus average bottom X% deciles."""
    n_top = max(1, round(N_DECILES * top_frac))
    n_bot = max(1, round(N_DECILES * bot_frac))
    top = np.nanmean(decile_ret[valid_mask, -n_top:], axis=1)
    bot = np.nanmean(decile_ret[valid_mask, :n_bot],  axis=1)
    return top - bot


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Performance Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance(ret_series):
    """Return dict of mean(%), t-stat, annualized SR, and sample T."""
    arr = np.asarray(ret_series, dtype=float)
    arr = arr[~np.isnan(arr)]
    T   = len(arr)
    if T < 2:
        return dict(mean=np.nan, t_stat=np.nan, SR=np.nan, T=T)
    mu  = arr.mean()
    sig = arr.std(ddof=1)
    return dict(
        mean   = mu * 100,
        t_stat = mu / sig * np.sqrt(T) if sig > 1e-10 else np.nan,
        SR     = mu / sig * np.sqrt(52) if sig > 1e-10 else np.nan,
        T      = T,
    )


def compute_prediction_r2(pred, actual, mask):
    """
    Average cross-sectional prediction R²:
      R²_t = 1 - SS_res_t / SS_tot_t,  averaged over t.
    Reported in %.
    """
    T, N = mask.shape
    r2s  = []
    ptr  = 0
    for t in range(T):
        n_t = int(mask[t].sum())
        if n_t < 3:
            ptr += n_t
            continue
        p_t = pred[ptr: ptr + n_t]
        a_t = actual[ptr: ptr + n_t]
        ptr += n_t
        ss_res = np.sum((a_t - p_t) ** 2)
        ss_tot = np.sum((a_t - a_t.mean()) ** 2)
        if ss_tot > 1e-12:
            r2s.append(1.0 - ss_res / ss_tot)
    return float(np.mean(r2s) * 100) if r2s else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Holding-Period Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_holding_period_stats(ew_decile_ret, valid_mask, hold_weeks_list):
    """
    Overlapping holding-period returns for the long-short portfolio.
    Returns dict: {s: {mean, std, SR, t_stat}}.
    """
    ls = (
        np.nanmean(ew_decile_ret[valid_mask, -2:], axis=1) -
        np.nanmean(ew_decile_ret[valid_mask, :2],  axis=1)
    )
    results = {}
    for s in hold_weeks_list:
        if s >= len(ls):
            continue
        cumrets = np.array([ls[t:t+s].mean() for t in range(len(ls) - s + 1)])
        mu  = cumrets.mean()
        sig = cumrets.std(ddof=1)
        T   = len(cumrets)
        results[s] = dict(
            mean   = mu * 100,
            std    = sig * 100,
            SR     = mu / sig * np.sqrt(52) if sig > 1e-10 else 0,
            t_stat = mu / sig * np.sqrt(T)  if sig > 1e-10 else 0,
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Variable Importance (numerical gradient)
# ─────────────────────────────────────────────────────────────────────────────

def compute_variable_importance(trainer, dl_test, n_features):
    """
    Sensitivity(z_k) = sqrt(mean((∂ŷ/∂z_k)²)), approximated numerically.
    Returns array of shape (n_features,).
    """
    import tensorflow as tf

    for I_macro, I, R, mask in dl_test.iterateOneEpoch():
        X_valid, _ = trainer._flatten_valid(I_macro, I, R, mask)
        if len(X_valid) == 0:
            return np.zeros(n_features)

        eps  = 1e-3
        sens = np.zeros(n_features)
        for k in range(n_features):
            X_p = X_valid.copy(); X_p[:, k] += eps
            X_m = X_valid.copy(); X_m[:, k] -= eps
            y_p = trainer.model(
                tf.constant(X_p, dtype=tf.float32), training=False
            ).numpy().flatten()
            y_m = trainer.model(
                tf.constant(X_m, dtype=tf.float32), training=False
            ).numpy().flatten()
            grad_k   = (y_p - y_m) / (2.0 * eps)
            sens[k]  = np.sqrt(np.mean(grad_k ** 2))
        return sens
    return np.zeros(n_features)


def compute_group_importance(sens, n_features):
    """Average sensitivity within each feature category."""
    group = {}
    for cat, indices in FEATURE_CATEGORIES.items():
        valid = [i for i in indices if i < n_features]
        group[cat] = float(np.mean([sens[i] for i in valid])) if valid else 0.0
    return group


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Interaction Effects
# ─────────────────────────────────────────────────────────────────────────────

def compute_interaction_grid(trainer, dl_test, feat_idx, macro_idx, n_grid=50):
    """
    Predicted return as a function of feat_idx for 5 fear_greed quantiles.
    All other features fixed at their median on the test set.

    Returns (x_grid, pred_matrix [5 × n_grid], sent_labels).
    """
    import tensorflow as tf

    for I_macro, I, R, mask in dl_test.iterateOneEpoch():
        X_valid, _ = trainer._flatten_valid(I_macro, I, R, mask)
        if len(X_valid) == 0:
            return None, None, None

        X_median = np.median(X_valid, axis=0)
        n_feat   = X_valid.shape[1]

        # Sentiment quantile values (10/25/50/75/90 %)
        if macro_idx < n_feat:
            q_vals = np.nanpercentile(X_valid[:, macro_idx], [10, 25, 50, 75, 90])
        else:
            q_vals = np.linspace(-1.0, 1.0, 5)

        x_grid = np.linspace(-0.45, 0.45, n_grid)
        pred_mat = np.zeros((5, n_grid))

        for q_idx, qv in enumerate(q_vals):
            batch = np.tile(X_median, (n_grid, 1)).copy()
            batch[:, feat_idx] = x_grid
            if macro_idx < n_feat:
                batch[:, macro_idx] = qv
            preds = trainer.model(
                tf.constant(batch.astype(np.float32)), training=False
            ).numpy().flatten()
            pred_mat[q_idx] = preds

        labels = [
            "fear_greed 10%", "fear_greed 25%", "fear_greed 50%",
            "fear_greed 75%", "fear_greed 90%",
        ]
        return x_grid, pred_mat, labels
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: Tables
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(val, decimals=2):
    if np.isnan(val):
        return "—"
    return f"{val:.{decimals}f}"


def make_table3(results_by_config):
    """Table 3 equivalent: Long-short portfolio performance by information set."""
    rows = []
    for cfg_name, res in results_by_config.items():
        pw = compute_performance(res["ls_pw"])
        ew = compute_performance(res["ls_ew"])
        r2 = res["R2"]
        rows.append({
            "Information set": cfg_name,
            "mean_PW (%)":     _fmt(pw["mean"]),
            "t-stat_PW":       _fmt(pw["t_stat"]),
            "SR_PW":           _fmt(pw["SR"]),
            "mean_EW (%)":     _fmt(ew["mean"]),
            "SR_EW":           _fmt(ew["SR"]),
            "R²_pred (%)":     _fmt(r2),
            "T (weeks)":       pw["T"],
        })
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "table3_long_short_performance.csv")
    df.to_csv(path, index=False)
    print("\n[Table 3] Long-short portfolio performance (test set, OOS):")
    print(df.to_string(index=False))
    return df


def make_table_A1(results_by_config):
    """Table A.1 equivalent: Top and bottom decile performance."""
    rows = []
    for cfg_name, res in results_by_config.items():
        vm = res["valid_mask"]
        for label, col, mat_key in [
            ("Top decile (PW)",    -1, "pw_decile_ret"),
            ("Bottom decile (PW)", 0,  "pw_decile_ret"),
            ("Top decile (EW)",    -1, "ew_decile_ret"),
            ("Bottom decile (EW)", 0,  "ew_decile_ret"),
        ]:
            ret  = res[mat_key][vm, col]
            perf = compute_performance(ret)
            rows.append({
                "Info set":  cfg_name,
                "Portfolio": label,
                "mean (%)":  _fmt(perf["mean"]),
                "t-stat":    _fmt(perf["t_stat"]),
                "SR":        _fmt(perf["SR"]),
                "T":         perf["T"],
            })
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "table_A1_decile_performance.csv")
    df.to_csv(path, index=False)
    print("\n[Table A.1] Top/Bottom decile performance:")
    print(df.to_string(index=False))
    return df


def make_table_B1():
    """Table B.1 equivalent: Hyperparameter configuration."""
    cfg = load_config()
    rows = [
        {"Parameter": "HL (hidden layers)",        "Optimal": cfg.get("num_layers", 1)},
        {"Parameter": "HU (units per layer)",       "Optimal": cfg.get("hidden_dim", [64])[0]},
        {"Parameter": "DR (dropout keep_prob)",     "Optimal": cfg.get("dropout", 0.95)},
        {"Parameter": "LR (learning rate)",         "Optimal": cfg.get("learning_rate", 0.001)},
        {"Parameter": "L1 regularization",          "Optimal": cfg.get("reg_l1", 0.0)},
        {"Parameter": "L2 regularization",          "Optimal": cfg.get("reg_l2", 0.001)},
        {"Parameter": "Num epochs",                 "Optimal": cfg.get("num_epochs", 300)},
        {"Parameter": "Ensemble seeds",             "Optimal": N_SEEDS},
        {"Parameter": "Train / Valid / Test split", "Optimal": "70% / 15% / 15%"},
        {"Parameter": "Model selection criterion",  "Optimal": "Validation Sharpe ratio"},
        {"Parameter": "Total features (full model)","Optimal": 33},
    ]
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "table_B1_hyperparameters.csv")
    df.to_csv(path, index=False)
    print("\n[Table B.1] Hyperparameter configuration:")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: Figures
# ─────────────────────────────────────────────────────────────────────────────

def _cumret(arr):
    """Cumulative sum (simple approximation for weekly returns)."""
    a = np.where(np.isnan(arr), 0.0, arr)
    return np.cumsum(a)


def fig01_macro_timeseries(data, dates):
    """Fig. 1 equivalent: Fear & Greed + BTC ETF Inflow time series."""
    date_arr = pd.to_datetime(dates)
    # Col in data: feature_index + 1 (col 0 = return)
    fear_col = 17   # fear_greed is feature 16 → col 17
    etf_col  = 28   # btc_etf_inflow_norm is feature 27 → col 28

    fear = np.where(data[:, 0, fear_col] == -99.99, np.nan, data[:, 0, fear_col])
    etf  = np.where(data[:, 0, etf_col]  == -99.99, np.nan, data[:, 0, etf_col])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(date_arr, fear, color="steelblue", linewidth=1)
    axes[0].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[0].set_title("(a) Fear & Greed Index (52-week rolling z-score)", fontsize=11)
    axes[0].set_ylabel("Standardised value")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(date_arr, etf, color="darkorange", linewidth=1)
    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1].set_title("(b) BTC ETF Inflow (52-week rolling z-score)", fontsize=11)
    axes[1].set_ylabel("Standardised value")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Fig. 1: Macro / Sentiment Time Series (Full Sample)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("fig01_macro_timeseries.png")


def fig02_data_split(dates, train_idx, valid_idx, test_idx):
    """Fig. 2 equivalent: Train / Validation / Test split timeline."""
    d = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(14, 2.5))

    colors = {"Train": "royalblue", "Valid": "darkorange", "Test": "mediumseagreen"}
    for label, idx, color in [
        ("Train", train_idx, colors["Train"]),
        ("Valid", valid_idx, colors["Valid"]),
        ("Test",  test_idx,  colors["Test"]),
    ]:
        ax.axvspan(d[idx[0]], d[idx[-1]], alpha=0.25, color=color,
                   label=f"{label} ({len(idx)} weeks)")
        ax.scatter(d[idx], np.zeros(len(idx)), c=color, s=4, alpha=0.6, zorder=3)

    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Fig. 2: Chronological Data Split — Train / Validation / Test", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    _save("fig02_data_split.png")


def fig05_cumulative_returns_decile(results_by_config, dates_test):
    """
    Fig. 5 equivalent: Cumulative returns sorted into prediction deciles.
    2 panels: prediction-weighted and equally-weighted.
    Uses the config with the most features as the main result.
    """
    best_name = list(results_by_config.keys())[-1]
    res   = results_by_config[best_name]
    vm    = res["valid_mask"]
    dates = pd.to_datetime(dates_test[vm])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    titles = [
        ("(a) Prediction-weighted returns", "pw_decile_ret"),
        ("(b) Equally-weighted returns",    "ew_decile_ret"),
    ]
    for ax, (title, key) in zip(axes, titles):
        mat = res[key][vm]                        # (T_valid, 10)
        for d in range(N_DECILES):
            lw    = 1.8 if d in (0, N_DECILES - 1) else 1.0
            ls    = "--" if d == 0 else "-"
            lbl   = f"Decile {d+1}" + (" (Top)" if d == N_DECILES-1 else (" (Bottom)" if d == 0 else ""))
            ax.plot(dates, _cumret(mat[:, d]), color=DECILE_COLORS[d],
                    linewidth=lw, linestyle=ls, label=lbl)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Date"); ax.set_ylabel("Cumulative return")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Fig. 5: Cumulative Returns by Prediction Decile  [{best_name}, Test Period, OOS]",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    _save("fig05_cumulative_returns_decile.png")


def fig07_info_sets_comparison(results_by_config, dates_test):
    """Fig. 7/8 equivalent: Long-short cumulative returns for each information set."""
    colors = ["royalblue", "tomato", "mediumseagreen", "darkorange", "darkviolet"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (ls_key, title) in zip(axes, [
        ("ls_pw", "(a) Prediction-weighted long-short"),
        ("ls_ew", "(b) Equally-weighted long-short"),
    ]):
        for (cfg_name, res), color in zip(results_by_config.items(), colors):
            vm   = res["valid_mask"]
            ret  = res[ls_key]
            dts  = pd.to_datetime(dates_test[vm])
            ax.plot(dts, _cumret(ret), color=color, linewidth=2, label=cfg_name)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Date"); ax.set_ylabel("Cumulative return")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle("Fig. 7/8: Long-Short Cumulative Returns by Information Set (Test Period, OOS)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save("fig07_info_sets_comparison.png")


def fig10_holding_period(results_by_config, dates_test):
    """Fig. 10 equivalent: Long-short performance for different holding periods."""
    hold_weeks = [1, 2, 4, 8, 12, 16]
    colors     = ["royalblue", "darkorange", "tomato", "mediumseagreen"]
    cfg_names  = list(results_by_config.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [
        ("mean",   "(a) Mean of returns (%)"),
        ("std",    "(b) Std of returns (%)"),
        ("SR",     "(c) Sharpe ratio"),
        ("t_stat", "(d) T-statistics"),
    ]

    for ax, (mkey, ylabel) in zip(axes.flatten(), metrics):
        for cfg_name, color in zip(cfg_names, colors):
            res = results_by_config[cfg_name]
            hp  = compute_holding_period_stats(
                res["ew_decile_ret"], res["valid_mask"], hold_weeks
            )
            xs = sorted(hp.keys())
            ys = [hp[s][mkey] for s in xs]
            ax.plot(xs, ys, color=color, marker="o", linewidth=2, label=cfg_name)
        ax.set_xlabel("Holding period (weeks)")
        ax.set_ylabel(ylabel.split(") ", 1)[1])
        ax.set_title(ylabel, fontsize=10)
        ax.set_xticks(hold_weeks)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("Fig. 10: Long-Short Performance for Different Holding Periods",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save("fig10_holding_period.png")


def fig12_variable_importance(trainer, dl_test, n_features, feat_names_used):
    """Fig. 12 equivalent: Variable importance bar chart."""
    print("    Computing sensitivities (numerical gradient)...")
    sens  = compute_variable_importance(trainer, dl_test, n_features)
    order = np.argsort(sens)[::-1]

    sorted_names  = [feat_names_used[i] if i < len(feat_names_used) else f"feat_{i}" for i in order]
    sorted_sens   = sens[order]

    # Colour by global feature category
    feat_to_cat = {i: cat for cat, idxs in FEATURE_CATEGORIES.items() for i in idxs}
    bar_colors  = [CATEGORY_COLORS.get(feat_to_cat.get(order[k], "Price Momentum"), "gray")
                   for k in range(len(order))]

    group_imp   = compute_group_importance(sens, n_features)
    sorted_cats = sorted(group_imp, key=group_imp.get, reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n_features * 0.35 + 1)))

    # Individual importance
    ax = axes[0]
    y  = np.arange(len(sorted_names))
    ax.barh(y, sorted_sens, color=bar_colors, alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlabel("Sensitivity"); ax.set_title("(a) Variable Importance", fontsize=11)
    legend_h = [mpatches.Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in CATEGORY_COLORS]
    ax.legend(handles=legend_h, fontsize=7, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)

    # Group importance
    ax2 = axes[1]
    gv  = [group_imp[c] for c in sorted_cats]
    gc  = [CATEGORY_COLORS[c] for c in sorted_cats]
    y2  = np.arange(len(sorted_cats))
    ax2.barh(y2, gv, color=gc, alpha=0.85)
    ax2.set_yticks(y2); ax2.set_yticklabels(sorted_cats, fontsize=9)
    ax2.set_xlabel("Group Sensitivity"); ax2.set_title("(b) Variable Group Importance", fontsize=11)
    ax2.grid(True, axis="x", alpha=0.3)

    plt.suptitle("Fig. 12: Variable Importance Ranking (Test Data)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save("fig12_variable_importance.png")
    return sens


def fig13_interaction_effects(trainer, dl_test, feat_names_used, n_features):
    """
    Fig. 13 equivalent: Predicted returns as function of key feature × fear_greed quantile.
    """
    # Determine which features to plot (prefer r12w, r1w, r52w, btc_etf_inflow_norm)
    prefer = ["r12w", "r1w", "r52w", "btc_etf_inflow_norm"]
    key_feats = []
    for name in prefer:
        if name in feat_names_used:
            key_feats.append((feat_names_used.index(name), name))
        if len(key_feats) == 4:
            break
    while len(key_feats) < 4 and len(key_feats) < n_features:
        idx = len(key_feats)
        key_feats.append((idx, feat_names_used[idx] if idx < len(feat_names_used) else f"feat_{idx}"))

    fear_idx = feat_names_used.index("fear_greed") if "fear_greed" in feat_names_used else min(16, n_features - 1)
    SENT_COLORS = ["navy", "royalblue", "forestgreen", "darkorange", "darkred"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (fidx, fname) in zip(axes.flatten(), key_feats):
        x_grid, pred_mat, labels = compute_interaction_grid(trainer, dl_test, fidx, fear_idx)
        if x_grid is None:
            continue
        for q, (preds, lbl, col) in enumerate(zip(pred_mat, labels, SENT_COLORS)):
            ax.scatter(x_grid, preds * 100, c=col, s=8, label=lbl)
        ax.set_xlabel(fname); ax.set_ylabel("Predicted return (%)")
        ax.set_title(f"Predicted returns as function of {fname}", fontsize=10)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle("Fig. 13: Interaction Effects (Predicted Returns × Fear & Greed Quantiles)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save("fig13_interaction_effects.png")


def fig14_3d_surface(trainer, dl_test, feat_names_used, n_features):
    """Fig. 14 equivalent: 3-D surface of predicted return vs r12w × fear_greed."""
    import tensorflow as tf

    feat_x = feat_names_used.index("r12w")      if "r12w"      in feat_names_used else 2
    feat_y = feat_names_used.index("fear_greed") if "fear_greed" in feat_names_used else min(16, n_features - 1)

    for I_macro, I, R, mask in dl_test.iterateOneEpoch():
        X_valid, _ = trainer._flatten_valid(I_macro, I, R, mask)
        if len(X_valid) == 0:
            return
        X_med = np.median(X_valid, axis=0)
        n_g   = 25
        xs    = np.linspace(-0.4, 0.4, n_g)
        ys    = np.linspace(-0.4, 0.4, n_g)
        XX, YY = np.meshgrid(xs, ys)
        ZZ     = np.zeros_like(XX)

        for i in range(n_g):
            batch = np.tile(X_med, (n_g, 1)).copy()
            batch[:, feat_x] = XX[i]
            if feat_y < batch.shape[1]:
                batch[:, feat_y] = YY[i]
            preds = trainer.model(
                tf.constant(batch.astype(np.float32)), training=False
            ).numpy().flatten()
            ZZ[i] = preds * 100

        fig  = plt.figure(figsize=(10, 7))
        ax3d = fig.add_subplot(111, projection="3d")
        surf = ax3d.plot_surface(XX, YY, ZZ, cmap="RdYlGn", alpha=0.85, edgecolor="none")
        fig.colorbar(surf, ax=ax3d, shrink=0.45, aspect=8, label="Predicted return (%)")
        ax3d.set_xlabel(feat_names_used[feat_x] if feat_x < len(feat_names_used) else "r12w")
        ax3d.set_ylabel(feat_names_used[feat_y] if feat_y < len(feat_names_used) else "fear_greed")
        ax3d.set_zlabel("Predicted return (%)")
        ax3d.set_title("Fig. 14: Predicted Return as Function of r12w × fear_greed", fontsize=11)
        plt.tight_layout()
        _save("fig14_3d_surface.png")
        return


def fig_training_curves():
    """Bonus: Training Sharpe / loss curves from training_log.csv files."""
    logs = glob.glob(os.path.join(CHECKPOINT_DIR, "folder_*", "*", "training_log.csv"))
    if not logs:
        print("    No training_log.csv files found; skipping.")
        return

    # Group by feature key
    grouped = {}
    for lp in logs:
        m = re.search(r"(feat\d+to\d+)", lp)
        key = m.group(1) if m else "unknown"
        grouped.setdefault(key, []).append(lp)

    for fkey, log_list in sorted(grouped.items()):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        tab10 = plt.cm.tab10(np.linspace(0, 1, max(len(log_list), 1)))
        for lp, color in zip(log_list, tab10):
            try:
                df  = pd.read_csv(lp)
                seed = re.search(r"folder_(\d+)", lp)
                lbl  = f"seed {seed.group(1)}" if seed else "?"
                axes[0].plot(df["epoch"], df["valid_loss"],   color=color, alpha=0.8,
                             linewidth=1, label=lbl)
                axes[1].plot(df["epoch"], df["valid_sharpe"], color=color, alpha=0.8,
                             linewidth=1, label=lbl)
            except Exception:
                continue
        for ax, title, ylabel in [
            (axes[0], "Validation MSE Loss", "MSE"),
            (axes[1], "Validation Sharpe Ratio", "Annualised Sharpe"),
        ]:
            ax.set_title(f"{title} ({fkey})"); ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        plt.suptitle(f"Training Curves — {fkey}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save(f"fig_training_curves_{fkey}.png")


def fig_transition_matrix(pred, actual, mask, dates_test, cfg_name):
    """Fig. A.10 equivalent: Prediction-decile transition probability matrix."""
    T, N = mask.shape
    decile_assign = np.full((T, N), -1, dtype=int)
    ptr = 0
    for t in range(T):
        n_t = int(mask[t].sum())
        if n_t == 0:
            continue
        pred_t = pred[ptr: ptr + n_t]
        ptr   += n_t
        if n_t < N_DECILES:
            continue
        asset_idx = np.where(mask[t])[0]
        order     = np.argsort(pred_t)
        for d, bucket in enumerate(np.array_split(order, N_DECILES)):
            for pos in bucket:
                decile_assign[t, asset_idx[pos]] = d

    trans  = np.zeros((N_DECILES, N_DECILES))
    counts = np.zeros(N_DECILES)
    for t in range(T - 1):
        for n in range(N):
            d0, d1 = decile_assign[t, n], decile_assign[t + 1, n]
            if d0 >= 0 and d1 >= 0:
                trans[d0, d1] += 1
                counts[d0]    += 1
    for i in range(N_DECILES):
        if counts[i] > 0:
            trans[i] /= counts[i]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(trans, cmap="YlOrRd", vmin=0, vmax=0.5)
    for i in range(N_DECILES):
        for j in range(N_DECILES):
            ax.text(j, i, f"{trans[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if trans[i, j] > 0.35 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(N_DECILES)); ax.set_xticklabels(range(1, N_DECILES + 1))
    ax.set_yticks(range(N_DECILES)); ax.set_yticklabels(range(1, N_DECILES + 1))
    ax.set_xlabel("Decile at t+1"); ax.set_ylabel("Decile at t")
    ax.set_title(f"Fig. A.10: Transition Matrix  [{cfg_name}]", fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save("fig_transition_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _save(fname):
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"    [Saved] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  BTC Deep Learning — FEN Paper Results Visualisation")
    print("=" * 65)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    data, dates, assets, variables, train_idx, valid_idx, test_idx = load_data()
    print(f"    Shape : {data.shape}  (T × N × features+1)")
    print(f"    Dates : {dates[0]} → {dates[-1]}")
    print(f"    Assets: {list(assets)}")
    print(f"    Split : Train {len(train_idx)} | Valid {len(valid_idx)} | Test {len(test_idx)} weeks")
    dates_test = dates[test_idx]

    # ── 2. Checkpoints ───────────────────────────────────────────────────────
    print("\n[2] Scanning checkpoints...")
    avail = detect_available_configs()
    if not avail:
        print("ERROR: No checkpoints found. Run train_btc.py first.")
        sys.exit(1)

    # ── 3. Ensemble predictions ──────────────────────────────────────────────
    print("\n[3] Generating ensemble predictions...")
    results = {}
    for fcfg in avail:
        name = fcfg["name"]
        print(f"  • {name}  ({fcfg['feat_key']}) ...")
        pred, actual, mask = get_ensemble_predictions(fcfg, test_idx)
        if pred is None:
            print(f"    Warning: skipped (no valid predictions).")
            continue

        ew_ret, pw_ret, vm = build_decile_portfolios(pred, actual, mask)
        ls_ew = build_long_short(ew_ret, vm)
        ls_pw = build_long_short(pw_ret, vm)
        r2    = compute_prediction_r2(pred, actual, mask)

        perf = compute_performance(ls_pw)
        print(f"    LS-PW  mean={perf['mean']:.2f}%  SR={perf['SR']:.2f}  "
              f"t={perf['t_stat']:.2f}  R²={r2:.2f}%  T={perf['T']}")

        results[name] = dict(
            pred=pred, actual=actual, mask=mask,
            ew_decile_ret=ew_ret, pw_decile_ret=pw_ret,
            valid_mask=vm,
            ls_ew=ls_ew, ls_pw=ls_pw, R2=r2,
            feat_cfg=fcfg,
        )

    if not results:
        print("ERROR: No valid results generated.")
        sys.exit(1)

    best_name = list(results.keys())[-1]
    best_res  = results[best_name]
    best_fcfg = best_res["feat_cfg"]
    print(f"\n  Best model: {best_name}  ({best_fcfg['feat_key']})")

    # ── 4. Load best model for gradient analysis ─────────────────────────────
    print("\n[4] Loading best model for importance / interaction analysis...")
    config = load_config()
    config["individual_feature_dim"] = len(best_fcfg["subset"])
    best_trainer = BTCTrainer(config)
    loaded_seed  = None
    for seed in range(1, N_SEEDS + 1):
        logdir = find_checkpoint_dir(f"folder_{seed}", best_fcfg["feat_key"])
        if logdir:
            try:
                best_trainer.load_best(logdir)
                loaded_seed = seed
                print(f"    Loaded seed {seed} from {logdir}")
                break
            except Exception:
                continue
    if loaded_seed is None:
        print("    Warning: could not load a model for gradient analysis.")

    dl_test_best    = DataInRamInputLayer(DATA_FILE, test_idx, best_fcfg["subset"])
    feat_names_used = [FEATURE_NAMES[i] for i in best_fcfg["subset"] if i < len(FEATURE_NAMES)]

    # ── 5. Tables ────────────────────────────────────────────────────────────
    print("\n[5] Generating tables...")
    make_table3(results)
    make_table_A1(results)
    make_table_B1()

    # ── 6. Figures ───────────────────────────────────────────────────────────
    print("\n[6] Generating figures...")

    print("  Fig 01: Macro time series")
    fig01_macro_timeseries(data, dates)

    print("  Fig 02: Data split")
    fig02_data_split(dates, train_idx, valid_idx, test_idx)

    print("  Fig 05: Cumulative returns by decile")
    fig05_cumulative_returns_decile(results, dates_test)

    print("  Fig 07: Information sets comparison")
    fig07_info_sets_comparison(results, dates_test)

    print("  Fig 10: Holding period analysis")
    fig10_holding_period(results, dates_test)

    if loaded_seed is not None:
        print("  Fig 12: Variable importance")
        fig12_variable_importance(
            best_trainer, dl_test_best,
            len(best_fcfg["subset"]), feat_names_used
        )

        print("  Fig 13: Interaction effects")
        fig13_interaction_effects(
            best_trainer, dl_test_best,
            feat_names_used, len(best_fcfg["subset"])
        )

        print("  Fig 14: 3D surface plot")
        fig14_3d_surface(
            best_trainer, dl_test_best,
            feat_names_used, len(best_fcfg["subset"])
        )
    else:
        print("  Skipping gradient-based figures (no model loaded).")

    print("  Training curves")
    fig_training_curves()

    print("  Transition matrix")
    fig_transition_matrix(
        best_res["pred"], best_res["actual"],
        best_res["mask"], dates_test, best_name
    )

    print("\n" + "=" * 65)
    print(f"  Done!  All outputs in: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
