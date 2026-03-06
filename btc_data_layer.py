"""
btc_data_layer.py
-----------------
BTC 多資產 Panel 資料載入層。

修改自原始 mutual fund 的 data_layer_cross.py，完全獨立於
deep_learning/src/ 目錄，不依賴任何 TF1 模組。

主要差異：
  - FirmChar → CryptoChar（5 大特徵類別，22 個特徵）
  - wficn    → 加密資產代碼（BTC, ETH, SOL, ...）
  - 缺失值標記為 UNK = -99.99（與原版一致）
  - 型別注解相容 Python 3.8+
"""

from __future__ import annotations

import os
import numpy as np


# ── 工具函數（獨立版，不依賴原始 src/utils.py）────────────────────────────────

def deco_print(line: str, end: str = "\n") -> None:
    print(">==================> " + str(line), end=end)


def squeeze_data(data: np.ndarray, UNK: float = -99.99):
    """
    移除報酬欄（col 0）全為 UNK 的資產（對應原版 squeeze_data）。

    Parameters
    ----------
    data : (T, N, M+1)
    UNK  : float  缺失值標記

    Returns
    -------
    data_filtered   : (T, N', M+1)
    lists_considered: list  保留的資產索引
    """
    returns = data[:, :, 0]  # (T, N)
    lists_considered = [
        i for i in range(data.shape[1])
        if np.any(returns[:, i] != UNK)
    ]
    return data[:, lists_considered, :], lists_considered


# ── 特徵類別定義 ───────────────────────────────────────────────────────────────

class CryptoChar:
    """
    加密資產特徵的分類定義（對應原版 FirmChar）。

    Category A - Price Momentum  : r1w, r4w, r12w, r26w, r52w
    Category B - Technical       : rsi_14, bb_pct, vol_ratio, atr_pct, obv_change
    Category C - On-chain        : active_addr, tx_count, nvt, exchange_net_flow, mvrv
    Category D - Macro/Sentiment : fear_greed, spx_ret, dxy_ret, vix
    Category E - ETF + Polymarket: etf_net_flow_norm, polymarket_btc, etf_net_flow_raw
    """

    def __init__(self) -> None:
        self._category = [
            "Price Momentum",
            "Technical",
            "On-chain",
            "Macro/Sentiment",
            "ETF & Polymarket",
        ]
        self._category2variables = {
            "Price Momentum":   ["r1w", "r4w", "r12w", "r26w", "r52w"],
            "Technical":        ["rsi_14", "bb_pct", "vol_ratio", "atr_pct", "obv_change"],
            "On-chain":         ["active_addr", "tx_count", "nvt", "exchange_net_flow", "mvrv"],
            "Macro/Sentiment":  ["fear_greed", "spx_ret", "dxy_ret", "vix"],
            "ETF & Polymarket": ["etf_net_flow_norm", "polymarket_btc", "etf_net_flow_raw"],
        }
        self._variable2category = {
            var: cat
            for cat, vars_ in self._category2variables.items()
            for var in vars_
        }
        self._category2color = {
            "Price Momentum":   "royalblue",
            "Technical":        "tomato",
            "On-chain":         "mediumseagreen",
            "Macro/Sentiment":  "darkviolet",
            "ETF & Polymarket": "darkorange",
        }
        self._color2category = {v: k for k, v in self._category2color.items()}

    def getColorLabelMap(self) -> dict:
        """回傳 {變數名: 顏色} 映射（用於 variable importance 視覺化）。"""
        return {
            var: self._category2color[self._variable2category[var]]
            for var in self._variable2category
        }


# ── 主資料層 ───────────────────────────────────────────────────────────────────

class DataInRamInputLayer:
    """
    BTC 多資產 Panel 資料載入層（對應原版 DataInRamInputLayer）。

    資料格式：NPZ 檔案，包含：
      data     : (T, N, M+1)  col 0 = return，col 1..M = features
      date     : (T,)          週結束日字串
      wficn    : (N,)          資產代碼（BTC, ETH, ...）
      variable : (M,)          特徵名稱

    Parameters
    ----------
    pathIndividualFeature : str
        NPZ 檔路徑（訓練/驗證/測試集透過 idx_list 從同一檔切分）
    idx_list : list
        時間索引（哪些週作為此分割的資料）
    subset   : list
        使用的特徵索引（0-based，對應 variable 陣列）
    """

    def __init__(
        self,
        pathIndividualFeature: str,
        idx_list,
        subset,
        pathMacroFeature: "Optional[str]" = None,   # 保留以維持 API 相容，BTC 版不使用
        macroIdx=None,
        meanMacroFeature=None,
        stdMacroFeature=None,
        normalizeMacroFeature: bool = True,
    ) -> None:
        self.idx_list = list(idx_list)
        self.subset   = list(subset)
        self._UNK     = -99.99

        self._load_individual_feature(pathIndividualFeature)
        self._load_macro_feature()
        self._crypto_char = CryptoChar()

    # ── 索引字典建立 ──────────────────────────────────────────────────────────

    @staticmethod
    def _make_idx_dict(arr):
        idx2var = {idx: val for idx, val in enumerate(arr)}
        var2idx = {val: idx for idx, val in enumerate(arr)}
        return idx2var, var2idx

    # ── 資料載入 ──────────────────────────────────────────────────────────────

    def _load_individual_feature(self, path: str) -> None:
        tmp  = np.load(path, allow_pickle=True)
        data = tmp["data"]  # (T_all, N_all, M+1)

        # 選取特徵欄（col 0 = return 固定保留）
        cols = [0] + [x + 1 for x in self.subset]
        data = data[:, :, cols]

        # 選取時間切片，移除全缺失資產
        data, list_considered = squeeze_data(data[self.idx_list], UNK=self._UNK)

        self._return            = data[:, :, 0]   # (T, N)
        self._individualFeature = data[:, :, 1:]  # (T, N, len(subset))
        self._mask              = (self._return != self._UNK)

        # 索引字典
        raw_dates = tmp["date"]
        dates_slice = raw_dates[self.idx_list] if len(raw_dates) > 0 else []
        self._idx2date,   self._date2idx   = self._make_idx_dict(dates_slice)

        self._idx2permno, self._permno2idx = self._make_idx_dict(
            tmp["wficn"][list_considered]
        )
        self._idx2var,    self._var2idx    = self._make_idx_dict(
            tmp["variable"][self.subset]
        )

        self._dateCount   = data.shape[0]
        self._permnoCount = data.shape[1]
        self._varCount    = data.shape[2] - 1  # 扣除 return 欄

    def _load_macro_feature(self) -> None:
        """BTC 版本不使用分離的 macro feature 檔，設置空陣列。"""
        self._macroFeature     = np.empty((self._dateCount, 0), dtype=np.float32)
        self._meanMacroFeature = None
        self._stdMacroFeature  = None

    # ── 特徵查詢接口（與原版相容）────────────────────────────────────────────

    def getIndividualFeatureByIdx(self, idx: int) -> str:
        return self._idx2var.get(idx, f"feature_{idx}")

    def getFeatureByIdx(self, idx: int) -> str:
        if idx < self._varCount:
            return self.getIndividualFeatureByIdx(idx)
        return f"macro_{idx - self._varCount}"

    def getMacroFeatureMeanStd(self):
        return self._meanMacroFeature, self._stdMacroFeature

    def getIndividualFeatureColarLabelMap(self):
        return (
            self._crypto_char.getColorLabelMap(),
            self._crypto_char._color2category,
        )

    def getDateCountList(self) -> np.ndarray:
        """計算每個時間點有效資產數（用於加權損失函數）。"""
        return self._mask.astype(np.float32)

    def getAssets(self) -> list:
        return list(self._idx2permno.values())

    # ── 資料迭代器（與原版相容）──────────────────────────────────────────────

    def iterateOneEpoch(self, subEpoch=False):
        """
        回傳 (macro_feature, individual_feature, return, mask)。
        格式與原版完全相容，可直接傳入 BTCTrainer。

        macro_feature      : (T, 0)         空陣列
        individual_feature : (T, N, M)      特徵矩陣
        return             : (T, N)          報酬（缺失為 -99.99）
        mask               : (T, N) bool
        """
        batch = (
            self._macroFeature,
            self._individualFeature,
            self._return,
            self._mask,
        )
        if subEpoch:
            for _ in range(int(subEpoch)):
                yield batch
        else:
            yield batch

    # ── 統計資訊 ──────────────────────────────────────────────────────────────

    def summary(self) -> None:
        deco_print("BTC Panel Dataset Summary")
        deco_print(f"  T (weeks)       = {self._dateCount}")
        deco_print(f"  N (assets)      = {self._permnoCount}")
        deco_print(f"  M (features)    = {self._varCount}")
        deco_print(f"  Valid obs ratio = {self._mask.mean():.2%}")
        deco_print(f"  Assets  : {self.getAssets()}")
        deco_print(f"  Features: {list(self._idx2var.values())}")
