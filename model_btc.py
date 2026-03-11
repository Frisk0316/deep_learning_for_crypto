"""
model_btc.py
------------
BTC 版前饋神經網路（FFN + Gated Interaction），基於 TensorFlow 2.x / Keras API。

架構演進（相對原始 FEN 論文）：
  v1  原版 FEN：純 FFN，所有特徵拼接後過 Dense → ReLU → Dense
  v2  新增 Gated Interaction Layer：
      - 參考 FEN 論文 Fig. 12-13 中 sentiment × fund characteristics
        交互作用是最重要預測因子的發現
      - 將輸入特徵拆分為「個別資產特徵」與「市場狀態特徵」
      - 市場狀態特徵（macro/DeFi/衍生品）通過 sigmoid gate 調控
        個別資產特徵的權重，顯式建模 interaction effect
      - 同時保留原始 concatenation 路徑（殘差連接）

模型結構：
  Input: x = [z_asset || z_market]

  Gate path:  g = σ(W_g · z_market + b_g)         ∈ (0,1)^K
              z_gated = g ⊙ (W_a · z_asset + b_a)  ∈ R^K

  Main path:  z_concat = [z_asset || z_market]
              h = ReLU(W_h · z_concat + b_h)       ∈ R^H

  Combined:   h_combined = [h || z_gated]
              output = W_o · h_combined + b_o       ∈ R^1

訓練邏輯不變：
  - Panel 資料 (T × N × M)，以布林遮罩過濾缺失值
  - MSE 損失 + L1/L2 正則化 + Dropout
  - 驗證集 Sharpe Ratio 為 checkpoint 選取準則
  - 支援 'Factor_sharpe' / 'natural' 兩種模式

外部依賴：tensorflow >= 2.2, numpy, pandas
"""

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV


# ── 工具函數（獨立版，不依賴原始 src/utils.py）────────────────────────────────

def deco_print(line, end="\n"):
    print(">==================> " + str(line), end=end)


def sharpe_ratio(returns: np.ndarray, annualize: bool = True) -> float:
    """
    計算 Sharpe Ratio。
    週頻資料乘以 sqrt(52) 年化；序列過短或標準差為零時回傳 0。
    """
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0 or np.std(arr) < 1e-10:
        return 0.0
    sr = np.mean(arr) / np.std(arr)
    return float(sr * np.sqrt(52)) if annualize else float(sr)


def construct_long_short_portfolio(
    R_pred: np.ndarray,
    R_actual: np.ndarray,
    mask: np.ndarray,
    low: float = 0.2,
    high: float = 0.2,
) -> np.ndarray:
    """
    依預測報酬建立等權多空投資組合（對應原版 construct_long_short_portfolio）。

    Parameters
    ----------
    R_pred   : (n_valid,)  預測報酬（已展平，僅含有效觀測，row-major 順序）
    R_actual : (n_valid,)  實際報酬
    mask     : (T, N) bool 原始遮罩，用於還原每個時間點的資產集合
    low/high : float       多空各取的比例（0.2 = 20%）

    Returns
    -------
    portfolio_returns : (n_timesteps_with_enough_assets,) 每期多空報酬
    """
    T, N = mask.shape
    portfolio_returns = []
    ptr = 0  # 在 R_pred / R_actual 中的指標

    for t in range(T):
        n_t = int(mask[t].sum())
        if n_t == 0:
            continue

        pred_t   = R_pred[ptr: ptr + n_t]
        actual_t = R_actual[ptr: ptr + n_t]
        ptr += n_t

        if n_t < 5:  # 資產數太少，跳過
            continue

        n_long  = max(1, int(n_t * high))
        n_short = max(1, int(n_t * low))

        order      = np.argsort(pred_t)
        long_ret   = float(np.mean(actual_t[order[-n_long:]]))
        short_ret  = float(np.mean(actual_t[order[:n_short]]))
        portfolio_returns.append(long_ret - short_ret)

    return np.array(portfolio_returns, dtype=np.float64)


# ── 原版 FFN 模型（保留向後相容）──────────────────────────────────────────────

class FFNModel(tf.keras.Model):
    """
    原版前饋神經網路（無交互機制）：
        輸入 → [Dense(ReLU) → Dropout] × num_layers → Dense(線性) → 預測報酬
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dims: list,
        dropout_rate: float,
        l1: float = 0.0,
        l2: float = 0.001,
    ):
        super().__init__()
        reg = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

        self.hidden_layers  = []
        self.dropout_layers = []

        for h_dim in hidden_dims:
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    h_dim, activation="relu", kernel_regularizer=reg
                )
            )
            self.dropout_layers.append(tf.keras.layers.Dropout(dropout_rate))

        self.output_layer = tf.keras.layers.Dense(1, kernel_regularizer=reg)

    def call(self, x, training=False):
        h = x
        for dense, drop in zip(self.hidden_layers, self.dropout_layers):
            h = dense(h)
            h = drop(h, training=training)
        return self.output_layer(h)  # shape: (batch, 1)


# ── Gated Interaction FFN 模型（新版）─────────────────────────────────────────

class GatedInteractionFFN(tf.keras.Model):
    """
    帶有 Gated Interaction 機制的前饋神經網路。

    參考 FEN 論文 (Kaniel et al., 2023) 的核心發現：
      - sentiment/macro × fund characteristics 的交互作用
        是預測 abnormal returns 的最重要因子 (Fig. 12-13, Table 6)
      - 純 FFN 雖能隱式學到交互效應，但需要更多參數
      - 顯式的 gating 機制能更高效地學習 state-dependent 特徵重要性

    架構：
      1) Gate path:  market state → sigmoid → element-wise gate on asset features
      2) Main path:  全特徵拼接 → Dense(ReLU) → Dropout (原版 FEN)
      3) Combined:   [main_hidden || gated_features] → Dense(1)

    Parameters
    ----------
    asset_dim   : int   個別資產特徵維度 (Category A~C, features 0~15)
    market_dim  : int   市場狀態特徵維度 (Category D~F, features 16~48)
    gate_dim    : int   Gate 輸出維度 (預設 = asset_dim)
    hidden_dims : list  主路徑每層節點數
    dropout_rate: float
    l1, l2      : float
    """

    def __init__(
        self,
        asset_dim: int,
        market_dim: int,
        gate_dim: int = 0,
        hidden_dims: list = None,
        dropout_rate: float = 0.05,
        l1: float = 0.0,
        l2: float = 0.001,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64]
        if gate_dim == 0:
            gate_dim = asset_dim

        self.asset_dim  = asset_dim
        self.market_dim = market_dim
        self.gate_dim   = gate_dim

        reg = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

        # ── Gate path: market_state → gate vector ──
        # 市場狀態特徵通過線性 → sigmoid 生成門控向量
        self.gate_proj = tf.keras.layers.Dense(
            gate_dim, activation="sigmoid",
            kernel_regularizer=reg, name="gate_proj"
        )
        # 資產特徵投影到 gate 空間
        self.asset_proj = tf.keras.layers.Dense(
            gate_dim, activation="relu",
            kernel_regularizer=reg, name="asset_proj"
        )

        # ── Main path: 與原版 FEN 完全相同 ──
        self.main_hidden_layers  = []
        self.main_dropout_layers = []

        total_dim = asset_dim + market_dim
        for h_dim in hidden_dims:
            self.main_hidden_layers.append(
                tf.keras.layers.Dense(
                    h_dim, activation="relu", kernel_regularizer=reg
                )
            )
            self.main_dropout_layers.append(
                tf.keras.layers.Dropout(dropout_rate)
            )

        # ── Output: 合併 main path + gated features ──
        combined_dim = hidden_dims[-1] + (gate_dim if market_dim > 0 else 0)
        self.output_layer = tf.keras.layers.Dense(
            1, kernel_regularizer=reg, name="output"
        )

    def call(self, x, training=False):
        """
        Parameters
        ----------
        x : (batch, asset_dim + market_dim)
            前 asset_dim 維 = 個別資產特徵
            後 market_dim 維 = 市場狀態特徵

        Returns
        -------
        output : (batch, 1)
        """
        # Main path: 與原版 FEN 相同（全特徵拼接 → Dense → ReLU）
        h = x
        for dense, drop in zip(self.main_hidden_layers, self.main_dropout_layers):
            h = dense(h)
            h = drop(h, training=training)

        # Gate path: 僅在 market_dim > 0 時啟用
        if self.market_dim > 0:
            z_asset  = x[:, :self.asset_dim]    # (batch, asset_dim)
            z_market = x[:, self.asset_dim:]    # (batch, market_dim)
            gate = self.gate_proj(z_market)     # (batch, gate_dim) ∈ (0,1)
            z_a  = self.asset_proj(z_asset)     # (batch, gate_dim)
            z_gated = gate * z_a                # element-wise gating
            h_combined = tf.concat([h, z_gated], axis=-1)
        else:
            # 無市場特徵時退化為純 FFN
            h_combined = h

        return self.output_layer(h_combined)  # (batch, 1)


# ── 特徵降維前處理（§12.5.2 Feature selection pre-processing）─────────────────

class FeatureReducer:
    """
    針對小橫截面（N ≤ 50）的特徵降維器。

    解決 §12.1 維度詛咒問題：14 個資產 × 49 個特徵 → 過擬合。
    支援三種模式：
      - "pca":   PCA 保留指定比例的變異
      - "lasso": LASSO 回歸篩選非零係數特徵
      - "none":  不做降維（向後相容）

    在訓練集上 fit，驗證/測試集上 transform。
    """

    def __init__(self, method: str = "none", n_components: float = 0.95,
                 lasso_alpha: str = "cv"):
        """
        Parameters
        ----------
        method       : "pca" | "lasso" | "none"
        n_components : PCA 模式下保留的累積變異比例（0~1）或固定維度（int）
        lasso_alpha  : "cv" 表示用交叉驗證選 alpha
        """
        self.method = method
        self.n_components = n_components
        self._reducer = None
        self._selected_mask = None  # LASSO 用
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """在訓練資料上擬合降維器。"""
        if self.method == "pca":
            self._reducer = PCA(n_components=self.n_components)
            self._reducer.fit(X)
            deco_print(
                f"PCA: {X.shape[1]} → {self._reducer.n_components_} dims "
                f"(explained variance: {self._reducer.explained_variance_ratio_.sum():.2%})"
            )
        elif self.method == "lasso":
            if y is None:
                raise ValueError("LASSO 降維需要 target y")
            lasso = LassoCV(cv=5, max_iter=5000, n_jobs=-1)
            lasso.fit(X, y)
            self._selected_mask = np.abs(lasso.coef_) > 1e-8
            n_selected = self._selected_mask.sum()
            if n_selected == 0:
                # 退回全部特徵
                self._selected_mask = np.ones(X.shape[1], dtype=bool)
                n_selected = X.shape[1]
            deco_print(
                f"LASSO: {X.shape[1]} → {n_selected} features "
                f"(alpha={lasso.alpha_:.6f})"
            )
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """轉換特徵矩陣。"""
        if self.method == "pca" and self._reducer is not None:
            return self._reducer.transform(X).astype(np.float32)
        elif self.method == "lasso" and self._selected_mask is not None:
            return X[:, self._selected_mask].astype(np.float32)
        return X.astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    @property
    def output_dim(self) -> int:
        """降維後的特徵維度。"""
        if self.method == "pca" and self._reducer is not None:
            return self._reducer.n_components_
        elif self.method == "lasso" and self._selected_mask is not None:
            return int(self._selected_mask.sum())
        return -1  # 未知，需要在 fit 後才能確定


# ── 輕量門控 FFN（§12.5.5 Lightweight gated architecture）────────────────────

class LightweightGatedFFN(tf.keras.Model):
    """
    針對小橫截面設計的輕量門控 FFN。

    與 GatedInteractionFFN 的差異：
      1) 參數量大幅減少：使用線性瓶頸層（bottleneck）壓縮特徵
      2) BatchNorm 穩定小樣本訓練
      3) 殘差連接防止梯度消失
      4) Gate 使用 softmax（而非 sigmoid）實現稀疏注意力

    架構：
      Input → BatchNorm → Bottleneck(Linear) → [Gate + Main] → Output

    適用場景：N < 50, features > 15
    """

    def __init__(
        self,
        input_dim: int,
        bottleneck_dim: int = 8,
        hidden_dim: int = 32,
        dropout_rate: float = 0.10,
        l2: float = 0.005,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        reg = tf.keras.regularizers.L2(l2)

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.input_bn = tf.keras.layers.BatchNormalization(name="input_bn")

        # 瓶頸層：壓縮高維特徵
        self.bottleneck = tf.keras.layers.Dense(
            bottleneck_dim, activation="relu",
            kernel_regularizer=reg, name="bottleneck"
        )

        # 稀疏注意力門控：softmax 讓 gate weights 加總為 1
        self.gate = tf.keras.layers.Dense(
            bottleneck_dim, activation="softmax",
            kernel_regularizer=reg, name="sparse_gate"
        )

        # 主路徑
        self.hidden = tf.keras.layers.Dense(
            hidden_dim, activation="relu",
            kernel_regularizer=reg, name="hidden"
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # 輸出
        self.output_layer = tf.keras.layers.Dense(
            1, kernel_regularizer=reg, name="output"
        )

    def call(self, x, training=False):
        # 輸入正規化
        h = self.input_bn(x, training=training) if self.use_batch_norm else x

        # 瓶頸壓縮
        z = self.bottleneck(h)          # (batch, bottleneck_dim)

        # 稀疏門控
        g = self.gate(h)                # (batch, bottleneck_dim), sums to 1
        z_gated = z * g                 # 稀疏加權

        # 主路徑 + 殘差
        h_main = self.hidden(h)         # (batch, hidden_dim)
        h_main = self.dropout(h_main, training=training)

        # 合併
        h_combined = tf.concat([h_main, z_gated], axis=-1)
        return self.output_layer(h_combined)


# ── 訓練器 ────────────────────────────────────────────────────────────────────

class BTCTrainer:
    """
    BTC 深度學習訓練器。

    支援兩種模型：
      - "ffn":   原版 FEN FFN (向後相容)
      - "gated": Gated Interaction FFN (新版，預設)

    使用 tf.GradientTape 取代 TF1 Session，其餘邏輯與論文保持一致：
      - 全量批次訓練（每個 epoch 一次前向 + 反向）
      - 驗證集 Sharpe Ratio 最高的 epoch 存為最佳 checkpoint
      - 訓練過程記錄至 training_log.csv

    Parameters
    ----------
    config : dict  對應 config_btc.json 的超參數字典
    """

    def __init__(self, config: dict):
        self.config = config

        # keep_prob → dropout rate
        keep_prob    = config.get("dropout", 0.95)
        dropout_rate = 1.0 - keep_prob

        feature_dim = (
            config["individual_feature_dim"]
            + config.get("macro_feature_dim", 0)
        )

        # ── 特徵降維器（§12.5.2）──────────────────────────────────────────
        reduce_method = config.get("feature_reduce", "none")
        pca_variance  = config.get("pca_variance", 0.95)
        self.feature_reducer = FeatureReducer(
            method=reduce_method,
            n_components=pca_variance,
        )
        self._reducer_fitted = False

        # ── 自適應隱藏層維度（§12.1 維度詛咒對策）──────────────────────────
        hidden_dims = config["hidden_dim"]
        if config.get("adaptive_hidden", False):
            # 特徵數少時自動縮小隱藏層，避免過擬合
            scale = max(0.25, min(1.0, feature_dim / 32))
            hidden_dims = [max(8, int(h * scale)) for h in hidden_dims]
            deco_print(f"Adaptive hidden: {config['hidden_dim']} → {hidden_dims}")

        # ── Early stopping 參數（§12.4 過擬合防範）─────────────────────────
        self.early_stopping_patience = config.get("early_stopping_patience", 0)

        model_type = config.get("model_type", "gated")

        if model_type == "lightweight":
            # 輕量門控 FFN（§12.5.5）
            bottleneck = config.get("bottleneck_dim", 8)
            self.model = LightweightGatedFFN(
                input_dim      = feature_dim,
                bottleneck_dim = bottleneck,
                hidden_dim     = hidden_dims[0],
                dropout_rate   = dropout_rate,
                l2             = config.get("reg_l2", 0.005),
                use_batch_norm = config.get("use_batch_norm", True),
            )
            deco_print(
                f"Model: LightweightGatedFFN "
                f"(input={feature_dim}, bottleneck={bottleneck}, "
                f"hidden={hidden_dims[0]})"
            )
        elif model_type == "gated":
            # Gated Interaction FFN
            asset_dim  = config.get("asset_feature_dim", 16)   # features 0~15
            market_dim = feature_dim - asset_dim                # features 16~
            gate_dim   = config.get("gate_dim", asset_dim)

            self.model = GatedInteractionFFN(
                asset_dim    = asset_dim,
                market_dim   = market_dim,
                gate_dim     = gate_dim,
                hidden_dims  = hidden_dims,
                dropout_rate = dropout_rate,
                l1           = config.get("reg_l1", 0.0),
                l2           = config.get("reg_l2", 0.001),
            )
            deco_print(
                f"Model: GatedInteractionFFN "
                f"(asset={asset_dim}, market={market_dim}, "
                f"gate={gate_dim}, hidden={hidden_dims})"
            )
        else:
            # 原版 FFN (向後相容)
            self.model = FFNModel(
                feature_dim  = feature_dim,
                hidden_dims  = hidden_dims,
                dropout_rate = dropout_rate,
                l1           = config.get("reg_l1", 0.0),
                l2           = config.get("reg_l2", 0.001),
            )
            deco_print(
                f"Model: FFNModel "
                f"(feature_dim={feature_dim}, hidden={hidden_dims})"
            )

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.get("learning_rate", 0.001)
        )

        self._best_sharpe   = -float("inf")
        self._ckpt          = None
        self._ckpt_manager  = None

    # ── Checkpoint 管理 ───────────────────────────────────────────────────────

    def _setup_checkpoint(self, logdir: str):
        os.makedirs(logdir, exist_ok=True)
        ckpt_dir = os.path.join(logdir, "ckpt")
        self._ckpt = tf.train.Checkpoint(
            model=self.model, optimizer=self.optimizer
        )
        self._ckpt_manager = tf.train.CheckpointManager(
            self._ckpt, directory=ckpt_dir, max_to_keep=1
        )

    def save_best(self):
        if self._ckpt_manager is not None:
            self._ckpt_manager.save()

    def load_best(self, logdir: str):
        """載入指定目錄下的最佳 checkpoint（推論 / 集成時使用）。"""
        ckpt     = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        ckpt_dir = os.path.join(logdir, "ckpt")
        manager  = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=1)
        if manager.latest_checkpoint:
            ckpt.restore(manager.latest_checkpoint).expect_partial()
            deco_print(f"Restored: {manager.latest_checkpoint}")
        else:
            raise FileNotFoundError(f"Checkpoint 不存在：{ckpt_dir}")

    # ── 資料展平（Panel → 有效觀測）─────────────────────────────────────────

    def _flatten_valid(self, I_macro, I, R, mask):
        """
        將 Panel 資料攤平，只取有效（非缺失）觀測。

        Parameters
        ----------
        I_macro : (T, macro_dim)
        I       : (T, N, M)
        R       : (T, N)
        mask    : (T, N) bool

        Returns
        -------
        X_valid : (n_valid, feature_dim)  輸入特徵矩陣
        R_valid : (n_valid,)              對應報酬
        """
        T, N, M = I.shape
        macro_dim = I_macro.shape[-1]

        if macro_dim > 0:
            # 廣播 macro 至每個資產
            macro_tile = np.broadcast_to(
                I_macro[:, np.newaxis, :], (T, N, macro_dim)
            ).copy()
            I_cat = np.concatenate([I, macro_tile], axis=-1)  # (T, N, M+macro_dim)
        else:
            I_cat = I  # (T, N, M)

        X_valid = I_cat[mask]   # (n_valid, feature_dim)
        R_valid = R[mask]       # (n_valid,)

        # 特徵降維（§12.5.2）
        if self.feature_reducer.method != "none":
            if not self._reducer_fitted:
                X_valid = self.feature_reducer.fit_transform(X_valid, R_valid)
                self._reducer_fitted = True
            else:
                X_valid = self.feature_reducer.transform(X_valid)

        return X_valid.astype(np.float32), R_valid.astype(np.float32)

    # ── 訓練步驟（tf.function 加速）──────────────────────────────────────────

    @tf.function
    def _train_step(self, X: tf.Tensor, R: tf.Tensor):
        with tf.GradientTape() as tape:
            R_pred    = tf.squeeze(self.model(X, training=True), axis=-1)
            mse_loss  = tf.reduce_mean(tf.square(R - R_pred))
            reg_loss  = tf.add_n(self.model.losses) if self.model.losses else 0.0
            total     = mse_loss + reg_loss
        grads = tape.gradient(total, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables)
        )
        return total

    # ── 評估函數 ──────────────────────────────────────────────────────────────

    def evaluate_loss(self, dl) -> float:
        """計算 MSE 損失（不含正則化項）。"""
        for I_macro, I, R, mask in dl.iterateOneEpoch():
            X_v, R_v = self._flatten_valid(I_macro, I, R, mask)
            if len(R_v) == 0:
                return 0.0
            R_pred = tf.squeeze(
                self.model(tf.constant(X_v), training=False), axis=-1
            ).numpy()
            return float(np.mean((R_v - R_pred) ** 2))
        return 0.0

    def evaluate_sharpe(self, dl) -> float:
        """計算多空投資組合 Sharpe Ratio（對應原版 evaluate_sharpe）。"""
        for I_macro, I, R, mask in dl.iterateOneEpoch():
            X_v, R_v = self._flatten_valid(I_macro, I, R, mask)
            if len(R_v) == 0:
                return 0.0
            R_pred = tf.squeeze(
                self.model(tf.constant(X_v), training=False), axis=-1
            ).numpy()
            portfolio = construct_long_short_portfolio(R_pred, R_v, mask)
            return sharpe_ratio(portfolio)
        return 0.0

    def get_prediction(self, dl) -> np.ndarray:
        """取得全部有效觀測的預測值（用於 ensemble 與分析）。"""
        for I_macro, I, R, mask in dl.iterateOneEpoch():
            X_v, _ = self._flatten_valid(I_macro, I, R, mask)
            if len(X_v) == 0:
                return np.array([])
            return tf.squeeze(
                self.model(tf.constant(X_v), training=False), axis=-1
            ).numpy()
        return np.array([])

    # ── 主訓練迴圈 ────────────────────────────────────────────────────────────

    def train(
        self,
        dl,
        dl_valid,
        logdir: str,
        dl_test=None,
        print_on_console: bool = True,
        print_freq: int = 10,
        model_selection: str = "Factor_sharpe",
    ):
        """
        主訓練迴圈（對應原版 FeedForwardModelWithNA_Return.train）。

        Parameters
        ----------
        dl               : DataInRamInputLayer  訓練資料
        dl_valid         : DataInRamInputLayer  驗證資料
        logdir           : str                  checkpoint & log 儲存目錄
        dl_test          : DataInRamInputLayer | None
        print_on_console : bool
        print_freq       : int                  每幾個 epoch 列印一次
        model_selection  : 'Factor_sharpe' | 'natural'
        """
        self._setup_checkpoint(logdir)
        self._best_sharpe = -float("inf")
        num_epochs = self.config.get("num_epochs", 300)
        patience = self.early_stopping_patience
        no_improve_count = 0

        # 訓練 log（CSV）
        log_path = os.path.join(logdir, "training_log.csv")
        header   = "epoch,train_loss,valid_loss,train_sharpe,valid_sharpe"
        if dl_test is not None:
            header += ",test_sharpe"
        with open(log_path, "w") as f:
            f.write(header + "\n")

        time_start = time.time()

        for epoch in range(num_epochs):

            # ── 訓練一個 epoch ────────────────────────────────────────────────
            for I_macro, I, R, mask in dl.iterateOneEpoch():
                X_v, R_v = self._flatten_valid(I_macro, I, R, mask)
                if len(R_v) == 0:
                    continue
                self._train_step(
                    tf.constant(X_v, dtype=tf.float32),
                    tf.constant(R_v, dtype=tf.float32),
                )

            # ── 評估 ─────────────────────────────────────────────────────────
            train_loss   = self.evaluate_loss(dl)
            valid_loss   = self.evaluate_loss(dl_valid)
            train_sharpe = self.evaluate_sharpe(dl)
            valid_sharpe = self.evaluate_sharpe(dl_valid)
            test_sharpe  = self.evaluate_sharpe(dl_test) if dl_test else None

            # ── 列印 ─────────────────────────────────────────────────────────
            if print_on_console and epoch % print_freq == 0:
                test_str = (
                    f"/{test_sharpe:.4f}" if test_sharpe is not None else ""
                )
                elapsed = time.time() - time_start
                est     = elapsed / (epoch + 1) * num_epochs
                deco_print(
                    f"Epoch {epoch:4d} | "
                    f"Loss {train_loss:.4f}/{valid_loss:.4f} | "
                    f"Sharpe {train_sharpe:.4f}/{valid_sharpe:.4f}{test_str} | "
                    f"{elapsed:.0f}s/{est:.0f}s"
                )

            # ── Checkpoint 選取 ───────────────────────────────────────────────
            if model_selection == "natural":
                self.save_best()
            elif model_selection == "Factor_sharpe":
                if valid_sharpe > self._best_sharpe:
                    self._best_sharpe = valid_sharpe
                    self.save_best()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

            # ── Early stopping（§12.4 防止過擬合）─────────────────────────
            if patience > 0 and no_improve_count >= patience:
                deco_print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs)"
                )
                break

            # ── 寫入 log ──────────────────────────────────────────────────────
            with open(log_path, "a") as f:
                row = (
                    f"{epoch},{train_loss:.6f},{valid_loss:.6f},"
                    f"{train_sharpe:.6f},{valid_sharpe:.6f}"
                )
                if test_sharpe is not None:
                    row += f",{test_sharpe:.6f}"
                f.write(row + "\n")

        deco_print(
            f"Training complete. Best valid Sharpe: {self._best_sharpe:.4f}"
        )
