"""
model_btc.py
------------
BTC 版前饋神經網路（FFN），完全基於 TensorFlow 2.x / Keras API。

取代原始論文的 FeedForwardModelWithNA_Return（TF 1.x Session 模式），
保留完全相同的輸入格式與訓練邏輯：
  - Panel 資料 (T × N × M)，以布林遮罩過濾缺失值
  - MSE 損失 + L1/L2 正則化 + Dropout
  - 驗證集 Sharpe Ratio 為 checkpoint 選取準則
  - 支援 'Factor_sharpe' / 'natural' 兩種模式

外部依賴：tensorflow >= 2.2, numpy, pandas（均不依賴原始 deep_learning/ 目錄）
"""

import os
import time
import numpy as np
import tensorflow as tf


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


# ── 主模型（Keras subclass API）───────────────────────────────────────────────

class FFNModel(tf.keras.Model):
    """
    前饋神經網路，架構與論文完全相同：
        輸入 → [Dense(ReLU) → Dropout] × num_layers → Dense(線性) → 預測報酬

    Parameters
    ----------
    feature_dim  : int    輸入特徵數
    hidden_dims  : list   每層節點數，例如 [64]
    dropout_rate : float  Dropout rate（= 1 - keep_prob；原版 keep_prob=0.95 → rate=0.05）
    l1, l2       : float  正則化係數
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


# ── 訓練器 ────────────────────────────────────────────────────────────────────

class BTCTrainer:
    """
    BTC 深度學習訓練器（對應原版 FeedForwardModelWithNA_Return）。

    使用 tf.GradientTape 取代 TF1 Session，其餘邏輯與論文保持一致：
      - 全量批次訓練（每個 epoch 一次前向 + 反向，對應原版 sub_epoch=False）
      - 驗證集 Sharpe Ratio 最高的 epoch 存為最佳 checkpoint
      - 訓練過程記錄至 training_log.csv

    Parameters
    ----------
    config : dict  對應 config_btc.json 的超參數字典
    """

    def __init__(self, config: dict):
        self.config = config

        # keep_prob → dropout rate（原版 config 存的是 keep_prob）
        keep_prob    = config.get("dropout", 0.95)
        dropout_rate = 1.0 - keep_prob

        feature_dim = (
            config["individual_feature_dim"]
            + config.get("macro_feature_dim", 0)
        )

        self.model = FFNModel(
            feature_dim  = feature_dim,
            hidden_dims  = config["hidden_dim"],
            dropout_rate = dropout_rate,
            l1           = config.get("reg_l1", 0.0),
            l2           = config.get("reg_l2", 0.001),
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
