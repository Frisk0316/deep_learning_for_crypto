#!/usr/bin/env bash
# =============================================================================
# setup.sh — 新機器一鍵環境建立腳本
# 適用：Ubuntu / macOS，Python 3.7–3.10
# =============================================================================

set -e  # 任何指令失敗即停止

echo "========================================"
echo "  BTC Deep Learning 環境設定"
echo "========================================"

# ── 1. 確認 Python 版本 ──────────────────────────────────────────────────────
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "[ERR] 找不到 Python，請先安裝 Python 3.7+"
    exit 1
fi
echo "[OK] Python: $($PYTHON --version)"

# ── 2. 建立虛擬環境 ──────────────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "[INFO] 建立虛擬環境 venv/ ..."
    $PYTHON -m venv venv
else
    echo "[OK] 虛擬環境 venv/ 已存在"
fi

# 啟動虛擬環境
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# ── 3. 安裝套件 ──────────────────────────────────────────────────────────────
echo "[INFO] 更新 pip ..."
pip install --upgrade pip -q

echo "[INFO] 安裝 requirements.txt ..."
pip install -r requirements.txt

# ── 4. 嘗試安裝 TensorFlow 1.x（若失敗則提示改用 TF 2.x 相容模式）────────────
echo "[INFO] 嘗試安裝 TensorFlow 1.15 ..."
if pip install "tensorflow==1.15.0" -q 2>/dev/null; then
    echo "[OK] TensorFlow 1.15 安裝成功"
else
    echo "[WARN] TensorFlow 1.15 安裝失敗（常見於 Python 3.9+ 或 Apple Silicon）"
    echo "       改安裝 TensorFlow 2.x + 相容層 ..."
    pip install "tensorflow>=2.10.0,<2.14.0" -q
    echo "[OK] TensorFlow 2.x 安裝成功"
    echo ""
    echo "[NOTE] 使用 TF 2.x 時，請將原始程式碼最上方的："
    echo "         import tensorflow as tf"
    echo "       改為："
    echo "         import tensorflow.compat.v1 as tf"
    echo "         tf.disable_v2_behavior()"
fi

# ── 5. 建立必要目錄 ──────────────────────────────────────────────────────────
echo "[INFO] 建立目錄結構 ..."
mkdir -p datasets
mkdir -p sampling_folds
mkdir -p checkpoints
mkdir -p result_saved

# ── 6. 驗證安裝 ──────────────────────────────────────────────────────────────
echo ""
echo "[INFO] 驗證安裝 ..."
$PYTHON -c "
import numpy, pandas, scipy, requests, yfinance, matplotlib
print(f'  numpy:      {numpy.__version__}')
print(f'  pandas:     {pandas.__version__}')
print(f'  scipy:      {scipy.__version__}')
print(f'  yfinance:   {yfinance.__version__}')
print(f'  matplotlib: {matplotlib.__version__}')
try:
    import tensorflow as tf
    print(f'  tensorflow: {tf.__version__}')
except Exception as e:
    print(f'  tensorflow: 安裝失敗 - {e}')
"

echo ""
echo "========================================"
echo "  設定完成！"
echo "  下一步："
echo "  1. source venv/bin/activate"
echo "  2. python prepare_btc_data.py --skip_onchain --skip_polymarket"
echo "  3. python train_btc.py --config config_btc.json --logdir ./checkpoints"
echo "========================================"
