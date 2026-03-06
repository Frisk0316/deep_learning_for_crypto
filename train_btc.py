"""
train_btc.py
------------
BTC 多資產 Panel 深度學習訓練腳本（TensorFlow 2.x 版本）。

使用本目錄下的 model_btc.BTCTrainer 取代原始 TF1 版的
FeedForwardModelWithNA_Return，其餘架構（Panel 資料、Sharpe 選模、
Ensemble 策略）與論文保持完全一致。

訓練策略：時序分割（Chronological Split），避免 look-ahead bias：
  70% train / 15% valid / 15% test

使用方式：
  python train_btc.py \
      --config   config_btc.json \
      --logdir   ./checkpoints \
      --max_num_process 0

特徵子集說明（在 get_tuned_network() 中修改 subset）：
  range(0, 10)  : 僅價格動能 + 技術指標
  range(0, 15)  : 加入鏈上指標
  range(0, 22)  : 全部 22 個特徵
  list(range(0,5)) + list(range(15,22)) : 動能 + 總體情緒 + ETF
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

# 確保 btc_data_layer 與 model_btc 可被找到
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from btc_data_layer import DataInRamInputLayer
from model_btc import BTCTrainer, deco_print


# ── CLI 參數 ───────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="BTC 深度學習模型訓練（TF2）")
parser.add_argument("--config",          type=str, default="config_btc.json",
                    help="Config JSON 路徑")
parser.add_argument("--logdir",          type=str, default="./checkpoints",
                    help="Checkpoint & Log 儲存目錄")
parser.add_argument("--max_num_process", type=str, default="0",
                    help="平行 Process 數（0 = 單進程）")
parser.add_argument("--printOnConsole",  action="store_true", default=True)
parser.add_argument("--saveLog",         action="store_true", default=True)
parser.add_argument("--printFreq",       type=int,  default=10)
FLAGS = parser.parse_args()


# ── 時序 Fold 生成 ─────────────────────────────────────────────────────────────

def generate_chronological_folds(
    n_total: int,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    save_path: str = "sampling_folds/btc_chronological_folds.npy",
) -> list:
    """
    將 n_total 個時間點按比例切分為 train / valid / test。
    儲存至 .npy 以利重現。

    Returns
    -------
    folds : [(train_idx, valid_idx, test_idx)]  單一 fold 的列表
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    train_end = int(n_total * train_ratio)
    valid_end = train_end + int(n_total * valid_ratio)

    train_idx = list(range(0, train_end))
    valid_idx = list(range(train_end, valid_end))
    test_idx  = list(range(valid_end, n_total))

    folds = [[train_idx, valid_idx, test_idx]]
    np.save(save_path, np.array(folds, dtype=object))

    print(f"  [OK] Fold 儲存至 {save_path}")
    print(f"       Train: {len(train_idx)} 週 | "
          f"Valid: {len(valid_idx)} 週 | "
          f"Test:  {len(test_idx)} 週")
    return folds


def _get_or_create_folds(config: dict) -> list:
    """讀取已儲存的 fold，或依資料長度重新生成。"""
    fold_path = "sampling_folds/btc_chronological_folds.npy"

    if os.path.exists(fold_path):
        folds = np.load(fold_path, allow_pickle=True).tolist()
        print(f"  [OK] 讀取現有 fold：{fold_path}")
        return folds

    tmp     = np.load(config["individual_feature_file"], allow_pickle=True)
    n_total = tmp["data"].shape[0]
    return generate_chronological_folds(n_total, save_path=fold_path)


# ── 訓練主函數 ────────────────────────────────────────────────────────────────

def run_code(train_lists: list):
    """
    執行一組訓練任務（每個 entry 對應一組超參數 × 一個隨機種子）。
    使用 BTCTrainer（TF2）取代原版 TF1 Session 訓練流程。
    """
    for train_model in train_lists:
        [subset, num_layers, hidden_dim, dropout, _max_hidden,
         l1_penalty, l2_penalty, lr, model_selection, folder_idx] = train_model

        subset_list = list(subset)
        start_feat  = subset_list[0]
        end_feat    = subset_list[-1]

        # 讀取並覆寫 config
        with open(FLAGS.config, "r") as f:
            config = json.load(f)

        config["num_layers"]             = num_layers
        config["hidden_dim"]             = hidden_dim
        config["dropout"]                = dropout
        config["reg_l1"]                 = l1_penalty
        config["reg_l2"]                 = l2_penalty
        config["learning_rate"]          = lr
        config["individual_feature_dim"] = len(subset_list)

        deco_print("Config:")
        print(json.dumps(
            {k: v for k, v in config.items() if not k.startswith("_")},
            indent=4
        ))

        folds = _get_or_create_folds(config)

        for fold_i, (train_idx, valid_idx, test_idx) in enumerate(folds):
            logdir = os.path.join(
                FLAGS.logdir, "btc", f"folder_{folder_idx}",
                f"feat{start_feat}to{end_feat}"
                f"_L{num_layers}_H{hidden_dim[0]}"
                f"_drop{dropout}_l2{l2_penalty}_lr{lr}"
                f"_{model_selection}_split{fold_i}"
            )
            os.makedirs(logdir, exist_ok=True)

            deco_print(f"Building data layers (fold {fold_i})...")
            feat_file = config["individual_feature_file"]
            dl       = DataInRamInputLayer(feat_file, train_idx, subset_list)
            dl_valid = DataInRamInputLayer(feat_file, valid_idx, subset_list)
            dl_test  = DataInRamInputLayer(feat_file, test_idx,  subset_list)

            if fold_i == 0:
                dl.summary()

            deco_print(
                f"Training: folder={folder_idx}, fold={fold_i}, "
                f"features={subset_list}"
            )

            trainer = BTCTrainer(config)
            trainer.train(
                dl            = dl,
                dl_valid      = dl_valid,
                logdir        = logdir,
                dl_test       = dl_test,
                print_on_console = FLAGS.printOnConsole,
                print_freq       = FLAGS.printFreq,
                model_selection  = model_selection,
            )


# ── 超參數設定 ────────────────────────────────────────────────────────────────

def get_tuned_network() -> list:
    """
    定義要訓練的模型配置列表。每個 entry 格式：
      [subset, num_layers, hidden_dim, dropout, max_hidden,
       l1_penalty, l2_penalty, lr, model_selection]

    (A) 僅價格動能 + 技術指標（不需外部 API，最快）
    (B) 加入鏈上指標
    (C) 全部 22 個特徵
    (D) 動能 + 總體情緒 + ETF（簡約模型）
    """
    temp_results = [
        # (A) Price Momentum + Technical  [0~9]
        [range(0, 10),  1, [64], 0.95, 6, 0.0, 0.001, 0.001, "Factor_sharpe"],

        # (B) + On-chain  [0~14]
        [range(0, 15),  1, [64], 0.95, 6, 0.0, 0.001, 0.001, "Factor_sharpe"],

        # (C) All features  [0~21]
        [range(0, 22),  1, [64], 0.95, 6, 0.0, 0.001, 0.001, "Factor_sharpe"],

        # (D) Momentum + Macro + ETF（parsimonious）
        [list(range(0, 5)) + list(range(15, 22)),
         1, [64], 0.95, 6, 0.0, 0.001, 0.001, "Factor_sharpe"],
    ]

    result = []
    for temp in temp_results:
        for seed_idx in range(1, 9):  # 8 個隨機種子 → ensemble
            result.append(temp + [seed_idx])

    return result


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    lst_t_sample = get_tuned_network()
    print(f"共 {len(lst_t_sample)} 個訓練任務")

    max_proc = int(FLAGS.max_num_process)

    if max_proc > 0:
        mp.set_start_method("spawn", force=True)
        pool = mp.Pool(processes=max_proc)

        for i, task in enumerate(lst_t_sample):
            pool.apply_async(run_code, args=([task],))

        pool.close()
        pool.join()
    else:
        run_code(lst_t_sample)


if __name__ == "__main__":
    main()
