"""
visualize_raw_data.py
---------------------
此腳本根據 `prepare_btc_data.py` 所使用的原始資料來源，
建立可視化圖表，用以直觀地檢查原始特徵。

本腳本會：
1. 呼叫 `data_sources` 中的函式來重新獲取原始資料。
2. 針對 BTC 資產的各類原始特徵（價格、技術指標等）繪製時間序列圖。
3. 將圖表儲存於 `visualizations/` 目錄下。
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 將 data_sources 模組加入搜尋路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_sources.fetch_prices import fetch_ohlcv, compute_features

# --- 設定 ---
TARGET_ASSET = "BTC"  # 指定要可視化的資產
START_DATE = "2020-01-01"
END_DATE = None
OUTPUT_DIR = "visualizations/price_features"

# --- 主程式 ---
def main():
    """主執行函式"""
    print("=" * 60)
    print(f"開始為資產 {TARGET_ASSET} 建立原始特徵可視化圖表...")
    print("=" * 60)

    # 建立輸出目錄
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Step 1: 獲取並可視化原始價格/成交量 ---
    print("\n[1/3] 正在獲取 OHLCV 資料...")
    try:
        raw_ohlcv_data = fetch_ohlcv(start=START_DATE, end=END_DATE)
        if TARGET_ASSET not in raw_ohlcv_data:
            print(f"[錯誤] 找不到資產 {TARGET_ASSET} 的資料。")
            return
        
        asset_df = raw_ohlcv_data[TARGET_ASSET]
        print(f"  > 成功獲取 {TARGET_ASSET} 的 {len(asset_df)} 筆週資料。")

        # 設定繪圖風格
        sns.set_style("darkgrid")

        # 繪製收盤價
        plt.figure(figsize=(15, 7))
        plt.plot(asset_df.index, asset_df['Close'], label='Close Price')
        plt.title(f'{TARGET_ASSET} Weekly Close Price ({START_DATE} to Present)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{TARGET_ASSET}_raw_close_price.png'))
        plt.close()
        print(f"  > 圖表已儲存: {TARGET_ASSET}_raw_close_price.png")

        # 繪製成交量
        plt.figure(figsize=(15, 7))
        plt.bar(asset_df.index, asset_df['Volume'], color='skyblue', width=5)
        plt.title(f'{TARGET_ASSET} Weekly Volume ({START_DATE} to Present)')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{TARGET_ASSET}_raw_volume.png'))
        plt.close()
        print(f"  > 圖表已儲存: {TARGET_ASSET}_raw_volume.png")

    except Exception as e:
        print(f"[錯誤] 獲取或繪製 OHLCV 資料時發生問題: {e}")
        return

    # --- Step 2: 計算並可視化價格/技術指標特徵 ---
    print("\n[2/3] 正在計算並可視化價格/技術指標特徵...")
    try:
        features_df = compute_features(asset_df)
        print(f"  > 成功計算 {len(features_df.columns)} 個特徵。")

        for feature in features_df.columns:
            plt.figure(figsize=(15, 7))
            plt.plot(features_df.index, features_df[feature], label=feature)
            plt.title(f'{TARGET_ASSET} - Raw Feature: {feature}')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            # 儲存圖檔，使用符合檔案系統的名稱
            safe_feature_name = feature.replace('/', '_')
            plt.savefig(os.path.join(OUTPUT_DIR, f'{TARGET_ASSET}_feature_{safe_feature_name}.png'))
            plt.close()
            print(f"  > 圖表已儲存: {TARGET_ASSET}_feature_{safe_feature_name}.png")
            
    except Exception as e:
        print(f"[錯誤] 計算或繪製特徵時發生問題: {e}")
        
    # --- Step 3: (未來擴展) 可在此加入其他資料來源的可視化 ---
    print("\n[3/3] 價格與技術指標圖表建立完成。")
    print("\n若要可視化鏈上、情緒等其他數據，可擴展此腳本。")

    print("\n" + "=" * 60)
    print(f"所有圖表已儲存於 '{OUTPUT_DIR}' 目錄。")
    print("=" * 60)

if __name__ == "__main__":
    main()
