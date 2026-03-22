# 02-1~3の分析結果を読み込み、作品ごとに統合してマスターデータセットを作成するコード
import os
import pandas as pd
from config import *

# ===== config =====
OUTPUT_MASTER = D02_MASTER_DATA_CSV

def load_and_merge():
    print("各分析データを読み込み中...")
    
    # 1. 各データの読み込み
    df_shape = pd.read_csv(D021b_TRAJECTORY)
    df_morph = pd.read_csv(D022a_STYLE)
    df_topic = pd.read_csv(D023_TOPIC)
    
    # 2. 結合の前に不要な重複列や、形式を整える
    topic_cols = ["title", "author"] + [c for c in df_topic.columns if "Topic_" in c]
    df_topic_sub = df_topic[topic_cols]
    
    # 3. 結合 (Merge) 
    print("データを統合(JOIN)しています...")
    
    # まず「構造」と「スタイル」を結合
    master = pd.merge(
        df_shape, 
        df_morph, 
        on=["title", "author"], 
        how="inner", 
        suffixes=('', '_drop')
    )
    
    # 次に「トピック」を結合
    master = pd.merge(
        master, 
        df_topic_sub, 
        on=["title", "author"], 
        how="inner"
    )
    
    # 重複してしまった列（suffixes='_drop'がついたもの）を削除
    master = master.drop([c for c in master.columns if "_drop" in c], axis=1)
    
    # 4. 最終的な列の並び替え（見やすさのため）
    # メタデータ -> 構造 -> 内容 -> スタイル の順に整理
    metadata_cols = ["title", "author", "year", "decade", "length_category"]
    shape_cols = ["shape_name", "distance_to_center"]+ [c for c in master.columns if c.startswith("dist_to_")]
    topic_cols = [c for c in master.columns if "Topic_" in c]
    morph_cols = [c for c in master.columns if c not in metadata_cols + shape_cols + topic_cols]
    
    final_cols = metadata_cols + shape_cols + topic_cols + morph_cols
    master = master[final_cols]
    
    print(f"統合完了: 全 {len(master)} 作品, {len(master.columns)} 特徴量")
    
    # 5. 保存
    master.to_csv(OUTPUT_MASTER, index=False, encoding="utf-8-sig")
    print(f"マスターデータセットを保存しました: {OUTPUT_MASTER}")

if __name__ == "__main__":
    if all(os.path.exists(f) for f in [D021b_TRAJECTORY, D022a_STYLE, D023_TOPIC]):
        load_and_merge()
    else:
        print("エラー: 必要なCSVファイルが見つかりません。パスを確認してください。")