# 2-1.「物語の軌跡（Narrative Trajectory）」や「物語の弧（Narrative Arc）」を計算機で解明しようとする
# 2-1a. 物語の軌跡の抽出と可視化

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns
import random
from config import *

# ===== seed =====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ===== config =====
INPUT_JSON = D01_LITERATURE
OUTPUT_PKL = D021a_TRAJECTORY
ID_FILE = get_file_prefix(os.path.basename(__file__))
MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"

print(f"モデル {MODEL_NAME} をロード中...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
EMB_DIM = model.get_sentence_embedding_dimension()

# ===== sentence split =====
def sentence_split(text):
    sents = re.split(r"[。！？\n]", text)
    return [s.strip() for s in sents if len(s.strip()) > 5]

# ===== 1. sentence split (長文の強制分割対応) =====
def sentence_split(text, max_length=100):
    """
    句点や改行で分割し、それでも長すぎる文は max_length で強制分割する安全設計
    """
    # 1. まず句点や改行でざっくり分割
    raw_sents = re.split(r"[。！？\n]", text)
    
    processed_sents = []
    for s in raw_sents:
        s = s.strip()
        if len(s) == 0:
            continue
            
        # 2. 「。」がない異常に長い文（150文字超過）を強制的に叩き切る
        if len(s) > max_length:
            for i in range(0, len(s), max_length):
                chunk = s[i:i+max_length]
                if len(chunk) > 5: # 短すぎる破片（5文字以下）は除外
                    processed_sents.append(chunk)
        else:
            if len(s) > 5:
                processed_sents.append(s)
                
    return processed_sents

# ===== 軌跡ベクトル（Trajectory）の取得 =====
def get_trajectory_embeddings(text):
    sentences = sentence_split(text)
    
    # 短すぎるテキストは除外（20分割できないため）
    if len(sentences) < NUM_SEGMENTS:
        return None, f"有効な文が不足 ({len(sentences)}文 / 必要数: {NUM_SEGMENTS})"
        
    try:
        # 【爆速化】forループを廃止し、全文章を一括でエンコード
        all_embs = model.encode(
            sentences,
            batch_size=64, 
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False
        )
        
        # エンコード済みの配列を NUM_SEGMENTS 個の塊に均等分割
        segments_embs = np.array_split(all_embs, NUM_SEGMENTS)
        
        trajectory = []
        for seg_emb in segments_embs:
            # セグメント内の全ベクトルの平均（重心）を計算
            seg_vec = np.mean(seg_emb, axis=0)
            
            # 単位ベクトルに正規化（コサイン類似度計算のため）
            norm = np.linalg.norm(seg_vec)
            if norm > 1e-9:
                seg_vec = seg_vec / norm
            else:
                seg_vec = np.zeros(EMB_DIM)
                
            trajectory.append(seg_vec)
            
        return np.array(trajectory), "成功"
        
    except Exception as e:
        return None, f"エンコード中にエラー発生: {str(e)}"

# ===== dataset load =====
if not os.path.exists(INPUT_JSON):
    raise FileNotFoundError(INPUT_JSON)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"解析開始: {len(dataset)}件 (各作品を{NUM_SEGMENTS}分割します)")

features = []
skipped_records = [] # スキップされた作品の救済用ログ
for data in tqdm(dataset):
    text = data.get("text_normalized", "")
    trajectory_matrix, status_msg = get_trajectory_embeddings(text)
    
    if trajectory_matrix is not None:
        features.append({
            "title": data["title"],
            "author": data["author"],
            "year": data["year"],
            "trajectory": trajectory_matrix # 20個のベクトルのリスト
        })
    else:
        # 失敗した作品はログに記録
        skipped_records.append({
            "title": data.get("title", "不明"),
            "author": data.get("author", "不明"),
            "reason": status_msg
        })

# ===== 4. データの保存 =====
df_features = pd.DataFrame(features)
df_features.to_pickle(OUTPUT_PKL)
print(f"保存完了: {OUTPUT_PKL} (成功: {len(df_features)}件)")

# スキップされたデータの保存（安全対策）
if len(skipped_records) > 0:
    df_skipped = pd.DataFrame(skipped_records)
    skipped_csv_path = OUTPUT_PKL.replace(".pkl", "_skipped_log.csv")
    df_skipped.to_csv(skipped_csv_path, index=False, encoding="utf-8-sig")
    print(f"警告: {len(skipped_records)}件の作品がスキップされました。詳細は {skipped_csv_path} を確認してください。")
else:
    print("すべての作品が正常に処理されました！")

# ===== Plot: 物語の軌跡の可視化 =====
print("\n軌跡のプロットを生成します...")
plt.figure(figsize=(15, 10))
plt.rcParams["font.family"] = "MS Gothic"

# 全作品描画すると潰れるため、ランダムに最大9作品をサンプリングして比較
sample_size = min(9, len(df_features))
sample_df = df_features.sample(n=sample_size, random_state=42).reset_index(drop=True)

for i, row in sample_df.iterrows():
    plt.subplot(3, 3, i + 1)
    
    traj = row["trajectory"]
    base_vec = traj[0] # 基準点：第1セグメント（冒頭）
    
    # 各セグメントの冒頭とのコサイン類似度を計算
    similarities = [np.dot(base_vec, v) for v in traj]
    
    # X軸：物語の進行度 (1〜20)
    x = np.arange(1, NUM_SEGMENTS + 1)
    
    sns.lineplot(x=x, y=similarities, marker="o", color="royalblue")
    
    plt.title(f"『{row['title']}』\n({row['author']}, {row['year']}年)", fontsize=10)
    plt.ylim(0.0, 1.05)
    plt.xticks([1, 5, 10, 15, 20])
    
    if i % 3 == 0:
        plt.ylabel("冒頭からの意味的類似度")
    if i >= 6:
        plt.xlabel("物語の進行度 (セグメント)")

plt.tight_layout()
plt.suptitle(f"物語の軌跡 (Narrative Trajectories): 冒頭からの意味的変容", fontsize=16, y=1.02)
plt.savefig(os.path.join(PLOT_DIR, f"{ID_FILE}_Narrative_Trajectories.png"), dpi=300, bbox_inches='tight')
plt.show()

print("\n"+"="*50)
print("完了しました。")
print("="*50)