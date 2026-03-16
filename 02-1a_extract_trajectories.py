# 2-1.「物語の軌跡（Narrative Trajectory）」や「物語の弧（Narrative Arc）」を計算機で解明しようとする
# 2-1a. 物語の軌跡の抽出と可視化
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ===== seed =====
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ===== config =====
INPUT_JSON = "data/01_literature.json"
OUTPUT_PKL = "data/02-1a_narrative_trajectories.pkl"
PLOT_DIR = "data/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
NUM_SEGMENTS = 20  # 作品を20等分する

print(f"モデル {MODEL_NAME} をロード中...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)
EMB_DIM = model.get_sentence_embedding_dimension()

# ===== sentence split =====
def sentence_split(text):
    sents = re.split(r"[。！？\n]", text)
    return [s.strip() for s in sents if len(s.strip()) > 5]

# ===== 軌跡ベクトル（Trajectory）の取得 =====
def get_trajectory_embeddings(text):
    sentences = sentence_split(text)
    
    # 短すぎるテキストは除外（20分割できないため）
    if len(sentences) < NUM_SEGMENTS:
        return None
        
    # 文のリストを20個のセグメントに均等分割
    segments = np.array_split(sentences, NUM_SEGMENTS)
    
    trajectory = []
    
    with torch.no_grad():
        for seg in segments:
            # セグメント内の各文をベクトル化
            embs = model.encode(
                seg.tolist(),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=False
            )
            
            # セグメントの意味的重心（平均ベクトル）を計算
            seg_vec = np.mean(embs, axis=0)
            
            # 単位ベクトルに正規化（コサイン類似度計算のため）
            norm = np.linalg.norm(seg_vec)
            if norm > 1e-9:
                seg_vec = seg_vec / norm
            else:
                seg_vec = np.zeros(EMB_DIM)
                
            trajectory.append(seg_vec)
            
    # shape: (20, 768) の配列を返す
    return np.array(trajectory)

# ===== dataset load =====
if not os.path.exists(INPUT_JSON):
    raise FileNotFoundError(INPUT_JSON)

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"解析開始: {len(dataset)}件 (各作品を{NUM_SEGMENTS}分割します)")

features = []

for data in tqdm(dataset):
    text = data.get("text_normalized", "")
    trajectory_matrix = get_trajectory_embeddings(text)
    
    if trajectory_matrix is not None:
        features.append({
            "title": data["title"],
            "author": data["author"],
            "year": data["year"],
            "trajectory": trajectory_matrix # 20個のベクトルのリスト
        })

df_features = pd.DataFrame(features)
df_features.to_pickle(OUTPUT_PKL)
print(f"保存完了: {OUTPUT_PKL}")

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
plt.savefig(f"{PLOT_DIR}02-1a_Narrative_Trajectories.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n"+"="*50)
print("完了しました。")
print("="*50)