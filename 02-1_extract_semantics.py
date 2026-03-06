import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ===== 1. 正しいファイル名でロード =====
INPUT_PKL = "data/02-1_features_semantics.pkl"
PLOT_DIR = "data/plots/"
# ===== 設定 =====
INPUT_JSON = "data/01_literature.json"
OUTPUT_PKL = "data/02-1_features_semantics.pkl"

# モデル指定（日本語の汎用Sentence-BERT）
MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"
# 1チャンクあたりの文字数（BERTの512トークン上限に収まる安全な文字数として設定）
CHUNK_SIZE = 500 

# ===== 1. モデルのロード =====
print(f"モデル {MODEL_NAME} をロード中...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用デバイス: {device}")
model = SentenceTransformer(MODEL_NAME, device=device)

# ===== 2. データの読み込み =====
print("データを読み込んでいます...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# ===== 3. Chunking & Pooling 関数 =====
def get_document_embedding(text, model, chunk_size=CHUNK_SIZE):
    """長文を分割してベクトル化し、平均プーリングを行う"""
    if not text:
        # テキストが空の場合はゼロベクトルを返す（モデルの次元数に合わせる）
        dim = model.get_sentence_embedding_dimension()
        return np.zeros(dim)

    # 1. Chunking（テキストを chunk_size 文字ごとに分割）
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    
    # 2. 各チャンクをベクトル化
    # encode() はバッチ処理に対応しているため、リストをそのまま渡すと高速です
    chunk_embeddings = model.encode(chunks, show_progress_bar=False)
    
    # 3. Mean Pooling（チャンクごとのベクトルの平均を計算して1つのベクトルにする）
    # axis=0 で縦方向（チャンク間）の平均をとります
    doc_embedding = np.mean(chunk_embeddings, axis=0)
    
    return doc_embedding

# ===== 4. 特徴抽出の実行 =====
print(f"ベクトル化を開始します（チャンクサイズ: {CHUNK_SIZE}文字）...")

features = []
for data in tqdm(dataset):
    # 作品全体のベクトルを取得
    doc_vector = get_document_embedding(data["text_normalized"], model)
    
    features.append({
        "title": data["title"],
        "author": data["author"],
        "year": data["year"],
        "semantic_vector": doc_vector
    })

# ===== 5. 保存 =====
# ベクトル（NumPy配列）を含むため、Pickle形式で保存
df_features = pd.DataFrame(features)
df_features.to_pickle(OUTPUT_PKL)

print(f"\n完了: {len(df_features)}件の意味ベクトルを {OUTPUT_PKL} に保存しました。")

# ===== 6. 可視化：意味空間の漂流 =====
df = pd.read_pickle(OUTPUT_PKL)
print(df.head())

# ===== 1. 次元圧縮 (PCA) で2次元に落とす =====
print("高次元ベクトルを可視化用に圧縮中...")
# semantic_vector (768次元) を 2次元に変換
vectors = np.stack(df["semantic_vector"].values)
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)

df["x"] = coords[:, 0]
df["y"] = coords[:, 1]

# ===== 2. 可視化：意味空間の漂流 =====
plt.figure(figsize=(12, 8))
plt.rcParams['font.family'] = 'MS Gothic' # 環境に合わせて変更

# 年代ごとに色を変えてプロット
scatter = plt.scatter(df["x"], df["y"], c=df["year"], cmap="viridis", alpha=0.6, s=50)
plt.colorbar(scatter, label="年代 (Year)")

# 時代ごとの平均的な「重心」を線で結ぶ
df['decade'] = (df['year'] // 10) * 10
decade_centers = df.groupby('decade')[['x', 'y']].mean().sort_index()
plt.plot(decade_centers['x'], decade_centers['y'], color='red', marker='o', linestyle='-', linewidth=2, label="時代の重心移動")

plt.title("日本近代文学：意味ベクトルの空間分布と変遷", fontsize=15)
plt.xlabel("PCA Component 1 (概念の広がり)")
plt.ylabel("PCA Component 2 (概念の広がり)")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig(f"{PLOT_DIR}02-1_semantic_space_map.png")
plt.show()