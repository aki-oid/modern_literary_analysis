import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# ===== 設定 =====
INPUT_JSON = "data/literature_1890_1945.json"
OUTPUT_PKL = "data/features_semantics.pkl"

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
    doc_vector = get_document_embedding(data["text"], model)
    
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

df = pd.read_pickle("data/features_semantics.pkl")
print(df.head())