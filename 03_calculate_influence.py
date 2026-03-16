import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# ===== 設定 =====
SEMANTICS_PKL = "data/02-1_features_semantics.pkl"
STYLE_CSV = "data/02-2_features_style.csv"
TOPICS_CSV = "data/02-3_features_topics.csv"
OUTPUT_CSV = "data/03_influence_edges.csv"

W_SEM = 0.25 
W_STY = 0.35 
W_TOP = 0.40 

HALFLIFE = 20.0
THRESHOLD = 0.15

# ===== 1. データのロードと統合 =====
print("データを読み込んでいます...")
with open(SEMANTICS_PKL, 'rb') as f:
    df_sem = pickle.load(f)
df_sty = pd.read_csv(STYLE_CSV)
df_top = pd.read_csv(TOPICS_CSV)

df = pd.merge(df_sem, df_sty, on=["title", "author", "year"])
df = pd.merge(df, df_top, on=["title", "author", "year"])
df = df.sort_values("year").reset_index(drop=True)
num_works = len(df)

# ===== 2. 特徴量行列の準備（高速化の鍵） =====
print("行列演算の準備中...")

# 1. 意味行列
sem_matrix = np.stack(df["semantic_vector"].values)
# L2正規化（内積でコサイン類似度を計算するため）
sem_matrix = sem_matrix / np.linalg.norm(sem_matrix, axis=1, keepdims=True)

# 2. 文体行列（新指標を追加）
style_cols = [
    "平均文長", "読点頻度", "語彙多様度_TTR", "旧字比率", 
    "和語比率", "漢語比率", "外来語比率", 
    "名詞割合", "動詞割合", "助詞割合"
]
style_vectors = StandardScaler().fit_transform(df[style_cols])
style_matrix = style_vectors / np.linalg.norm(style_vectors, axis=1, keepdims=True)

# 3. 主題行列
top_cols = [c for c in df.columns if c.startswith("topic_") or c.startswith("Topic_")]
top_matrix = df[top_cols].values
top_matrix = top_matrix / np.linalg.norm(top_matrix, axis=1, keepdims=True)

# 年代行列（時間差計算用）
years = df["year"].values

# ===== 3. 高速計算（行列演算） =====
print(f"影響ネットワークを計算中 (1,585 x 1,584)... ")

# 全ペアの類似度行列を計算 (dot product)
sim_sem = np.dot(sem_matrix, sem_matrix.T)
sim_sty = np.dot(style_matrix, style_matrix.T)
sim_top = np.dot(top_matrix, top_matrix.T)

# 総合類似度
sim_total = (W_SEM * sim_sem) + (W_STY * sim_sty) + (W_TOP * sim_top)

# ===== 4. 時間減衰の適用とエッジ抽出 =====
edges = []
for i in tqdm(range(num_works)):
    # 過去(i)から未来(j)への影響のみ抽出
    # j > i のインデックスを取得
    target_indices = np.arange(i + 1, num_works)
    if len(target_indices) == 0: continue
    
    dt = years[target_indices] - years[i]
    # 時間減衰: 半減期モデル
    decay = np.exp(-dt * np.log(2) / HALFLIFE)
    
    # 影響スコア算出
    scores = sim_total[i, target_indices] * decay
    
    # 閾値超えのインデックスを抽出
    valid_mask = scores > THRESHOLD
    for idx_in_target, score in zip(np.where(valid_mask)[0], scores[valid_mask]):
        j = target_indices[idx_in_target]
        edges.append({
            "source_title": df.iloc[i]["title"],
            "source_author": df.iloc[i]["author"],
            "source_year": df.iloc[i]["year"],
            "target_title": df.iloc[j]["title"],
            "target_author": df.iloc[j]["author"],
            "target_year": df.iloc[j]["year"],
            "weight": score,
            "dt": dt[idx_in_target],
            # デバッグ用に内訳も少し残すと可視化が捗る
            "sim_sem": sim_sem[i, j],
            "sim_sty": sim_sty[i, j],
            "sim_top": sim_top[i, j]
        })

# ===== 5. 保存 =====
df_edges = pd.DataFrame(edges)
df_edges.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n完了: {len(df_edges)} 本の影響エッジを保存しました。")