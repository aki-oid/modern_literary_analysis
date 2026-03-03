import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ===== 設定 =====
SEMANTICS_PKL = "data/02-1_features_semantics.pkl"
STYLE_CSV = "data/02-2_features_style.csv"
TOPICS_CSV = "data/02-3_features_topics.csv"
OUTPUT_CSV = "data/03_influence_edges.csv"

# 重み設定 (合計が1になるように調整)
W_SEM = 0.4  # 意味の類似度
W_STY = 0.3  # 文体の類似度
W_TOP = 0.3  # 主題の類似度

# 時間減衰のパラメータ（影響の半減期：20年と仮定）
HALFLIFE = 20.0

# スコアのしきい値（これ以下の微弱な影響は保存しない）
THRESHOLD = 0.1

# ===== 1. データのロードと統合 =====
print("各特徴量を読み込んでいます...")
df_sem = pd.read_pickle(SEMANTICS_PKL)
df_sty = pd.read_csv(STYLE_CSV)
df_top = pd.read_csv(TOPICS_CSV)

# メタデータ(title, author, year)をキーに結合
df = pd.merge(df_sem, df_sty, on=["title", "author", "year"])
df = pd.merge(df, df_top, on=["title", "author", "year"])

# 年代順にソート（過去から未来への影響のみ計算するため）
df = df.sort_values("year").reset_index(drop=True)
num_works = len(df)

# ===== 2. 特徴量の準備（標準化など） =====
print("特徴量を準備しています...")

# 意味ベクトル (NumPy行列へ)
sem_vectors = np.stack(df["semantic_vector"].values)

# 文体特徴量 (数値列のみ抽出して標準化)
style_cols = ["平均文長", "読点頻度", "語彙多様度_TTR", "名詞割合", "動詞割合", "形容詞割合", "助詞割合", "助動詞割合"]
scaler = StandardScaler()
style_vectors = scaler.fit_transform(df[style_cols])

# 主題特徴量 (topic_prob_ で始まる列を抽出)
top_cols = [c for c in df.columns if c.startswith("Topic_") or c.startswith("topic_prob_")]
top_vectors = df[top_cols].values

# ===== 3. 影響スコアの計算 =====
print(f"影響ネットワークの計算を開始します（全ペア数: {num_works * (num_works-1) // 2}）...")

edges = []

# すべての作品ペアに対してループ (i: 過去, j: 未来)
for i in tqdm(range(num_works)):
    for j in range(i + 1, num_works):
        
        # 1. 時間差の計算
        dt = float(df.iloc[j]["year"] - df.iloc[i]["year"])
        
        # 2. 特徴量ごとの類似度 (コサイン類似度)
        # 意味類似度
        s_sem = cosine_similarity(sem_vectors[i:i+1], sem_vectors[j:j+1])[0][0]
        # 文体類似度
        s_sty = cosine_similarity(style_vectors[i:i+1], style_vectors[j:j+1])[0][0]
        # 主題類似度
        s_top = cosine_similarity(top_vectors[i:i+1], top_vectors[j:j+1])[0][0]
        
        # 3. 加重平均類似度
        total_sim = (W_SEM * s_sem) + (W_STY * s_sty) + (W_TOP * s_top)
        
        # 4. 時間減衰関数 f(dt) = exp(-dt * ln(2) / HalfLife)
        # 時間差がHalfLife(20年)のとき、影響力が半分(0.5)になる
        decay = np.exp(-dt * np.log(2) / HALFLIFE)
        
        # 最終的な影響スコア
        influence_score = total_sim * decay
        
        # しきい値以上のエッジのみ保存
        if influence_score > THRESHOLD:
            edges.append({
                "source_title": df.iloc[i]["title"],
                "source_author": df.iloc[i]["author"],
                "source_year": df.iloc[i]["year"],
                "target_title": df.iloc[j]["title"],
                "target_author": df.iloc[j]["author"],
                "target_year": df.iloc[j]["year"],
                "weight": influence_score,
                "dt": dt
            })

# ===== 4. 保存 =====
df_edges = pd.DataFrame(edges)
df_edges.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n完了: {len(df_edges)} 本の影響エッジを {OUTPUT_CSV} に保存しました。")