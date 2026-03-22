# 2-3. トピック抽出と通時的分析：LDAによるテーマの動向分析

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import fugashi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from config import *

# ===== config =====
INPUT_JSON = D01_LITERATURE
OUTPUT_CSV = D023_TOPIC
ID_FILE = get_file_prefix(os.path.basename(__file__))

CONFIG = {
    "random_seed": 42,
    "max_iter": 50,  # EMアルゴリズムが十分収束する反復回数
    "doc_topic_prior": 0.1,  # Alpha: 文書ごとのトピック分布のスパース性（1未満で少数のトピックに集中）
    "topic_word_prior": 0.01 # Beta/Eta: トピックごとの単語分布のスパース性（より限定的な語彙を重視）
}
plt.rcParams['font.family'] = 'MS Gothic' # 環境に合わせて調整

# ===== 1. 言語学的処理：精密な形態素解析（Linguistic Preprocessing） =====
tagger = fugashi.Tagger()

# Zipfの法則に従い高頻度かつ無意味な語、およびドメイン特有の定型語を除外
ACADEMIC_STOP_WORDS = {
    "こと", "もの", "自分", "ため", "とき", "よう", "ほう", "わけ", "なか", "ところ",
    "それ", "これ", "あれ", "どれ", "ここ", "そこ", "あそこ", "どこ",
    "さん", "くん", "ちゃん", "ある", "いる", "なる", "する", "みる", "いく", "くる"
}

def extract_academic_lemmas(text):
    """
    形態素解析の学術的妥当性を確保：
    1. 表層形ではなく『語彙素（Lemma）』を使用し、表記揺れや活用を吸収
    2. 内容語（Content Words）である「名詞」「動詞」「形容詞」に限定
    3. 数詞、代名詞、非自立語、接尾辞を厳密に除外
    """
    if not text or not isinstance(text, str): return ""
    
    tokens = []
    for word in tagger(text):
        pos = word.feature.pos1
        pos_detail = word.feature.pos2
        
        # 抽出対象：意味を担う内容語に限定
        if pos in ["名詞", "動詞", "形容詞"]:
            # 除外対象：分析にノイズを与える機能語的要素
            if pos_detail not in ["数詞", "代名詞", "非自立", "接尾"]:
                # UniDicの語彙素(lemma)を取得。なければ表層形
                lemma = word.feature.lemma if word.feature.lemma else word.surface
                # ハイフン区切りの語彙素（例：見上げる-見る）から先頭を取得
                lemma = lemma.split("-")[0] 
                
                # 長さ制約とストップワード処理
                if len(lemma) > 1 and lemma not in ACADEMIC_STOP_WORDS:
                    tokens.append(lemma)
                    
    return " ".join(tokens)

# ===== 2. データの構造化（Data Preparation） =====
print("Loading dataset and performing linguistic analysis...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    df = pd.DataFrame(json.load(f))

tqdm.pandas(desc="Linguistic Processing")
df["processed_text"] = df["text_no_person"].progress_apply(extract_academic_lemmas)

# ===== 3. 統計的モデル構築：LDA（Probabilistic Topic Modeling） =====
# 文書頻度（DF）に基づく次元削減（Zipfの法則に基づくカットオフ）
vectorizer = CountVectorizer(
    max_df=0.7,  # コーパスの70%以上に出現する一般的な語（ストップワード漏れ等）を排除
    min_df=5,    # 5文書未満にしか出現しない低頻度語（外れ値・誤字）を排除
    max_features=5000 # 特徴量空間の適正化
)
dtm = vectorizer.fit_transform(df["processed_text"])

print(f"Executing LDA with {NUM_TOPICS} topics...")
lda = LatentDirichletAllocation(
    n_components=NUM_TOPICS,
    doc_topic_prior=CONFIG["doc_topic_prior"],
    topic_word_prior=CONFIG["topic_word_prior"],
    learning_method='batch', # データ全体を用いた厳密な変分推論
    max_iter=CONFIG["max_iter"],
    random_state=CONFIG["random_seed"],
    n_jobs=-1
)

doc_topic_dist = lda.fit_transform(dtm)

# 訓練データに対する評価指標
print(f"Training Log Likelihood: {lda.score(dtm):.2f}")
print(f"Training Perplexity: {lda.perplexity(dtm):.2f}")

# ===== 4. 分析結果の解釈と保存（Interpretation） =====
feature_names = vectorizer.get_feature_names_out()
topic_labels = []

print("\n" + "="*40)
print("TOPIC ANALYSIS: TOP TERMS")
print("="*40)
for i, topic in enumerate(lda.components_):
    # トピック内の単語重要度（確率分布）が高い順に抽出
    top_indices = topic.argsort()[:-11:-1]
    top_terms = [feature_names[j] for j in top_indices]
    label = f"T{i:02}: " + "/".join(top_terms[:3])
    topic_labels.append(label)
    print(f"[{i:02}] {', '.join(top_terms)}")

# トピック分布のDataFrame結合
for i in range(NUM_TOPICS):
    df[f"Topic_{i}"] = doc_topic_dist[:, i]
# --- データフレーム全体の改行コードをスペースに置換 ---
df = df.replace({'\n': ' ', '\r': ' '}, regex=True)
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# ===== 5. 通時的分析：トピック・ダイナミクス（Diachronic Analysis） =====
# 1. 1年単位で平均値を集計
topic_cols = [f"Topic_{i}" for i in range(NUM_TOPICS)]
df_yearly = df.groupby('year')[topic_cols].mean()

# 2. 移動平均の計算（「見やすさ」のための平滑化）
# window=5 は「前後5年間の平均」をとる設定です。
window_size = 3
df_trend_smooth = df_yearly.rolling(window=window_size, center=True, min_periods=1).mean()

# 3. 可視化
plt.figure(figsize=(15, 7), dpi=150)
colors = sns.color_palette(n_colors=NUM_TOPICS)

for i, col in enumerate(topic_cols):
    plt.plot(
        df_trend_smooth.index, 
        df_trend_smooth[col], 
        marker='',
        linewidth=2.5,
        alpha=0.9,
        label=topic_labels[i], 
        color=colors[i]
    )

plt.title(f"Yearly Trend of Topic Popularity ({window_size}-year Moving Average)", fontsize=16, pad=20)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Mean Topic Probability", fontsize=12)

# 凡例を整理して右側に配置
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Topics", frameon=False, fontsize=10)

# X軸の目盛りを5年刻みにして見やすくする
start_year = int(df['year'].min())
end_year = int(df['year'].max())
plt.xticks(np.arange(start_year, end_year + 1, 5), rotation=45)

plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{ID_FILE}-1_yearly_topic_trend.png"), dpi=300, bbox_inches='tight')
plt.show()

# ===== 6. 代表的文書の特定（Exemplary Document Extraction） =====
print("\n" + "="*40)
print("REPRESENTATIVE WORKS PER TOPIC")
print("="*40)

for i in range(NUM_TOPICS):
    print(f"\n{topic_labels[i]}")
    # 各トピックへの適合度が最も高い文書を抽出（質的分析への接続）
    top_works = df.nlargest(10, f"Topic_{i}")
    for _, row in top_works.iterrows():
        print(f" - {row[f'Topic_{i}']:0.3f}: {row['title']} ({row['author']}, {int(row['year'])})")

# ===== 7. トピック数 K の評価（Model Selection） =====
# 候補となるトピック数の範囲
k_candidates = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
perplexity_scores = []
log_likelihood_scores = []

print("\nSearching for the optimal number of topics (K)...")

# 学術的妥当性：過学習を防ぐため、モデル評価は必ずホールドアウト（テスト）データで行う
dtm_train, dtm_test = train_test_split(dtm, test_size=0.2, random_state=CONFIG["random_seed"])

for k in tqdm(k_candidates, desc="Evaluating K"):
    lda_eval = LatentDirichletAllocation(
        n_components=k,
        learning_method='batch',
        max_iter=30, # 探索用のため反復回数を絞る
        random_state=CONFIG["random_seed"],
        n_jobs=-1
    )
    # 学習は訓練データでのみ実行
    lda_eval.fit(dtm_train)
    
    # 評価は未知のテストデータに対して実行（汎化性能の計測）
    log_likelihood_scores.append(lda_eval.score(dtm_test))
    perplexity_scores.append(lda_eval.perplexity(dtm_test))

# ===== 8. 評価指標の可視化（Elbow Method） =====
fig, ax1 = plt.subplots(figsize=(10, 6))

# Log-Likelihoodのプロット（高いほど良い）
color = 'tab:blue'
ax1.set_xlabel('Number of Topics (K)')
ax1.set_ylabel('Log-Likelihood (Held-out)', color=color)
ax1.plot(k_candidates, log_likelihood_scores, marker='o', color=color, label='Log-Likelihood')
ax1.tick_params(axis='y', labelcolor=color)

# Perplexityのプロット（低いほど良い）
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Perplexity (Held-out)', color=color)
ax2.plot(k_candidates, perplexity_scores, marker='s', color=color, label='Perplexity')
ax2.tick_params(axis='y', labelcolor=color)

plt.title("Evaluation of LDA Topic Models by Topic Count (K) on Held-out Data", fontsize=14)
fig.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(PLOT_DIR, f"{ID_FILE}-2_model_selection_metrics.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"全解析完了。結果は {OUTPUT_CSV} および {PLOT_DIR} に保存されました。")