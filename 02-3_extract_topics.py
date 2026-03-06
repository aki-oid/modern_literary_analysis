import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import fugashi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ===== 設定 =====
INPUT_JSON = "data/01_literature.json"
OUTPUT_CSV = "data/02-3_features_topics.csv"
PLOT_DIR = "data/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

NUM_TOPICS = 10
plt.rcParams['font.family'] = 'MS Gothic'

# ===== 1. 形態素解析と名詞抽出 =====
print("形態素解析エンジン(MeCab)を初期化中...")
tagger = fugashi.Tagger()

STOP_WORDS = {
    "こと", "もの", "自分", "ところ", "ため", "二人", "一人", "まま", "うち", 
    "とき", "やつ", "なか", "あと", "わけ", "今日", "先生", "言葉", "人間",
    "あり", "なし", "つた", "つて", "なかつ", "それ", "これ", "あれ", "どれ",
    "もん", "ほう", "あたり", "まゝ", "今度", "相手", "主人", "夫人", "子供", "旦那",
    "そう", "こう", "ああ", "どう", "ため", "よう", "さん"
}

def extract_nouns(text):
    if not text: return ""
    nouns = []
    for word in tagger(text):
        if word.feature.pos1 == "名詞":
            surface = word.surface
            if word.feature.pos2 not in ["数詞", "代名詞", "非自立"] and surface not in STOP_WORDS:
                if len(surface) > 1:
                    nouns.append(surface)
    return " ".join(nouns)

# ===== 2. データの準備 =====
print("データを読み込み、名詞を抽出しています...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

df = pd.DataFrame(dataset)
tqdm.pandas()
df["nouns"] = df["text_normalized"].progress_apply(extract_nouns)

# ===== 3. LDAモデルの学習 =====
print("トピックモデル(LDA)を作成中...")
vectorizer = CountVectorizer(max_df=0.9, min_df=5, max_features=5000)
dtm = vectorizer.fit_transform(df["nouns"])

# トピックラベルの作成（特徴語の上位3つをラベルにする）
feature_names = vectorizer.get_feature_names_out()

print(f"LDAモデル(トピック数: {NUM_TOPICS})を学習中...")
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42, learning_method='batch')
doc_topic_dist = lda.fit_transform(dtm)

print("\n" + "="*30)
print("【各トピックの詳細分析：上位15語】")
print("="*30)
topic_labels = []
for topic_idx, topic in enumerate(lda.components_):
    # コンソール出力用（上位15語）
    top_indices = topic.argsort()[:-16:-1]
    top_features_15 = [feature_names[i] for i in top_indices]
    print(f"トピック {topic_idx:02}: {', '.join(top_features_15)}")
    
    # グラフ凡例用（上位5語）
    top_features_5 = top_features_15[:5]
    topic_labels.append(f"T{topic_idx}: " + "/".join(top_features_5))
print("="*60 + "\n")

# ===== 4. 結果の保存 =====
for i in range(NUM_TOPICS):
    df[f"Topic_{i}"] = doc_topic_dist[:, i]

save_cols = ["title", "author", "year"] + [f"Topic_{i}" for i in range(NUM_TOPICS)]
df[save_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# ===== 5. 可視化：年代別トピックシェアの変遷 =====
print("年代別トピック推移グラフを生成中...")

# 10年単位の年代（Decade）カラムを作成
df['decade'] = (df['year'] // 10) * 10
topic_cols = [f"Topic_{i}" for i in range(NUM_TOPICS)]

# 年代ごとに各トピックの平均値を集計
df_trend = df.groupby('decade')[topic_cols].mean()

# プロット
plt.figure(figsize=(15, 9))
colors = sns.color_palette("tab10", NUM_TOPICS)

# 積層グラフ作成
plt.stackplot(df_trend.index, df_trend.T, labels=topic_labels, colors=colors, alpha=0.8)

plt.title("日本近代文学におけるトピック変遷 (1868-1975)", fontsize=16, pad=20)
plt.xlabel("年代 (10年単位)", fontsize=12)
plt.ylabel("トピックの相対的なシェア", fontsize=12)
plt.xlim(df['year'].min(), df['year'].max())
plt.ylim(0, 1)

# 凡例をグラフの右側に詳細に表示
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=9, title="トピック(上位5語)")

plt.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}02-3_topic_evolution_stacked.png")
plt.show()

print(f"完了！詳細なトピック分析結果とグラフを保存しました。")