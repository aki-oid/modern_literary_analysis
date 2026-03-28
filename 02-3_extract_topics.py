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
INPUT_STOPWORDS = "data/INPUT/slothlib.txt"
INPUT_THOUGHT = D02_INPUT_THOUGHT
OUTPUT_CSV = D023_TOPIC
OUTPUT_SUMMARY_CSV = os.path.join(DATA_DIR, "02-3-2_topic_summary.csv")
ID_FILE = get_file_prefix(os.path.basename(__file__))

CONFIG = {
    "random_seed": 42,
    "max_iter": 300,  # EMアルゴリズムが十分収束する反復回数
}
plt.rcParams['font.family'] = 'MS Gothic' # 環境に合わせて調整

# ===== 1. 言語学的処理：精密な形態素解析（Linguistic Preprocessing） =====
tagger = fugashi.Tagger()

try:
    with open(INPUT_STOPWORDS, "r", encoding="utf-8") as f:
        slothlib_words = set(line.strip() for line in f if line.strip())
except FileNotFoundError:
    print(f"Warning: {INPUT_STOPWORDS} が見つかりません。")
    slothlib_words = set()
lliberary_whitelist = {
    "私", "自分", "僕", "俺", "彼", "彼女",  # 一人称・三人称
    "思う", "考える", "知る", "感じる",      # 内面描写
    "心", "魂", "死", "愛", "孤独",          # 抽象概念
    "しかし", "けれど", "やはり", "ふと"     # 接続詞・副詞（展開の鍵）
}

custom_words = {
    "トウキョウ", "エド", "コウベ", "ナカツ", "ミト","オオサカ", "サイタマ", "ヒロシマ", "フクシマ", "キョウト", "ナゴヤ", "サッポロ",
    "マモル", "ケン", "リョウ" ,"ハツ", "マン", "キヨシ", "シホ", "ミサ", "タロウ", "メグル", "ジン", "ウジ", "イヨ","モク", "ヒツ"
}
STOP_WORDS = slothlib_words | custom_words - lliberary_whitelist
print(f"ストップワードを {len(STOP_WORDS)} 件読み込みました。")

def extract_academic_lemmas(text):
    """
    形態素解析の学術的妥当性を確保：
    1. 表層形ではなく『語彙素（Lemma）』を使用し、表記揺れや活用を吸収
    2. 内容語（Content Words）である「名詞」に限定
    3. 数詞、代名詞、非自立語、接尾辞を厳密に除外
    """
    if not text or not isinstance(text, str): return ""
    
    tokens = []
    for word in tagger(text):
        pos = word.feature.pos1
        pos_detail = word.feature.pos2
        # UniDicの語彙素(lemma)を取得。なければ表層形
        lemma = word.feature.lemma if word.feature.lemma else word.surface
        # ハイフン区切りの語彙素（例：見上げる-見る）から先頭を取得
        lemma = lemma.split("-")[0] 
        
        # 特殊トークンの保護
        if "[PERSON]" in text and word.surface in ["[", "person", "]"]:
            continue
        # 抽出対象：意味を担う内容語に限定
        if pos in ["名詞"]:#, "動詞", "形容詞"
            # 除外対象：分析にノイズを与える機能語的要素
            if pos_detail not in ["数詞", "代名詞", "非自立", "接尾"]:
                # 長さ制約とストップワード処理
                if len(lemma) > 1 and lemma not in STOP_WORDS:
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
    min_df=0.01,    # 5文書未満にしか出現しない低頻度語（外れ値・誤字）を排除
    max_features=3500 # 特徴量空間の適正化
)
dtm = vectorizer.fit_transform(df["processed_text"])

print(f"Executing LDA with {NUM_TOPICS} topics...")
lda = LatentDirichletAllocation(
    n_components=NUM_TOPICS,
    learning_method='batch', # データ全体を用いた厳密な変分推論
    max_iter=CONFIG["max_iter"],
    random_state=CONFIG["random_seed"],
    n_jobs=-1,
    evaluate_every=20,          # 20イテレーションごとに評価を計算
    #verbose=1                   # 1イテレーションごとにログ出力 
)

doc_topic_dist = lda.fit_transform(dtm)

# 訓練データに対する評価指標
print(f"Vocabulary size: {dtm.shape[1]}")
print(f"Total word count: {dtm.sum()}")
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

for i in range(NUM_TOPICS):
    df[f"Topic_{i}"] = doc_topic_dist[:, i]

# トピック分布のDataFrame結合
topic_cols = [f"Topic_{i}" for i in range(NUM_TOPICS)]
df["Primary_Topic"] = df[topic_cols].idxmax(axis=1)
df["Primary_Prob"] = df[topic_cols].max(axis=1)

drop_cols = ["text_original","text_normalized","text_no_person","person_names","processed_text"]
df_output = df.drop(columns=drop_cols, errors='ignore')
df_output = df_output.replace({'\n': ' ', '\r': ' '}, regex=True)
df_output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# ===== 5. 通時的分析：トピック・ダイナミクス（Diachronic Analysis） =====
eras_order = list(ERA_LABELS.keys())
df['era'] = df['year'].apply(get_era)

df_filtered = df[df['era'].isin(ERA_LABELS.keys())].copy()
df_filtered['era'] = pd.Categorical(df_filtered['era'], categories=eras_order, ordered=True)

# 2. 時代ごとのトピック平均値を計算
topic_cols = [f"Topic_{i}" for i in range(NUM_TOPICS)]
era_topic_mean = df_filtered.groupby('era', observed=True)[topic_cols].mean()

# 3. 可視化（積層棒グラフ）
plt.figure(figsize=(12, 7))
era_topic_mean.plot(kind='bar', stacked=True, ax=plt.gca(), colormap='tab10', edgecolor='white')

plt.title('時代区分別トピック構成比の推移 (平均確率)', fontsize=15)
plt.xlabel('時代', fontsize=12)
plt.ylabel('平均トピック占有率', fontsize=12)
plt.legend(topic_labels, title="トピック内容", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, f"{ID_FILE}-1_era_topic_trend.png"), dpi=300, bbox_inches='tight')
plt.show()

# ===== 6. 代表的文書の特定（Exemplary Document Extraction） =====
summary_rows = []
print("\nGenerating final summary CSV...")
print("各トピックの代表的な作品と著者を抽出中...")
print("=著者：作品数(当該著者の全作品の割合)の順で表示=")
target_df = df_filtered 
for i in range(NUM_TOPICS):
    row_data = {
        "Topic_ID": f"T{i:02d}",
        "Label": topic_labels[i]
    }
    for era in eras_order:
        row_data[f"Share_{era}"] = era_topic_mean.loc[era, f"Topic_{i}"]
    
    # 該当単語（上位20語）
    top_20_indices = lda.components_[i].argsort()[:-21:-1]
    top_20_words = [feature_names[j] for j in top_20_indices]
    row_data["Top_20_Words"] = ", ".join(top_20_words)
    
    # 該当作品（上位20作品）
    top_works = df.nlargest(20, f"Topic_{i}")
    works_list = []
    for _, row in top_works.iterrows():
        works_list.append(f"{row['title']}({row['author']}, {int(row['year'])})")
    row_data["Top_20_Works"] = " / ".join(works_list)
    
    # 著者（上位10名）
    topic_id = f"Topic_{i}"
    print(f"\n{topic_labels[i]}")
    topic_docs = target_df[target_df["Primary_Topic"] == topic_id]
    author_counts = topic_docs["author"].value_counts().head(10)
    if not author_counts.empty:
        for author, count in author_counts.items():
            # その著者の全作品のうち、このトピックに属する割合（％）も出すと面白い
            total_author_works = len(target_df[target_df["author"] == author])
            ratio = (count / total_author_works) * 100
            print(f" - {author}: {count}作品({ratio:.1f}%)")
    else:
        print(" - (該当作品なし)")

    summary_rows.append(row_data)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
print(f"完了！サマリーを {OUTPUT_SUMMARY_CSV} に保存しました。")

# ===== 7. トピック別・派閥認定チェック (Topic-Faction Alignment) =====
print("\n" + "="*60)
print("【トピック別】派閥・名乗り判定チェック（閾値：40%以上）")
print("="*60)

print(f"JSONデータを読み込み中: {INPUT_THOUGHT}")
with open(INPUT_THOUGHT, "r", encoding="utf-8") as f:
    thought_data = json.load(f)

# 1. 思考データ（JSON）を作家単位の逆引き辞書に変換
author_to_faction = {}
faction_metadata = {}

for faction, info in thought_data.items():
    persons = [p.strip() for p in info["person"].split(",")]
    faction_label = f"{faction}({info['sub_label']})" if info['sub_label'] else faction
    faction_metadata[faction] = faction_label
    for p in persons:
        if p not in author_to_faction:
            author_to_faction[p] = []
        author_to_faction[p].append(faction)

# 2. LDAの結果（Primary_Topic）と派閥を紐付け
topic_faction_list = []
for _, row in df_filtered.iterrows():
    factions = author_to_faction.get(row['author'], ["無所属"])
    for f in factions:
        topic_faction_list.append({
            "topic_id": row["Primary_Topic"],
            "faction": f
        })

df_tf = pd.DataFrame(topic_faction_list)

# 3. トピックごとの集計
topic_totals = df_tf.groupby('topic_id').size().reset_index(name='topic_total')
tf_counts = df_tf.groupby(['topic_id', 'faction']).size().reset_index(name='count')
tf_alignment = pd.merge(tf_counts, topic_totals, on='topic_id')
tf_alignment['ratio'] = tf_alignment['count'] / tf_alignment['topic_total']

# 4. 判定と出力
for i in range(NUM_TOPICS):
    t_id = f"Topic_{i}"
    t_label = topic_labels[i]
    print(f">>> {t_label}")
    
    # 該当トピックのデータを抽出
    res = tf_alignment[tf_alignment['topic_id'] == t_id].sort_values('ratio', ascending=False)
    
    if res.empty:
        print("    (該当作品なし)")
        continue
        
    # 4割以上の派閥をチェック
    winners = res[(res['ratio'] >= 0.4) & (res['faction'] != "無所属")]
    
    if not winners.empty:
        for _, w in winners.iterrows():
            f_full_name = faction_metadata.get(w['faction'], w['faction'])
            print(f"    ★ 認定：このトピックは【{f_full_name}】の専売特許です。（派閥含有率: {w['ratio']*100:.1f}%）")
    else:
        top_f = res.iloc[0]
        # 無所属が最大の場、2番目の派閥も参考表示
        print(f"    × 認定不可（最大勢力: {top_f['faction']} {top_f['ratio']*100:.1f}%）")

    second_f = res.iloc[1]
    print(f"    （次点派閥: {second_f['faction']} {second_f['ratio']*100:.1f}%）")
    third_f = res.iloc[2] if len(res) > 2 else None
    print(f"    （3位派閥: {third_f['faction']} {third_f['ratio']*100:.1f}%）" if third_f is not None else "    （3位派閥: 該当なし）")

print("\n" + "="*60)
# ===== 8. トピック数 K の評価（Model Selection） =====
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
    lda_eval.fit(dtm_train)
    
    # 評価は未知のテストデータに対して実行（汎化性能の計測）
    log_likelihood_scores.append(lda_eval.score(dtm_test))
    perplexity_scores.append(lda_eval.perplexity(dtm_test))

# ===== 9. 評価指標の可視化（Elbow Method） =====
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