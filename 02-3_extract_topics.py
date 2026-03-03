import json
import pandas as pd
from tqdm import tqdm
import fugashi
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# ===== 設定 =====
INPUT_JSON = "data/01_literature_1890_1945.json"
OUTPUT_CSV = "data/02-3_features_topics.csv"

# トピック数の設定
NUM_TOPICS = 7

# ===== 1. 形態素解析エンジン(MeCab)の初期化 =====
print("形態素解析エンジン(MeCab)を初期化中...")
tagger = fugashi.Tagger()

STOP_WORDS = {
    "こと", "もの", "自分", "ところ", "ため", "二人", "一人", "まま", "うち", 
    "とき", "やつ", "なか", "あと", "わけ", "今日", "先生", "言葉", "人間",
    "あり", "なし", "つた", "つて", "なかつ", "それ", "これ", "あれ"
}

def extract_nouns(text):
    """テキストから名詞のみを抽出し、ストップワードを除外して返す"""
    if not text:
        return ""
    nouns = []
    for word in tagger(text):
        # 名詞のみを抽出
        if word.feature.pos1 == "名詞":
            surface = word.surface
            # 数詞、代名詞、非自立名詞（こと、もの等）や、ストップワードリストに含まれるものを除外
            if word.feature.pos2 not in ["数詞", "代名詞", "非自立"] and surface not in STOP_WORDS:
                # 1文字の単語（ひらがな1文字のゴミなど）も除外することが多いです
                if len(surface) > 1:
                    nouns.append(surface)
    return " ".join(nouns)

# ===== 2. データの読み込みと名詞の抽出 =====
print("データを読み込んでいます...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

df = pd.DataFrame(dataset)

print("各作品から名詞を抽出しています...")
tqdm.pandas()
df["nouns"] = df["text"].progress_apply(extract_nouns)

# ===== 3. CountVectorizerで頻度行列の作成 =====
print("単語の頻度行列(Document-Term Matrix)を作成中...")
# max_df: 90%以上の文書に出現する単語は無視（一般的なノイズの除去）
# min_df: 5つ未満の文書にしか出現しない単語は無視（極端な希少語の除去）
vectorizer = CountVectorizer(max_df=0.9, min_df=5, max_features=5000)
dtm = vectorizer.fit_transform(df["nouns"])

feature_names = vectorizer.get_feature_names_out()

# ===== 4. LDAモデルの学習 =====
print(f"LDAモデル(トピック数: {NUM_TOPICS})を学習中...")
lda = LatentDirichletAllocation(n_components=NUM_TOPICS, random_state=42)
doc_topic_dist = lda.fit_transform(dtm)

# ===== 5. トピックごとの特徴語の出力 =====
print("\n=== 各トピックの特徴的な単語 ===")
for topic_idx, topic in enumerate(lda.components_):
    # 値が大きい上位15単語を取得
    top_features_ind = topic.argsort()[: -15 - 1 : -1]
    top_features = [feature_names[i] for i in top_features_ind]
    print(f"トピック {topic_idx}: {', '.join(top_features)}")
print("===============================\n")

# ===== 6. 結果の保存 =====
print("作品ごとのトピック分布を保存します...")
for i in range(NUM_TOPICS):
    df[f"Topic_{i}"] = doc_topic_dist[:, i]

# 必要な列（メタデータ + トピック分布）を選択
save_cols = ["title", "author", "year"] + [f"Topic_{i}" for i in range(NUM_TOPICS)]
df_topics = df[save_cols]
df_topics.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"完了: トピック特徴を {OUTPUT_CSV} に保存しました。")