import json
import pandas as pd
from tqdm import tqdm
import fugashi

# ===== 設定 =====
INPUT_JSON = "data/01_literature_1890_1945.json"
OUTPUT_CSV = "data/02-2_features_style.csv" # 文体特徴は数値の表なのでCSVが扱いやすい

# ===== 1. MeCab(fugashi)の初期化 =====
print("形態素解析エンジン(MeCab)を初期化中...")
# ipadic辞書を使用する標準的なTagger
tagger = fugashi.Tagger()

# ===== 2. データの読み込み =====
print("データを読み込んでいます...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# ===== 3. 文体抽出関数 =====
def extract_stylometry(text):
    if not text:
        return {}

    # 1. 基礎的な文字レベルの統計
    total_chars = len(text)
    sentence_count = text.count("。") if text.count("。") > 0 else 1
    comma_count = text.count("、")
    
    avg_sentence_length = total_chars / sentence_count
    comma_rate = comma_count / total_chars # 1文字あたりの読点出現率

    # 2. 形態素解析による品詞レベルの統計
    word_count = 0
    pos_counts = {"名詞": 0, "動詞": 0, "形容詞": 0, "助詞": 0, "助動詞": 0}
    unique_words = set()

    # tagger(text) で形態素に分割
    for word in tagger(text):
        word_count += 1
        # word.surface は単語の文字列、word.feature.pos1 は品詞の大分類
        surface = word.surface
        pos1 = word.feature.pos1
        
        unique_words.add(surface)

        if pos1 in pos_counts:
            pos_counts[pos1] += 1

    # 語彙多様度 (TTR: Type-Token Ratio)
    # 異なり語数 / 総語数 (数値が大きいほど語彙が豊富)
    ttr = len(unique_words) / word_count if word_count > 0 else 0

    # 各品詞の割合（総語数に対する割合）
    pos_rates = {f"{k}割合": (v / word_count if word_count > 0 else 0) for k, v in pos_counts.items()}

    # 結果をまとめる
    style_features = {
        "総文字数": total_chars,
        "平均文長": avg_sentence_length,
        "読点頻度": comma_rate,
        "語彙多様度_TTR": ttr,
    }
    style_features.update(pos_rates)
    
    return style_features

# ===== 4. 特徴抽出の実行 =====
print("文体特徴の抽出を開始します...")

features = []
for data in tqdm(dataset):
    style_dict = extract_stylometry(data["text"])
    
    # メタデータと結合
    row = {
        "title": data["title"],
        "author": data["author"],
        "year": data["year"],
    }
    row.update(style_dict)
    features.append(row)

# ===== 5. 保存 =====
# 今回は単なる数値データの表なので、視認性の高いCSVで保存します
df_style = pd.DataFrame(features)
df_style.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n完了: {len(df_style)}件の文体特徴を {OUTPUT_CSV} に保存しました。")