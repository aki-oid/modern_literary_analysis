import json
import pandas as pd
import re
from tqdm import tqdm
import fugashi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import LogLocator, ScalarFormatter

# ===== 1. 設定 =====
INPUT_JSON = "data/01_literature.json"
OUTPUT_CSV = "data/02-2_features_style.csv"
PLOT_DIR = "data/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

# 旧字判定用の文字セット
OLD_KANJI_SET = set("體國會實氣獨與變寫廣讀學禮盡驛鐵應觀歸舊晝顯燒條状乘浄眞粹")

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

# ===== 1. 文体抽出関数の改善 =====
tagger = fugashi.Tagger()

def extract_stylometry(text_no_person, text_original):
    if not text_no_person or len(text_no_person) < 10:
        return {}

    # --- A. 基礎的な文字レベル・表記統計 ---
    clean_original = re.sub(r'\s+', '', text_original)
    total_chars = len(clean_original)
    
    # 句点等で分割（改行も区切りに含む）
    sentences = re.split(r'[。！？\n]', text_original)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    avg_sentence_length = np.mean([len(s) for s in sentences]) if sentences else 0
    comma_count = clean_original.count("、")
    comma_rate = comma_count / total_chars if total_chars > 0 else 0

    kanji_chars = re.findall(r'[一-龠]', clean_original)
    old_kanji_ratio = sum(1 for char in kanji_chars if char in OLD_KANJI_SET) / len(kanji_chars) if kanji_chars else 0

    # --- B. 形態素解析統計 ---
    word_count = 0
    pos_counts = {"名詞": 0, "動詞": 0, "形容詞": 0, "助詞": 0, "助動詞": 0}
    goshu_counts = {"和": 0, "漢": 0, "外": 0, "混": 0}
    unique_words = set()

    for word in tagger(text_no_person):
        pos1 = word.feature.pos1
        if pos1 == "補助記号" or pos1 == "空白":
            continue
            
        word_count += 1
        unique_words.add(word.surface)
        if pos1 in pos_counts:
            pos_counts[pos1] += 1
        
        goshu = word.feature.goshu
        if goshu in goshu_counts:
            goshu_counts[goshu] += 1

    ttr = len(unique_words) / word_count if word_count > 0 else 0
    
    features = {
        "平均文長": avg_sentence_length,
        "読点頻度": comma_rate,
        "旧字比率": old_kanji_ratio,
        "語彙多様度_TTR": ttr,
        "和語比率": goshu_counts["和"] / word_count if word_count > 0 else 0,
        "漢語比率": goshu_counts["漢"] / word_count if word_count > 0 else 0,
        "外来語比率": goshu_counts["外"] / word_count if word_count > 0 else 0,
    }
    for pos, count in pos_counts.items():
        features[f"{pos}割合"] = count / word_count if word_count > 0 else 0
    return features

# ===== 2. 実行・データ集計 =====
print("データを読み込んでいます...")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print("文体特徴の抽出を開始します...")
features_list = []
for data in tqdm(dataset):
    style_dict = extract_stylometry(data["text_no_person"], data["text_original"])
    if not style_dict: continue
    
    style_dict.update({"title": data["title"], "author": data["author"], "year": data["year"]})
    features_list.append(style_dict)

df = pd.DataFrame(features_list).sort_values("year")
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# ===== 3. 可視化：強化版グラフ生成 =====
print("グラフを生成中...")

numeric_cols = ["平均文長", "読点頻度", "旧字比率", "語彙多様度_TTR", "和語比率", "漢語比率", "外来語比率"]
df_rolling = df[["year"] + numeric_cols].rolling(window=30, on='year', min_periods=5).mean()

# --- Graph 1: 語種の変遷 ---
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
ax1.plot(df_rolling["year"], df_rolling["和語比率"], label="和語比率", color="darkgreen", lw=2)
ax1.plot(df_rolling["year"], df_rolling["漢語比率"], label="漢語比率", color="crimson", lw=2)
ax2.plot(df_rolling["year"], df_rolling["外来語比率"], label="外来語比率 (右軸)", color="royalblue", lw=1.5, ls="--")
ax1.set_title("語彙構成の変遷", fontsize=14)
ax1.set_xlabel("年代")
ax1.set_ylabel("比率")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
plt.savefig(f"{PLOT_DIR}02-2-1_word_origin_trend.png")

# --- Graph 2: 文章構造 (対数軸) ---
fig2, ax3 = plt.subplots(figsize=(12, 6))
ax4 = ax3.twinx()
ax3.set_yscale('log') 
ax3.plot(df_rolling["year"], df_rolling["平均文長"], color="orange", label="平均文長 (対数軸)", lw=2)
ax4.plot(df_rolling["year"], df_rolling["旧字比率"], color="purple", label="旧字比率 (右軸)", lw=2, alpha=0.6)
ax3.yaxis.set_major_formatter(ScalarFormatter())
ax3.set_title("文章構造と表記の近代化", fontsize=14)
ax3.set_ylabel("平均文長 (文字数)")
ax4.set_ylabel("旧字比率")
lines3, labels3 = ax3.get_legend_handles_labels()
lines4, labels4 = ax4.get_legend_handles_labels()
ax3.legend(lines3 + lines4, labels3 + labels4, loc="upper right")
plt.savefig(f"{PLOT_DIR}02-2-2_sentence_modernization.png")

# --- Graph 3: 語彙多様度 (TTR) の変遷 ---
# 
plt.figure(figsize=(12, 6))
plt.plot(df_rolling["year"], df_rolling["語彙多様度_TTR"], color="teal", lw=2, label="語彙多様度 (TTR)")
plt.fill_between(df_rolling["year"], df_rolling["語彙多様度_TTR"], color="teal", alpha=0.1)
plt.title("語彙多様度 (TTR) の変遷：表現の豊かさの推移", fontsize=14)
plt.xlabel("年代")
plt.ylabel("TTR (Unique Words / Total Words)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(f"{PLOT_DIR}02-2-3_vocabulary_diversity_ttr.png")

plt.show()
print(f"解析完了！3つのグラフを {PLOT_DIR} に保存しました。")