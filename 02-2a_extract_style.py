# 2-2a. 文体分析のための特徴量抽出と年代別スタイルの可視化

import os
import json
import pandas as pd
import re
from tqdm import tqdm
import fugashi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
from matplotlib.ticker import ScalarFormatter
from config import *

# ===== 1. 設定 & 定数 =====
INPUT_JSON1 = D01_LITERATURE
INPUT_JSON2 = D00_KANJI_MAPPING
OUTPUT_CSV = D022a_STYLE
ID_FILE = get_file_prefix(os.path.basename(__file__))

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

def load_kyuji_mapping(json_path):
    print("旧字・異体字マッピング辞書を読み込んでいます...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        print(f"完了: {len(mapping)} 件の旧字・異体字データを登録しました。")
        return mapping
    except Exception as e:
        print(f"エラー: マッピング辞書の読み込みに失敗しました ({e})。")
        return {}
    
KYUJI_MAPPING = load_kyuji_mapping(INPUT_JSON2)

# ===== 2. 分析補助関数 =====
tagger = fugashi.Tagger()

def is_kanji(char):
    # CJK統合漢字、拡張A、互換漢字の範囲
    return ('\u4E00' <= char <= '\u9FFF') or ('\u3400' <= char <= '\u4DBF') or ('\uF900' <= char <= '\uFAFF')

def split_sentences_robust(text):
    """
    カギカッコ内を保護しつつ、文を分割する。
    """
    if not text:
        return []

    # 1. カギカッコ内の句点を一時的に保護
    # 最小一致 (.*?) でカギカッコ内を特定し、その中の句点を置換
    protected_text = re.sub(
        r'([「『])(.*?)([」』])',
        lambda m: m.group(1) + m.group(2).replace('。', '<PER>').replace('！', '<EXC>').replace('？', '<QUE>') + m.group(3),
        text
    )

    # 2. 本来の文境界（。！？ \n）で分割
    # 分割後の句点等を消さないためにカッコで括る（re.splitの仕様）
    raw_sentences = re.split(r'([。！？\n])', protected_text)
    
    # 3. 分割した記号を文末に結合し、保護した記号を元に戻す
    sentences = []
    for i in range(0, len(raw_sentences) - 1, 2):
        s = raw_sentences[i] + raw_sentences[i+1]
        s = s.replace('<PER>', '。').replace('<EXC>', '！').replace('<QUE>', '？').strip()
        if s:
            sentences.append(s)
            
    # 最後の残余部分の処理
    last_part = raw_sentences[-1].replace('<PER>', '。').replace('<EXC>', '！').replace('<QUE>', '？').strip()
    if last_part:
        sentences.append(last_part)

    return sentences

def calculate_mattr(words, window_size=500):
    """
    MATTRの計算。
    $$MATTR = \frac{1}{N-L+1} \sum_{i=1}^{N-L+1} \frac{V_{window,i}}{L}$$
    """
    # サンプルサイズが窓幅に満たない場合は比較不能（NaN）とするのが学術的に厳密
    if len(words) < window_size:
        return np.nan
    
    distinct_counts = [len(set(words[i : i + window_size])) for i in range(len(words) - window_size + 1)]
    return np.mean(distinct_counts) / window_size

def extract_stylometry(data, mattr_window=500):
    orig = data.get("text_original", "")
    norm = data.get("text_normalized", "")
    no_p = data.get("text_no_person", "")

    # 文長100文字未満は統計的ノイズとして除外
    if not orig or len(orig) < 100:
        return {}

    # --- A. 表記統計 ---
    clean_orig = re.sub(r'\s+', '', orig)
    kanji_chars = [c for c in clean_orig if is_kanji(c)]
    
    old_kanji_count = sum(1 for c in kanji_chars if c in KYUJI_MAPPING)
    old_kanji_ratio = old_kanji_count / len(kanji_chars) if kanji_chars else 0

    # --- B. 文構造統計 ---
    sentences = split_sentences_robust(norm)
    
    if sentences:
        avg_sentence_length = np.mean([len(s) for s in sentences])
        # 文長のばらつき（標準偏差）も、文体の「リズム」を知る上で重要な指標です
        std_sentence_length = np.std([len(s) for s in sentences])
    else:
        avg_sentence_length = 0
        std_sentence_length = 0

    # --- C. 形態素解析統計 ---
    words = []
    pos_targets = {"名詞", "動詞", "形容詞", "助詞", "助動詞", "副詞", "接続詞"}
    pos_counts = {p: 0 for p in pos_targets}
    goshu_counts = {"和": 0, "漢": 0, "外": 0, "混": 0}
    unk_count = 0
    for word in tagger(no_p):
        pos1 = word.feature.pos1
        if pos1 in ["補助記号", "空白"]: continue
        if word.is_unk:
            unk_count += 1
        try:
            # UniDic環境を想定し、lemma（語彙素）を取得
            lemma = word.feature.lemma
            if not lemma or lemma == "*":
                lemma = word.surface
            # 辞書によっては「食べる-動詞」のようにハイフンで付加情報がつく場合の安全策
            lemma = lemma.split('-')[0]
        except AttributeError:
            # lemma属性がない場合は表層形をフォールバックとして使用
            lemma = word.surface
            
        words.append(lemma)
        if pos1 in pos_counts: pos_counts[pos1] += 1
        
        try:
            g = word.feature.goshu
            if g and g[0] in goshu_counts:
                goshu_counts[g[0]] += 1
        except: continue

    word_count = len(words)
    # MATTRの窓幅に満たない作品は多様度解析から外す（または窓幅を小さく設定）
    mattr_val = calculate_mattr(words, window_size=mattr_window)

    features = {
        "平均文長": avg_sentence_length,"文長標準偏差": std_sentence_length,
        "読点頻度": orig.count("、") / len(orig) if len(orig) > 0 else 0,
        "旧字比率": old_kanji_ratio,
        "語彙多様度_MATTR": mattr_val,
        "和語比率": goshu_counts["和"] / word_count if word_count > 0 else 0,
        "漢語比率": goshu_counts["漢"] / word_count if word_count > 0 else 0,
        "外来語比率": goshu_counts["外"] / word_count if word_count > 0 else 0,
        "未知語率": unk_count / word_count if word_count > 0 else 0, # 【追加】
    }
    for pos in pos_targets:
        features[f"{pos}割合"] = pos_counts[pos] / word_count if word_count > 0 else 0

    return features

# ===== 3. 実行・データ集計 =====
print("データを読み込んでいます...")
try:
    with open(INPUT_JSON1, "r", encoding="utf-8") as f:
        dataset = json.load(f)
except FileNotFoundError:
    print(f"Error: {INPUT_JSON1} が見つかりません。")
    dataset = []

features_list = []
for data in tqdm(dataset, desc="Stylometry Extraction"):
    style_dict = extract_stylometry(data)
    if not style_dict: continue
    
    style_dict.update({
        "title": data.get("title", "Unknown"),
        "author": data.get("author", "Unknown"),
        "year": data.get("year", 0)
    })
    features_list.append(style_dict)

df = pd.DataFrame(features_list).sort_values("year")
front_cols = ["title", "author", "year"]
other_cols = [col for col in df.columns if col not in front_cols]
df = df[front_cols + other_cols]
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# ===== 4. 可視化 =====
print("グラフを生成中...")
# 移動平均の計算（データが少ない箇所のノイズを抑えるため window=20）
numeric_cols = ["平均文長", "読点頻度", "旧字比率", "語彙多様度_MATTR", "和語比率", "漢語比率", "外来語比率", "名詞割合", "動詞割合", "形容詞割合", "助詞割合", "助動詞割合", "副詞割合", "接続詞割合"]
if not df.empty:
    # 1. 同じ年の作品群を平均化
    df_yearly = df.groupby("year")[numeric_cols].mean()
    # 2. データの存在しない年をNaNで埋めて、連続した年系列を作成
    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    df_yearly = df_yearly.reindex(range(min_year, max_year + 1))
    # 3. 5「年」の窓幅で厳密な移動平均を計算
    df_rolling = df_yearly.rolling(window=3, min_periods=2).mean().reset_index()
    df_rolling = df_rolling.rename(columns={"index": "year"})
else:
    df_rolling = pd.DataFrame(columns=["year"] + numeric_cols)

# [Graph 1: 語種構成の積み上げ推移]
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.stackplot(df_rolling["year"], df_rolling["和語比率"], df_rolling["漢語比率"], 
              labels=["和語比率", "漢語比率"], colors=["#81b214", "#d72323"], alpha=0.6)
ax2 = ax1.twinx()
ax2.plot(df_rolling["year"], df_rolling["外来語比率"], color="#132743", lw=2, label="外来語比率 (右軸)")
ax1.set_title("語種構成の通時的推移", fontsize=14)
ax1.set_xlabel("年代")
ax1.set_ylabel("構成比")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, f"{ID_FILE}-1_word_origin_stack.png"), dpi=300, bbox_inches='tight')

# [Graph 2: 語彙多様度の年代別プロット]
plt.figure(figsize=(12, 6))
scatter = plt.scatter(df["year"], df["語彙多様度_MATTR"], c=df["漢語比率"], cmap="coolwarm", alpha=0.5)
plt.colorbar(scatter, label="漢語比率")
sns.regplot(data=df, x="year", y="語彙多様度_MATTR", scatter=False,lowess=True, color="black", line_kws={"color": "black", "lw": 2, "ls": "--", "label": "LOESS（局所回帰トレンド）"})
plt.title("年代別語彙多様度 (MATTR500) と漢語依存度の相関", fontsize=14)
plt.savefig(os.path.join(PLOT_DIR, f"{ID_FILE}-2_vocabulary_mattr_enhanced.png"), dpi=300, bbox_inches='tight')

# [Graph 3: 文章の近代化指標]
fig, ax3 = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_rolling, x="year", y="平均文長", ax=ax3, color="#f0a500", lw=2.5, label="平均文長")
ax4 = ax3.twinx()
sns.lineplot(data=df_rolling, x="year", y="旧字比率", ax=ax4, color="#5c2a9d", lw=2.5, label="旧字比率 (右軸)")
ax3.set_title("文長と表記（旧字）の近代化相関", fontsize=14)
ax3.set_ylabel("平均文長 (文字)")
ax4.set_ylabel("旧字比率")
plt.savefig(os.path.join(PLOT_DIR, f"{ID_FILE}-3_modernization_trend.png"), dpi=300, bbox_inches='tight')
print(f"全解析完了。結果は {OUTPUT_CSV} および {PLOT_DIR} に保存されました。")