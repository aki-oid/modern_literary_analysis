import json
import pandas as pd
import re
from tqdm import tqdm
import fugashi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import unicodedata
from matplotlib.ticker import ScalarFormatter

# ===== 1. 設定 & 定数 =====
INPUT_JSON = "data/01_literature.json"
OUTPUT_CSV = "data/02-2_features_style.csv"
PLOT_DIR = "data/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

# 学術的補完：頻出旧字体・異体字リスト（判定精度向上用）
OLD_KANJI_STR = (
    "亞惡壓已卑喝嘆器塀墨層屮悔慨憎懲敏既暑梅海渚漢煮爫琢碑社祉祈祐祖祝禍禎穀突節練縉繁署者臭艹艹著褐視謁謹賓贈辶逸難響頻"
    "體國會實氣獨與變寫廣讀學禮盡驛鐵應觀歸舊晝顯燒條狀乘淨眞粹衞驛圓緣艷奧橫歐黃假價畫魁海繪慨概擴殼覺樂渴褐"
    "勸卷寬歡觀貫關陷巖顏歸僞戲犧據擧虛峽狹曉勤謹近驅勳群軍郡係繼惠掲攜溪經莖螢輕鷄藝擊缺儉劍險驗顯獻權厳源縣"
    "效恆煌廣鑛碎劑濟祭齋細宰裁判最際雜產算贊殘慘仕仔使刺司姉始指死視試詞誌諮資飼事似侍兒字時磁治爾自慈濕質實"
    "寫捨奢煮勺灼爵若弱主取守手朱殊狩授樹収終秋週愁酬衆集住十柔熟術述純准順遵處諸署緒助敍叙徐除傷償勝匠昇昭晶"
    "松沼照燒獎條狀乘淨剩場疊穰蒸讓釀嘱觸辱神眞寢震慎新薪審刃人仁盡迅甚陣尋甚杉親身進帥推水垂錘數枢趨雛据杉澄"
    "寸世制勢聖誠齊靜稅説攝節絶舌羨鮮前善漸然全禪繕撰踐遷選薦銭閃戦潜船尖智置逐蓄築畜竹筑衷忠中仲駐昼柱鋳著"
    "貯庁鳥張朝潮町超暢頂長牒跳徴挺釣寵聴帳脹直朕沈珍賃鎮陳追墜通痛塚掴漬低停呈廷弟定底貞庭挺提程締釘鼎滴的"
    "笛適適鏑敵嫡溺哲徹撤鉄天転展店添典点伝澱殿土吐徒途都度渡塗杜屠党冬凍刀唐塔島悼投搭東桃棟盗陶湯灯当独読"
    "特督内南難軟二尼弐肉日入如任忍認妊念燃粘農濃脳能覇派把波馬婆排廃牌背輩配拝杯梅売倍買媒陪博白伯泊拍舶縛"
    "麦爆漠莫箱八発抜髪伐罰範販汎繁彼比筆非卑碑扉飛皮備微美鼻俵標氷表評描病秒品不付府負婦浮敷普賦部武舞復幅"
    "複覆腹沸仏物分文聞兵平並閉塀弊弊社米弁勉歩保報宝抱放方法泡砲豊亡忘忙傍防妨某棒貌冒望紡房北牧本翻凡末摩"
    "魔麻毎妹枚埋幕膜万慢満漫未味密妙民眠名命明盟銘鳴迷冥務無夢霧黙門問夜野也弥矢役約訳薬由輸予余与誉預容曜"
    "様葉陽養来頼雷乱覧利理履裏離陸律率略流留硫粒隆龍呂慮旅虜僚両量領良療糧力緑林倫輪隣臨類令礼零例冷励嶺"
    "鈴隷齢麗黎戻連練錬路露労廊楼朗浪老録論和話歪"
)
OLD_KANJI_SET = set(OLD_KANJI_STR)

# ===== 2. 分析補助関数 =====
tagger = fugashi.Tagger()

def is_old_kanji(char):
    # Unicodeのより広範な漢字範囲をカバー
    if not (('\u4E00' <= char <= '\u9FFF') or ('\u3400' <= char <= '\u4DBF')):
        return False
    if char in OLD_KANJI_SET:
        return True
    # 互換漢字
    return 0xF900 <= ord(char) <= 0xFAFF

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
    kanji_chars = [c for c in clean_orig if is_old_kanji(c) or ('\u4E00' <= c <= '\u9FFF')]
    old_kanji_count = sum(1 for c in kanji_chars if is_old_kanji(c))
    old_kanji_ratio = old_kanji_count / len(kanji_chars) if kanji_chars else 0

    # --- B. 文構造統計 ---
    sentences = [s.strip() for s in re.split(r'[。！？\n]', norm) if len(s.strip()) > 0]
    avg_sentence_length = np.mean([len(s) for s in sentences]) if sentences else 0

    # --- C. 形態素解析統計 ---
    words = []
    pos_targets = {"名詞", "動詞", "形容詞", "助詞", "助動詞", "副詞", "接続詞"}
    pos_counts = {p: 0 for p in pos_targets}
    goshu_counts = {"和": 0, "漢": 0, "外": 0, "混": 0}
    
    for word in tagger(no_p):
        pos1 = word.feature.pos1
        if pos1 in ["補助記号", "空白"]: continue
        
        words.append(word.surface)
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
        "平均文長": avg_sentence_length,
        "読点頻度": norm.count("、") / len(norm) if len(norm) > 0 else 0,
        "旧字比率": old_kanji_ratio,
        "語彙多様度_MATTR": mattr_val,
        "和語比率": goshu_counts["和"] / word_count if word_count > 0 else 0,
        "漢語比率": goshu_counts["漢"] / word_count if word_count > 0 else 0,
        "外来語比率": goshu_counts["外"] / word_count if word_count > 0 else 0,
    }
    for pos in pos_targets:
        features[f"{pos}割合"] = pos_counts[pos] / word_count if word_count > 0 else 0

    return features

# ===== 3. 実行・データ集計 =====
print("データを読み込んでいます...")
try:
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)
except FileNotFoundError:
    print(f"Error: {INPUT_JSON} が見つかりません。")
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
df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# ===== 4. 可視化 =====
print("グラフを生成中...")
# 移動平均の計算（データが少ない箇所のノイズを抑えるため window=20）
numeric_cols = ["平均文長", "読点頻度", "旧字比率", "語彙多様度_MATTR", "和語比率", "漢語比率", "外来語比率", "名詞割合", "動詞割合", "形容詞割合", "助詞割合", "助動詞割合", "副詞割合", "接続詞割合"]
df_rolling = df[["year"] + numeric_cols].rolling(window=20, on='year', min_periods=5).mean()

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
plt.savefig(f"{PLOT_DIR}02-2-1_word_origin_stack.png")

# [Graph 2: 語彙多様度の年代別プロット]
plt.figure(figsize=(12, 6))
scatter = plt.scatter(df["year"], df["語彙多様度_MATTR"], c=df["漢語比率"], cmap="coolwarm", alpha=0.5)
plt.colorbar(scatter, label="漢語比率")
sns.regplot(data=df, x="year", y="語彙多様度_MATTR", scatter=False, color="black", line_kws={"ls":"--"})
plt.title("年代別語彙多様度 (MATTR500) と漢語依存度の相関", fontsize=14)
plt.savefig(f"{PLOT_DIR}02-2-2_vocabulary_mattr_enhanced.png")

# [Graph 3: 文章の近代化指標]
fig, ax3 = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_rolling, x="year", y="平均文長", ax=ax3, color="#f0a500", lw=2.5, label="平均文長")
ax4 = ax3.twinx()
sns.lineplot(data=df_rolling, x="year", y="旧字比率", ax=ax4, color="#5c2a9d", lw=2.5, label="旧字比率 (右軸)")
ax3.set_title("文長と表記（旧字）の近代化相関", fontsize=14)
ax3.set_ylabel("平均文長 (文字)")
ax4.set_ylabel("旧字比率")
plt.savefig(f"{PLOT_DIR}02-2-3_modernization_trend.png")

print(f"全解析完了。結果は {OUTPUT_CSV} および {PLOT_DIR} に保存されました。")