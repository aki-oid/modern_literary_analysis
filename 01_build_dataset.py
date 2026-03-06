import os
import re
import json
import requests
import pandas as pd
from tqdm import tqdm
import zipfile
import io
import time
import spacy

# ===== 1. 設定 =====
CSV_PATH = "data/list_person_all_extended_utf8.csv"
OUTPUT_JSON = "data/01_literature.json"

YEAR_MIN = 1868
YEAR_MAX = 1975
MAX_WORKS_TOTAL = 2000
MIN_WORKS_PER_AUTHOR = 3
MAX_WORKS_PER_AUTHOR = 10

PRIORITY_TITLES = [
    "こころ", "吾輩は猫である", "坊っちゃん", "三四郎", "それから", "門", 
    "羅生門", "蜘蛛の糸", "河童", "歯車", "或る阿呆の一生", "地獄変",
    "舞姫", "高瀬舟", "阿部一族", "人間失格", "斜陽", "走れメロス",
    "山月記", "李陵", "破戒", "蒲団", "高野聖", "金色夜叉", "或る女"
]

# NLPモデルのロード（人名抽出用）
print("NLPモデルをロード中...")
nlp = spacy.load("ja_ginza") 

# 旧字→新字変換用マッピング（より網羅的に拡充）
OLD_KANJI = "體國會實氣獨與變寫廣讀學禮盡驛廣鐵應觀歸舊晝顯燒條狀乘淨眞粹體"
NEW_KANJI = "体国会実気独与変写広読学礼尽駅広鉄応観帰旧昼顕焼条状乗浄真粋体"
KANJI_TABLE = str.maketrans(OLD_KANJI, NEW_KANJI)

# ===== 2. 前処理関数 =====
def extract_year(text):
    if pd.isna(text): return None
    match = re.search(r"\d{4}", str(text))
    return int(match.group()) if match else None

def clean_text(text):
    text = re.split(r'\n底本：', text)[0]
    borders = list(re.finditer(r'-{10,}', text))
    if len(borders) >= 2: text = text[borders[1].end():]
    text = re.sub(r"《.*?》|［＃.*?］|｜|\r", "", text)
    return text.strip()

def process_text_variants(text):
    """
    1. 人名を除去したテキスト
    2. 抽出された人名リスト
    3. 旧字を正規化したテキスト
    を作成する。Sudachiの制限を避けるため、テキストを分割して処理。
    """
    # 1チャンクあたりの文字数（安全のため10,000文字程度に設定）
    CHUNK_SIZE = 10000
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    
    all_person_names = []
    processed_no_person_chunks = []
    
    for chunk in chunks:
        doc = nlp(chunk)
        # 人名を抽出（後ろから置換してインデックスずれを防ぐ）
        entities = sorted([ent for ent in doc.ents if ent.label_ == "Person"], key=lambda x: x.start_char, reverse=True)
        
        chunk_no_person = chunk
        for ent in entities:
            all_person_names.append(ent.text)
            chunk_no_person = chunk_no_person[:ent.start_char] + chunk_no_person[ent.end_char:]
        
        processed_no_person_chunks.append(chunk_no_person)
    
    text_no_person = "".join(processed_no_person_chunks)
    text_normalized = text_no_person.translate(KANJI_TABLE)
    
    return {
        "original": text,
        "no_person": text_no_person,
        "normalized": text_normalized,
        "person_names": list(set(all_person_names))
    }

# ===== 3. データ読み込みとスコアリング =====
print("CSVを読み込み中...")
df = pd.read_csv(CSV_PATH, encoding="utf-8")
df["year"] = df["初出"].apply(extract_year).fillna(df["底本初版発行年1"].apply(extract_year))
df["author"] = df["姓"].fillna("") + df["名"].fillna("")

# 重複排除用の「クリーンなタイトル」を作成 (例: "こころ(新字新仮名)" -> "こころ")
df["title_clean"] = df["作品名"].str.replace(r"\(.*?\)|（.*?）", "", regex=True).str.strip()

# フィルタリング
df = df[
    (df["分類番号"].str.contains(r"NDC 913", na=False)) &
    (df["役割フラグ"] == "著者") &
    (df["人物著作権フラグ"] == "なし") &
    (df["テキストファイルURL"].notna()) &
    (df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)
]

def calculate_priority(row):
    score = 0
    if row["title_clean"] in PRIORITY_TITLES: score += 1000
    if pd.notna(row.get("テキストファイルURL")): score += 100
    if pd.notna(row.get("初出")): score += 50
    if "全集" in str(row.get("底本名1", "")): score += 20
    return score

df["priority"] = df.apply(calculate_priority, axis=1)
# 重複削除: 著者名とクリーンなタイトルの組み合わせで、最も優先度が高いものだけ残す
df = df.sort_values("priority", ascending=False).drop_duplicates(subset=["author", "title_clean"])

# 作家あたりの最低作品数でフィルタリング
author_counts = df["author"].value_counts()
valid_authors = author_counts[author_counts >= MIN_WORKS_PER_AUTHOR].index
df = df[df["author"].isin(valid_authors)].copy()

# 作家あたりの上限
df = df.groupby("author").head(MAX_WORKS_PER_AUTHOR)

# 全体の上限
df = df.head(MAX_WORKS_TOTAL)

# ===== 4. ダウンロードと多層プロセシング =====
def download_and_extract(url):
    try:
        r = requests.get(url, timeout=20)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for name in z.namelist():
                if name.endswith(".txt"):
                    return z.read(name).decode("cp932", errors="ignore")
    except: return None
    return None

dataset = []
print(f"{len(df)}作品の抽出・多層プロセシングを開始します...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    raw_text = download_and_extract(row["テキストファイルURL"])
    if not raw_text: continue
    
    clean = clean_text(raw_text)
    if len(clean) < 100: continue

    # 多層プロセシング（人名除去、正規化、人名リスト抽出）
    variants = process_text_variants(clean)

    dataset.append({
        "title": row["作品名"],
        "author": row["author"],
        "year": int(row["year"]),
        "text_original": variants["original"],
        "text_no_person": variants["no_person"],
        "text_normalized": variants["normalized"],
        "person_names": variants["person_names"]
    })
    #time.sleep(0.1)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\n完了: {len(dataset)}件の多層構造データを '{OUTPUT_JSON}' に保存しました。")