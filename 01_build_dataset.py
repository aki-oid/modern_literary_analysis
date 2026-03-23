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
from config import *

# ===== 1. 設定 =====
INPUT_CSV = D00_INPUT_DATA
INPUT_KANJI_MAPPING = D00_KANJI_MAPPING
OUTPUT_JSON = D01_LITERATURE

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

print("旧字マッピングデータをロード中...")
with open(INPUT_KANJI_MAPPING, "r", encoding="utf-8") as f:
    kanji_json = json.load(f)

kanji_dict = {}
skipped_keys = []
for old_char, val in kanji_json.items():
    if "shinji" in val and val["shinji"]:
        if len(old_char) == 1:
            kanji_dict[old_char] = val["shinji"]
        else:
            skipped_keys.append(old_char)
if skipped_keys:
    print(f"注意: 以下のキーは1文字ではないため変換テーブルから除外されました: {skipped_keys}")

KANJI_TABLE = str.maketrans(kanji_dict)

# ===== 2. 前処理関数 =====
def extract_year(text):
    if pd.isna(text) or text == "":
        return None
    
    # 1. まず「1893」や「1956」のような4桁の数字を探す
    years = re.findall(r'\b(18\d{2}|19\d{2}|20\d{2})\b', str(text))
    if years:
        # 複数ある場合は、一番古い年（執筆年に近い方）を採用
        return int(min(years))
    
    # 2. 数字がない場合、明治・大正・昭和の和暦表記から計算する（オプション）
    era_map = {"明治": 1867, "大正": 1911, "昭和": 1925, "平成": 1988}
    for era, start_year in era_map.items():
        match = re.search(f'{era}(\d+)年', str(text))
        if match:
            return start_year + int(match.group(1))
            
    return None

def clean_text(text):
    text = re.split(r'\n底本：', text)[0]
    borders = list(re.finditer(r'-{10,}', text))
    if len(borders) >= 2: text = text[borders[1].end():]
    
    text = re.sub(r"※?［＃.*?］", "", text)
    text = re.sub(r"《.*?》|｜|\r", "", text)
    text = re.sub(r'　{2,}', '　', text) 
    text = re.sub(r'(―|ー|…|・|〜){3,}', r'\1\1', text)
    
    # 段落区切り（空行）は残しつつ、文の途中の単なる改行は削除して繋げる
    text = re.sub(r'(?<!\n)\n(?!\n)', '', text)
    # 連続しすぎる改行（3つ以上）は2つにまとめる
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def process_text_variants(text):
    """
    1. 元テキスト (original)
    2. 旧字を正規化したテキスト (normalized)
    3. 正規化テキストから人名をプレースホルダーに置換したテキスト (no_person)
    4. 抽出された人名リスト
    """
    text_normalized = text.translate(KANJI_TABLE)
    
    MAX_LEN = 10000
    chunks = []
    current_chunk = ""
    
    # まず改行を保持したまま行ごとに分割
    for line in text_normalized.splitlines(keepends=True):
        # 行を追加しても上限を超えない場合
        if len(current_chunk) + len(line) <= MAX_LEN:
            current_chunk += line
        else:
            # 行自体が長すぎる場合（改行のない長文）は「。」で分割を試みる
            if len(line) > MAX_LEN:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                sentences = line.split('。')
                for i, sent in enumerate(sentences):
                    # 最後の要素以外は「。」を戻す
                    if i < len(sentences) - 1:
                        sent += '。'
                    
                    if len(current_chunk) + len(sent) <= MAX_LEN:
                        current_chunk += sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        # それでも1文が5000文字を超える異常なケースの最終手段（強制分割）
                        if len(sent) > MAX_LEN:
                            for j in range(0, len(sent), MAX_LEN):
                                chunks.append(sent[j:j+MAX_LEN])
                            current_chunk = ""
                        else:
                            current_chunk = sent
            else:
                # 行は短いが、足すと上限を超える場合
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
                
    if current_chunk:
        chunks.append(current_chunk)
    # -----------------------------------------------------------

    all_person_names = []
    processed_no_person_chunks = []
    
    # チャンクごとにGiNZAで処理
    docs = nlp.pipe(chunks, batch_size=50)
    for chunk, doc in zip(chunks, docs):
        entities = sorted([ent for ent in doc.ents if ent.label_ == "Person"], key=lambda x: x.start_char, reverse=True)
        
        chunk_no_person = chunk
        for ent in entities:
            all_person_names.append(ent.text)
            chunk_no_person = chunk_no_person[:ent.start_char] + "[PERSON]" + chunk_no_person[ent.end_char:]
            
        processed_no_person_chunks.append(chunk_no_person)
    text_no_person = "".join(processed_no_person_chunks)

    return {
        "original": text,
        "normalized": text_normalized,
        "no_person": text_no_person,
        "person_names": list(set(all_person_names))
    }

# ===== 3. データ読み込みとスコアリング =====
print("CSVを読み込み中...")
df = pd.read_csv(INPUT_CSV, encoding="utf-8")
df["year_shutsude"] = df["初出"].apply(extract_year)
df["year_oyamoto"] = df["底本の親本初版発行年1"].apply(extract_year)
df["year_teihon"] = df["底本初版発行年1"].apply(extract_year)
df["year"] = df[["year_shutsude", "year_oyamoto", "year_teihon"]].min(axis=1)

df["author"] = df["姓"].fillna("") + df["名"].fillna("")
# 重複排除用の「クリーンなタイトル」を作成 (例: "こころ(新字新仮名)" -> "こころ")
df["title_clean"] = df["作品名"].str.replace(r"\(.*?\)|（.*?）", "", regex=True).str.strip()

# フィルタリング
df = df[
    (df["分類番号"].str.contains(r"NDC K?913", na=False)) &
    (df["役割フラグ"] == "著者") &
    (df["人物著作権フラグ"] == "なし") &
    (df["テキストファイルURL"].notna()) &
    (df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)
]

def calculate_priority(row):
    score = 0
    if row["title_clean"] in PRIORITY_TITLES: score += 1000
    kana_type = str(row.get("文字遣い種別", ""))
    if kana_type == "旧字旧仮名": score += 500
    elif kana_type == "新字旧仮名": score += 300
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
skipped_logs = []
print(f"{len(df)}作品の抽出・多層プロセシングを開始します...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    raw_text = download_and_extract(row["テキストファイルURL"])
    if not raw_text: 
            skipped_logs.append(f"【DL失敗】{row['author']} - {row['作品名']}")
            continue
    clean = clean_text(raw_text)
    if len(clean) < 100: 
            skipped_logs.append(f"【文字数不足({len(clean)}文字)】{row['author']} - {row['作品名']}")
            continue

    # 多層プロセシング（人名除去、正規化、人名リスト抽出）
    variants = process_text_variants(clean)

    dataset.append({
        "title": row["作品名"],
        "author": row["author"],
        "year": int(row["year"]),
        "text_original": variants["original"],
        "text_normalized": variants["normalized"],
        "text_no_person": variants["no_person"],
        "person_names": variants["person_names"]
    })
    time.sleep(0.2)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\n完了: {len(dataset)}件の多層構造データを '{OUTPUT_JSON}' に保存しました。")
print("\n--- スキップされた作品リスト ---")
for log in skipped_logs[:30]:
    print(log)