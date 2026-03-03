import os
import re
import json
import requests
import pandas as pd
from tqdm import tqdm
import zipfile
import io
import time

# ===== 設定 =====
CSV_PATH = "data/list_person_all_extended_utf8.csv"
OUTPUT_JSON = "data/literature_1890_1945.json"

YEAR_MIN = 1890
YEAR_MAX = 1945
DEATH_YEAR_MAX = 1955
MAX_WORKS = 1000
MAX_WORKS_PER_AUTHOR = 5

# ===== CSV読み込み =====
df = pd.read_csv(CSV_PATH, encoding="utf-8")

# ===== 年抽出 =====
def extract_year(text):
    if pd.isna(text):
        return None
    match = re.search(r"\d{4}", str(text))
    return int(match.group()) if match else None

df["first_pub_year"] = df["初出"].apply(extract_year)
df["teihon_year"] = df["底本初版発行年1"].apply(extract_year)
df["year"] = df["first_pub_year"].fillna(df["teihon_year"])
df["death_year"] = pd.to_datetime(df["没年月日"], errors="coerce").dt.year

# ===== フィルタ =====
df = df[df["分類番号"].str.contains(r"NDC 91", na=False)]
df = df[
    (df["year"].between(YEAR_MIN, YEAR_MAX)) &
    (df["death_year"] <= DEATH_YEAR_MAX)
]
df = df[df["役割フラグ"] == "著者"]
df = df[df["人物著作権フラグ"] == "なし"]

# 作家名の結合（欠損値対策済）
df["author"] = df["姓"].fillna("") + df["名"].fillna("")
# 作家ごとに5作品を絞る「前」に、同作家の同名作品を排除する
df = df.drop_duplicates(subset=["author", "作品名"], keep="first")

# その後で年代順ソートと上限カットを行う
df = df.sort_values("year")
df = df.groupby("author").head(MAX_WORKS_PER_AUTHOR)

# 全体上限
if len(df) > MAX_WORKS:
    df = df.head(MAX_WORKS)

print("抽出予定作品数:", len(df))

# ===== テキストDL & 解凍（リトライ機能付き） =====
def download_and_extract(url, retries=3):
    for i in range(retries):
        try:
            # タイムアウトを30秒に延長
            r = requests.get(url, timeout=30)
            r.raise_for_status() # HTTPエラー（404など）があれば例外を出す
            
            z = zipfile.ZipFile(io.BytesIO(r.content))
            for name in z.namelist():
                if name.endswith(".txt"):
                    # shift_jisより安全なcp932を使用
                    return z.read(name).decode("cp932", errors="ignore")
        except requests.exceptions.RequestException as e:
            print(f"通信エラー({url}) - リトライ {i+1}/{retries}: {e}")
            time.sleep(2) # 2秒待ってから再挑戦
        except Exception as e:
            print(f"解凍・読込エラー({url}): {e}")
            return None
    
    # 規定回数リトライしてもダメだった場合
    print(f"ダウンロード失敗（スキップします）: {url}")
    return None

def clean_text(text):
    # 1. ヘッダー・フッターの除去
    # 青空文庫は末尾の「底本：」以降が書誌情報・入力者情報
    text = re.split(r'\n底本：', text)[0]
    
    # ヘッダー（ハイフン等の連続線）の除去
    borders = list(re.finditer(r'-{10,}', text))
    if len(borders) >= 2:
        text = text[borders[1].end():]
    elif len(borders) == 1:
        text = text[borders[0].end():]

    # 2. ルビと注記の除去
    text = re.sub(r"《.*?》", "", text)      # ルビ除去
    text = re.sub(r"［＃.*?］", "", text)    # 注記除去
    text = re.sub(r"｜", "", text)           # ルビ開始記号の除去
    text = re.sub(r"\r", "", text)           # CR除去

    return text.strip()

dataset = []
error_count = 0
empty_count = 0

# ===== データ取得 =====
for _, row in tqdm(df.iterrows(), total=len(df)):
    url = row["テキストファイルURL"]
    if pd.isna(url):
        continue

    text = download_and_extract(url)
    if text is None:
        error_count += 1
        continue

    text = clean_text(text)
    
    if not text:
        empty_count += 1
        continue

    dataset.append({
        "title": row["作品名"],
        "author": row["author"],
        "year": int(row["year"]),
        "text": text
    })
    
    # サーバー負荷軽減のためのスリープ（必須）
    time.sleep(1)

# ===== 保存 =====
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"\n完了: 保存 {len(dataset)}件, DLエラー {error_count}件, 本文消失 {empty_count}件")