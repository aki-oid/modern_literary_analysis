# 2-2b. 漢字の激変ランキング：新字（親字）ごとに、旧字のシェアの変動幅を計算してランキング化

import json
import pandas as pd
import requests
import re
from collections import defaultdict
from tqdm import tqdm
from config import *

# ===== 1. 設定 & 定数 =====
INPUT_JSON1 = D01_LITERATURE
INPUT_JSON2 = D00_KANJI_MAPPING
OUTPUT_RANKING_CSV = "data/02-2b_kyuji_fluctuation_ranking.csv"
# 時代区分（1945年以前を「初期」、1946年以降を「後期」とする）
ERA_BOUNDARY = 1945
# ノイズ除去のための足切り（初期・後期の合計出現回数がこの値未満の漢字群は除外）
MIN_TOTAL_COUNT = 200

def load_kyuji_mapping(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ===== 2. 集計ロジック =====
def calculate_share_fluctuation():
    kyuji_mapping = load_kyuji_mapping(INPUT_JSON2)
    
    shinji_to_kyuji = defaultdict(list)
    for kyuji, data in kyuji_mapping.items():
        shinji_to_kyuji[data["shinji"]].append(kyuji)
    
    # カウンターの初期化
    counts = defaultdict(lambda: {"early_shinji": 0, "early_kyuji": 0, "late_shinji": 0, "late_kyuji": 0})
    
    print("テキストデータを走査して漢字の出現回数をカウントしています...")
    try:
        with open(INPUT_JSON1, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_JSON1} が見つかりません。")
        return

    for data in tqdm(dataset, desc="Counting Kanji"):
        year = data.get("year", 0)
        text = re.sub(r'\s+', '', data.get("text_original", ""))
        
        era_prefix = "early" if year <= ERA_BOUNDARY else "late"
        
        for char in text:
            # パターンA: 文字が旧字・異体字である場合
            if char in kyuji_mapping:
                target_shinji = kyuji_mapping[char]["shinji"]
                counts[target_shinji][f"{era_prefix}_kyuji"] += 1
                
            # パターンB: 文字が新字（親字）である場合
            elif char in shinji_to_kyuji:
                counts[char][f"{era_prefix}_shinji"] += 1

    # ===== 3. シェアと変動幅の計算 =====
    print("激変ランキングを計算中...")
    results = []
    
    for shinji, cnt in counts.items():
        early_total = cnt["early_shinji"] + cnt["early_kyuji"]
        late_total = cnt["late_shinji"] + cnt["late_kyuji"]
        total_uses = early_total + late_total
        
        # 足切り: 出現回数が少なすぎるマイナー漢字は除外
        if total_uses < MIN_TOTAL_COUNT:
            continue
            
        # 初期と後期、それぞれの旧字シェア（0.0 〜 1.0）を計算
        early_share = cnt["early_kyuji"] / early_total if early_total > 0 else 0
        late_share = cnt["late_kyuji"] / late_total if late_total > 0 else 0
        
        # シェアの変動幅（ポイント差）
        # プラスなら「旧字が駆逐された」、マイナスなら「後期になって旧字が増えた（復古）」
        share_diff = early_share - late_share
        
        results.append({
            "新字(親字)": shinji,
            "主な旧字": "・".join(shinji_to_kyuji[shinji][:3]), # 代表的な旧字を3つまで表示
            "合計出現回数": total_uses,
            "初期_総数": early_total,
            "初期_旧字シェア": round(early_share * 100, 2),
            "後期_総数": late_total,
            "後期_旧字シェア": round(late_share * 100, 2),
            "シェア下落幅(pt)": round(share_diff * 100, 2)
        })

    # データフレーム化してソート
    df = pd.DataFrame(results)
    if df.empty:
        print("条件を満たすデータがありませんでした。")
        return
        
    # 下落幅が大きい（旧字が使われなくなった）順にソート
    df_sorted = df.sort_values("シェア下落幅(pt)", ascending=False).reset_index(drop=True)
    df_sorted.to_csv(OUTPUT_RANKING_CSV, index=False, encoding="utf-8-sig")
    
    # ===== 4. 結果の出力 =====
    print("\n 【旧字が駆逐された漢字 トップ10】（時代による変化が最も激しい）")
    print(df_sorted.head(10)[["新字(親字)", "主な旧字", "初期_旧字シェア", "後期_旧字シェア", "シェア下落幅(pt)"]])
    print("\n 【旧字がしぶとく生き残った/復活した漢字 トップ10】（変動が少ない、または逆行）")
    print(df_sorted.tail(10)[["新字(親字)", "主な旧字", "初期_旧字シェア", "後期_旧字シェア", "シェア下落幅(pt)"]])
    print(f"\n完了: ランキングの全データは {OUTPUT_RANKING_CSV} に保存されました。")

def main():
    calculate_share_fluctuation()

if __name__ == "__main__":
    main()