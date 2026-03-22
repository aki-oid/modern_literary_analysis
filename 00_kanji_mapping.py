import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from config import *

# ===== 設定 =====
INPUT_JOYO_JSON = "data/INPUT/00_kyuuzi/常用漢字表本表.json"
INPUT_KANKEN_CSV = "data/INPUT/00_kyuuzi/漢検漢字辞典漢字.csv"
OUTPUT_JSON = D00_KANJI_MAPPING

def fetch_data():
    print("データを取得中...")
    with open(INPUT_JOYO_JSON, 'r', encoding='utf-8') as f:
        joyo_data = json.load(f)
    
    df_kanken = pd.read_csv(INPUT_KANKEN_CSV)
    return joyo_data, df_kanken

def build_mapping(joyo_data, df_kanken):
    kyuji_map = {}
    conflict_count = 0

    print("1. 常用漢字表（本表）からのマッピングを構築中...")
    # READMEの仕様: 「通用字体」に新字、「康熙字典体」に旧字（丸括弧内の字）が入っている想定
    for item in joyo_data:
        kanji_info = item.get("漢字", {})
        shinjitai = kanji_info.get("通用字体", "")
        kyujitai = kanji_info.get("康熙字典体", "")
        
        # 康熙字典体が存在し、かつ新字体と異なる場合のみ登録
        if kyujitai and shinjitai and kyujitai != shinjitai:
            kyuji_map[kyujitai] = {
                "shinji": shinjitai,
                "type": "旧字 (常用漢字表)"
            }

    joyo_count = len(kyuji_map)
    print(f"  -> 常用漢字表から {joyo_count} 件の旧字ペアを抽出しました。")

    print("2. 漢検データからのマッピングを構築中（表外字の補完）...")
    # 漢検データは「字種ID」で親字と旧字が紐づいているリレーショナル構造
    
    # まず、各「字種ID」に対応する「親字」の辞書を作る
    parents = df_kanken[df_kanken['字体'] == '親字']
    parent_dict = pd.Series(parents['漢字テキスト'].values, index=parents['字種ID']).to_dict()

    # 次に、「旧字」としてマークされているものを回してマッピング
    variants = df_kanken[df_kanken['字体'] != '親字']
    
    kanken_added_count = 0
    for _, row in variants.iterrows():
        kyujitai = str(row['漢字テキスト'])
        jishu_id = row['字種ID']
        shinjitai = parent_dict.get(jishu_id)
        zitai_type = str(row['字体'])
        
        # NaN（欠損値）や文字列の 'nan' のスキップ処理
        if pd.isna(kyujitai) or pd.isna(shinjitai) or kyujitai == 'nan':
            continue

        if shinjitai and kyujitai != shinjitai:
            # 常用漢字表の定義を優先し、上書きしない
            if kyujitai in kyuji_map:
                if kyuji_map[kyujitai] != shinjitai:
                    conflict_count += 1
            else:
                kyuji_map[kyujitai] = {
                    "shinji": shinjitai,
                    "type": f"{zitai_type} (漢検)"
                }
                kanken_added_count += 1

    print(f"  -> 漢検データから {kanken_added_count} 件の旧字ペアを補完しました。")
    if conflict_count > 0:
        print(f"  ※注: 常用漢字と漢検で新字の定義が衝突した文字が {conflict_count} 件ありました（常用漢字を優先）。")

    return kyuji_map

def main():
    try:
        joyo_data, df_kanken = fetch_data()
        kyuji_map = build_mapping(joyo_data, df_kanken)
        
        # JSONとして保存
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(kyuji_map, f, ensure_ascii=False, indent=2)
            
        print(f"\n完了: 合計 {len(kyuji_map)} 件のマッピングを '{OUTPUT_JSON}' に保存しました。")
        
        # サンプル出力
        print("\n【出力サンプル】")
        sample_keys = list(kyuji_map.keys())[:5]
        for k in sample_keys:
            print(f"  {k} -> {kyuji_map[k]}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
    
if __name__ == "__main__":
    main()