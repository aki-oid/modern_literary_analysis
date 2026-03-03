import pandas as pd
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
# データフレームの復元

import pandas as pd

PKL_PATH = "data/features_semantics.pkl"

# データの読み込み
df = pd.read_pickle(PKL_PATH)
print(f"重複排除前: {len(df)}件")

# 著者(author)と作品名(title)が同じ場合、最初の1件だけを残して削除
df_dedup = df.drop_duplicates(subset=["author", "title"], keep="first")

# 上書き保存
df_dedup.to_pickle(PKL_PATH)
print(f"重複排除後: {len(df_dedup)}件 に修正して上書きしました。")

df = pd.read_pickle("data/features_semantics.pkl")

# 各作品の意味ベクトルがそのままNumPy配列として入っている
print(df.head())