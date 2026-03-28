import os

# ディレクトリ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_INPUT_DIR = os.path.join(DATA_DIR, "INPUT")
PLOT_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# データファイル
D00_KANJI_MAPPING = os.path.join(DATA_DIR, "00_kanji_mapping.json")
D00_INPUT_DATA = os.path.join(DATA_INPUT_DIR, "list_person_all_extended_utf8.csv")
D01_LITERATURE = os.path.join(DATA_DIR, "01_literature.json")
D02_INPUT_THOUGHT = os.path.join(DATA_INPUT_DIR, "group-thoughts.json")

D021a_TRAJECTORY = os.path.join(DATA_DIR, "02-1a_narrative_trajectories.pkl")
D021b_TRAJECTORY = os.path.join(DATA_DIR, "02-1b_clustering_results.csv")
D022a_STYLE = os.path.join(DATA_DIR, "02-2a_features_style.csv")
D023_TOPIC = os.path.join(DATA_DIR, "02-3-1_features_topics.csv")
D02_MASTER_DATA_CSV = os.path.join(DATA_DIR, "02_master_dataset.csv")
D02_MASTER_SCALED_DATA_CSV = os.path.join(DATA_DIR, "02_master_dataset_scaled.csv")

# 分析パラメータ
YEAR_MIN = 1868
YEAR_MAX = 1975
ERA_LABELS = {
    "明治": (1868, 1912),
    "大正": (1912, 1926),
    "昭和戦前": (1926, 1945),
    "昭和戦後": (1945, 1989)
}
def get_era(year):
    for era, (start, end) in ERA_LABELS.items():
        if start <= year < end:
            return era
    return "その他"

NUM_CLUSTERS = 5
NUM_SEGMENTS = 20
NUM_TOPICS = 18

def get_file_prefix(path):
    # ファイルパスからファイル名（02-1b_... .pkl）だけを取り出す
    base_name = os.path.basename(path)
    return base_name.split('_')[0]