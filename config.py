import os

# ディレクトリ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_INPUT_DIR = os.path.join(DATA_DIR, "INPUT")
PLOT_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# データファイル
D00_INPUT_DATA = os.path.join(DATA_INPUT_DIR, "list_person_all_extended_utf8.csv")
D01_LITERATURE = os.path.join(DATA_DIR, "01_literature.json")

D021a_TRAJECTORY = os.path.join(DATA_DIR, "02-1a_narrative_trajectories.pkl")
D021b_TRAJECTORY = os.path.join(DATA_DIR, "02-1b_clustering_results.csv")
D022a_KANJI_MAPPING = os.path.join(DATA_DIR, "02-2a-1_kanji_mapping.json")
D022b_STYLE = os.path.join(DATA_DIR, "02-2b_features_style.csv")
D023_TOPIC = os.path.join(DATA_DIR, "02-3_features_topics.csv")
D02_MASTER_DATA_CSV = os.path.join(DATA_DIR, "02_master_dataset.csv")

# 分析パラメータ
YEAR_MIN = 1868
YEAR_MAX = 1975

NUM_CLUSTERS = 5
NUM_SEGMENTS = 20
NUM_TOPICS = 8

def get_file_prefix(path):
    # ファイルパスからファイル名（02-1b_... .pkl）だけを取り出す
    base_name = os.path.basename(path)
    return base_name.split('_')[0]