import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# ===== 設定 =====
INPUT_CSV = "data/03_influence_edges.csv"
TOP_N = 20 # 上位何件を表示するか

# ===== 1. ネットワークの構築 =====
print("ネットワークを構築中...")
df_edges = pd.read_csv(INPUT_CSV)

# 指向性グラフ（Directed Graph）を作成
G = nx.from_pandas_edgelist(
    df_edges, 
    source='source_title', 
    target='target_title', 
    edge_attr='weight', 
    create_using=nx.DiGraph()
)

# 作家情報を紐付けるための辞書作成（表示用）
author_map = pd.concat([
    df_edges[['source_title', 'source_author']].rename(columns={'source_title':'title', 'source_author':'author'}),
    df_edges[['target_title', 'target_author']].rename(columns={'target_title':'title', 'target_author':'author'})
]).drop_duplicates().set_index('title')['author'].to_dict()

# ===== 2. 指標の計算 =====
print("中心性指標を計算中...")

# PageRank (影響の質を重視)
pagerank = nx.pagerank(G, weight='weight')

# 出次数中心性 (どれだけ多くの作品の「源流」になったか)
out_degree = dict(G.out_degree(weight='weight'))

# ===== 3. 結果の統合と表示 =====
results = []
for title in G.nodes():
    results.append({
        "作品名": title,
        "著者": author_map.get(title, "不明"),
        "象徴性スコア(PageRank)": pagerank.get(title, 0),
        "源流スコア(Out-Degree)": out_degree.get(title, 0)
    })

df_res = pd.DataFrame(results)

# 象徴性スコアでソート
df_iconic = df_res.sort_values("象徴性スコア(PageRank)", ascending=False).head(TOP_N)

print(f"\n=== 日本近代文学：最も象徴的な作品 TOP {TOP_N} ===")
print(df_iconic.to_string(index=False))

# ===== 4. 簡単な可視化 (上位のみ) =====
print("\nネットワークのプレビューを生成中...")
plt.figure(figsize=(12, 8))
# スコア上位のノードとその周辺だけを抽出して描画
top_nodes = df_iconic["作品名"].tolist()
sub_G = G.subgraph(top_nodes)
pos = nx.spring_layout(sub_G, k=0.5)
nx.draw(sub_G, pos, with_labels=True, node_size=2000, node_color="skyblue", font_family="MS Gothic", font_size=9, edge_color="gray", alpha=0.6)
plt.title("日本近代文学：象徴的作品ネットワーク（上位）")
plt.show()

# 保存
df_res.to_csv("data/04_final_analysis_results.csv", index=False, encoding="utf-8-sig")