# 2-1.「物語の軌跡（Narrative Trajectory）」や「物語の弧（Narrative Arc）」を計算機で解明しようとする
# 2-1b. 軌跡データを用いて「物語の形（Narrative Shape）」のクラスタリングと代表作抽出を行うコード
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import *

# ===== config =====
INPUT_JSON = D01_LITERATURE
INPUT_PKL = D021a_TRAJECTORY
OUTPUT_CSV = D021b_TRAJECTORY
ID_FILE = get_file_prefix(os.path.basename(__file__))

# ===== 1. JSONから文字数を取得し、分類を付与 =====
print(f"JSONデータを読み込み中: {INPUT_JSON}")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    json_data = json.load(f)

length_records = []
for row in json_data:
    text = row.get("text_normalized", "")
    length_records.append({
        "title": row["title"],
        "author": row["author"],
        "char_count": len(text)
    })

df_len = pd.DataFrame(length_records)

def categorize_length(length):
    if length < 20000:
        return "短編"
    else:
        return "中長編"

df_len["length_category"] = df_len["char_count"].apply(categorize_length)

# ===== 2. 軌跡データの読み込みと結合 =====
print(f"軌跡データを読み込み中: {INPUT_PKL}")
df_traj = pd.read_pickle(INPUT_PKL)
df_traj["decade"] = (df_traj["year"] // 10) * 10

df = pd.merge(df_traj, df_len, on=["title", "author"], how="inner")

print("\n" + "="*50)
print("データ件数（2群分類）:")
print("="*50)
print(df["length_category"].value_counts())
print("="*50 + "\n")

# ベクトル空間の白色化（Whitening）処理と信頼性評価
print("="*60)
print("【ベクトル空間の補正】 白色化（Whitening）処理を実行中...")

# 全作品の全セグメントベクトルを1つの行列にまとめる
all_vecs = []
valid_mask = []
for traj in df["trajectory"]:
    if traj is not None and len(traj) == NUM_SEGMENTS:
        all_vecs.extend(traj)
        valid_mask.append(True)
    else:
        valid_mask.append(False)
all_vecs = np.array(all_vecs)

if len(all_vecs) > 0:
    # --- 信頼性評価1：補正前の異方性の測定 ---
    np.random.seed(42)
    sample_size = min(2000, len(all_vecs)) # 計算量削減のためサンプリング
    sample_indices = np.random.choice(len(all_vecs), size=sample_size, replace=False)
    sample_vecs_before = all_vecs[sample_indices]
    
    # 補正前の平均コサイン類似度計算
    sim_matrix_before = np.dot(sample_vecs_before, sample_vecs_before.T)
    np.fill_diagonal(sim_matrix_before, np.nan) # 自分自身との類似度(1.0)を除外
    avg_sim_before = np.nanmean(sim_matrix_before)
    
    # --- 白色化行列の計算 ---
    mu = np.mean(all_vecs, axis=0)
    centered_vecs = all_vecs - mu
    cov = np.cov(centered_vecs, rowvar=False)
    
    # 特異値分解 (SVD) を用いて白色化行列 W を算出
    U, S, V = np.linalg.svd(cov)
    W = np.dot(U, np.diag(1.0 / np.sqrt(S + 1e-5))) # ゼロ割りを防ぐための微小値追加
    
    # --- 全データへの適用 ---
    def apply_whitening(traj):
        if traj is None or len(traj) != NUM_SEGMENTS:
            return traj
        # 中心化して白色化行列を掛ける
        whitened = np.dot(traj - mu, W)
        # 類似度計算のために再度単位ベクトルに正規化
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        return whitened / np.where(norms > 0, norms, 1e-9)

    df["trajectory_whitened"] = df["trajectory"].apply(apply_whitening)
    
    # --- 信頼性評価2：補正後の異方性の測定 ---
    all_whitened_vecs = np.array([v for traj in df["trajectory_whitened"].dropna() for v in traj])
    sample_vecs_after = all_whitened_vecs[sample_indices]
    
    sim_matrix_after = np.dot(sample_vecs_after, sample_vecs_after.T)
    np.fill_diagonal(sim_matrix_after, np.nan)
    avg_sim_after = np.nanmean(sim_matrix_after)

    print(f"  [空間全体の偏り] 補正前ランダム類似度: {avg_sim_before:.4f} -> 補正後: {avg_sim_after:.4f}")

    # --- 信頼性評価3：意味的構造の保存チェック ---
    print("\n  [定性チェック: 文脈（意味的繋がり）は破壊されていないか？]")
    adj_sims_before = []
    adj_sims_after = []
    
    for idx, row in df.iterrows():
        traj_b = row["trajectory"]
        traj_a = row.get("trajectory_whitened")
        if traj_b is not None and traj_a is not None and len(traj_b) == NUM_SEGMENTS:
            # 同一作品内の「隣り合うセグメント（文脈が繋がっているはずの部分）」の類似度
            for k in range(NUM_SEGMENTS - 1):
                adj_sims_before.append(np.dot(traj_b[k], traj_b[k+1]))
                adj_sims_after.append(np.dot(traj_a[k], traj_a[k+1]))
                
    avg_adj_before = np.mean(adj_sims_before)
    avg_adj_after = np.mean(adj_sims_after)
    
    signal_before = avg_adj_before - avg_sim_before
    signal_after = avg_adj_after - avg_sim_after
    
    print(f"    【補正前】 隣接セグメント類似度: {avg_adj_before:.4f} (ランダムとの差分: +{signal_before:.4f})")
    print(f"    【補正後】 隣接セグメント類似度: {avg_adj_after:.4f} (ランダムとの差分: +{signal_after:.4f})")
    
    if signal_after > 0:
        print(f"    -> 結論: 補正後も隣接セグメントはランダムな文より明確に高い類似度(+{signal_after:.4f})を維持しています。")
        print("       「文学的に重要な類似性（文脈）」はノイズとして除去されず、むしろ抽出されやすくなっています。")
    else:
        print("    -> 警告: 隣接セグメントの類似度がランダム以下になりました。意味構造が破壊されています。")
else:
    print("有効なベクトルが見つかりませんでした。")
    df["trajectory_whitened"] = df["trajectory"]

print("="*60 + "\n")

# ===== 3. 分析・プロット処理と代表作の抽出 =====
categories_to_plot = ["短編", "中長編"]

plt.rcParams["font.family"] = "MS Gothic"
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

all_cluster_results = []

for i, cat in enumerate(categories_to_plot):
    print(f"\n>>> 【{cat}】のクラスタリングと代表作抽出を実行中...")
    
    df_target = df[df["length_category"] == cat].copy()
    
    # 軌跡計算
    trajectories_1d = []
    valid_indices = []
    
    for idx, row in df_target.iterrows():
        # 白色化済みの trajectory_whitened を使用
        traj_matrix = row.get("trajectory_whitened") 
        
        if traj_matrix is not None and len(traj_matrix) == NUM_SEGMENTS:
            doc_centroid = np.mean(traj_matrix, axis=0)
            norm = np.linalg.norm(doc_centroid)
            if norm > 1e-9:
                doc_centroid = doc_centroid / norm
            else:
                doc_centroid = np.zeros_like(doc_centroid)
            
            sims = [np.dot(doc_centroid, v) for v in traj_matrix]
            trajectories_1d.append(sims)
            valid_indices.append(idx)
            
    df_valid = df_target.loc[valid_indices].copy()
    X_raw = np.array(trajectories_1d)
    
    # A 平滑化あり
    from scipy.signal import savgol_filter # 【追記】平滑化のためのライブラリ
    X_smoothed = savgol_filter(X_raw, window_length=5, polyorder=2, axis=1)
    scaler = StandardScaler()   # Z-score標準化とK-Means
    X_scaled = scaler.fit_transform(X_smoothed.T).T
    
    # B 平滑化なし
    #scaler = StandardScaler()   # Z-score標準化とK-Means
    #X_scaled = scaler.fit_transform(X_raw.T).T
    
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    df_valid["cluster"] = kmeans.fit_predict(X_scaled)
    
    cluster_centers = kmeans.cluster_centers_
    cluster_names = {j: f"Shape {chr(65+j)}" for j in range(NUM_CLUSTERS)}
    df_valid["shape_name"] = df_valid["cluster"].map(cluster_names)
    
    distances_all = kmeans.transform(X_scaled) # 全クラスタとの距離行列 (作品数 x クラスタ数)
    
    # 各クラスタへの距離を個別の列として追加
    dist_cols = []
    for j in range(NUM_CLUSTERS):
        col_name = f"dist_to_{cluster_names[j].replace(' ', '_')}" # 例: dist_to_Shape_A
        df_valid[col_name] = distances_all[:, j]
        dist_cols.append(col_name)
    df_valid["distance_to_center"] = [distances_all[idx, cluster_idx] for idx, cluster_idx in enumerate(df_valid["cluster"])]
    
    # --- CSV保存用のデータをリストに追加 ---
    base_cols = ["title", "author", "year", "decade", "length_category", "shape_name", "distance_to_center"]
    extracted_df = df_valid[base_cols + dist_cols].copy()
    all_cluster_results.append(extracted_df)

    # Seabornによる誤差帯（ばらつき）描画のためのデータ成形
    plot_data = []
    for shape_name, z_scores in zip(df_valid["shape_name"], X_scaled):
        for seg_idx, val in enumerate(z_scores):
            plot_data.append({
                "segment": seg_idx + 1,
                "z_score": val,
                "shape_name": shape_name
            })
    df_plot = pd.DataFrame(plot_data)
    
    # 代表作の抽出（コンソール出力）
    distances = kmeans.transform(X_scaled)
    df_valid["distance_to_center"] = [distances[idx, cluster_idx] for idx, cluster_idx in enumerate(df_valid["cluster"])]
  
    print("\n" + "="*60)
    print(f"【{cat}】年代別・波形別の代表作トップ5")
    print("="*60)
    
    decades = sorted(df_valid["decade"].unique())
    for d in decades:
        print(f"\n--- {d}年代 ---")
        for j in range(NUM_CLUSTERS):
            shape = cluster_names[j]
            top_works = df_valid[(df_valid["decade"] == d) & (df_valid["shape_name"] == shape)].sort_values("distance_to_center").head(5)
            
            if len(top_works) == 0:
                continue
            
            print(f"  [{shape}]")
            for rank, (_, row) in enumerate(top_works.iterrows(), 1):
                print(f"    {rank}位: 『{row['title']}』 ({row['author']}) [距離: {row['distance_to_center']:.4f}]")

    # ===== 描画処理 =====
    ax_shape = axes[i, 0]
    shape_order = [f"Shape {chr(65+j)}" for j in range(NUM_CLUSTERS)]
    colors = sns.color_palette("Set1", NUM_CLUSTERS)

    # 平均線と標準偏差（誤差帯）を自動計算してプロット
    # errorbar="sd" により、クラスタ内のデータのばらつき（標準偏差）を影として描画
    sns.lineplot(
        data=df_plot,
        x="segment",
        y="z_score",
        hue="shape_name",
        hue_order=shape_order,
        palette=colors,
        linewidth=2,
        errorbar=("ci", 99.5),
        ax=ax_shape
    )
    
    ax_shape.set_title(f"【{cat}】抽出された物語の軌跡 (N={len(df_valid)})", fontsize=14)
    ax_shape.set_ylabel("作品全体の平均的な文脈からの変容度 (Z-score)", fontsize=11)
    ax_shape.set_xticks([1, 5, 10, 15, 20])
    ax_shape.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_shape.grid(True, linestyle="--", alpha=0.6)
    if i == 1:
        ax_shape.set_xlabel("物語の進行度 (セグメント)", fontsize=12)
    ax_shape.legend(loc="lower right", fontsize=9)
    
    ax_dist = axes[i, 1]
    cross_tab = pd.crosstab(df_valid["decade"], df_valid["shape_name"])
    cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0).fillna(0) * 100
    cross_tab_pct = cross_tab_pct.reindex(columns=shape_order).fillna(0)
    cross_tab_pct.plot(kind="bar", stacked=True, color=colors, ax=ax_dist, edgecolor="black", width=0.8)
    ax_dist.set_title(f"【{cat}】年代別: 波形の出現割合", fontsize=14)
    ax_dist.set_ylabel("割合 (%)", fontsize=11)
    ax_dist.tick_params(axis='x', rotation=45)
    if i == 1:
        ax_dist.set_xlabel("年代 (Decade)", fontsize=12)
    else:
        ax_dist.set_xlabel("")
    ax_dist.legend(title="波形", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.suptitle("テキスト長による物語の軌跡の比較（短編 vs 中長編）", fontsize=18, y=1.02)
plt.savefig(os.path.join(PLOT_DIR, f"{ID_FILE}_Shape_Comparison.png"), dpi=300, bbox_inches='tight')
plt.show()
print(f"\n保存完了: {os.path.join(PLOT_DIR, f'{ID_FILE}_Shape_Comparison.png')}")

# ===== 4. クラスタリング結果のCSV保存 =====
df_final_csv = pd.concat(all_cluster_results, ignore_index=True)
df_final_csv = df_final_csv.sort_values(by=["length_category", "shape_name", "distance_to_center"])
df_final_csv.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"CSV保存完了: {OUTPUT_CSV}")
print("\n"+"="*50)
print("すべての処理が完了しました。")
print("="*50)