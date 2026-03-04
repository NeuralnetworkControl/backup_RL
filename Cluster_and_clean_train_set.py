# cluster_and_clean_trainset.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_store import FounderDataStore
from get_observation import FounderState

# ------------------ Config ------------------
DATA_DIR = "data"
TRAIN_LABELS = "data/labels_train.csv"
OUT_CLEAN_TRAIN = "data/labels_train_clean.csv"
OUT_FIG = "cluster_vis.png"

N_CLUSTERS = 8
OUTLIER_STD_K = 1.2
RANDOM_SEED = 42
# --------------------------------------------


def build_state_vectors(train_df, data_store):
    xs, ys, fids = [], [], []

    for _, row in train_df.iterrows():
        fid = row["founder_uuid"]
        y = int(row["success"])

        state = FounderState(fid, data_store)

        # ✅ 关键：query 所有 slot
        for slot in ["edu", "role", "exec", "industry", "depth"]:
            state.query(slot)

        x = state.get_state_vector()

        xs.append(x)
        ys.append(y)
        fids.append(fid)

    return np.stack(xs), np.array(ys), np.array(fids)



def main():
    np.random.seed(RANDOM_SEED)

    # ---- load data ----
    train_df = pd.read_csv(TRAIN_LABELS)
    data_store = FounderDataStore(data_dir=DATA_DIR)

    print(f"Loaded train set: {len(train_df)} samples")

    # ---- build feature matrix ----
    X, y, fids = build_state_vectors(train_df, data_store)

    # ---- standardize ----
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ---- clustering ----
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_SEED,
        n_init=10
    )
    cluster_id = kmeans.fit_predict(Xs)
    centers = kmeans.cluster_centers_

    # ---- compute distances ----
    dists = np.linalg.norm(Xs - centers[cluster_id], axis=1)

    # ---- outlier detection (cluster-wise) ----
    keep_mask = np.ones(len(Xs), dtype=bool)

    for k in range(N_CLUSTERS):
        idx = np.where(cluster_id == k)[0]
        if len(idx) < 10:
            # 太小的 cluster 直接全丢
            keep_mask[idx] = False
            continue

        d_k = dists[idx]
        thr = d_k.mean() + OUTLIER_STD_K * d_k.std()
        keep_mask[idx] = d_k <= thr

    n_removed = int((~keep_mask).sum())
    print(f"Removed outliers: {n_removed}")

    # ---- PCA for visualization ----
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X2 = pca.fit_transform(Xs)

    # ---- plot ----
    plt.figure(figsize=(8, 6))

    # label = 0 → orange
    idx0 = (y == 0) & keep_mask
    plt.scatter(X2[idx0, 0], X2[idx0, 1],
                s=8, c="orange", alpha=0.6, label="fail (0)")

    # label = 1 → blue
    idx1 = (y == 1) & keep_mask
    plt.scatter(X2[idx1, 0], X2[idx1, 1],
                s=8, c="blue", alpha=0.6, label="success (1)")

    # outliers → red x
    idx_out = ~keep_mask
    plt.scatter(X2[idx_out, 0], X2[idx_out, 1],
                s=15, c="red", marker="x", label="outliers")

    plt.legend()
    plt.title("Clustering visualization (PCA)")
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.close()

    print(f"Saved visualization to {OUT_FIG}")

    # ---- rebuild clean train set ----
    clean_df = train_df.iloc[keep_mask].reset_index(drop=True)

    n_success = int(clean_df["success"].sum())
    n_fail = int((clean_df["success"] == 0).sum())

    clean_df.to_csv(OUT_CLEAN_TRAIN, index=False)

    print("Clean train set saved:", OUT_CLEAN_TRAIN)
    print(f"After cleaning: success={n_success}, fail={n_fail}")


if __name__ == "__main__":
    main()
