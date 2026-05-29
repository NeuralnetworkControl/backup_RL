import pandas as pd
import torch
import numpy as np
from data_store import FounderDataStore
from get_observation import FounderState
from Networks import Classifier

# ------------------ Config ------------------
STATE_DIM = 1543
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SLOTS = ["edu", "role", "exec", "industry", "depth"]
THR = 0.3

BATCH_SIZE = 16
EPOCHS = 50
VAL_RATIO = 0.1
SEED = 42
# -------------------------------------------


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(df: pd.DataFrame, label_col: str, val_ratio: float, seed: int):
    """
    与 train.py 中一致的分层切分方法
    """
    rng = np.random.default_rng(seed)
    parts_train, parts_val = [], []

    for y, g in df.groupby(label_col):
        idx = np.arange(len(g))
        rng.shuffle(idx)
        n_val = int(round(len(g) * val_ratio))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        parts_val.append(g.iloc[val_idx])
        parts_train.append(g.iloc[train_idx])

    train_df = pd.concat(parts_train, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    val_df = pd.concat(parts_val, ignore_index=True).sample(
        frac=1, random_state=seed + 1
    ).reset_index(drop=True)

    return train_df, val_df


def build_full_state(fid, data_store):
    state = FounderState(fid, data_store)
    for s in SLOTS:
        state.query(s)
    return state.get_state_vector()


def compute_loss(df, data_store, clf, loss_fn):
    """
    full-info validation loss（不做 threshold）
    """
    clf.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _, row in df.iterrows():
            x = torch.from_numpy(
                build_full_state(row["founder_uuid"], data_store)
            ).float().to(DEVICE)
            y = torch.tensor(row["success"], dtype=torch.float32).to(DEVICE)

            loss = loss_fn(clf(x).squeeze(), y)
            total_loss += loss.item()

    return total_loss / max(1, len(df))


def eval_full_info(df, data_store, clf):
    tp = fp = tn = fn = 0
    clf.eval()

    with torch.no_grad():
        for _, row in df.iterrows():
            x = torch.from_numpy(
                build_full_state(row["founder_uuid"], data_store)
            ).float().to(DEVICE)

            y = int(row["success"])
            prob = torch.sigmoid(clf(x)).item()
            pred = 1 if prob >= THR else 0

            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            else:
                fn += 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f05 = 1.25 * precision * recall / (0.25 * precision + recall + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
        "accuracy": acc,
    }


def main():
    set_seed(SEED)

    # ---- data ----
    full_train_df = pd.read_csv("data/labels_train.csv")
    test_df = pd.read_csv("data/labels_test.csv")  # 保持和你原来一致
    data_store = FounderDataStore("data")

    # ---- split train / val（来自 train.py 的逻辑）----
    train_df, val_df = stratified_split(
        full_train_df, label_col="success", val_ratio=VAL_RATIO, seed=SEED
    )

    print(f"[INFO] Train size = {len(train_df)}, Val size = {len(val_df)}")

    # ---- model ----
    clf = Classifier(STATE_DIM).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # ---- training ----
    for epoch in range(EPOCHS):
        clf.train()
        epoch_loss = 0.0
        batch_x, batch_y = [], []

        for _, row in train_df.iterrows():
            x = torch.from_numpy(
                build_full_state(row["founder_uuid"], data_store)
            ).float()
            y = torch.tensor(row["success"], dtype=torch.float32)

            batch_x.append(x)
            batch_y.append(y)

            if len(batch_x) == BATCH_SIZE:
                bx = torch.stack(batch_x).to(DEVICE)
                by = torch.stack(batch_y).to(DEVICE)

                loss = loss_fn(clf(bx).squeeze(), by)
                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                batch_x, batch_y = [], []

        # last batch
        if len(batch_x) > 0:
            bx = torch.stack(batch_x).to(DEVICE)
            by = torch.stack(batch_y).to(DEVICE)

            loss = loss_fn(clf(bx).squeeze(), by)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / max(1, len(train_df) // BATCH_SIZE)
        val_loss = compute_loss(val_df, data_store, clf, loss_fn)

        print(
            f"[Epoch {epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

    # ---- final eval on test ----
    metrics = eval_full_info(test_df, data_store, clf)
    print("\n=== Full-Info Baseline (Batch=16) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
