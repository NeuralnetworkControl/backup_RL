# pretrain_classifier_mask_curriculum.py
import argparse
import random

import numpy as np
import pandas as pd
import torch

from data_store import FounderDataStore
from get_observation import FounderState
from Networks import Classifier


# ------------------ Config ------------------
STATE_DIM = 1543
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SLOTS = ["edu", "role", "exec", "industry", "depth"]
FIXED_THRESHOLD = 0.3
BATCH_SIZE = 16

CURRICULUM = [
    {"name": "full", "min_k": 5, "max_k": 5, "frac": 0.3},
    {"name": "medium", "min_k": 2, "max_k": 4, "frac": 0.4},
    {"name": "light", "min_k": 1, "max_k": 2, "frac": 0.3},
]
# ------------------------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_masked_state(fid, data_store, k):
    state = FounderState(fid, data_store)
    slots = random.sample(SLOTS, k=k)
    for s in slots:
        state.query(s)
    return state.get_state_vector()


def get_curriculum_phase(epoch, total_epochs):
    t = epoch / total_epochs
    acc = 0.0
    for phase in CURRICULUM:
        acc += phase["frac"]
        if t <= acc:
            return phase
    return CURRICULUM[-1]


def stratified_split(df: pd.DataFrame, label_col: str, val_ratio: float, seed: int):
    """Split labels into train/validation while keeping class ratios stable."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []

    for _, g in df.groupby(label_col):
        idx = np.arange(len(g))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(g) * val_ratio)))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        train_parts.append(g.iloc[train_idx])
        val_parts.append(g.iloc[val_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    val_df = pd.concat(val_parts, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    return train_df, val_df


def f05_score(precision, recall):
    return 1.25 * precision * recall / (0.25 * precision + recall + 1e-9)


def eval_on_full_info(df, data_store, clf, thr=FIXED_THRESHOLD):
    """Evaluate with all slots queried and return confusion, F0.5, and prob stats."""
    tp = fp = tn = fn = 0
    pos_probs = []
    neg_probs = []
    clf.eval()

    with torch.no_grad():
        for _, row in df.iterrows():
            fid = row["founder_uuid"]
            y = int(row["success"])

            state = FounderState(fid, data_store)
            for s in SLOTS:
                state.query(s)

            x = torch.from_numpy(state.get_state_vector()).float().to(DEVICE)
            prob = torch.sigmoid(clf(x)).item()
            pred = 1 if prob >= thr else 0

            if y == 1:
                pos_probs.append(prob)
            else:
                neg_probs.append(prob)

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
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    all_probs = pos_probs + neg_probs

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f0.5": f05_score(precision, recall),
        "accuracy": acc,
        "prob_pos_mean": float(np.mean(pos_probs)) if pos_probs else 0.0,
        "prob_neg_mean": float(np.mean(neg_probs)) if neg_probs else 0.0,
        "prob_min": float(np.min(all_probs)) if all_probs else 0.0,
        "prob_max": float(np.max(all_probs)) if all_probs else 0.0,
    }


def eval_fixed_threshold(df, data_store, clf):
    metrics = eval_on_full_info(df, data_store, clf, thr=FIXED_THRESHOLD)
    return FIXED_THRESHOLD, metrics


def compute_loss(df, data_store, clf, loss_fn):
    """Full-info validation loss, matching baseline_1_oneshot-nn.py."""
    clf.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _, row in df.iterrows():
            state = FounderState(row["founder_uuid"], data_store)
            for s in SLOTS:
                state.query(s)

            x = torch.from_numpy(state.get_state_vector()).float().to(DEVICE)
            y = torch.tensor(row["success"], dtype=torch.float32).to(DEVICE)

            loss = loss_fn(clf(x).squeeze(), y)
            total_loss += loss.item()

    return total_loss / max(1, len(df))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--save_path", type=str, default="pretrained_classifier.pt")
    args = parser.parse_args()

    set_seed(args.seed)

    labels_train_df = pd.read_csv("data/labels_train.csv")
    train_df, val_df = stratified_split(
        labels_train_df,
        label_col="success",
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    test_df = pd.read_csv("data/labels_test.csv")
    data_store = FounderDataStore("data")

    clf = Classifier(STATE_DIM).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    print(f"[DATA] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    for epoch in range(args.epochs):
        phase = get_curriculum_phase(epoch, args.epochs)
        clf.train()
        epoch_loss = 0.0
        batch_x = []
        batch_y = []

        epoch_df = train_df.sample(frac=1, random_state=args.seed + epoch).reset_index(drop=True)

        for _, row in epoch_df.iterrows():
            fid = row["founder_uuid"]
            y = torch.tensor([float(row["success"])], dtype=torch.float32).to(DEVICE)

            k = random.randint(phase["min_k"], phase["max_k"])
            x_np = build_masked_state(fid, data_store, k)
            x = torch.from_numpy(x_np).float()

            batch_x.append(x)
            batch_y.append(y.cpu().squeeze(0))

            if len(batch_x) == args.batch_size:
                bx = torch.stack(batch_x).to(DEVICE)
                by = torch.stack(batch_y).to(DEVICE)

                loss = loss_fn(clf(bx).squeeze(), by)
                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_loss += loss.item()
                batch_x = []
                batch_y = []

        if len(batch_x) > 0:
            bx = torch.stack(batch_x).to(DEVICE)
            by = torch.stack(batch_y).to(DEVICE)

            loss = loss_fn(clf(bx).squeeze(), by)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss / max(1, len(train_df) // args.batch_size)
        val_loss = compute_loss(val_df, data_store, clf, loss_fn)
        epoch_thr, metrics = eval_fixed_threshold(val_df, data_store, clf)

        print(
            f"[Epoch {epoch + 1:02d}/{args.epochs}] "
            f"phase={phase['name']} "
            f"k=[{phase['min_k']},{phase['max_k']}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"thr={epoch_thr:.2f} "
            f"val_acc={metrics['accuracy']:.3f} "
            f"val_p/r/f05={metrics['precision']:.3f}/{metrics['recall']:.3f}/{metrics['f0.5']:.3f} "
            f"cm={metrics['tp']},{metrics['fp']},{metrics['tn']},{metrics['fn']} "
            f"prob_pos/neg={metrics['prob_pos_mean']:.3f}/{metrics['prob_neg_mean']:.3f} "
            f"prob_min/max={metrics['prob_min']:.3f}/{metrics['prob_max']:.3f}"
        )

    test_metrics = eval_on_full_info(test_df, data_store, clf, thr=FIXED_THRESHOLD)
    print("\n=== Final Test ===")
    print(f"threshold={FIXED_THRESHOLD:.2f}")
    print("Confusion:", test_metrics["tp"], test_metrics["fp"], test_metrics["tn"], test_metrics["fn"])
    print("Precision:", test_metrics["precision"])
    print("Recall:", test_metrics["recall"])
    print("F0.5:", test_metrics["f0.5"])
    print("Accuracy:", test_metrics["accuracy"])
    print("Prob pos/neg mean:", test_metrics["prob_pos_mean"], test_metrics["prob_neg_mean"])
    print("Prob min/max:", test_metrics["prob_min"], test_metrics["prob_max"])

    torch.save(
        {
            "state_dict": clf.state_dict(),
            "config": vars(args),
            "best_threshold": FIXED_THRESHOLD,
            "test_metrics": test_metrics,
        },
        args.save_path,
    )
    print(f"\nSaved pretrained classifier to {args.save_path}")


if __name__ == "__main__":
    main()
