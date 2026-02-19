# pretrain_classifier_mask_curriculum.py
import random
import argparse
import pandas as pd
import numpy as np
import torch

from data_store import FounderDataStore
from get_observation import FounderState
from Networks import Classifier

# ------------------ Config ------------------
STATE_DIM = 1543
DEVICE = torch.device("cpu")
SLOTS = ["edu", "role", "exec", "industry", "depth"]

# curriculum phases (fraction of total epochs)
CURRICULUM = [
    {"name": "full",   "min_k": 5, "max_k": 5, "frac": 0.3},
    {"name": "medium", "min_k": 2, "max_k": 4, "frac": 0.4},
    {"name": "light",  "min_k": 1, "max_k": 2, "frac": 0.3},
]

# ------------------------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_masked_state(fid, data_store, k):
    """
    随机选择 k 个 slot query，其余置零
    这是对 RL 中“中途 stop state”的直接模拟
    """
    state = FounderState(fid, data_store)
    slots = random.sample(SLOTS, k=k)
    for s in slots:
        state.query(s)
    return state.get_state_vector()


def get_curriculum_phase(epoch, total_epochs):
    """
    根据 epoch 返回当前 curriculum phase
    """
    t = epoch / total_epochs
    acc = 0.0
    for phase in CURRICULUM:
        acc += phase["frac"]
        if t <= acc:
            return phase
    return CURRICULUM[-1]


def eval_on_full_info(df, data_store, clf, thr=0.3):
    """
    sanity check：full-info 上不要太离谱
    """
    tp = fp = tn = fn = 0
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

            if pred == 1 and y == 1: tp += 1
            elif pred == 1 and y == 0: fp += 1
            elif pred == 0 and y == 0: tn += 1
            else: fn += 1

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return {
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="pretrained_classifier.pt")
    args = parser.parse_args()

    set_seed(args.seed)

    # ---- data ----
    train_df = pd.read_csv("labels_train_clean.csv")
    val_df = pd.read_csv("data2/labels_val.csv")
    data_store = FounderDataStore("data2")

    # ---- model ----
    clf = Classifier(STATE_DIM).to(DEVICE)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # ---- training ----
    for epoch in range(args.epochs):
        phase = get_curriculum_phase(epoch, args.epochs)
        clf.train()

        losses = []

        for _, row in train_df.iterrows():
            fid = row["founder_uuid"]
            y = torch.tensor(row["success"], dtype=torch.float32).to(DEVICE)

            k = random.randint(phase["min_k"], phase["max_k"])
            x_np = build_masked_state(fid, data_store, k)
            x = torch.from_numpy(x_np).float().to(DEVICE)

            logit = clf(x)
            loss = loss_fn(logit, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        avg_loss = float(np.mean(losses))

        metrics = eval_on_full_info(val_df, data_store, clf)

        print(
            f"[Epoch {epoch+1:02d}/{args.epochs}] "
            f"phase={phase['name']} "
            f"k=[{phase['min_k']},{phase['max_k']}] "
            f"loss={avg_loss:.4f} | "
            f"val_acc={metrics['accuracy']:.3f} "
            f"val_f05~={metrics['precision']:.3f}/{metrics['recall']:.3f}"
        )

    # ---- save ----
    torch.save(
        {"state_dict": clf.state_dict(), "config": vars(args)},
        args.save_path
    )
    print(f"\nSaved pretrained classifier to {args.save_path}")


if __name__ == "__main__":
    main()
