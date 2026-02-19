import argparse
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch

from data_store import FounderDataStore
from Networks import PolicyNet, Classifier
from main_loop import eval_on_val
from get_observation import FounderState
from main_loop import INFO_MAP, STOP_ACTION


# ------------------ Config ------------------
STATE_DIM = 1543
N_INFO_ACTIONS = 5
ACTION_DIM = N_INFO_ACTIONS + 1
DEVICE = torch.device("cpu")
# --------------------------------------------

def collect_decision_paths(
    test_df,
    data_store,
    policy,
    clf,
    device,
    max_steps=5,
    greedy=True,
    min_queries=1,
    thr=0.3,
    n_samples=20,
):
    """
    从 test_df 中随机选 10 TP + 10 TN，
    输出决策路径并返回记录列表
    """

    tp_records = []
    tn_records = []

    policy.eval()
    clf.eval()

    for _, row in test_df.iterrows():
        fid = row["founder_uuid"]
        y_true = int(row["success"])

        state = FounderState(fid, data_store)
        actions = []
        queried = 0

        for step in range(max_steps):
            x = torch.from_numpy(state.get_state_vector()).float().to(device)
            pi = policy(x).detach().cpu().numpy()

            a = int(pi.argmax()) if greedy else int(np.random.choice(len(pi), p=pi))

            if a == STOP_ACTION and queried < min_queries:
                pi[STOP_ACTION] = -1
                a = int(pi.argmax())

            if a == STOP_ACTION:
                actions.append("stop")
                break

            action_name = INFO_MAP[a]
            actions.append(action_name)
            state.query(action_name)
            queried += 1

        x_final = torch.from_numpy(state.get_state_vector()).float().to(device)
        with torch.no_grad():
            logit = clf(x_final)
            prob = float(torch.sigmoid(logit).item())
            y_pred = 1 if prob >= thr else 0

        record = {
            "fid": fid,
            "true_label": y_true,
            "pred": y_pred,
            "prob": prob,
            "actions": actions,
        }

        if y_true == 1 and y_pred == 1:
            tp_records.append(record)
        if y_true == 0 and y_pred == 0:
            tn_records.append(record)

    # 随机抽样
    tp_selected = random.sample(tp_records, min(n_samples, len(tp_records)))
    tn_selected = random.sample(tn_records, min(n_samples, len(tn_records)))

    return tp_selected + tn_selected


def load_model(ckpt_path: str, device):
    """
    加载已训练好的 policy + classifier
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    policy = PolicyNet(STATE_DIM, ACTION_DIM).to(device)
    clf = Classifier(STATE_DIM).to(device)

    policy.load_state_dict(ckpt["policy_state_dict"])
    clf.load_state_dict(ckpt["clf_state_dict"])

    policy.eval()
    clf.eval()

    return policy, clf


def build_test_set(
    df_all: pd.DataFrame,
    n_success: int,
    n_fail: int,
    seed: int,
):
    """
    构造一个可控的 test set
    """
    pos = df_all[df_all["success"] == 1]
    neg = df_all[df_all["success"] == 0]

    assert n_success <= len(pos), "n_success > available positive samples"
    assert n_fail <= len(neg), "n_fail > available negative samples"

    pos_df = pos.sample(n=n_success, replace=False, random_state=seed)
    neg_df = neg.sample(n=n_fail, replace=False, random_state=seed + 1)

    test_df = pd.concat([pos_df, neg_df], ignore_index=True)
    test_df = test_df.sample(frac=1, random_state=seed + 2).reset_index(drop=True)

    return test_df


def parse_args():
    p = argparse.ArgumentParser()

    # 模型
    p.add_argument("--ckpt", type=str, required=True,
                   help="path to trained checkpoint (.pt)")

    # 数据
    p.add_argument("--labels_pool", type=str, default="data2/labels_val.csv",
                   help="CSV containing founder_uuid + success (holdout pool)")
    p.add_argument("--data_dir", type=str, default="data2",
                   help="data directory for FounderDataStore")

    # test set 构造
    p.add_argument("--n_success", type=int, default=90)
    p.add_argument("--n_fail", type=int, default=910)
    p.add_argument("--test_seeds", type=int, nargs="+",
                   default=list(range(1, 11)),
                   help="list of seeds for test set sampling")

    # 输出
    p.add_argument("--out_csv", type=str, default="multi_test_results2.csv")

    return p.parse_args()


def main():
    args = parse_args()

    # ---- load data ----
    labels_df = pd.read_csv(args.labels_pool)
    data_store = FounderDataStore(data_dir=args.data_dir)

    # ---- load model ----
    policy, clf = load_model(args.ckpt, DEVICE)

    results = []

    print("\n=== Running multiple test sets ===")
    print(f"Model checkpoint: {args.ckpt}")
    print(f"Test config: n_success={args.n_success}, n_fail={args.n_fail}")
    print(f"Test seeds: {args.test_seeds}\n")

    for idx, seed in enumerate(args.test_seeds):
        # 只在最后一个 seed 时保存路径
        if idx == len(args.test_seeds) - 1:
            print("\nCollecting decision paths for last seed...")

            path_records = collect_decision_paths(
                test_df,
                data_store,
                policy,
                clf,
                DEVICE,
                max_steps=5,
                greedy=True,
                min_queries=1,
                thr=0.3,
                n_samples=20,
            )

            with open("decision_paths_last_seed.json", "w", encoding="utf-8") as f:
                json.dump(path_records, f, indent=2)

            print("Saved decision paths to decision_paths_last_seed.json")

        print(f"[Test seed {seed}] building test set...")

        test_df = build_test_set(
            df_all=labels_df,
            n_success=args.n_success,
            n_fail=args.n_fail,
            seed=seed,
        )

        metrics = eval_on_val(
            test_df,
            data_store,
            policy,
            clf,
            DEVICE,
            min_queries=1
        )

        rec = {
            "test_seed": seed,
            "n_success": args.n_success,
            "n_fail": args.n_fail,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f0.5": metrics["f0.5"],
            "accuracy": metrics["accuracy"],
            "total_info_used": metrics["total_info_used"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
        }

        results.append(rec)

        print(
            f"  Precision={rec['precision']:.4f} "
            f"Recall={rec['recall']:.4f} "
            f"F0.5={rec['f0.5']:.4f} "
            f"(TP={rec['tp']}, FP={rec['fp']})"
            f"Info_used={rec['total_info_used']:.4f} "
        )

    # ---- save results ----
    df = pd.DataFrame(results)
    out_path = Path(args.out_csv)
    df.to_csv(out_path, index=False)

    print("\n=== Summary (mean ± std) ===")
    summary = df[["precision", "recall", "f0.5", "total_info_used"]].agg(["mean", "std"])

    print(summary)

    print(f"\nSaved results to {out_path.resolve()}")


if __name__ == "__main__":
    main()

# python run_multi_seeds.py --ckpt runs/model_llm_supervisor/final_model.pt --test_seeds 0 1 2 3 4 5 6 7 8 9
# python run_multi_seeds.py --ckpt runs/model_without_llm/final_model.pt --test_seeds 0 1 2 3 4 5 6 7 8 9
