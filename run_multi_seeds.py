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
from llm_next_action_supervisor import llm_prefer_next_actions_from_merged
from main_loop import (
    INFO_MAP, STOP_ACTION, LLM_BIAS, UNCERTAIN_DELTA, LLM_CACHE
)


# ------------------ Config ------------------
STATE_DIM = 1543
N_INFO_ACTIONS = 5
ACTION_DIM = N_INFO_ACTIONS + 1
DEVICE = torch.device("cpu")
# --------------------------------------------


def apply_test_time_llm_bias(pi, state, fid, actions_taken, use_llm, merged_csv_path):
    if not use_llm:
        return pi

    pi = pi.copy()
    pi_info = pi[:N_INFO_ACTIONS].copy()
    pi_stop = pi[STOP_ACTION]

    order = np.argsort(-pi_info)
    uncertain = (pi_info[order[0]] - pi_info[order[1]]) < UNCERTAIN_DELTA
    if not uncertain:
        return pi

    mask = tuple(int(state.observed.get(s, 0)) for s in INFO_MAP.values())
    cache_key = (merged_csv_path, fid, mask)

    if cache_key in LLM_CACHE:
        prefer = LLM_CACHE[cache_key]
    else:
        prefer = llm_prefer_next_actions_from_merged(
            fid=fid,
            actions_taken=actions_taken,
            merged_csv_path=merged_csv_path,
        )
        LLM_CACHE[cache_key] = prefer

    info_mass = float(pi_info.sum())
    for k, slot in INFO_MAP.items():
        if slot in prefer:
            pi_info[k] += LLM_BIAS

    pi_info = np.clip(pi_info, 1e-8, None)
    pi_info = pi_info / pi_info.sum() * info_mass

    pi[:N_INFO_ACTIONS] = pi_info
    pi[STOP_ACTION] = pi_stop
    return pi


def predict_one_for_eval(
    fid,
    data_store,
    policy,
    clf,
    device,
    max_steps,
    greedy,
    min_queries,
    thr,
    use_llm=False,
    merged_csv_path="merged.csv",
):
    state = FounderState(fid, data_store)
    queried = 0
    used_actions = set()
    actions_taken = []

    for step in range(max_steps):
        x = torch.from_numpy(state.get_state_vector()).float().to(device)
        pi = policy(x).detach().cpu().numpy()
        pi = apply_test_time_llm_bias(
            pi, state, fid, actions_taken, use_llm, merged_csv_path
        )

        a = int(pi.argmax()) if greedy else int(np.random.choice(len(pi), p=pi))

        if a == STOP_ACTION and queried < min_queries:
            pi[STOP_ACTION] = -1
            a = int(pi.argmax())

        if a == STOP_ACTION:
            actions_taken.append({"step": step, "action_dim": a, "action": "stop"})
            break

        action_name = INFO_MAP[a]
        actions_taken.append({"step": step, "action_dim": a, "action": action_name})
        used_actions.add(a)
        state.query(action_name)
        queried += 1

    x_final = torch.from_numpy(state.get_state_vector()).float().to(device)
    logit = clf(x_final)
    prob = float(torch.sigmoid(logit).item())
    pred = 1 if prob >= thr else 0

    return pred, prob, used_actions


def eval_on_val_for_test(
    val_df,
    data_store,
    policy,
    clf,
    device,
    max_steps=5,
    greedy=True,
    min_queries=1,
    thr=0.3,
    use_llm=False,
    merged_csv_path="merged.csv",
):
    if not use_llm:
        return eval_on_val(
            val_df, data_store, policy, clf, device,
            max_steps=max_steps, greedy=greedy, min_queries=min_queries, thr=thr,
        )

    tp = fp = tn = fn = 0
    info_counts = []
    action_counter = {}

    policy.eval()
    clf.eval()

    with torch.no_grad():
        for _, row in val_df.iterrows():
            fid = row["founder_uuid"]
            y = int(row["success"])

            pred, prob, used_actions = predict_one_for_eval(
                fid, data_store, policy, clf, device,
                max_steps=max_steps, greedy=greedy, min_queries=min_queries, thr=thr,
                use_llm=use_llm, merged_csv_path=merged_csv_path,
            )

            info_counts.append(len(used_actions))
            for a in used_actions:
                action_counter[a] = action_counter.get(a, 0) + 1

            if pred == 1 and y == 1:
                tp += 1
            elif pred == 1 and y == 0:
                fp += 1
            elif pred == 0 and y == 0:
                tn += 1
            else:
                fn += 1

    total = tp + fp + tn + fn
    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f05 = 1.25 * precision * recall / (0.25 * precision + recall + 1e-9)
    acc = (tp + tn) / (total + 1e-9)

    fpr = fp / (fp + tn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    fnr = fn / (fn + tp + 1e-9)
    total_info_used = sum(info_counts)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
        "accuracy": acc,
        "avg_step": total_info_used / max(len(info_counts), 1),
        "total_info_used": total_info_used,
        "info_distribution": info_counts,
        "action_usage": action_counter,
        "rates_total": {
            "tp_rate_total": tp / (total + 1e-9),
            "fp_rate_total": fp / (total + 1e-9),
            "tn_rate_total": tn / (total + 1e-9),
            "fn_rate_total": fn / (total + 1e-9),
        },
        "rates_common": {"TPR": recall, "FPR": fpr, "TNR": tnr, "FNR": fnr},
    }


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
    use_llm=False,
    merged_csv_path="merged.csv",
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
        actions_taken = []
        queried = 0

        for step in range(max_steps):
            x = torch.from_numpy(state.get_state_vector()).float().to(device)
            pi = policy(x).detach().cpu().numpy()
            pi = apply_test_time_llm_bias(
                pi, state, fid, actions_taken, use_llm, merged_csv_path
            )

            a = int(pi.argmax()) if greedy else int(np.random.choice(len(pi), p=pi))

            if a == STOP_ACTION and queried < min_queries:
                pi[STOP_ACTION] = -1
                a = int(pi.argmax())

            if a == STOP_ACTION:
                actions.append("stop")
                actions_taken.append({"step": step, "action_dim": a, "action": "stop"})
                break

            action_name = INFO_MAP[a]
            actions.append(action_name)
            actions_taken.append({"step": step, "action_dim": a, "action": action_name})
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
    p.add_argument("--labels_pool", type=str, default="data/labels_test.csv",
                   help="CSV containing founder_uuid + success (holdout pool)")
    p.add_argument("--data_dir", type=str, default="data",
                   help="data directory for FounderDataStore")

    # test set 构造
    p.add_argument("--n_success", type=int, default=180)
    p.add_argument("--n_fail", type=int, default=1820)
    p.add_argument("--test_seeds", type=int, nargs="+",
                   default=list(range(1, 11)),
                   help="list of seeds for test set sampling")

    # 输出
    p.add_argument("--out_csv", type=str, default="multi_test_results.csv")
    p.add_argument("--use_llm_eval", action="store_true",
                   help="enable LLM supervisor during test-time rollout")
    p.add_argument("--merged_csv_path", type=str, default="merged.csv",
                   help="raw merged CSV used by the LLM supervisor")

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
    print(f"LLM eval: {args.use_llm_eval}")
    if args.use_llm_eval:
        print(f"LLM merged CSV: {args.merged_csv_path}")
    print(f"Test seeds: {args.test_seeds}\n")

    for idx, seed in enumerate(args.test_seeds):
        print(f"[Test seed {seed}] building test set...")

        test_df = build_test_set(
            df_all=labels_df,
            n_success=args.n_success,
            n_fail=args.n_fail,
            seed=seed,
        )

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
                use_llm=args.use_llm_eval,
                merged_csv_path=args.merged_csv_path,
            )

            with open("decision_paths_last_seed.json", "w", encoding="utf-8") as f:
                json.dump(path_records, f, indent=2)

            print("Saved decision paths to decision_paths_last_seed.json")

        metrics = eval_on_val_for_test(
            test_df,
            data_store,
            policy,
            clf,
            DEVICE,
            min_queries=1,
            use_llm=args.use_llm_eval,
            merged_csv_path=args.merged_csv_path,
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
# python run_multi_seeds.py --ckpt runs/model_baseline2_myopic_RL/final_model.pt --test_seeds 0 1 2 3 4 5 6 7 8 9
# python run_multi_seeds.py --ckpt runs/20260514_163837/final_model.pt --test_seeds 0 1 2 3 4 5 6 7 8 9