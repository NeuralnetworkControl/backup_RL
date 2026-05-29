
from data_store import FounderDataStore
from get_observation import FounderState
from baseline_myoptic_no_rollout import loop_per_founder_myopic
from Networks import PolicyNet, Classifier
from network_trainers import train_policy_from_spv, train_classifier_from_samples
from main_loop import (
    loop_per_founder, rollout_final_state, eval_on_val, sample_list
)
from main_loop import INFO_MAP, STOP_ACTION
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from policy_diagnostics import PolicyDiagnostics



# ------------------ Config ------------------
STATE_DIM = 1543
N_INFO_ACTIONS = 5
ACTION_DIM = N_INFO_ACTIONS + 1  # infos + stop
DEVICE = torch.device("cuda")



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(save_dir: Path, name: str,
                    policy, clf, pol_opt, clf_opt,
                    step: int, epoch: int, config: dict, extra: dict | None = None):
    payload = {
        "step": step,
        "epoch": epoch,
        "config": config,
        "policy_state_dict": policy.state_dict(),
        "clf_state_dict": clf.state_dict(),
        "pol_opt_state_dict": pol_opt.state_dict() if pol_opt is not None else None,
        "clf_opt_state_dict": clf_opt.state_dict() if clf_opt is not None else None,
        "extra": extra or {},
    }
    path = save_dir / f"{name}.pt"
    torch.save(payload, path)
    torch.save(payload, save_dir / "latest.pt")


def parse_args():
    p = argparse.ArgumentParser()

    # 训练集抽样：指定 success/fail founder 数量（None 或 -1 表示全量）
    p.add_argument("--n_success_train", type=int, default=-1,
                   help="训练中使用的 success founder 数量（-1 表示全量）")
    p.add_argument("--n_fail_train", type=int, default=-1,
                   help="训练中使用的 fail founder 数量（-1 表示全量）")

    # 把原 labels_val.csv 按比例切成 val/test（默认对半）
    p.add_argument("--val_ratio", type=float, default=0.5,
                   help="从原 labels_val.csv 中划给 validation 的比例（其余为 test），默认 0.5")

    # 训练过程中每隔多少次 update 做一次 validation 评估
    p.add_argument("--eval_every", type=int, default=1,
                   help="每多少次 UPDATE 做一次 validation 评估（默认每次都评估）")

    # 其他
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def stratified_split(df: pd.DataFrame, label_col: str, val_ratio: float, seed: int):
    """
    按 label_col 分层抽样，保持 success/fail 比例不变地拆成 val/test。
    不依赖 sklearn，避免环境没有 sklearn 的情况。
    """
    assert 0.0 < val_ratio < 1.0
    rng = np.random.default_rng(seed)

    parts_val = []
    parts_test = []
    for y, g in df.groupby(label_col):
        idx = np.arange(len(g))
        rng.shuffle(idx)
        n_val = int(round(len(g) * val_ratio))
        val_idx = idx[:n_val]
        test_idx = idx[n_val:]
        parts_val.append(g.iloc[val_idx])
        parts_test.append(g.iloc[test_idx])

    val_df = pd.concat(parts_val, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(parts_test, ignore_index=True).sample(frac=1, random_state=seed + 1).reset_index(drop=True)
    return val_df, test_df

def eval_and_print_TP(
    test_df,
    data_store,
    policy,
    clf,
    device,
    max_steps,
    greedy,
    min_queries,
    thr,
    max_print  # 可选：最多打印多少个
):
    """
    打印 test set 中所有 TP 的 founder：
      - founder_uuid
      - prob
      - rollout actions
    """
    tp_records = []

    policy.eval()
    clf.eval()

    for _, row in test_df.iterrows():
        fid = row["founder_uuid"]
        y_true = int(row["success"])

        # ---- rollout + 记录 actions ----
        state = FounderState(fid, data_store)
        actions = []
        queried = 0

        for step in range(max_steps):
            x = torch.from_numpy(state.get_state_vector()).float().to(device)
            pi = policy(x).detach().cpu().numpy()

            a = int(pi.argmax()) if greedy else int(np.random.choice(len(pi), p=pi))

            # early-stop constraint
            if a == STOP_ACTION and queried < min_queries:
                pi[STOP_ACTION] = -1
                a = int(pi.argmax())

            if a == STOP_ACTION:
                actions.append({"step": step, "action": "stop"})
                break

            action_name = INFO_MAP[a]
            actions.append({"step": step, "action": action_name})
            state.query(action_name)
            queried += 1

        # ---- classifier prediction ----
        x_final = torch.from_numpy(state.get_state_vector()).float().to(device)
        with torch.no_grad():
            logit = clf(x_final)
            prob = float(torch.sigmoid(logit).item())
            y_pred = 1 if prob >= thr else 0

        # ---- TP only ----
        if y_true == 1 and y_pred == 1:
            tp_records.append({
                "founder_uuid": fid,
                "prob": prob,
                "actions": actions
            })

    # ---- 打印 ----
    print(f"\n=== TRUE POSITIVES ({len(tp_records)}) ===")

    for i, rec in enumerate(tp_records):
        if max_print is not None and i >= max_print:
            break

        print(f"\n[{i+1}] founder={rec['founder_uuid']}")
        print(f"    prob={rec['prob']:.4f}")
        print("    actions:")
        for a in rec["actions"]:
            print(f"      step {a['step']}: {a['action']}")

    return tp_records


def main():
    args = parse_args()

    # ---- run dir（每次跑自动新建）----
    run_id = time.strftime("%Y%m%d_%H%M%S")
    save_dir = Path("runs") / run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- seed ----
    seed = args.seed
    set_seed(seed)

    # ---- data ----
    train_df = pd.read_csv("labels_train_clean.csv")
    holdout_df = pd.read_csv("data2/labels_val.csv")  # 你说的“原本测试集”
    data_store = FounderDataStore(data_dir="data2")

    # 把 holdout 拆成 validation / test（比例保持不变）
    val_df, test_df = stratified_split(holdout_df, label_col="success", val_ratio=args.val_ratio, seed=seed)

    # ---- models ----
    policy = PolicyNet(STATE_DIM, ACTION_DIM).to(DEVICE)
    clf = Classifier(STATE_DIM).to(DEVICE)

    # ====== NEW: load pretrained classifier ======
    pretrain_path = "pretrained_classifier.pt"
    if Path(pretrain_path).exists():
        ckpt = torch.load(pretrain_path, map_location=DEVICE)
        clf.load_state_dict(ckpt["state_dict"])
        print(f"[INFO] Loaded pretrained classifier from {pretrain_path}")
    else:
        print("[INFO] No pretrained classifier found, training from scratch")

    pol_opt = torch.optim.Adam(policy.parameters(), lr=5e-5)
    clf_opt = torch.optim.Adam(clf.parameters(), lr=5e-5)

    # --------- FAST CONFIG ----------
    FAST_MAX_STEPS = 5
    FAST_MAX_DEPTH = 4
    FAST_N_ROLLOUTS = 10

    UPDATE_EVERY = 25
    POLICY_SAMPLE = 2000
    CLF_SAMPLE = 4000

    POLICY_BATCH = 512 # 128
    CLF_BATCH = 512
    EPOCHS_PER_UPDATE = 1

    REPLAY_MAX_SPV = 20000
    REPLAY_MAX_CLF = 40000

    REPLAY_MAX_FINAL = 20000
    FINAL_SAMPLE = 0
    USE_LLM = 1
    # --------------------------------

    config = {
        "seed": seed,
        "STATE_DIM": STATE_DIM,
        "ACTION_DIM": ACTION_DIM,
        "FAST_MAX_STEPS": FAST_MAX_STEPS,
        "FAST_MAX_DEPTH": FAST_MAX_DEPTH,
        "FAST_N_ROLLOUTS": FAST_N_ROLLOUTS,
        "UPDATE_EVERY": UPDATE_EVERY,
        "POLICY_SAMPLE": POLICY_SAMPLE,
        "CLF_SAMPLE": CLF_SAMPLE,
        "POLICY_BATCH": POLICY_BATCH,
        "CLF_BATCH": CLF_BATCH,
        "EPOCHS_PER_UPDATE": EPOCHS_PER_UPDATE,
        "REPLAY_MAX_SPV": REPLAY_MAX_SPV,
        "REPLAY_MAX_CLF": REPLAY_MAX_CLF,
        "REPLAY_MAX_FINAL": REPLAY_MAX_FINAL,
        "FINAL_SAMPLE": FINAL_SAMPLE,
        "device": str(DEVICE),

        # 新增：训练集抽样 & holdout 拆分配置
        "n_success_train": args.n_success_train,
        "n_fail_train": args.n_fail_train,
        "val_ratio_from_holdout": args.val_ratio,
        "eval_every": args.eval_every,
        "train_size_total": int(len(train_df)),
        "val_size": int(len(val_df)),
        "test_size": int(len(test_df)),
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ---- buffers ----
    policy_replay = []
    clf_replay = []
    final_buf = []  # {"x": state_vec, "y": label}

    # ---- choose subset for training ----
    pos_all = train_df[train_df["success"] == 1].reset_index(drop=True)
    neg_all = train_df[train_df["success"] == 0].reset_index(drop=True)

    if args.n_success_train is None or args.n_success_train < 0:
        pos_df = pos_all
    else:
        n_pos = min(args.n_success_train, len(pos_all))
        pos_df = pos_all.sample(n=n_pos, replace=False, random_state=seed).reset_index(drop=True)

    if args.n_fail_train is None or args.n_fail_train < 0:
        neg_df = neg_all
    else:
        n_neg = min(args.n_fail_train, len(neg_all))
        neg_df = neg_all.sample(n=n_neg, replace=False, random_state=seed + 1).reset_index(drop=True)

    # ---- training ----
    FOUNDER_EPOCHS = 2
    global_step = 0

    # ====== NEW: freeze classifier at early stage ======
    FREEZE_CLF_UPDATES = 10  # 你可以调：5~20 都合理
    clf_frozen = True

    for p in clf.parameters():
        p.requires_grad = False

    print(f"[INFO] Classifier frozen for first {FREEZE_CLF_UPDATES} updates")
    # ==================================================

    for ep in range(FOUNDER_EPOCHS):
        balanced_df = pd.concat([pos_df, neg_df], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=seed + 2000 * ep).reset_index(drop=True)

        # ===== INIT POLICY DIAGNOSTICS (ONLY ONCE) =====
        if ep == 0:
            N_EVAL_STATES = 200
            eval_fids = balanced_df["founder_uuid"].sample(
                n=N_EVAL_STATES, random_state=seed
            ).tolist()

            eval_states = []
            for fid_eval in eval_fids:
                state_eval = FounderState(fid_eval, data_store)
                for s in ["edu", "role", "exec", "industry", "depth"]:
                    state_eval.query(s)

                eval_states.append(
                    torch.from_numpy(state_eval.get_state_vector())
                    .float()
                    .to(DEVICE)
                )

            diag = PolicyDiagnostics(
                eval_states=eval_states,
                device=DEVICE,
                use_llm=USE_LLM,  # 你原来用来开关 LLM 的 flag
            )

            print(f"[INFO] Policy diagnostics initialized with {len(eval_states)} states")
        # =============================================

        print(f"\n[FounderEpoch {ep + 1}/{FOUNDER_EPOCHS}] size={len(balanced_df)} "
              f"pos={int(balanced_df['success'].sum())} neg={int((balanced_df['success'] == 0).sum())}")
        print(f"[Holdout split] val={len(val_df)} test={len(test_df)} "
              f"(val_ratio={args.val_ratio})")

        for i, row in balanced_df.iterrows():
            fid = row["founder_uuid"]
            label = int(row["success"])

            S_Pai_V, S_Labels = loop_per_founder_myopic(
                fid, label, data_store, policy, clf, DEVICE,
                max_steps=FAST_MAX_STEPS,
                eps=0.1
            )

            final_x = rollout_final_state(
                fid, data_store, policy, DEVICE,
                max_steps=FAST_MAX_STEPS, greedy=True
            )
            final_buf.append({"x": final_x, "y": label})
            if len(final_buf) > REPLAY_MAX_FINAL:
                final_buf = final_buf[-REPLAY_MAX_FINAL:]

            policy_replay.extend(S_Pai_V)
            clf_replay.extend(S_Labels)

            if len(policy_replay) > REPLAY_MAX_SPV:
                policy_replay = policy_replay[-REPLAY_MAX_SPV:]
            if len(clf_replay) > REPLAY_MAX_CLF:
                clf_replay = clf_replay[-REPLAY_MAX_CLF:]

            # ---- update ----
            if (i + 1) % UPDATE_EVERY == 0:
                global_step += 1

                # ====== NEW: unfreeze classifier ======
                if clf_frozen and global_step >= FREEZE_CLF_UPDATES:
                    for p in clf.parameters():
                        p.requires_grad = True
                    clf_frozen = False
                    print(f"[INFO] Unfroze classifier at update {global_step}")
                # =====================================

                spv_batch = sample_list(policy_replay, POLICY_SAMPLE)
                clf_batch = sample_list(clf_replay, CLF_SAMPLE)

                final_batch = sample_list(final_buf, FINAL_SAMPLE)
                clf_batch = clf_batch + final_batch

                pol_loss = train_policy_from_spv(
                    policy, spv_batch, DEVICE,
                    batch_size=POLICY_BATCH, epochs=EPOCHS_PER_UPDATE, opt=pol_opt
                )

                diag.log(policy, update=global_step)

                if not clf_frozen:
                    clf_loss, clf_acc = train_classifier_from_samples(
                        clf, clf_batch, DEVICE,
                        batch_size=CLF_BATCH, epochs=EPOCHS_PER_UPDATE, opt=clf_opt
                    )
                else:
                    clf_loss, clf_acc = 0.0, 0.0

                log = (f"[{i + 1}/{len(balanced_df)}] "
                       f"replay_spv={len(policy_replay)} replay_clf={len(clf_replay)} | "
                       f"pol_loss={pol_loss:.4f} clf_loss={clf_loss:.4f} clf_acc={clf_acc:.3f}")
                print(log)

                # ---- validation eval during training ----
                if args.eval_every > 0 and (global_step % args.eval_every == 0):
                    policy.eval()
                    clf.eval()
                    metrics_val = eval_on_val(val_df, data_store, policy, clf, DEVICE)
                    print(f"  [VAL @update {global_step}] "
                          f"Precision={metrics_val['precision']:.4f} "
                          f"Recall={metrics_val['recall']:.4f} "
                          f"F0.5={metrics_val['f0.5']:.4f} "
                          f"Acc={metrics_val['accuracy']:.4f}")
                    policy.train()
                    clf.train()

    # ---- final eval: val + test ----
    policy.eval()
    clf.eval()

    metrics_val = eval_on_val(val_df, data_store, policy, clf, DEVICE)
    print("\n=== Final Validation ===")
    print("Confusion:", metrics_val["tp"], metrics_val["fp"], metrics_val["tn"], metrics_val["fn"])
    print("Precision:", metrics_val["precision"])
    print("Recall:", metrics_val["recall"])
    print("F0.5:", metrics_val["f0.5"])
    print("Accuracy:", metrics_val["accuracy"])

    metrics_test = eval_on_val(test_df, data_store, policy, clf, DEVICE)
    print("\n=== Final Test ===")
    print("Confusion:", metrics_test["tp"], metrics_test["fp"], metrics_test["tn"], metrics_test["fn"])
    print("Precision:", metrics_test["precision"])
    print("Recall:", metrics_test["recall"])
    print("F0.5:", metrics_test["f0.5"])
    print("Accuracy:", metrics_test["accuracy"])

    tp_records = eval_and_print_TP(
        test_df,
        data_store,
        policy,
        clf,
        DEVICE,
        max_steps=FAST_MAX_STEPS,
        greedy=True,
        min_queries=3,
        thr=0.3,
        max_print=20  # 先别全打，容易刷屏
    )

    save_checkpoint(
        save_dir=save_dir,
        name="final_model",
        policy=policy,
        clf=clf,
        pol_opt=pol_opt,
        clf_opt=clf_opt,
        step=global_step,
        epoch=FOUNDER_EPOCHS,
        config=config,
    )

    diag.save(
        "policy_metrics_with_llm.jsonl" if USE_LLM
        else "policy_metrics_no_llm.jsonl"
    )


if __name__ == "__main__":
    main()
# python train.py --n_success_train 100 --n_fail_train 500 --val_ratio 0.5 --eval_every 5
# python train.py --n_success_train 270 --n_fail_train 2730 --val_ratio 0.5 --eval_every 5
# python train.py  --val_ratio 0.5 --eval_every 5
# python train.py --n_success_train 270 --n_fail_train 2730 --val_ratio 0.5 --eval_every 5 --seed 123
