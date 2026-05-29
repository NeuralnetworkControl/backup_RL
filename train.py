from data_store import FounderDataStore
from get_observation import FounderState
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

    # Training subset sampling. -1 means use all available founders.
    p.add_argument("--n_success_train", type=int, default=-1,
                   help="number of success founders to use for training; -1 means all")
    p.add_argument("--n_fail_train", type=int, default=-1,
                   help="number of fail founders to use for training; -1 means all")
    """

    # 训练集抽样：指定 success/fail founder 数量（None 或 -1 表示全量）
    p.add_argument("--n_success_train", type=int, default=-1,
                   help="训练中使用的 success founder 数量（-1 表示全量）")
                   help="训练中使用的 fail founder 数量（-1 表示全量）")

    """
    # Build validation from labels_train.csv; labels_test.csv is used only after training.
    p.add_argument("--cv_folds", type=int, default=5,
                   help="number of stratified cross-validation folds built from data/labels_train.csv")
    p.add_argument("--cv_fold", type=int, default=0,
                   help="which fold to use as validation; the other folds are used for training")

    # Validation/test monitoring cadence during training.
    p.add_argument("--eval_every", type=int, default=5,
                   help="every N updates, run validation evaluation with fixed threshold; 0 disables")
    """

    # 训练过程中每隔多少次 update 做一次 validation 评估
    p.add_argument("--eval_every", type=int, default=5,
                   help="每多少次 UPDATE 做一次 validation 评估（默认每次都评估）")
    """
    p.add_argument("--threshold_min", type=float, default=0.01,
                   help="minimum classifier threshold to try on validation")
    p.add_argument("--threshold_max", type=float, default=0.90,
                   help="maximum classifier threshold to try on validation")
    p.add_argument("--threshold_steps", type=int, default=30,
                   help="number of thresholds to try on validation")
    p.add_argument("--eval_threshold", type=float, default=0.3,
                   help="fixed classifier threshold used for all train/val/test evaluation")

    # 其他
    p.add_argument("--test_eval_every", type=int, default=5,
                   help="every N updates, print held-out test metrics for monitoring only; 0 disables")

    # Tunable training parameters. Defaults are conservative for an imbalanced
    # precision-oriented setup: keep policy learning steady and fine-tune the
    # pretrained classifier more slowly.
    p.add_argument("--policy_lr", type=float, default=5e-5)
    p.add_argument("--clf_lr", type=float, default=1e-5)
    p.add_argument("--founder_epochs", type=int, default=1)
    p.add_argument("--update_every", type=int, default=25)
    p.add_argument("--policy_sample", type=int, default=2000)
    p.add_argument("--clf_sample", type=int, default=4000)
    p.add_argument("--policy_batch", type=int, default=128)
    p.add_argument("--clf_batch", type=int, default=512)
    p.add_argument("--epochs_per_update", type=int, default=1)
    p.add_argument("--freeze_clf_updates", type=int, default=10)
    p.add_argument("--clf_target_update_every", type=int, default=5,
                   help="copy online classifier weights to target classifier every N updates; 0 disables")
    p.add_argument("--fast_max_steps", type=int, default=5)
    p.add_argument("--fast_max_depth", type=int, default=4)
    p.add_argument("--fast_n_rollouts", type=int, default=10)
    p.add_argument("--final_sample", type=int, default=1000)
    p.add_argument("--replay_max_spv", type=int, default=20000)
    p.add_argument("--replay_max_clf", type=int, default=40000)
    p.add_argument("--replay_max_final", type=int, default=20000)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def stratified_kfold_split(df: pd.DataFrame, label_col: str, n_folds: int, fold: int, seed: int):
    """Build one stratified CV split without requiring sklearn."""
    if n_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if not 0 <= fold < n_folds:
        raise ValueError(f"cv_fold must be in [0, {n_folds - 1}]")
    rng = np.random.default_rng(seed)

    parts_train = []
    parts_val = []
    for y, g in df.groupby(label_col):
        idx = np.arange(len(g))
        rng.shuffle(idx)
        fold_ids = np.arange(len(idx)) % n_folds
        val_idx = idx[fold_ids == fold]
        train_idx = idx[fold_ids != fold]
        parts_train.append(g.iloc[train_idx])
        parts_val.append(g.iloc[val_idx])

    train_df = pd.concat(parts_train, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(parts_val, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, val_df


def find_best_threshold(
    val_df,
    data_store,
    policy,
    clf,
    device,
    threshold_min,
    threshold_max,
    threshold_steps,
):
    if threshold_steps < 1:
        raise ValueError("threshold_steps must be at least 1")

    best_thr = threshold_min
    best_metrics = None
    for thr in np.linspace(threshold_min, threshold_max, threshold_steps):
        metrics = eval_on_val(val_df, data_store, policy, clf, device, thr=float(thr))
        if best_metrics is None or metrics["f0.5"] > best_metrics["f0.5"]:
            best_thr = float(thr)
            best_metrics = metrics

    return best_thr, best_metrics


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
    labels_train_df = pd.read_csv("data/labels_train.csv")
    test_df = pd.read_csv("data/labels_test.csv")
    data_store = FounderDataStore(data_dir="data")

    train_df, val_df = stratified_kfold_split(
        labels_train_df,
        label_col="success",
        n_folds=args.cv_folds,
        fold=args.cv_fold,
        seed=seed,
    )

    # ---- models ----
    policy = PolicyNet(STATE_DIM, ACTION_DIM).to(DEVICE)
    clf = Classifier(STATE_DIM).to(DEVICE)
    eval_threshold = args.eval_threshold

    # ====== NEW: load pretrained classifier ======
    pretrain_path = "pretrained_classifier.pt"
    if Path(pretrain_path).exists():
        ckpt = torch.load(pretrain_path, map_location=DEVICE)
        clf.load_state_dict(ckpt["state_dict"])
        print(f"[INFO] Loaded pretrained classifier from {pretrain_path}")
        print(f"[INFO] Fixed eval threshold = {eval_threshold:.4f}")
    else:
        print("[INFO] No pretrained classifier found, training from scratch")

    clf_target = Classifier(STATE_DIM).to(DEVICE)
    clf_target.load_state_dict(clf.state_dict())
    clf_target.eval()
    for p in clf_target.parameters():
        p.requires_grad = False
    print(f"[INFO] Initialized target classifier; copy every {args.clf_target_update_every} updates")

    pol_opt = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    clf_opt = torch.optim.Adam(clf.parameters(), lr=args.clf_lr)

    # --------- FAST CONFIG ----------
    FAST_MAX_STEPS = args.fast_max_steps
    FAST_MAX_DEPTH = args.fast_max_depth
    FAST_N_ROLLOUTS = args.fast_n_rollouts

    UPDATE_EVERY = args.update_every
    POLICY_SAMPLE = args.policy_sample
    CLF_SAMPLE = args.clf_sample

    POLICY_BATCH = args.policy_batch
    CLF_BATCH = args.clf_batch
    EPOCHS_PER_UPDATE = args.epochs_per_update

    REPLAY_MAX_SPV = args.replay_max_spv
    REPLAY_MAX_CLF = args.replay_max_clf

    REPLAY_MAX_FINAL = args.replay_max_final
    FINAL_SAMPLE = args.final_sample
    USE_LLM = 0
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
        "policy_lr": args.policy_lr,
        "clf_lr": args.clf_lr,
        "founder_epochs": args.founder_epochs,
        "freeze_clf_updates": args.freeze_clf_updates,

        # 新增：训练集抽样 & holdout 拆分配置
        "n_success_train": args.n_success_train,
        "n_fail_train": args.n_fail_train,
        "cv_folds": args.cv_folds,
        "cv_fold": args.cv_fold,
        "eval_every": args.eval_every,
        "test_eval_every": args.test_eval_every,
        "threshold_min": args.threshold_min,
        "threshold_max": args.threshold_max,
        "threshold_steps": args.threshold_steps,
        "eval_threshold": args.eval_threshold,
        "clf_target_update_every": args.clf_target_update_every,
        "initial_eval_threshold": eval_threshold,
        "labels_train_size_total": int(len(labels_train_df)),
        "train_size": int(len(train_df)),
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
    FOUNDER_EPOCHS = args.founder_epochs
    global_step = 0

    # ====== NEW: freeze classifier at early stage ======
    FREEZE_CLF_UPDATES = args.freeze_clf_updates
    clf_frozen = True
    best_val_f05 = -1.0
    best_val_threshold = eval_threshold

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
        print(f"[CV split] train={len(train_df)} val={len(val_df)} "
              f"(validation fold {args.cv_fold + 1}/{args.cv_folds})")

        for i, row in balanced_df.iterrows():
            fid = row["founder_uuid"]
            label = int(row["success"])

            S_Pai_V, S_Labels, _ = loop_per_founder(
                fid, label, data_store, policy, clf_target, DEVICE,
                max_steps=FAST_MAX_STEPS,
                max_depth=FAST_MAX_DEPTH,
                n_rollouts=FAST_N_ROLLOUTS,
                eps=0.1, USE_LLM = USE_LLM,
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

                if (
                    args.clf_target_update_every > 0
                    and global_step % args.clf_target_update_every == 0
                ):
                    clf_target.load_state_dict(clf.state_dict())
                    clf_target.eval()
                    print(f"[INFO] Synced target classifier at update {global_step}")

                log = (f"[{i + 1}/{len(balanced_df)}] "
                       f"replay_spv={len(policy_replay)} replay_clf={len(clf_replay)} | "
                       f"pol_loss={pol_loss:.4f} clf_loss={clf_loss:.4f} clf_acc={clf_acc:.3f}")
                print(log)

                # ---- validation eval during training ----
                if args.eval_every > 0 and (global_step % args.eval_every == 0):
                    policy.eval()
                    clf.eval()
                    metrics_val = eval_on_val(
                        val_df,
                        data_store,
                        policy,
                        clf,
                        DEVICE,
                        thr=eval_threshold,
                    )
                    if metrics_val["f0.5"] > best_val_f05:
                        best_val_f05 = metrics_val["f0.5"]
                        best_val_threshold = eval_threshold
                    print(f"  [VAL @update {global_step}] "
                          f"Thr={eval_threshold:.4f} "
                          f"Precision={metrics_val['precision']:.4f} "
                          f"Recall={metrics_val['recall']:.4f} "
                          f"F0.5={metrics_val['f0.5']:.4f} "
                          f"Acc={metrics_val['accuracy']:.4f} "
                          f"AvgStep={metrics_val['avg_step']:.2f}")
                    policy.train()
                    clf.train()

                # ---- held-out test eval for monitoring only ----
                # This uses the fixed threshold and never updates model weights
                # or best-val tracking.
                if args.test_eval_every > 0 and (global_step % args.test_eval_every == 0):
                    policy.eval()
                    clf.eval()
                    metrics_test_monitor = eval_on_val(
                        test_df,
                        data_store,
                        policy,
                        clf,
                        DEVICE,
                        thr=eval_threshold,
                    )
                    print(f"  [TEST monitor @update {global_step}] "
                          f"Thr={eval_threshold:.4f} "
                          f"Precision={metrics_test_monitor['precision']:.4f} "
                          f"Recall={metrics_test_monitor['recall']:.4f} "
                          f"F0.5={metrics_test_monitor['f0.5']:.4f} "
                          f"Acc={metrics_test_monitor['accuracy']:.4f} "
                          f"AvgStep={metrics_test_monitor['avg_step']:.2f}")
                    policy.train()
                    clf.train()

    # ---- final eval: CV validation + held-out test ----
    policy.eval()
    clf.eval()

    final_threshold = eval_threshold
    metrics_val = eval_on_val(
        val_df,
        data_store,
        policy,
        clf,
        DEVICE,
        thr=final_threshold,
    )
    print("\n=== Final CV Validation ===")
    print("Threshold:", final_threshold)
    print("Confusion:", metrics_val["tp"], metrics_val["fp"], metrics_val["tn"], metrics_val["fn"])
    print("Precision:", metrics_val["precision"])
    print("Recall:", metrics_val["recall"])
    print("F0.5:", metrics_val["f0.5"])
    print("Accuracy:", metrics_val["accuracy"])
    print("AvgStep:", metrics_val["avg_step"])

    metrics_test = eval_on_val(test_df, data_store, policy, clf, DEVICE, thr=final_threshold)
    print("\n=== Final Test ===")
    print("Threshold:", final_threshold)
    print("Confusion:", metrics_test["tp"], metrics_test["fp"], metrics_test["tn"], metrics_test["fn"])
    print("Precision:", metrics_test["precision"])
    print("Recall:", metrics_test["recall"])
    print("F0.5:", metrics_test["f0.5"])
    print("Accuracy:", metrics_test["accuracy"])
    print("AvgStep:", metrics_test["avg_step"])

    tp_records = eval_and_print_TP(
        test_df,
        data_store,
        policy,
        clf,
        DEVICE,
        max_steps=FAST_MAX_STEPS,
        greedy=True,
        min_queries=1,
        thr=final_threshold,
        max_print=20,
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
        extra={
            "best_val_threshold_during_training": best_val_threshold,
            "best_val_f0.5_during_training": best_val_f05,
            "final_threshold": final_threshold,
            "final_val_metrics": metrics_val,
            "final_test_metrics": metrics_test,
        },
    )

    diag.save(
        "policy_metrics_with_llm.jsonl" if USE_LLM
        else "policy_metrics_no_llm.jsonl"
    )


if __name__ == "__main__":
    main()

# python train.py --n_success_train 100 --n_fail_train 500 --cv_folds 5 --cv_fold 0 --eval_every 5
# python train.py --cv_folds 5 --cv_fold 0 --eval_every 5 --test_eval_every 5 --policy_lr 5e-5 --clf_lr 1e-5 --freeze_clf_updates 10 --founder_epochs 3
# set OPENAI_API_KEY=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