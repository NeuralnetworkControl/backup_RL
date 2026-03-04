import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data_store import FounderDataStore
from get_observation import FounderState
from Tree_value_map import get_reward
from network_trainers import train_classifier_from_samples
from main_loop import eval_on_val, INFO_MAP, STOP_ACTION

# ------------------ Config ------------------
STATE_DIM = 1543
N_INFO_ACTIONS = 5
ACTION_DIM = N_INFO_ACTIONS + 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_STEPS = 7
MIN_QUERIES = 3
GAMMA = 0.99
CLIP_EPS = 0.2
LR = 3e-4
EPOCHS = 10
VAL_RATIO = 0.5
ENTROPY_COEF = 0.01
# -------------------------------------------


# ---------- utils ----------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(df, label_col, val_ratio, seed):
    rng = np.random.default_rng(seed)
    parts_val, parts_test = [], []

    for y, g in df.groupby(label_col):
        idx = np.arange(len(g))
        rng.shuffle(idx)
        n_val = int(round(len(g) * val_ratio))
        parts_val.append(g.iloc[idx[:n_val]])
        parts_test.append(g.iloc[idx[n_val:]])

    val_df = pd.concat(parts_val, ignore_index=True).sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)

    test_df = pd.concat(parts_test, ignore_index=True).sample(
        frac=1, random_state=seed + 1
    ).reset_index(drop=True)

    return val_df, test_df


# ---------- PPO Policy ----------
class PPOPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.pi = nn.Linear(256, action_dim)
        self.v = nn.Linear(256, 1)

    def forward(self, x):
        h = self.backbone(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value

    def act(self, x):
        logits, _ = self.forward(x)
        return torch.softmax(logits, dim=-1)


# ---------- Policy adapter for eval ----------
class PPOPolicyAdapter:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, x):
        return self.policy.act(x)


# ---------- Rollout ----------
def rollout_episode(fid, label, data_store, policy, clf):
    state = FounderState(fid, data_store)
    traj = []
    queried = 0
    done = False

    for step_idx in range(MAX_STEPS):
        x = torch.from_numpy(state.get_state_vector()).float().to(DEVICE)
        logits, value = policy(x)

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logp = dist.log_prob(action)

        a = int(action.item())
        reward = 0.0
        sample = None

        # early-stop constraint
        if a == STOP_ACTION and queried < MIN_QUERIES:
            probs[STOP_ACTION] = 0
            probs = probs / probs.sum()
            a = int(torch.argmax(probs).item())

        if a == STOP_ACTION:
            reward, sample = get_reward(state, clf, label, DEVICE)
            done = True
        else:
            state.query(INFO_MAP[a])
            queried += 1

        traj.append({
            "x": x,
            "a": torch.tensor(a, device=DEVICE),
            "logp": logp,
            "v": value,
            "r": reward,
            "done": done,
            "sample": sample,
            "is_stop": (a == STOP_ACTION),
        })

        if done:
            break

    # force stop if max_steps reached
    if not done:
        x = torch.from_numpy(state.get_state_vector()).float().to(DEVICE)
        logits, value = policy(x)
        reward, sample = get_reward(state, clf, label, DEVICE)

        traj.append({
            "x": x,
            "a": torch.tensor(STOP_ACTION, device=DEVICE),
            "logp": torch.zeros((), device=DEVICE),
            "v": value,
            "r": reward,
            "done": True,
            "sample": sample,
            "is_stop": True,
        })

    return traj


# ---------- PPO Update ----------
def ppo_update(policy, optimizer, trajs, entropy_coef):
    xs, actions, logps_old, returns, values = [], [], [], [], []

    for traj in trajs:
        R = 0.0
        for step in reversed(traj):
            R = step["r"] + GAMMA * R
            returns.insert(0, R)
            xs.insert(0, step["x"])
            actions.insert(0, step["a"])
            logps_old.insert(0, step["logp"])
            values.insert(0, step["v"])

    xs = torch.stack(xs)
    actions = torch.stack(actions)
    logps_old = torch.stack(logps_old).detach()
    returns = torch.tensor(returns, device=DEVICE)
    values = torch.stack(values)

    adv = returns - values.detach()

    logits, values_new = policy(xs)
    dist = torch.distributions.Categorical(logits=logits)
    logps = dist.log_prob(actions)

    ratio = torch.exp(logps - logps_old)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv

    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = (values_new - returns).pow(2).mean()
    entropy = dist.entropy().mean()

    loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return (
        loss.item(),
        values_new.mean().item(),
        values_new.std().item(),
    )


# ---------- Main ----------
def main():
    # ---- args ----
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    # ---- data ----
    train_df = pd.read_csv("data/labels_train_clean.csv")
    holdout_df = pd.read_csv("data/labels_val.csv")
    data_store = FounderDataStore("data")

    val_df, test_df = stratified_split(
        holdout_df, "success", VAL_RATIO, args.seed
    )

    # ---- models ----
    policy = PPOPolicy(STATE_DIM, ACTION_DIM).to(DEVICE)

    ckpt = torch.load("pretrained_classifier.pt", map_location=DEVICE)
    from Networks import Classifier
    classifier = Classifier(STATE_DIM).to(DEVICE)
    classifier.load_state_dict(ckpt["state_dict"])
    classifier.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    # ---- buffers & logs ----
    clf_replay = []
    REPLAY_MAX_CLF = 20000
    CLF_BATCH = 512

    stop_curve = []
    value_mean_curve = []
    value_std_curve = []

    print(f"[INFO] PPO + online classifier | entropy={ENTROPY_COEF} | seed={args.seed}")

    # ---- training ----
    for epoch in range(EPOCHS):
        policy.train()
        trajs = []

        for _, row in train_df.iterrows():
            trajs.append(
                rollout_episode(
                    row["founder_uuid"],
                    int(row["success"]),
                    data_store,
                    policy,
                    classifier,
                )
            )

        # ---- stop statistics ----
        stop_steps = []
        for traj in trajs:
            for i, step in enumerate(traj):
                if step["is_stop"]:
                    stop_steps.append(i)
                    break
        # treat trajectories without a stop as MAX_STEPS
        if len(stop_steps) < len(trajs):
            # count those as MAX_STEPS
            missing = len(trajs) - len(stop_steps)
            stop_steps.extend([MAX_STEPS] * missing)

        avg_stop = float(np.mean(stop_steps))
        stop_curve.append(avg_stop)

        # ---- collect classifier samples ----
        for traj in trajs:
            for step in traj:
                if step["sample"] is not None:
                    clf_replay.append(step["sample"])
        if len(clf_replay) > REPLAY_MAX_CLF:
            clf_replay = clf_replay[-REPLAY_MAX_CLF:]

        # ---- PPO update ----
        loss, v_mean, v_std = ppo_update(
            policy, optimizer, trajs, ENTROPY_COEF
        )
        value_mean_curve.append(v_mean)
        value_std_curve.append(v_std)

        # ---- classifier update ----
        if len(clf_replay) > 0:
            clf_loss, clf_acc = train_classifier_from_samples(
                classifier,
                clf_replay,
                DEVICE,
                batch_size=CLF_BATCH,
                epochs=1
            )
        else:
            clf_loss, clf_acc = 0.0, 0.0

        # ---- validation ----
        policy.eval()
        metrics_val = eval_on_val(
            val_df,
            data_store,
            PPOPolicyAdapter(policy),
            classifier,
            DEVICE,
        )

        print(
            f"[Epoch {epoch+1}/{EPOCHS}] "
            f"ppo_loss={loss:.4f} | "
            f"Vmean={v_mean:.3f} Vstd={v_std:.3f} | "
            f"stop={avg_stop:.2f} | "
            f"clf_loss={clf_loss:.4f} | "
            f"F0.5={metrics_val['f0.5']:.4f}"
        )

    # ---- final test ----
    policy.eval()
    metrics_test = eval_on_val(
        test_df,
        data_store,
        PPOPolicyAdapter(policy),
        classifier,
        DEVICE,
    )

    print("\n=== Final PPO + Online Classifier (Test) ===")
    for k, v in metrics_test.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # ---- plotting ----
    epochs = list(range(1, len(stop_curve) + 1))

    # Fig 1: stop curve
    plt.figure()
    plt.plot(epochs, stop_curve)
    plt.xlabel("Epoch")
    plt.ylabel("Average Stop Step")
    plt.title("PPO Stop Behavior (avg stop step per epoch)")
    plt.grid(True)
    plt.savefig("ppo_stop_curve.png", bbox_inches="tight")
    plt.close()

    # Fig 2: value mean & std
    plt.figure()
    plt.plot(epochs, value_mean_curve, label="mean V(s)")
    plt.plot(epochs, value_std_curve, label="std V(s)")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("PPO Value Function Mean and Std")
    plt.legend()
    plt.grid(True)
    plt.savefig("ppo_value_collapse.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
