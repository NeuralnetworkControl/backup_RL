# baseline_myopic_no_rollout.py
"""
Ablation A3: - Rollout (Myopic / One-step RL)

- 不使用 tree_value_map / Monte Carlo rollout
- 用当前 classifier 对 next state 直接估计 Q
- 其余训练、distillation、eval 全部复用原代码
"""

import numpy as np
import torch

from get_observation import FounderState
from Tree_value_map import get_reward, STEP_PENALTY
from main_loop import INFO_MAP, STOP_ACTION


def compute_myopic_targets(
    state_obj,
    policy,
    clf,
    label,
    device,
    tau_info=1.0,
    tau_stop=4.0,
):
    """
    用 one-step / myopic 方式计算 π*
    """
    state_vec = state_obj.get_state_vector()

    # ---------- root policy（算 v 用） ----------
    x = torch.from_numpy(state_vec).float().to(device)
    with torch.no_grad():
        pi_root = policy(x).detach().cpu().numpy()

    # ---------- info action Q（one-step） ----------
    Qi = []
    dataset = []

    for a_dim, slot in INFO_MAP.items():
        # clone state
        child = FounderState(state_obj.founder_id, state_obj.data)
        child.observed = dict(state_obj.observed)
        child.cache = dict(state_obj.cache)

        # 执行一次 query
        child.query(slot)

        # 立刻终止，用 classifier 给 reward
        r, sample = get_reward(child, clf, label, device)

        # 加一步查询的 penalty
        q = r + STEP_PENALTY
        Qi.append(float(q))

        if sample is not None:
            dataset.append(sample)

    Qi = np.asarray(Qi, dtype=np.float32)

    # ---------- stop Q ----------
    q_stop, sample = get_reward(state_obj, clf, label, device)
    q_stop = float(q_stop)

    if sample is not None:
        dataset.append(sample)

    # ---------- dual-temperature soft targets ----------
    Qi_scaled = Qi / tau_info
    q_stop_scaled = q_stop / tau_stop

    # info probs
    p_info = torch.softmax(torch.from_numpy(Qi_scaled), dim=0)

    # stop prob
    p_stop = torch.sigmoid(torch.tensor(q_stop_scaled))

    p_all = torch.cat([p_info, p_stop.view(1)], dim=0)
    pai = p_all / p_all.sum()

    # ---------- value target ----------
    Q_all = np.concatenate([Qi_scaled, [q_stop_scaled]])
    v = float(np.dot(pi_root[: len(Q_all)], Q_all))

    s_pai_v = {
        "s": state_vec.copy() if hasattr(state_vec, "copy") else state_vec,
        "pai": pai,
        "v": v,
    }

    return s_pai_v, dataset


def loop_per_founder_myopic(
    fid,
    label,
    data_store,
    policy,
    clf,
    device,
    max_steps,
    eps,
):
    """
    与 main_loop.loop_per_founder 等价，但：
    - target 来自 myopic Q
    """
    state_obj = FounderState(fid, data_store)
    S_Pai_V = []
    S_Labels = []

    for step in range(max_steps):
        # --- policy action ---
        x = torch.from_numpy(state_obj.get_state_vector()).float().to(device)
        with torch.no_grad():
            pi = policy(x).detach().cpu().numpy()

        if np.random.rand() < eps:
            a_dim = int(np.random.randint(len(pi)))
        else:
            a_dim = int(np.argmax(pi))

        # --- stop ---
        if a_dim == STOP_ACTION:
            r, sample = get_reward(state_obj, clf, label, device)
            if sample is not None:
                S_Labels.append(sample)
            break

        # --- info ---
        action_name = INFO_MAP[a_dim]
        state_obj.query(action_name)

        s_pai_v, samples = compute_myopic_targets(
            state_obj, policy, clf, label, device
        )

        S_Pai_V.append(s_pai_v)
        S_Labels.extend(samples)

    return S_Pai_V, S_Labels
