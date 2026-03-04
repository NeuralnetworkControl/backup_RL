# Tree_value_map.py
import numpy as np
import torch

INFO_MAP = {0: "edu", 1: "role", 2: "exec", 3: "industry", 4: "depth"}
STEP_PENALTY = -0.1
REPEAT_PENALTY = -5  # 你可以调大/调小，比如 -0.2、-0.5、-2


def _is_repeated_action(state_obj, action_dim):
    slot = INFO_MAP[action_dim]
    return bool(state_obj.observed.get(slot, 0))


def _clone_state(state_obj):
    """深拷贝 FounderState，避免 rollout 修改同一个对象导致串味。"""
    new_state = state_obj.__class__(state_obj.founder_id, state_obj.data)
    new_state.observed = dict(state_obj.observed)
    new_state.cache = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in state_obj.cache.items()}
    return new_state


def get_state(state_obj, action_dim):
    """执行一次 info query，返回新的 state_obj（不改原对象）。"""
    slot = INFO_MAP[action_dim]
    child = _clone_state(state_obj)
    child.query(slot)
    return child


def get_reward(state_obj, Classifier, label, device):
    """终止时 reward：Classifier 吃 torch tensor，输出 logits -> sigmoid -> 0/1。"""
    REWARDS = {"TP": 4, "FP": -16, "TN": 0, "FN": -0.25}

    state_vec = state_obj.get_state_vector()
    x = torch.from_numpy(state_vec).float().to(device)

    with torch.no_grad():
        logits = Classifier(x)
        prob = torch.sigmoid(logits)
        pred_label = int((prob >= 0.3).item())

    label = int(label)

    if pred_label == 1 and label == 1:
        result = "TP"
    elif pred_label == 1 and label == 0:
        result = "FP"
    elif pred_label == 0 and label == 0:
        result = "TN"
    elif pred_label == 0 and label == 1:
        result = "FN"
    else:
        raise ValueError("prediction or label must be 0 or 1")

    reward = REWARDS[result]
    mask = np.array([state_obj.observed[s] for s in ["edu", "role", "exec", "industry", "depth"]], dtype=np.int64)

    sample = {
        "x": state_vec.copy() if hasattr(state_vec, "copy") else state_vec,
        "y": label,
        "mask": mask.copy()
    }
    return reward, sample


def _policy_pi(state_obj, Policynet, device):
    """给定 state_obj，返回 numpy 的动作概率 pi（长度 6：5 infos + stop）。"""
    state_vec = state_obj.get_state_vector()
    x = torch.from_numpy(state_vec).float().to(device)
    with torch.no_grad():
        pi_t = Policynet(x)  # 期望输出已经是 softmax 概率
    pi = pi_t.detach().cpu().numpy().astype(np.float64)

    # 稳健处理：避免数值问题导致 sum!=1 或 nan
    pi = np.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0)
    s = pi.sum()
    if s <= 0:
        pi[:] = 1.0 / len(pi)
    else:
        pi /= s
    return pi


def _rollout_once(start_state_obj, Policynet, max_depth, Classifier, label, device, dataset, depth_start=0):
    state_obj = _clone_state(start_state_obj)

    depth = depth_start
    repeat_penalty_sum = 0.0

    while True:
        # 强制 stop（到最大深度）
        if depth >= max_depth:
            reward, sample = get_reward(state_obj, Classifier, label, device)
            reward = reward + STEP_PENALTY * depth + repeat_penalty_sum
            if sample is not None:
                dataset.append(sample)
            return float(reward)

        pi = _policy_pi(state_obj, Policynet, device)
        n_info = len(pi) - 1
        a = int(np.random.choice(len(pi), p=pi))

        # stop
        if a == n_info:
            reward, sample = get_reward(state_obj, Classifier, label, device)
            reward = reward + STEP_PENALTY * depth + repeat_penalty_sum
            if sample is not None:
                dataset.append(sample)
            return float(reward)

        # info action：如果动作重复，额外加惩罚
        if _is_repeated_action(state_obj, a):
            repeat_penalty_sum += REPEAT_PENALTY
            # 这里你有两种选择：
            # 选择1（推荐）：重复查询没有新信息，就不改变状态，只增加深度
            depth += 1
            continue

            # 选择2：仍然执行 query（如果你的 query 有副作用/费用，也可以）
            # state_obj = get_state(state_obj, a)
            # depth += 1
            # continue

        # 非重复：正常 query
        state_obj = get_state(state_obj, a)
        depth += 1



def tree_value_map(x0, Policynet, n_rollouts, max_depth, Classifier, label, device, seed=None):
    """
    Monte Carlo rollout 版：
    对每个 root info action a：
      - 先执行 a 得到 child
      - 从 child 开始 rollout n_rollouts 次（每步按 pi 采样）
      - 均值作为 Q_i[a]

    返回：
      Q_i: 长度 5
      dataset: 每次 rollout 终止时收集的 sample（规模 ~ 5*n_rollouts）
    """
    if seed is not None:
        np.random.seed(seed)

    dataset = []

    # 用 policy 输出维度确定 n_info
    pi0 = _policy_pi(x0, Policynet, device)
    n_info = len(pi0) - 1

    Q_i = [None] * n_info
    for a in range(n_info):
        child = get_state(x0, a)

        init_repeat = REPEAT_PENALTY if _is_repeated_action(x0, a) else 0.0

        returns = []
        for _ in range(int(n_rollouts)):
            g = _rollout_once(
                child, Policynet, max_depth, Classifier, label, device,
                dataset=dataset, depth_start=1
            )
            returns.append(g + init_repeat)  # ✅ 把 root 的重复惩罚加进回报
        Q_i[a] = float(np.mean(returns)) if returns else None

    return Q_i, dataset
