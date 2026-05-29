# policy_loop_core.py
import numpy as np
import torch

from get_observation import FounderState
from Tree_value_map import tree_value_map, get_reward, STEP_PENALTY
from llm_next_action_supervisor import llm_prefer_next_actions_from_merged

# ---- Action map ----
INFO_MAP = {0: "edu", 1: "role", 2: "exec", 3: "industry", 4: "depth"}
N_INFO_ACTIONS = len(INFO_MAP)
STOP_ACTION = N_INFO_ACTIONS  # 5
# ========== LLM next-action 相关 ==========
LLM_BIAS = 0.05       # small tie-break bias for info actions
UNCERTAIN_DELTA = 0.01  # top-2 概率差阈值
LLM_CACHE = {}          # key: (merged_csv_path, fid, mask tuple), value: prefer list
# =========================================


def get_new_state(
    state_obj,
    policy,
    device,
    eps,
    *,
    fid,
    actions_taken,
    USE_LLM,
    merged_csv_path="merged.csv",   # This is our private dataset, LLM need the raw data instead of the embedded data.
                        # if you want to activate the llm_supervisor, you need to use vcbench_final_public.csv instead.
):
    """
    - policy 负责给出基础 π
    - LLM 只在 policy 对 info action 犹豫时介入
    - LLM 使用 merged.csv + actions_taken（人类可读信息）
    - LLM 只给 next-action 的弱偏好，不碰 stop
    """

    # ---------- policy 前向 ----------
    state_vec = state_obj.get_state_vector()
    x = torch.from_numpy(state_vec).float().to(device)

    with torch.no_grad():
        pi_t = policy(x)                  # shape [6]
    pi = pi_t.detach().cpu().numpy()     # shape [6]

    # ---------- 拆分 info / stop ----------
    pi_info = pi[:N_INFO_ACTIONS].copy()   # 0~4
    pi_stop = pi[STOP_ACTION]              # 5

    # ---------- 判断 policy 是否“犹豫” ----------
    order = np.argsort(-pi_info)
    uncertain = (pi_info[order[0]] - pi_info[order[1]]) < UNCERTAIN_DELTA

    # ---------- 仅在犹豫时调用 LLM ----------
    if uncertain and USE_LLM:
        # Cache must be profile-specific. The same observed mask can need
        # different next actions for different founders or source files.
        mask = tuple(int(state_obj.observed.get(s, 0)) for s in INFO_MAP.values())
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

        # 对 LLM 提出的 slot 加很小的 bias，只在 info actions 内部重分配概率。
        # Keep the original stop probability unchanged, so LLM only suggests
        # what to ask next and does not implicitly change when to stop.
        info_mass = float(pi_info.sum())
        for k, slot in INFO_MAP.items():
            if slot in prefer:
                pi_info[k] += LLM_BIAS

        # 重新归一化 info action while preserving the original info-vs-stop mass.
        pi_info = np.clip(pi_info, 1e-8, None)
        pi_info = pi_info / pi_info.sum() * info_mass

        # 拼回完整 π（stop 不动）
        pi[:N_INFO_ACTIONS] = pi_info
        pi[STOP_ACTION] = pi_stop

    # ---------- epsilon-greedy 选动作 ----------
    if np.random.rand() < eps:
        action_dim = int(np.random.randint(len(pi)))
    else:
        action_dim = int(np.argmax(pi))

    # ---------- 执行动作 ----------
    stop_flag = 0
    if action_dim in INFO_MAP:
        action_name = INFO_MAP[action_dim]
        state_obj.query(action_name)
    elif action_dim == STOP_ACTION:
        action_name = "stop"
        stop_flag = 1
    else:
        raise ValueError(f"Invalid action dim: {action_dim}")

    return state_obj, stop_flag, action_dim, action_name


def compute_tree_targets(
    state_obj, policy, clf, label, device,
    max_depth, n_rollouts,
    tau_info=1.0,
    tau_stop=4.0,
):
    state_vec = state_obj.get_state_vector()

    # root policy（用于算 v）
    x = torch.from_numpy(state_vec).float().to(device)
    with torch.no_grad():
        pi_root = policy(x).detach().cpu().numpy()   # (6,)

    # -------- tree search for info actions --------
    Qi, dataset = tree_value_map(
        state_obj, policy, n_rollouts, max_depth, clf, label, device
    )
    Qi_np = np.asarray(Qi, dtype=np.float32)        # (5,)

    min_queries = 3
    n_observed = sum(state_obj.observed.values())

    # -------- stop value --------
    if n_observed < min_queries:
        q_stop = -1e9  # 等价于不允许 stop
    else:
        q_stop, _ = get_reward(state_obj, clf, label, device)
    q_stop = float(q_stop)                           # scalar

    # -------- dual-temperature softmax --------
    Q_info_scaled = Qi_np / tau_info
    Q_stop_scaled = q_stop / tau_stop

    Q_all = np.concatenate([Q_info_scaled, [Q_stop_scaled]])  # (6,)

    Q_all_t = torch.from_numpy(Q_all)
    # info probs
    p_info = torch.softmax(torch.from_numpy(Qi_np / tau_info), dim=0)  # (5,)

    # stop prob
    p_stop = torch.sigmoid(torch.tensor(q_stop / tau_stop))  # scalar

    # combine & normalize
    p_all = torch.cat([p_info, p_stop.view(1)], dim=0)
    pai = p_all / p_all.sum()

    # -------- value target --------
    pi_root = pi_root[:len(Q_all)]
    v = float(np.dot(pi_root, Q_all))

    s_pai_v = {
        "s": state_vec.copy() if hasattr(state_vec, "copy") else state_vec,
        "pai": pai,
        "v": v
    }
    return s_pai_v, dataset


# max_steps=10, max_depth=6,
def main_loop(fid, data_store, policy, clf, label, device,
              max_steps, max_depth, n_rollouts, eps, USE_LLM):
    state_obj = FounderState(fid, data_store)
    S_Pai_V = []
    S_Labels = []
    actions_taken = []

    for step in range(max_steps):
        state_obj, stop_flag, a_dim, a_name = get_new_state(
            state_obj,
            policy,
            device,
            eps=eps,
            fid=fid,
            actions_taken=actions_taken,
            USE_LLM = USE_LLM,
        )

        actions_taken.append({"step": step, "action_dim": a_dim, "action": a_name})

        if stop_flag == 1:
            break

        s_pai_v, s_labels = compute_tree_targets(
            state_obj, policy, clf, label, device,
            max_depth=max_depth, n_rollouts=n_rollouts
        )

        S_Pai_V.append(s_pai_v)
        S_Labels.extend(s_labels)

    return S_Pai_V, S_Labels, actions_taken


def loop_per_founder(fid, label, data_store, policy, clf, device,
                     max_steps, max_depth, n_rollouts, eps, USE_LLM):
    return main_loop(
        fid, data_store, policy, clf, label, device,
        max_steps=max_steps, max_depth=max_depth, n_rollouts=n_rollouts, eps=eps, USE_LLM=USE_LLM
    )


@torch.no_grad()
def rollout_final_state(fid, data_store, policy, device,
                        max_steps, greedy, min_queries=3):
    state = FounderState(fid, data_store)
    queried = 0

    for _ in range(max_steps):
        x = torch.from_numpy(state.get_state_vector()).float().to(device)
        pi = policy(x).detach().cpu().numpy()

        if greedy:
            a = int(pi.argmax())
        else:
            p = pi.astype(np.float64)
            p = p / (p.sum() + 1e-12)
            a = int(np.random.choice(len(p), p=p))

        # 禁止太早 stop
        if a == STOP_ACTION and queried < min_queries:
            pi[STOP_ACTION] = -1
            a = int(pi.argmax())

        if a == STOP_ACTION:
            break

        state.query(INFO_MAP[a])
        queried += 1

    return state.get_state_vector().copy()


@torch.no_grad()
def predict_one(fid, data_store, policy, clf, device,
                max_steps, greedy, min_queries, thr):
    """
    用 policy 逐步 query slot，直到 stop 或 max_steps，然后用 clf 预测标签
    返回:
        pred (0/1),
        prob,
        used_actions (set)  # 新增
    """

    state = FounderState(fid, data_store)
    queried = 0
    used_actions = set()   # ⭐ 新增：记录 unique action

    for _ in range(max_steps):
        x = torch.from_numpy(state.get_state_vector()).float().to(device)
        pi = policy(x).detach().cpu().numpy()

        a = int(pi.argmax()) if greedy else int(np.random.choice(len(pi), p=pi))

        # 不允许太早 stop
        if a == STOP_ACTION and queried < min_queries:
            pi[STOP_ACTION] = -1
            a = int(pi.argmax())

        if a == STOP_ACTION:
            break

        # ⭐ 记录 action（只统计 info action）
        used_actions.add(a)

        state.query(INFO_MAP[a])
        queried += 1

    x_final = torch.from_numpy(state.get_state_vector()).float().to(device)
    logit = clf(x_final)
    prob = float(torch.sigmoid(logit).item())
    pred = 1 if prob >= thr else 0

    return pred, prob, used_actions


def eval_on_val(val_df, data_store, policy, clf, device,
                max_steps=5, greedy=True, min_queries=1, thr=0.3):

    tp = fp = tn = fn = 0

    info_counts = []          # 每个 founder 用了多少种信息
    action_counter = {}       # 每种 action 被多少 founder 用过

    for _, row in val_df.iterrows():
        fid = row["founder_uuid"]
        y = int(row["success"])

        pred, prob, used_actions = predict_one(
            fid, data_store, policy, clf, device,
            max_steps=max_steps, greedy=greedy,
            min_queries=min_queries, thr=thr
        )

        # ===== 统计信息使用 =====
        n_info = len(used_actions)
        info_counts.append(n_info)

        for a in used_actions:
            action_counter[a] = action_counter.get(a, 0) + 1

        # ===== 分类统计 =====
        if pred == 1 and y == 1: tp += 1
        elif pred == 1 and y == 0: fp += 1
        elif pred == 0 and y == 0: tn += 1
        else: fn += 1

    total = tp + fp + tn + fn
    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    f05 = 1.25 * precision * recall / (0.25 * precision + recall + 1e-9)
    acc = (tp + tn) / (total + 1e-9)

    fpr = fp / (fp + tn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    fnr = fn / (fn + tp + 1e-9)

    total_info_used = sum(info_counts)
    avg_step = total_info_used / max(len(info_counts), 1)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f0.5": f05,
        "accuracy": acc,
        "avg_step": avg_step,
        "total_info_used": total_info_used,       # ⭐ 新增
        "info_distribution": info_counts,
        "action_usage": action_counter,   # ⭐ 每种 action 用了多少 founder
        "rates_total": {
            "tp_rate_total": tp / (total + 1e-9),
            "fp_rate_total": fp / (total + 1e-9),
            "tn_rate_total": tn / (total + 1e-9),
            "fn_rate_total": fn / (total + 1e-9),
        },
        "rates_common": {"TPR": recall, "FPR": fpr, "TNR": tnr, "FNR": fnr},
    }



def sample_list(buf, k):
    if k <= 0:
        return []
    if len(buf) <= k:
        return list(buf)
    idx = np.random.choice(len(buf), size=k, replace=False)
    return [buf[i] for i in idx]
