import json
import pandas as pd
import matplotlib.pyplot as plt

def load(path):
    return pd.read_json(path, lines=True)

def smooth(df, col, window=20):
    return df[col].rolling(window=window, min_periods=1).mean()

df_llm = load("policy_metrics_with_llm.jsonl")
df_nollm = load("policy_metrics_no_llm.jsonl")

# plt.figure(figsize=(12, 4))
plt.figure(figsize=(10, 5))

# ---------- Entropy ----------
plt.subplot(1, 2, 1)
plt.plot(df_llm["update"], df_llm["entropy"], label="With LLM")
plt.plot(df_nollm["update"], df_nollm["entropy"], label="No LLM")

# window = 20  # 你可以试 10 / 20 / 50
#
# plt.plot(
#     df_llm["update"],
#     smooth(df_llm, "entropy", window),
#     label="With LLM (smoothed)"
# )
#
# plt.plot(
#     df_nollm["update"],
#     smooth(df_nollm, "entropy", window),
#     label="No LLM (smoothed)"
# )

plt.xlabel("Update")
plt.ylabel("Policy Entropy")
plt.legend()
plt.title("Policy Entropy")


# # ---------- KL ----------
plt.subplot(1, 2, 2)
# plt.plot(df_llm["update"], df_llm["kl"], label="With LLM")
# plt.plot(df_nollm["update"], df_nollm["kl"], label="No LLM")
# plt.xlabel("Update")
# plt.ylabel("KL(pi_t || pi_{t-1})")
# plt.legend()
# plt.title("Policy Activity")

window = 50  # 推荐 10 / 20 / 50 都可以试
#
kl_llm_smooth = df_llm["kl"].rolling(window=window, min_periods=1).mean()
kl_nollm_smooth = df_nollm["kl"].rolling(window=window, min_periods=1).mean()

plt.plot(df_llm["update"], kl_llm_smooth, label="With LLM")
plt.plot(df_nollm["update"], kl_nollm_smooth, label="No LLM")

plt.xlabel("Update")
plt.ylabel(r"KL($\pi_t \,\|\, \pi_{t-1}$)")
plt.legend()
plt.title("Policy Activity")

# # ---------- Zoomed early stage ----------
# plt.subplot(1, 3, 3)
# plt.plot(df_llm["update"][:30], df_llm["entropy"][:30], label="With LLM")
# plt.plot(df_nollm["update"][:30], df_nollm["entropy"][:30], label="No LLM")
# plt.title("Early-stage Entropy")
# plt.legend()

plt.tight_layout()
plt.show()
