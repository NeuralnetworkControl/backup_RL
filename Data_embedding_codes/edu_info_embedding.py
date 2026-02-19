import numpy as np
import pandas as pd
import json
import os

# 文件名（改成你实际的路径/名字）
EDU_VEC_PATH = "data2/founder_edu_vecs.npy"     # (N, 384)
EDU_META_CSV = "data2/founder_edu_meta.csv"    # 包含列: max_degree_level, qs_rank （可能有缺失）
OUT_STATE_PATH = "data2/founder_edu_state.npy"
OUT_META_JSON = "data2/founder_edu_state_meta.json"

# 加载
edu_vecs = np.load(EDU_VEC_PATH)          # shape (N, D)
edu_meta = pd.read_csv(EDU_META_CSV)     # index 对应行顺序（若不同，请先 align）

N, D = edu_vecs.shape
print("Loaded edu_vecs:", edu_vecs.shape, "edu_meta:", edu_meta.shape)

# 确保长度一致
if len(edu_meta) != N:
    raise ValueError("edu_meta.csv length != edu_vecs rows. 请先确保行对齐（或使用 embedding_index 映射）。")

# 参数：degree ordinal 最大值（用于归一）
MAX_DEGREE_LEVEL = 3.0  # 0..3 -> divide by 3

# 准备输出数组
# 结构: [edu_vec (D), max_degree_level_norm (1), qs_rank_norm (1), has_degree (1), qs_present (1)]
OUT_DIM = D + 4
edu_states = np.zeros((N, OUT_DIM), dtype=np.float32)

def norm_degree(level):
    # level in {0,1,2,3}
    try:
        if pd.isna(level):
            return 0.0
        lv = float(level)
        return max(0.0, min(1.0, lv / MAX_DEGREE_LEVEL))
    except:
        return 0.0

# 若你知道QS的最大排名，可以用它；否则用 inverse transform
# 这里采用 inverse: qs_norm = 1/(rank + 1)  -> 更高的排名(1) 得到 0.5, 0->1.0
def norm_qs_rank(qs):
    try:
        if pd.isna(qs):
            return 0.0
        r = float(qs)
        if r <= 0:
            return 0.0
        return 1.0 / (r + 1.0)
    except:
        return 0.0

for i in range(N):
    vec = edu_vecs[i]
    meta_row = edu_meta.iloc[i] if i < len(edu_meta) else {}
    # degree
    max_deg = meta_row.get("max_degree_level", None) if "max_degree_level" in edu_meta.columns else None
    qs_rank = meta_row.get("qs_rank", None) if "qs_rank" in edu_meta.columns else None

    has_degree = 0.0
    qs_present = 0.0

    if pd.notna(max_deg) and float(max_deg) > 0:
        has_degree = 1.0
    if pd.notna(qs_rank):
        # if it's nan-like string, pd.notna handles; otherwise numeric
        try:
            if str(qs_rank).strip() != "":
                qs_present = 1.0
        except:
            qs_present = 0.0

    deg_norm = norm_degree(max_deg)
    qs_norm = norm_qs_rank(qs_rank)

    # build state
    s = np.zeros(OUT_DIM, dtype=np.float32)
    s[0:D] = vec.astype(np.float32)              # edu_vec
    s[D]   = deg_norm                            # normalized degree
    s[D+1] = qs_norm                             # normalized qs
    s[D+2] = has_degree                          # has_degree flag
    s[D+3] = qs_present                          # qs_present flag

    edu_states[i] = s

# 保存
np.save(OUT_STATE_PATH, edu_states)

# 保存 meta 描述
meta_desc = {
    "edu_vec_dim": int(D),
    "state_extra_fields": [
        {"name": "max_degree_level_norm", "type": "float", "desc": f"ordinal degree / {MAX_DEGREE_LEVEL}"},
        {"name": "qs_rank_norm", "type": "float", "desc": "1/(qs_rank+1) if present else 0"},
        {"name": "has_degree", "type": "float", "desc": "1 if degree present else 0"},
        {"name": "qs_present", "type": "float", "desc": "1 if qs_rank present else 0"}
    ],
    "state_total_dim": int(OUT_DIM)
}
with open(OUT_META_JSON, "w") as f:
    json.dump(meta_desc, f, indent=2)

print("Saved edu_state:", OUT_STATE_PATH, "shape:", edu_states.shape)
print("Saved meta json:", OUT_META_JSON)
