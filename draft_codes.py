import numpy as np
import pandas as pd
# 1. 读取 index（这是关键）
index_df = pd.read_csv("data/founder_index.csv")
id2idx = dict(zip(index_df["founder_uuid"], index_df["embedding_index"]))

# 2. 读取你需要的 embedding / feature
edu_state     = np.load("data/founder_edu_state.npy")   # (N, 388)
role_vecs     = np.load("data/founder_role_vecs.npy")      # (N, 384)
exec_vecs     = np.load("data/founder_exec_vecs.npy")      # (N, 384)
industry_vecs = np.load("data/founder_industry_vecs.npy")  # (N, 384)
depth_feats   = np.load("data/founder_depth_feats.npy")    # (N, 3)
exit_vecs     = np.load("data/founder_exit_feats.npy") # 3
orgscale_vecs     = np.load("data/founder_orgscale_vecs.npy") # 384
TEXT_vecs     = np.load("data/founder_text_vecs.npy") # 384
a = 1
