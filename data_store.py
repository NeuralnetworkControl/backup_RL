import numpy as np
import pandas as pd

class FounderDataStore:
    def __init__(self, data_dir="data"):
        # 1. index
        index_df = pd.read_csv(f"{data_dir}/founder_index.csv")
        self.id2idx = dict(zip(index_df["founder_uuid"],
                                index_df["embedding_index"]))

        # 2. embeddings / features
        self.edu_state     = np.load(f"{data_dir}/founder_edu_state.npy")
        self.role_vecs     = np.load(f"{data_dir}/founder_role_vecs.npy")
        self.exec_vecs     = np.load(f"{data_dir}/founder_exec_vecs.npy")
        self.industry_vecs = np.load(f"{data_dir}/founder_industry_vecs.npy")
        self.depth_feats   = np.load(f"{data_dir}/founder_depth_feats.npy")
        # self.exit_feats  = np.load(f"{data_dir}/data_exit_feats.npy")

    def get_slot(self, founder_id, slot):
        idx = self.id2idx[founder_id]

        if slot == "edu":
            return self.edu_state[idx]
        elif slot == "role":
            return self.role_vecs[idx]
        elif slot == "exec":
            return self.exec_vecs[idx]
        elif slot == "industry":
            return self.industry_vecs[idx]
        elif slot == "depth":
            return self.depth_feats[idx]
        # elif slot == "exit":
        #     return self.exit_feats[idx]
        else:
            raise ValueError(f"Unknown slot: {slot}")

    def get_slot_dim(self, slot):
        if slot == "edu":
            return self.edu_state.shape[1]
        elif slot in ["role", "exec", "industry"]:
            return self.role_vecs.shape[1]
        elif slot == "depth":
            return self.depth_feats.shape[1]
        # elif slot == "exit":
        #     return self.exit_feats.shape[1]
        else:
            raise ValueError(f"Unknown slot: {slot}")
