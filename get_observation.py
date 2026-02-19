import numpy as np

class FounderState:
    def __init__(self, founder_id, data_store):
        self.founder_id = founder_id
        self.data = data_store

        self.observed = {
            "edu": False,
            "role": False,
            "exec": False,
            "industry": False,
            "depth": False,
            # "exit": False
        }
        self.cache = {}

    def query(self, slot):
        if not self.observed[slot]:
            self.cache[slot] = self.data.get_slot(self.founder_id, slot)
            self.observed[slot] = True
        return self.cache[slot]

    def get_state_vector(self):
        vecs = []
        for slot in ["edu", "role", "exec", "industry", "depth"]:
            if self.observed[slot]:
                vecs.append(self.cache[slot])
            else:
                vecs.append(np.zeros(self.data.get_slot_dim(slot)))
        return np.concatenate(vecs)
