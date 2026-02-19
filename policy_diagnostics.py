# policy_diagnostics.py
import json
import math
from typing import List, Optional

import torch


class PolicyDiagnostics:
    """
    负责记录：
    - policy entropy
    - KL(pi_t || pi_{t-1})
    在一组固定 evaluation states 上

    用法：
        diag = PolicyDiagnostics(eval_states, device, use_llm=True)
        ...
        diag.log(policy, update=global_step)
        ...
        diag.save("policy_metrics.jsonl")
    """

    def __init__(
        self,
        eval_states: List[torch.Tensor],
        device: torch.device,
        *,
        use_llm: bool,
        eps: float = 1e-8,
    ):
        self.eval_states = eval_states
        self.device = device
        self.use_llm = bool(use_llm)
        self.eps = eps

        self.prev_pi: Optional[torch.Tensor] = None
        self.records = []

    @torch.no_grad()
    def _collect_pi(self, policy: torch.nn.Module) -> torch.Tensor:
        """
        返回 shape: [N_eval, action_dim]
        """
        policy.eval()
        pis = []

        for x in self.eval_states:
            if x.device != self.device:
                x = x.to(self.device)
            pi = policy(x).detach().cpu()
            pis.append(pi)

        return torch.stack(pis, dim=0)

    def _entropy(self, pi: torch.Tensor) -> torch.Tensor:
        """
        pi: [N, A]
        return: [N]
        """
        return -(pi * torch.log(pi + self.eps)).sum(dim=1)

    def _kl(self, pi: torch.Tensor, pi_prev: torch.Tensor) -> torch.Tensor:
        """
        KL(pi || pi_prev)
        """
        return (pi * (torch.log(pi + self.eps) - torch.log(pi_prev + self.eps))).sum(dim=1)

    def log(self, policy: torch.nn.Module, *, update: int):
        """
        在一次 policy update 后调用
        """
        pi = self._collect_pi(policy)

        # entropy
        ent = self._entropy(pi)
        mean_entropy = float(ent.mean().item())

        # KL
        mean_kl = None
        if self.prev_pi is not None:
            kl = self._kl(pi, self.prev_pi)
            mean_kl = float(kl.mean().item())

        self.prev_pi = pi

        record = {
            "update": int(update),
            "entropy": mean_entropy,
            "kl": mean_kl,
            "use_llm": self.use_llm,
        }
        self.records.append(record)

    def save(self, path: str):
        """
        保存为 jsonl
        """
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r) + "\n")

    # --------- utility: time-to-stable ---------

    @staticmethod
    def time_to_stable(
        records: List[dict],
        *,
        entropy_key: str = "entropy",
        eps: float = 0.5,
        window: int = 5,
    ) -> Optional[int]:
        """
        返回第一个满足：
        entropy < eps 持续 window 次
        的 update index（不是 update id）

        若未达到，返回 None
        """
        ents = [r[entropy_key] for r in records if r[entropy_key] is not None]

        for i in range(len(ents) - window):
            ok = True
            for j in range(window):
                if ents[i + j] >= eps:
                    ok = False
                    break
            if ok:
                return i
        return None
