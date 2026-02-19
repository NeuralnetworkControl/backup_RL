# import torch
# import torch.nn as nn
#
# class FeatureTransformerEncoder(nn.Module):
#     def __init__(
#         self,
#         state_dim: int,
#         d_model: int = 256,
#         nhead: int = 8,
#         num_layers: int = 4,
#         dim_feedforward: int = 1024,
#         dropout: float = 0.1,
#         pooling: str = "mean",   # "mean" or "cls"
#     ):
#         super().__init__()
#         assert pooling in ("mean", "cls")
#         self.state_dim = state_dim
#         self.d_model = d_model
#         self.pooling = pooling
#
#         self.in_proj = nn.Linear(1, d_model)
#
#         extra = 1 if pooling == "cls" else 0
#         self.pos_emb = nn.Parameter(torch.zeros(1, state_dim + extra, d_model))
#
#         if pooling == "cls":
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
#         else:
#             self.cls_token = None
#
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,
#             activation="gelu",
#             norm_first=True,
#         )
#         self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
#         self.norm = nn.LayerNorm(d_model)
#
#         nn.init.trunc_normal_(self.pos_emb, std=0.02)
#         if self.cls_token is not None:
#             nn.init.trunc_normal_(self.cls_token, std=0.02)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 兼容：(state_dim,) -> (1, state_dim)
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         elif x.dim() != 2:
#             raise ValueError(f"Expected x dim 1 or 2, got shape={tuple(x.shape)}")
#
#         B, D = x.shape
#         if D != self.state_dim:
#             raise ValueError(f"Expected state_dim={self.state_dim}, got {D}")
#
#         # (B, D, 1) -> (B, D, d_model)
#         h = self.in_proj(x.unsqueeze(-1))
#
#         if self.pooling == "cls":
#             cls = self.cls_token.expand(B, -1, -1)  # (B,1,C)
#             h = torch.cat([cls, h], dim=1)          # (B,1+D,C)
#
#         h = h + self.pos_emb[:, :h.size(1), :]
#         h = self.encoder(h)
#         h = self.norm(h)
#
#         if self.pooling == "cls":
#             return h[:, 0]        # (B, d_model)
#         else:
#             return h.mean(dim=1)  # (B, d_model)
#
#
# class PolicyNet(nn.Module):
#     def __init__(
#         self,
#         state_dim: int,
#         action_dim: int,
#         d_model: int = 256,
#         nhead: int = 8,
#         num_layers: int = 4,
#         dim_feedforward: int = 1024,
#         dropout: float = 0.1,
#         pooling: str = "mean",
#         return_probs: bool = True,
#     ):
#         super().__init__()
#         self.backbone = FeatureTransformerEncoder(
#             state_dim=state_dim,
#             d_model=d_model,
#             nhead=nhead,
#             num_layers=num_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             pooling=pooling,
#         )
#         self.head = nn.Linear(d_model, action_dim)
#         self.return_probs = return_probs
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         single = (x.dim() == 1)
#         h = self.backbone(x)          # (B, d_model)
#         logits = self.head(h)         # (B, action_dim)
#         out = self.softmax(logits) if self.return_probs else logits
#         return out.squeeze(0) if single else out   # 单条输入时返回 (action_dim,)
#
#
# class Classifier(nn.Module):
#     def __init__(
#         self,
#         state_dim: int,
#         d_model: int = 256,
#         nhead: int = 8,
#         num_layers: int = 4,
#         dim_feedforward: int = 1024,
#         dropout: float = 0.1,
#         pooling: str = "mean",
#     ):
#         super().__init__()
#         self.backbone = FeatureTransformerEncoder(
#             state_dim=state_dim,
#             d_model=d_model,
#             nhead=nhead,
#             num_layers=num_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             pooling=pooling,
#         )
#         self.head = nn.Linear(d_model, 1)
#
#     def forward(self, x):
#         single = (x.dim() == 1)
#         h = self.backbone(x)              # (B, d_model)
#         logits = self.head(h).squeeze(-1) # (B,)
#         return logits.squeeze(0) if single else logits




import torch
import torch.nn as nn


# ------------------ Models ------------------
# Policy network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)


# Classifier
class Classifier(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        # returns probability after sigmoid outside or logits depending on use
        return self.net(x).squeeze(-1)  # logits