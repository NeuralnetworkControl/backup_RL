# network_trainers.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def train_policy_from_spv(
    policy,
    S_Pai_V,
    device,
    lr=1e-4,
    batch_size=64,
    epochs=1,
    clip_grad=5.0,
    opt=None,                 # ✅ 复用外部 optimizer
):
    """
    Soft-label CE:
      L = -E_s [ sum_a pi_search(a|s) * log pi_theta(a|s) ]
    说明：你存的 pai 是长度 5（info actions），所以这里只监督前 5 维。
    """
    if S_Pai_V is None or len(S_Pai_V) == 0:
        return 0.0

    states = torch.from_numpy(np.stack([d["s"] for d in S_Pai_V])).float()
    targets = torch.stack([d["pai"] for d in S_Pai_V]).float()  # (N,K+1)

    ds = TensorDataset(states, targets)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    if opt is None:
        opt = torch.optim.Adam(policy.parameters(), lr=lr)

    policy.train()
    total_loss, n_batches = 0.0, 0

    for _ in range(epochs):
        for x, pi_tgt in dl:
            x = x.to(device)
            pi_tgt = pi_tgt.to(device)  # (B,5)

            pi_pred = policy(x)  # (B,6) probs
            pi_pred_info = pi_pred[:, :pi_tgt.size(1)]  # (B,5)

            eps = 1e-8
            loss = -(pi_tgt * torch.log(pi_pred_info + eps)).sum(dim=1).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad)
            opt.step()

            total_loss += float(loss.item())
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_classifier_from_samples(
    clf,
    samples,
    device,
    lr=1e-4,
    batch_size=128,
    epochs=1,
    clip_grad=5.0,
    opt=None,                 # ✅ 复用外部 optimizer
):
    """
    logits = clf(x)
    loss = BCEWithLogitsLoss(logits, y)
    samples: list of {"x": state_vec, "y": label, ...}
    """
    if samples is None or len(samples) == 0:
        return 0.0, 0.0

    xs = torch.from_numpy(np.stack([s["x"] for s in samples])).float()
    ys = torch.tensor([int(s["y"]) for s in samples], dtype=torch.float32)

    ds = TensorDataset(xs, ys)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    if opt is None:
        opt = torch.optim.Adam(clf.parameters(), lr=lr)

    criterion = torch.nn.BCEWithLogitsLoss()

    clf.train()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for _ in range(epochs):
        for x, y in dl:
            x = x.to(device)
            y = y.to(device)

            logits = clf(x)  # (B,)
            loss = criterion(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(clf.parameters(), clip_grad)
            opt.step()

            total_loss += float(loss.item())

            with torch.no_grad():
                pred = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += int((pred == y).sum().item())
                total_n += int(y.numel())

    acc = total_correct / max(total_n, 1)
    avg_loss = total_loss / max(len(dl) * epochs, 1)
    return avg_loss, acc
