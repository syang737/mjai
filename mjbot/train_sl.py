"""Supervised pretraining: imitate the recorded actions, learn the belief heads, fit the oracle critic.

    python -m mjbot.train_sl --data data/teacher --out checkpoints/sl.pt [--belief-to-policy]

Reports, per epoch on held-out games: move agreement, belief losses, and deal-in AUC of the learned
ron head vs the hand-crafted genbutsu/suji/kabe danger (all tiles the actor could discard, and the
subset where that opponent is in riichi).
"""

import argparse
import glob
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import load
from .features import N_ENGINE
from .model import Net, belief_loss, masked_logits, save


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    label = label.astype(bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def batch(data: dict, idx: np.ndarray) -> dict[str, torch.Tensor]:
    b = {k: torch.from_numpy(data[k][idx]) for k in ("x", "mask", "action", "tenpai", "ron", "holdings", "reward")}
    b["x"] = b["x"].float()
    b["oracle"] = torch.from_numpy(np.concatenate([data["holdings"][idx] / 4, data["ron"][idx]], 1).astype(np.float32))
    return b


def losses(net: Net, b: dict, belief_w: float, value_w: float) -> dict[str, torch.Tensor]:
    out = net(b["x"])
    ls = belief_loss(out, b["ron"], b["holdings"], b["tenpai"])
    ls["policy"] = F.cross_entropy(masked_logits(out["logits"], b["mask"]), b["action"].long())
    ls["value"] = F.mse_loss(net.value(b["x"], b["oracle"]), b["reward"])
    ls["total"] = ls["policy"] + belief_w * (ls["ron"] + ls["holdings"] + ls["tenpai"]) + value_w * ls["value"]
    return ls, out


@torch.no_grad()
def evaluate(net: Net, data: dict, idx: np.ndarray, bs: int = 2048) -> dict[str, float]:
    net.eval()
    sums: dict[str, float] = {}
    correct = 0
    ron_p, heur, ron_y, riichi = [], [], [], []
    for s in range(0, len(idx), bs):
        bi = idx[s : s + bs]
        b = batch(data, bi)
        ls, out = losses(net, b, 1.0, 1.0)
        for k, v in ls.items():
            sums[k] = sums.get(k, 0.0) + v.item() * len(bi)
        correct += (masked_logits(out["logits"], b["mask"]).argmax(1) == b["action"]).sum().item()
        # deal-in prediction on tiles the actor could discard right now
        can = np.broadcast_to(data["mask"][bi][:, None, :34], (len(bi), 3, 34))
        ron_p.append(out["ron"].sigmoid().numpy()[can])
        heur.append(data["x"][bi][:, N_ENGINE : N_ENGINE + 3].astype(np.float32)[can])
        ron_y.append(data["ron"][bi][can])
        riichi.append(np.broadcast_to(data["opp_riichi"][bi][:, :, None], (len(bi), 3, 34))[can])
    net.train()
    m = {k: v / len(idx) for k, v in sums.items()}
    m["move_acc"] = correct / len(idx)
    ron_p, heur, ron_y, riichi = map(np.concatenate, (ron_p, heur, ron_y, riichi))
    m["dealin_auc_learned"] = auc(ron_p, ron_y)
    m["dealin_auc_heuristic"] = auc(heur, ron_y)
    m["dealin_auc_learned_vs_riichi"] = auc(ron_p[riichi], ron_y[riichi])
    m["dealin_auc_heuristic_vs_riichi"] = auc(heur[riichi], ron_y[riichi])
    return m


def split(data: dict, val_frac: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    games = np.unique(data["game"])
    rng = np.random.default_rng(seed)
    val_games = set(rng.choice(games, max(1, int(len(games) * val_frac)), replace=False).tolist())
    is_val = np.array([g in val_games for g in data["game"]])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/teacher")
    ap.add_argument("--out", default="checkpoints/sl.pt")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--belief-w", type=float, default=1.0)
    ap.add_argument("--value-w", type=float, default=0.5)
    ap.add_argument("--belief-to-policy", action="store_true")
    ap.add_argument("--ch", type=int, default=96)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--max-samples", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(0)

    data = load(sorted(glob.glob(f"{a.data}/*.npz")))
    if a.max_samples:
        data = {k: v[: a.max_samples] for k, v in data.items()}
    tr, va = split(data)
    print(f"{len(tr)} train / {len(va)} val samples", flush=True)
    net = Net(a.ch, a.blocks, a.belief_to_policy)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (len(tr) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(0)
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        for s in range(0, len(perm) - a.bs + 1, a.bs):
            ls, _ = losses(net, batch(data, np.sort(perm[s : s + a.bs])), a.belief_w, a.value_w)
            opt.zero_grad()
            ls["total"].backward()
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{steps} " + " ".join(f"{k}={v.item():.3f}" for k, v in ls.items())
                      + f" ({time.time() - t0:.0f}s)", flush=True)
        m = evaluate(net, data, va)
        print(f"epoch {ep} val " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
        save(net, a.out)


if __name__ == "__main__":
    main()
