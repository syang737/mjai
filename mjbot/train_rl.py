"""PPO self-play fine-tuning from the SL checkpoint.

- Reward: Tenhou-style rank points at game end (placement EV, not raw score).
- Critic: sees oracle planes (opponents' hands and ronnable tiles); the actor never does.
- Belief heads keep training on simulator ground truth as an auxiliary loss.
- League: each game seats the learner against itself, the teacher, or frozen snapshots;
  only the learner's decisions are trained on.

    python -m mjbot.train_rl --init checkpoints/sl_b2p.pt --out checkpoints/rl.pt --iters 40
"""

import argparse
import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .agents import NetAgent
from .arena import duplicate_match, fmt
from .features import labels, oracle_planes
from .model import belief_loss, load, masked_logits, save
from .runner import RANK_REWARD, run_games
from .teacher import Teacher


class Buffer:
    def __init__(self):
        self.rows = {k: [] for k in ("x", "mask", "action", "logp", "oracle", "ron", "holdings", "tenpai", "key")}

    def on_act(self, decisions, x, a, logp) -> None:
        for i, d in enumerate(decisions):
            if d.mask.sum() <= 1:
                continue
            lab = labels(d)
            r = self.rows
            r["x"].append(x[i].numpy())
            r["mask"].append(d.mask)
            r["action"].append(int(a[i]))
            r["logp"].append(float(logp[i]))
            r["oracle"].append(oracle_planes(lab))
            r["ron"].append(lab.ron)
            r["holdings"].append(lab.holdings)
            r["tenpai"].append(lab.tenpai)
            r["key"].append((d.game, d.seat))

    def tensors(self) -> dict[str, torch.Tensor]:
        out = {k: torch.from_numpy(np.stack(v) if k in ("x", "mask", "oracle", "ron", "holdings", "tenpai")
                                   else np.array(v)) for k, v in self.rows.items() if k != "key"}
        return out


def advantages(keys, values: np.ndarray, rewards: dict, gamma: float, lam: float):
    """GAE per (game, seat) trajectory; reward only at the final decision."""
    adv = np.zeros(len(keys), np.float32)
    traj: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        traj.setdefault(k, []).append(i)
    for k, idx in traj.items():
        nxt_v, gae = 0.0, 0.0
        for j, i in enumerate(reversed(idx)):
            r = rewards[k] if j == 0 else 0.0
            delta = r + gamma * nxt_v - values[i]
            gae = delta + gamma * lam * gae
            adv[i] = gae
            nxt_v = values[i]
    return adv, adv + values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/sl_b2p.pt")
    ap.add_argument("--out", default="checkpoints/rl.pt")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games", type=int, default=64, help="games per iteration")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--aux", type=float, default=0.5)
    ap.add_argument("--anchor-kl", type=float, default=0.02, help="KL to the SL policy, against drift")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-seeds", type=int, default=40)
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)

    net = load(a.init)
    anchor = copy.deepcopy(net).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    teacher = Teacher()
    snapshots = [NetAgent(copy.deepcopy(net), greedy=False, name="snap0")]
    seed = 0
    for it in range(1, a.iters + 1):
        t0 = time.time()
        buf = Buffer()
        learner = NetAgent(net, greedy=False, name="learner", on_act=buf.on_act)
        lineups = []
        for _ in range(a.games):
            u = rng.random()
            other = teacher if u < 0.25 else rng.choice(snapshots) if u < 0.5 else learner
            n_learn = 4 if other is learner else rng.choice((1, 2))
            seats = [learner] * n_learn + [other] * (4 - n_learn)
            rng.shuffle(seats)
            lineups.append(seats)
        results = run_games(list(range(seed, seed + a.games)), lambda i: lineups[i],
                            n_parallel=a.games)
        seed += a.games
        rewards = {(r.game, s): RANK_REWARD[r.ranks[s] - 1] for r in results for s in range(4)}
        learner_ranks = [r.ranks[s] for r in results for s in range(4) if lineups[r.game][s] is learner]
        t_collect = time.time() - t0

        net.train()
        b = buf.tensors()
        with torch.no_grad():
            values = torch.cat([net.value(b["x"][s : s + 4096], b["oracle"][s : s + 4096])
                                for s in range(0, len(b["x"]), 4096)]).numpy()
        adv, ret = advantages(buf.rows["key"], values, rewards, a.gamma, a.lam)
        adv_t = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
        ret_t = torch.from_numpy(ret)
        n = len(adv)
        for _ in range(a.epochs):
            for idx in torch.randperm(n).split(a.bs):
                x, mask, act = b["x"][idx], b["mask"][idx], b["action"][idx]
                out = net(x)
                logits = masked_logits(out["logits"], mask)
                logp_all = logits.log_softmax(1)
                logp = logp_all.gather(1, act[:, None]).squeeze(1)
                ratio = (logp - b["logp"][idx]).exp()
                pg = -torch.min(ratio * adv_t[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t[idx]).mean()
                p = logp_all.exp()
                ent = -(p * logp_all.masked_fill(~mask, 0)).sum(1).mean()
                with torch.no_grad():
                    anchor_logp = masked_logits(anchor(x)["logits"], mask).log_softmax(1)
                kl = (p * (logp_all - anchor_logp).masked_fill(~mask, 0)).sum(1).mean()
                v = net.value(x, b["oracle"][idx])
                vl = F.mse_loss(v, ret_t[idx])
                bl = belief_loss(out, b["ron"][idx], b["holdings"][idx], b["tenpai"][idx])
                loss = pg - a.ent * ent + a.vf * vl + a.anchor_kl * kl + a.aux * sum(bl.values())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
        net.eval()
        print(f"iter {it}: {n} samples, learner avg rank {np.mean(learner_ranks):.3f} "
              f"pg {pg.item():.3f} v {vl.item():.3f} ent {ent.item():.3f} kl_anchor {kl.item():.4f} "
              f"ron {bl['ron'].item():.4f} (collect {t_collect:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
        if it % 5 == 0:
            snapshots = (snapshots + [NetAgent(copy.deepcopy(net), greedy=False, name=f"snap{it}")])[-4:]
        save(net, a.out)
        if a.eval_every and it % a.eval_every == 0:
            m = duplicate_match(NetAgent(net, greedy=True), teacher, list(range(10**6, 10**6 + a.eval_seeds)),
                                game_mode="4p-red-east")
            print(f"  eval vs 3x teacher (east): {fmt(m)}", flush=True)


if __name__ == "__main__":
    main()
