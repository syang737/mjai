"""PPO self-play fine-tuning from the SL checkpoint.

- Reward: Tenhou-style rank points at game end (placement EV, not raw score), optionally plus
  each round's score change (`--round-reward`, per 10k points) to give every round its own signal.
- Critic: sees oracle planes (opponents' hands and ronnable tiles); the actor never does.
- Belief heads keep training on simulator ground truth as an auxiliary loss.
- League: each game seats the learner against itself, the teacher, or frozen snapshots;
  only the learner's decisions are trained on.
- Games are collected by worker processes (one torch thread each); the update runs in the
  main process.

    python -m mjbot.train_rl --init checkpoints/sl_b2p.pt --out checkpoints/rl.pt --iters 40
"""

import argparse
import copy
import multiprocessing as mp
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .agents import NetAgent
from .arena import duplicate_match, fmt
from .features import labels, oracle_planes
from .model import Net, belief_loss, load, masked_logits, save
from .runner import RANK_REWARD, run_games
from .teacher import Teacher

_FIELDS = ("x", "mask", "action", "logp", "oracle", "ron", "holdings", "tenpai", "key", "round")


class Buffer:
    def __init__(self, chunk: int = 0):
        self.chunk = chunk
        self.rows = {k: [] for k in _FIELDS}

    def on_act(self, decisions, x, a, logp) -> None:
        for i, d in enumerate(decisions):
            if d.mask.sum() <= 1:
                continue
            lab = labels(d)
            r = self.rows
            r["x"].append(x[i].numpy().astype(np.float16))
            r["mask"].append(d.mask)
            r["action"].append(int(a[i]))
            r["logp"].append(float(logp[i]))
            r["oracle"].append(oracle_planes(lab).astype(np.float16))
            r["ron"].append(lab.ron)
            r["holdings"].append(lab.holdings)
            r["tenpai"].append(lab.tenpai)
            r["key"].append((self.chunk, d.game, d.seat))
            r["round"].append(d.round_index)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.stack(v) if k in ("x", "mask", "oracle", "ron", "holdings", "tenpai") else np.array(v)
                for k, v in self.rows.items() if k not in ("key", "round")}


def collect(job) -> tuple:
    """Play one chunk of games. `job` carries plain data (state dicts, lineup names) so it can be
    sent to a worker process. Returns buffer arrays, trajectory keys, round indices, final rewards
    per key, per-round score changes per key, and the learner's ranks."""
    chunk, config, states, lineups, seeds, threads = job
    torch.set_num_threads(threads)
    buf = Buffer(chunk)
    agents = {"teacher": Teacher()}
    for name, state in states.items():
        net = Net(**config)
        net.load_state_dict(state)
        agents[name] = NetAgent(net, greedy=False, name=name, on_act=buf.on_act if name == "learner" else None)
    results = run_games(seeds, lambda i: [agents[n] for n in lineups[i]], n_parallel=len(seeds))
    rewards = {(chunk, r.game, s): RANK_REWARD[r.ranks[s] - 1] for r in results for s in range(4)}
    deltas = {(chunk, r.game, s): [d[s] for d in r.round_deltas] for r in results for s in range(4)}
    learner_ranks = [r.ranks[s] for r in results for s in range(4) if lineups[r.game][s] == "learner"]
    return buf.arrays(), buf.rows["key"], buf.rows["round"], rewards, deltas, learner_ranks


def step_rewards(rounds: list[int], final: float, round_rewards: list[float]) -> np.ndarray:
    """Rewards for one trajectory: each round's reward lands on the seat's last decision in that
    round (rounds without a recorded decision roll forward), the final reward on the last decision."""
    r = np.zeros(len(rounds), np.float32)
    paid = 0  # rounds already credited
    for pos, rnd in enumerate(rounds):
        last = pos == len(rounds) - 1
        if last or rounds[pos + 1] != rnd:
            upto = len(round_rewards) if last else rnd + 1
            r[pos] = sum(round_rewards[paid:upto])
            paid = upto
    r[-1] += final
    return r


def advantages(keys, rounds, values: np.ndarray, rewards: dict, round_rewards: dict, gamma: float, lam: float):
    """GAE per (game, seat) trajectory."""
    adv = np.zeros(len(keys), np.float32)
    traj: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        traj.setdefault(k, []).append(i)
    for k, idx in traj.items():
        r = step_rewards([rounds[i] for i in idx], rewards[k], round_rewards.get(k, []))
        nxt_v, gae = 0.0, 0.0
        for j in range(len(idx) - 1, -1, -1):
            i = idx[j]
            delta = r[j] + gamma * nxt_v - values[i]
            gae = delta + gamma * lam * gae
            adv[i] = gae
            nxt_v = values[i]
    return adv, adv + values


def _batched(fn, *tensors, bs: int = 4096) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([fn(*(t[s : s + bs] for t in tensors)) for s in range(0, len(tensors[0]), bs)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="checkpoints/sl_b2p.pt")
    ap.add_argument("--out", default="checkpoints/rl.pt")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--games", type=int, default=64, help="games per iteration")
    ap.add_argument("--workers", type=int, default=4, help="collection processes (0 = in-process)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--aux", type=float, default=0.5)
    ap.add_argument("--anchor-kl", type=float, default=0.02, help="KL to the SL policy, against drift")
    ap.add_argument("--round-reward", type=float, default=0.0, help="reward per 10k points gained in a round")
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
    snapshots = {"snap0": copy.deepcopy(net.state_dict())}
    pool = mp.get_context("spawn").Pool(a.workers) if a.workers else None
    n_chunks = max(1, a.workers)
    seed = 0
    for it in range(1, a.iters + 1):
        t0 = time.time()
        lineups = []
        for _ in range(a.games):
            u = rng.random()
            other = "teacher" if u < 0.25 else rng.choice(list(snapshots)) if u < 0.5 else "learner"
            n_learn = 4 if other == "learner" else rng.choice((1, 2))
            seats = ["learner"] * n_learn + [other] * (4 - n_learn)
            rng.shuffle(seats)
            lineups.append(seats)
        states = {"learner": net.state_dict(), **snapshots}
        jobs = []
        for c in range(n_chunks):
            idx = list(range(c, a.games, n_chunks))
            jobs.append((c, net.config, states, [lineups[i] for i in idx], [seed + i for i in idx],
                         1 if pool else torch.get_num_threads()))
        seed += a.games
        outs = pool.map(collect, jobs) if pool else [collect(j) for j in jobs]
        b = {k: torch.from_numpy(np.concatenate([o[0][k] for o in outs])) for k in outs[0][0]}
        b["x"], b["oracle"] = b["x"].float(), b["oracle"].float()
        keys = [k for o in outs for k in o[1]]
        rounds = [r for o in outs for r in o[2]]
        rewards = {k: v for o in outs for k, v in o[3].items()}
        round_rewards = {k: [a.round_reward * d / 10000 for d in v] for o in outs for k, v in o[4].items()}
        learner_ranks = [r for o in outs for r in o[5]]
        t_collect = time.time() - t0

        torch.set_num_threads(max(1, a.workers))
        net.eval()
        values = _batched(net.value, b["x"], b["oracle"]).numpy()
        anchor_logp = _batched(lambda x, m: masked_logits(anchor(x)["logits"], m).log_softmax(1), b["x"], b["mask"])
        net.train()
        adv, ret = advantages(keys, rounds, values, rewards, round_rewards, a.gamma, a.lam)
        adv_t = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
        ret_t = torch.from_numpy(ret)
        n = len(adv)
        for _ in range(a.epochs):
            for idx in torch.randperm(n).split(a.bs):
                x, mask, act = b["x"][idx], b["mask"][idx], b["action"][idx]
                out = net(x)
                logp_all = masked_logits(out["logits"], mask).log_softmax(1)
                logp = logp_all.gather(1, act[:, None]).squeeze(1)
                ratio = (logp - b["logp"][idx]).exp()
                pg = -torch.min(ratio * adv_t[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * adv_t[idx]).mean()
                p = logp_all.exp()
                ent = -(p * logp_all.masked_fill(~mask, 0)).sum(1).mean()
                kl = (p * (logp_all - anchor_logp[idx]).masked_fill(~mask, 0)).sum(1).mean()
                vl = F.mse_loss(net.value(x, b["oracle"][idx]), ret_t[idx])
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
            snapshots[f"snap{it}"] = copy.deepcopy(net.state_dict())
            snapshots = dict(list(snapshots.items())[-4:])
        save(net, a.out)
        if a.eval_every and it % a.eval_every == 0:
            save(net, a.out.replace(".pt", f"_iter{it}.pt"))
            m = duplicate_match(NetAgent(net, greedy=True), Teacher(), list(range(10**6, 10**6 + a.eval_seeds)),
                                game_mode="4p-red-east")
            print(f"  eval vs 3x teacher (east): {fmt(m)}", flush=True)
    if pool:
        pool.close()


if __name__ == "__main__":
    main()
