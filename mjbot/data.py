"""Record (features, legal mask, action, belief labels, outcome) samples from self-play and save shards.

    python -m mjbot.data --games 1200 --out data/teacher --workers 4
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np

from .features import labels, planes
from .runner import RANK_REWARD, Decision, run_games


class Recorder:
    """`on_decision` callback for run_games. Skips forced moves (one legal action)."""

    def __init__(self, record_seats=None):
        self.record_seats = record_seats  # callable (game, seat) -> bool, or None for all
        self.rows: dict[str, list] = {k: [] for k in
                                      ("x", "mask", "action", "tenpai", "ron", "holdings", "payment",
                                       "opp_riichi", "game", "seat")}

    def __call__(self, d: Decision, a: int) -> None:
        if d.mask.sum() <= 1 or (self.record_seats and not self.record_seats(d.game, d.seat)):
            return
        lab = labels(d)
        r = self.rows
        r["x"].append(planes(d).astype(np.float16))
        r["mask"].append(d.mask)
        r["action"].append(a)
        r["tenpai"].append(lab.tenpai)
        r["ron"].append(lab.ron)
        r["holdings"].append(lab.holdings)
        r["payment"].append(np.minimum(lab.payment, 2**15 - 1).astype(np.int16))
        r["opp_riichi"].append([d.state.players[o].riichi_seq is not None for o in d.state.opponents(d.seat)])
        r["game"].append(d.game)
        r["seat"].append(d.seat)

    def arrays(self, results) -> dict[str, np.ndarray]:
        out = {k: np.array(v) if k in ("action", "game", "seat") else np.stack(v)
               for k, v in self.rows.items()}
        rank = {(res.game, s): res.ranks[s] for res in results for s in range(4)}
        out["reward"] = np.array([RANK_REWARD[rank[g, s] - 1] for g, s in zip(out["game"], out["seat"])], np.float32)
        return out


def _gen_shard(args) -> str:
    from .teacher import Teacher

    shard, seeds, out = args
    teacher = Teacher()
    rec = Recorder()
    results = run_games(seeds, lambda i: [teacher] * 4, on_decision=rec)
    arr = rec.arrays(results)
    arr["game"] += seeds[0]  # unique across shards
    path = Path(out) / f"shard{shard:04d}.npz"
    np.savez_compressed(path, **arr)
    return f"{path}: {len(arr['action'])} samples"


def load(paths) -> dict[str, np.ndarray]:
    parts = [dict(np.load(p)) for p in paths]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--per-shard", type=int, default=100)
    ap.add_argument("--out", default="data/teacher")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(i, list(range(a.seed + s, a.seed + s + a.per_shard)), a.out)
            for i, s in enumerate(range(0, a.games, a.per_shard))]
    with mp.Pool(a.workers) as pool:
        for msg in pool.imap_unordered(_gen_shard, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
