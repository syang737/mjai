"""Duplicate-style evaluation: one candidate vs three copies of a baseline, every wall played with
the candidate in each of the four seats (same seed, rotated), which removes most seat/wall luck.

    python -m mjbot.arena --a checkpoints/rl.pt --b teacher --seeds 100
    python -m mjbot.arena --a new.pt --c old.pt --b teacher   # paired: is new better than old vs teacher?
"""

import argparse

import numpy as np

from .agents import make_agent
from .runner import RANK_REWARD, run_games


def duplicate_match(candidate, baseline, seeds, game_mode: str = "4p-red-half", n_parallel: int = 64) -> dict:
    games = [s for s in seeds for _ in range(4)]

    def seats(i: int):
        agents = [baseline] * 4
        agents[i % 4] = candidate
        return agents

    results = run_games(games, seats, n_parallel=n_parallel, game_mode=game_mode)
    ranks = np.array([r.ranks[r.game % 4] for r in results])
    scores = np.array([r.scores[r.game % 4] for r in results])
    stats = [r.stats[r.game % 4] for r in results]
    rounds = sum(s["rounds"] for s in stats)
    pts = RANK_REWARD[ranks - 1] * 90
    # games sharing a seed are correlated, so the standard error is over seeds
    per_seed_rank = ranks.reshape(-1, 4).mean(1)
    per_seed_pts = pts.reshape(-1, 4).mean(1)
    return dict(
        games=len(results),
        avg_rank=ranks.mean(),
        avg_rank_se=per_seed_rank.std(ddof=1) / np.sqrt(len(per_seed_rank)),
        rank_pts=pts.mean(),
        rank_pts_se=per_seed_pts.std(ddof=1) / np.sqrt(len(per_seed_pts)),
        rank_dist=np.bincount(ranks, minlength=5)[1:] / len(ranks),
        avg_score=scores.mean(),
        win_rate=sum(s["wins"] for s in stats) / rounds,
        deal_in_rate=sum(s["deal_ins"] for s in stats) / rounds,
        riichi_rate=sum(s["riichi"] for s in stats) / rounds,
        call_rate=sum(s["calls"] for s in stats) / rounds,
        per_seed_rank=per_seed_rank,
        per_seed_pts=per_seed_pts,
    )


def paired(a: dict, c: dict) -> str:
    """Difference a - c from two matches on the same seeds; the SE is over per-seed differences, so
    wall/seat luck shared by both candidates cancels."""
    out = []
    for key, name in (("per_seed_rank", "avg_rank"), ("per_seed_pts", "rank_pts")):
        d = a[key] - c[key]
        out.append(f"{name} {d.mean():+.3f}±{d.std(ddof=1) / np.sqrt(len(d)):.3f}")
    return "  ".join(out)


def fmt(m: dict) -> str:
    return (f"avg_rank {m['avg_rank']:.3f}±{m['avg_rank_se']:.3f}  rank_pts {m['rank_pts']:+.1f}±{m['rank_pts_se']:.1f}  "
            f"dist {np.round(m['rank_dist'], 3).tolist()}  score {m['avg_score']:.0f}  win {m['win_rate']:.3f}  "
            f"deal-in {m['deal_in_rate']:.3f}  riichi {m['riichi_rate']:.3f}  calls/round {m['call_rate']:.3f}  "
            f"({m['games']} games)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="candidate: 'teacher' or checkpoint path")
    ap.add_argument("--b", default="teacher", help="baseline (3 seats)")
    ap.add_argument("--c", default="", help="second candidate, played on the same seeds for a paired comparison with --a")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=10**6)  # away from training seeds
    ap.add_argument("--mode", default="4p-red-half")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    baseline = make_agent(a.b)
    m = duplicate_match(make_agent(a.a), baseline, seeds, a.mode)
    print(f"{a.a} vs 3x {a.b}: {fmt(m)}", flush=True)
    if a.c:
        mc = duplicate_match(make_agent(a.c), baseline, seeds, a.mode)
        print(f"{a.c} vs 3x {a.b}: {fmt(mc)}")
        print(f"paired {a.a} - {a.c}: {paired(m, mc)}")


if __name__ == "__main__":
    main()
