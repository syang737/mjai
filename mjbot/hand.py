"""Own-hand math: shanten, waits, ukeire. Ukeire takes a `remaining` vector so it can be
computed on the belief-adjusted wall rather than the naive 4 - visible count."""

import numpy as np
from mahjong.shanten import Shanten

from .tiles import N_TYPES

_SHANTEN = Shanten()


def shanten(counts: np.ndarray, closed: bool = True) -> int:
    """-1 = complete. Chiitoitsu/kokushi only count for closed hands."""
    return _SHANTEN.calculate_shanten(list(map(int, counts)), use_chiitoitsu=closed, use_kokushi=closed)


def improving_tiles(counts: np.ndarray, closed: bool = True) -> list[int]:
    """For a 3n+1 hand: tile types whose draw lowers shanten. At tenpai these are the waits."""
    assert counts.sum() % 3 == 1, "expects a hand waiting for a draw (3n+1 tiles)"
    counts = counts.copy()
    base = shanten(counts, closed)
    out = []
    for t in range(N_TYPES):
        if counts[t] >= 4:
            continue
        counts[t] += 1
        if shanten(counts, closed) < base:
            out.append(t)
        counts[t] -= 1
    return out


def waits(counts: np.ndarray, closed: bool = True) -> list[int]:
    """Shape waits (no yaku check). Empty unless tenpai."""
    return improving_tiles(counts, closed) if shanten(counts, closed) == 0 else []


def ukeire(counts: np.ndarray, remaining: np.ndarray, closed: bool = True) -> float:
    """Expected number of drawable tiles that lower shanten."""
    return float(sum(remaining[t] for t in improving_tiles(counts, closed)))
