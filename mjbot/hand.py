"""Own-hand math: shanten, waits, ukeire. Ukeire takes a `remaining` vector so it can be
computed on the belief-adjusted wall rather than the naive 4 - visible count.

Shanten comes from riichienv's Rust implementation (~0.5us per call). Hands with melds simply
have fewer tiles; chiitoitsu/kokushi can't apply to them because they need 13 concealed tiles.
"""

import numpy as np
from riichienv import calculate_shanten

from .tiles import N_TYPES


def _ids(counts: np.ndarray) -> list[int]:
    return [t * 4 + k for t in range(N_TYPES) for k in range(counts[t])]


def shanten(counts: np.ndarray) -> int:
    """-1 = complete."""
    return calculate_shanten(_ids(counts))


def _improving(c: list[int], ids: list[int], base: int) -> list[int]:
    return [t for t in range(N_TYPES) if c[t] < 4 and calculate_shanten(ids + [t * 4 + c[t]]) < base]


def improving_tiles(counts: np.ndarray) -> list[int]:
    """For a 3n+1 hand: tile types whose draw lowers shanten. At tenpai these are the waits."""
    assert counts.sum() % 3 == 1, "expects a hand waiting for a draw (3n+1 tiles)"
    ids = _ids(counts)
    return _improving(counts.tolist(), ids, calculate_shanten(ids))


def waits(counts: np.ndarray) -> list[int]:
    """Shape waits (no yaku check). Empty unless tenpai."""
    return improving_tiles(counts) if shanten(counts) == 0 else []


def ukeire(counts: np.ndarray, remaining: np.ndarray) -> float:
    """Expected number of drawable tiles that lower shanten."""
    return float(sum(remaining[t] for t in improving_tiles(counts)))


def discard_table(counts: np.ndarray, remaining: np.ndarray) -> dict[int, tuple[int, float]]:
    """For a 3n+2 hand: {tile type: (shanten, ukeire) after discarding one copy of it}."""
    c = counts.tolist()
    rem = remaining.tolist()
    ids = _ids(counts)
    out = {}
    for t in range(N_TYPES):
        if not c[t]:
            continue
        c[t] -= 1
        h = [i for i in ids if i != t * 4 + c[t]]
        sh = calculate_shanten(h)
        out[t] = (sh, float(sum(rem[u] for u in _improving(c, h, sh))))
        c[t] += 1
    return out
