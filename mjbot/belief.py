"""Tile belief tensor: per-opponent, per-tile-type marginals of the hidden state.

Instead of a distribution over every opponent hand (~1e32 worlds), the model predicts a few
numbers per (opponent, tile type). `belief_labels` produces ground-truth targets from a
full-information state (Tenhou logs record every hand); `heuristic_danger` is the hand-crafted
genbutsu/suji/kabe baseline the learned head has to beat.

Opponent axis is relative seat order: 0 = shimocha, 1 = toimen, 2 = kamicha.
"""

from dataclasses import dataclass

import numpy as np

from .hand import shanten, waits
from .state import Player, RoundState
from .tiles import N_TYPES, is_honor, number


@dataclass
class BeliefLabels:
    holdings: np.ndarray  # (3, 34) int: concealed copies held
    tenpai: np.ndarray  # (3,) bool
    waits: np.ndarray  # (3, 34) bool: shape waits, no yaku check
    ron: np.ndarray  # (3, 34) bool: waits not blocked by furiten
    pon: np.ndarray  # (3, 34) bool: could pon this discard
    chi: np.ndarray  # (3, 34) bool: could chi this discard (shimocha only)


def furiten_tiles(state: RoundState, seat: int) -> set[int]:
    """Tiles `seat` cannot ron on: own discards, tiles passed since its last discard, and tiles
    passed after its riichi."""
    p = state.players[seat]
    blocked = {d.tile for d in p.discards}
    since = max((d.seq for d in p.discards), default=-1)
    if (r := p.riichi_seq) is not None:
        since = min(since, r)
    for other in state.players:
        if other is not p:
            blocked |= {d.tile for d in other.discards if d.seq > since}
    return blocked


def _chi_shapes(hand: np.ndarray, t: int) -> bool:
    if is_honor(t):
        return False
    n = number(t)
    return any(
        1 <= n + a <= 9 and 1 <= n + b <= 9 and hand[t + a] and hand[t + b]
        for a, b in ((-2, -1), (-1, 1), (1, 2))
    )


def belief_labels(state: RoundState, seat: int) -> BeliefLabels:
    shape = (3, N_TYPES)
    lab = BeliefLabels(
        holdings=np.zeros(shape, np.int8),
        tenpai=np.zeros(3, bool),
        waits=np.zeros(shape, bool),
        ron=np.zeros(shape, bool),
        pon=np.zeros(shape, bool),
        chi=np.zeros(shape, bool),
    )
    for j, opp in enumerate(state.opponents(seat)):
        p: Player = state.players[opp]
        closed = not p.is_open
        lab.holdings[j] = p.hand
        if shanten(p.hand, closed) == 0:
            lab.tenpai[j] = True
            w = waits(p.hand, closed)
            lab.waits[j, w] = True
            lab.ron[j, sorted(set(w) - furiten_tiles(state, opp))] = True
        if p.riichi_seq is None:
            lab.pon[j] = p.hand >= 2
            if j == 0:
                lab.chi[j] = [_chi_shapes(p.hand, t) for t in range(N_TYPES)]
    return lab


def genbutsu(state: RoundState, seat: int, opp: int) -> np.ndarray:
    """(34,) bool: tiles `opp` provably cannot ron, as seen from `seat` (no hidden info used)."""
    p = state.players[opp]
    safe = np.zeros(N_TYPES, bool)
    for d in p.discards:
        safe[d.tile] = True
    if (r := p.riichi_seq) is not None:
        for other in state.players:
            for d in other.discards:
                if d.seq > r:
                    safe[d.tile] = True
    return safe


def heuristic_danger(state: RoundState, seat: int) -> np.ndarray:
    """(3, 34) relative deal-in danger from public info only: genbutsu, suji, kabe (no-chance), honor counts.

    Scores are ordinal, not calibrated probabilities. Number tiles score by how many ryanmen
    shapes could still wait on them; other wait shapes (kanchan, penchan, shanpon, tanki) get a floor.
    """
    visible = state.visible_counts(seat)
    danger = np.zeros((3, N_TYPES))
    for j, opp in enumerate(state.opponents(seat)):
        safe = genbutsu(state, seat, opp)
        for t in range(N_TYPES):
            if safe[t] or visible[t] == 4:
                continue
            if is_honor(t):
                danger[j, t] = (0.25, 0.2, 0.12, 0.05)[visible[t]]
                continue
            n = number(t)
            sides = 0
            # left ryanmen (n-2, n-1) waits on n-3/n; right ryanmen (n+1, n+2) waits on n/n+3
            if n >= 4 and not safe[t - 3] and visible[t - 2] < 4 and visible[t - 1] < 4:
                sides += 1
            if n <= 6 and not safe[t + 3] and visible[t + 1] < 4 and visible[t + 2] < 4:
                sides += 1
            danger[j, t] = 0.1 + 0.3 * sides
    return danger


def adjusted_remaining(visible: np.ndarray, expected_holdings: np.ndarray) -> np.ndarray:
    """(34,) expected copies still in the wall: 4 - visible - copies believed held by opponents."""
    return np.clip(4 - visible - expected_holdings.sum(axis=0), 0, None)
