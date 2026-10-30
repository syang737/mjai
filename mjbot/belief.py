"""Tile belief tensor: per-opponent, per-tile-type marginals of the hidden state.

Instead of a distribution over every opponent hand (~1e32 worlds), the model predicts a few
numbers per (opponent, tile type). `belief_labels` produces ground-truth targets from a
full-information state (Tenhou logs record every hand); `heuristic_danger` is the hand-crafted
genbutsu/suji/kabe baseline the learned head has to beat.

Opponent axis is relative seat order: 0 = shimocha, 1 = toimen, 2 = kamicha.
"""

from dataclasses import dataclass

import numpy as np
from riichienv import Conditions, HandEvaluator, MeldType
from riichienv import Meld as RMeld

from .hand import shanten, waits
from .state import Player, RoundState
from .tiles import N_TYPES, RED_FIVES, IdAllocator

_MELD_TYPES = {
    "chi": MeldType.Chi,
    "pon": MeldType.Pon,
    "daiminkan": MeldType.Daiminkan,
    "kakan": MeldType.Kakan,
    "ankan": MeldType.Ankan,
}


@dataclass
class BeliefLabels:
    holdings: np.ndarray  # (3, 34) int: concealed copies held
    tenpai: np.ndarray  # (3,) bool
    waits: np.ndarray  # (3, 34) bool: shape waits, no yaku check
    ron: np.ndarray  # (3, 34) bool: waits not blocked by furiten and with a yaku
    payment: np.ndarray  # (3, 34) int: points paid on deal-in (incl. honba), 0 where no ron
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


def ron_payment(state: RoundState, seat: int, tile: int) -> int:
    """Points `seat` collects for ron on a (non-red) `tile`, including honba; 0 if that win has no yaku
    or every copy of the tile is already in `seat`'s hand, melds or the dora indicators.

    Ura dora and ippatsu are not counted, so riichi hands are underestimated.
    """
    p = state.players[seat]
    alloc = IdAllocator()

    def meld_ids(m) -> list[int]:
        red = next(i for i, t in enumerate(m.tiles) if t in RED_FIVES) if m.red else None
        return [alloc.take(t, red=i == red) for i, t in enumerate(m.tiles)]

    melds = [RMeld(_MELD_TYPES[m.kind], meld_ids(m), m.kind != "ankan") for m in p.melds]
    hand = [alloc.take(t, red=t in p.red and i == 0) for t in range(N_TYPES) for i in range(p.hand[t])]
    dora = [alloc.take(t) for t in state.dora_indicators]
    if alloc.free(tile) == 0:
        return 0
    win = alloc.take(tile)
    cond = Conditions(
        riichi=p.riichi_seq is not None,
        houtei=state.tiles_left == 0,
        player_wind=state.seat_wind(seat) - 27,
        round_wind=state.round_wind - 27,
        honba=state.honba,
    )
    result = HandEvaluator(hand + [win], melds).calc(win, dora, cond, [])
    return result.ron_agari if result.is_win else 0


def chi_tiles(hand: np.ndarray) -> np.ndarray:
    """(34,) bool: tiles this hand could chi."""
    h = hand[:27].reshape(3, 9) > 0
    c = np.zeros((3, 9), bool)
    c[:, 2:] |= h[:, :7] & h[:, 1:8]  # n-2, n-1
    c[:, 1:8] |= h[:, :7] & h[:, 2:]  # n-1, n+1
    c[:, :7] |= h[:, 1:8] & h[:, 2:]  # n+1, n+2
    return np.concatenate([c.ravel(), np.zeros(7, bool)])


def belief_labels(state: RoundState, seat: int) -> BeliefLabels:
    shape = (3, N_TYPES)
    lab = BeliefLabels(
        holdings=np.zeros(shape, np.int8),
        tenpai=np.zeros(3, bool),
        waits=np.zeros(shape, bool),
        ron=np.zeros(shape, bool),
        payment=np.zeros(shape, np.int32),
        pon=np.zeros(shape, bool),
        chi=np.zeros(shape, bool),
    )
    for j, opp in enumerate(state.opponents(seat)):
        p: Player = state.players[opp]
        lab.holdings[j] = p.hand
        if shanten(p.hand) == 0:
            lab.tenpai[j] = True
            w = waits(p.hand)
            lab.waits[j, w] = True
            for t in set(w) - furiten_tiles(state, opp):
                lab.payment[j, t] = ron_payment(state, opp, t)
            lab.ron[j] = lab.payment[j] > 0
        if p.riichi_seq is None:
            lab.pon[j] = p.hand >= 2
            if j == 0:
                lab.chi[j] = chi_tiles(p.hand)
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


_HONOR_DANGER = np.array([0.25, 0.2, 0.12, 0.05, 0.0])


def heuristic_danger(state: RoundState, seat: int) -> np.ndarray:
    """(3, 34) relative deal-in danger from public info only: genbutsu, suji, kabe (no-chance), honor counts.

    Scores are ordinal, not calibrated probabilities. Number tiles score by how many ryanmen
    shapes could still wait on them; other wait shapes (kanchan, penchan, shanpon, tanki) get a floor.
    """
    visible = state.visible_counts(seat)
    v = visible[:27].reshape(3, 9)
    kabe_ok = (v[:, 1:7] < 4) & (v[:, 2:8] < 4)  # both shape tiles of a ryanmen still available
    danger = np.zeros((3, N_TYPES))
    for j, opp in enumerate(state.opponents(seat)):
        safe = genbutsu(state, seat, opp)
        s9 = safe[:27].reshape(3, 9)
        sides = np.zeros((3, 9))
        # left ryanmen (n-2, n-1) waits on n-3/n; right ryanmen (n+1, n+2) waits on n/n+3
        sides[:, 3:] += ~s9[:, :6] & kabe_ok
        sides[:, :6] += ~s9[:, 3:] & kabe_ok
        danger[j, :27] = 0.1 + 0.3 * sides.ravel()
        danger[j, 27:] = _HONOR_DANGER[visible[27:]]
        danger[j, safe | (visible == 4)] = 0
    return danger


def adjusted_remaining(visible: np.ndarray, expected_holdings: np.ndarray) -> np.ndarray:
    """(34,) expected copies still in the wall: 4 - visible - copies believed held by opponents."""
    return np.clip(4 - visible - expected_holdings.sum(axis=0), 0, None)
