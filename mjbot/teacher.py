"""Rule-based bot: tile efficiency on offense, heuristic-danger folding against riichi.

It generates the bootstrap data for supervised pretraining (Tenhou logs aren't reachable from this
environment) and is the first baseline the learned agents have to beat.
"""

import numpy as np

from .actions import AGARI, PASS, PON, RED_DISCARD, RIICHI, RYUKYOKU
from .belief import heuristic_danger
from .hand import shanten, ukeire
from .runner import Decision
from .tiles import RED_FIVES, dora_from_indicator, is_honor, number

_RED_INDEX = {v: k for k, v in RED_DISCARD.items()}


def _keep_value(t: int, d: Decision, hand: np.ndarray) -> float:
    """How much we'd rather keep tile type t, all else equal."""
    st = d.state
    v = 2.0 if t in {dora_from_indicator(i) for i in st.dora_indicators} else 0.0
    if is_honor(t):
        yakuhai = t >= 31 or t in (st.round_wind, st.seat_wind(d.seat))
        return v + (0.8 if yakuhai and hand[t] >= 2 else 0.2 if yakuhai else 0.0)
    n = number(t)
    return v + (0.1 if n in (1, 9) else 0.3 if n in (2, 8) else 0.5)


class Teacher:
    name = "teacher"

    def act(self, decisions: list[Decision]) -> list[int]:
        return [self.choose(d) for d in decisions]

    def choose(self, d: Decision) -> int:
        m = d.mask
        for a in (AGARI, RIICHI, RYUKYOKU):
            if m[a]:
                return a
        if m[:37].any():
            return self.discard(d)
        if m[PON] and not self._threats(d):
            t = d.legal[PON].tile // 4
            st = d.state
            if t >= 31 or t in (st.round_wind, st.seat_wind(d.seat)):
                return PON
        return PASS if m[PASS] else int(np.flatnonzero(m)[0])

    def _threats(self, d: Decision) -> list[int]:
        st = d.state
        return [j for j, opp in enumerate(st.opponents(d.seat)) if st.players[opp].riichi_seq is not None]

    def discard(self, d: Decision) -> int:
        st = d.state
        hand = st.players[d.seat].hand
        remaining = np.clip(4 - st.visible_counts(d.seat), 0, None)
        threats = self._threats(d)
        danger = heuristic_danger(st, d.seat)[threats].max(0) if threats else np.zeros(34)
        options = []
        for i in np.flatnonzero(d.mask[:37]):
            t = _RED_INDEX.get(i, i)
            h = hand.copy()
            h[t] -= 1
            sh = shanten(h)
            offense = -100 * sh + ukeire(h, remaining) - 2 * _keep_value(t, d, hand)
            if i >= 34 or (t in RED_FIVES and t in st.players[d.seat].red and hand[t] == 1):
                offense -= 3  # throwing away the red five
            options.append((i, sh, offense, danger[t]))
        best_sh = min(o[1] for o in options)
        if threats and best_sh >= 1:  # fold: safest tile, efficiency breaks ties
            return int(min(options, key=lambda o: (o[3], -o[2]))[0])
        return int(max(options, key=lambda o: o[2] - 20 * o[3] * bool(threats))[0])
