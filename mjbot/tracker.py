"""mjai event stream -> RoundState.

Feed it one seat's stream (riichienv `Observation.new_events()`) and it holds that seat's view:
opponents' concealed hands stay empty because their tiles arrive as '?'. Feed it a full log
(Tenhou converted to mjai, riichienv `mjai_log`) and every hand is known.
"""

import json

import numpy as np

from .state import Discard, Meld, Player, RoundState
from .tiles import N_TYPES, from_mjai

_WIND = {"E": 27, "S": 28, "W": 29, "N": 30}


class Tracker:
    def __init__(self):
        self.state: RoundState | None = None
        self.known = [False] * 4
        self.scores = [25000] * 4
        self.kyoku = 0
        self.round_index = -1  # rounds started so far in this game, minus one
        self._seq = 0
        self._reach_pending = [False] * 4
        self.round_over = True

    def feed(self, ev: dict | str) -> None:
        if isinstance(ev, str):
            ev = json.loads(ev)
        getattr(self, f"_on_{ev['type']}", lambda e: None)(ev)

    # hand bookkeeping; no-ops for hands this stream can't see
    def _add(self, seat: int, pai: str) -> None:
        if self.known[seat] and pai != "?":
            t, red = from_mjai(pai)
            p = self.state.players[seat]
            p.hand[t] += 1
            if red:
                p.red.add(t)

    def _remove(self, seat: int, pai: str) -> None:
        if self.known[seat] and pai != "?":
            t, red = from_mjai(pai)
            p = self.state.players[seat]
            assert p.hand[t] > 0, f"seat {seat} discards {pai} it doesn't hold"
            p.hand[t] -= 1
            if red:
                p.red.discard(t)

    def _on_start_kyoku(self, e: dict) -> None:
        players = [Player(np.zeros(N_TYPES, np.int8)) for _ in range(4)]
        self.state = RoundState(
            players,
            dora_indicators=[from_mjai(e["dora_marker"])[0]],
            round_wind=_WIND[e["bakaze"]],
            dealer=e["oya"],
            honba=e["honba"],
            riichi_sticks=e["kyotaku"],
        )
        self.scores = list(e["scores"])
        self.kyoku = (_WIND[e["bakaze"]] - 27) * 4 + e["kyoku"] - 1
        self.round_index += 1
        self.known = [tehai[0] != "?" for tehai in e["tehais"]]
        self._reach_pending = [False] * 4
        self.round_over = False
        for seat, tehai in enumerate(e["tehais"]):
            for pai in tehai:
                self._add(seat, pai)

    def _on_tsumo(self, e: dict) -> None:
        self.state.tiles_left -= 1
        self._add(e["actor"], e["pai"])

    def _on_dahai(self, e: dict) -> None:
        a = e["actor"]
        self._remove(a, e["pai"])
        self.state.players[a].discards.append(
            Discard(from_mjai(e["pai"])[0], self._seq, tsumogiri=e["tsumogiri"], riichi=self._reach_pending[a])
        )
        self._reach_pending[a] = False
        self._seq += 1

    def _on_reach(self, e: dict) -> None:
        self._reach_pending[e["actor"]] = True

    def _on_reach_accepted(self, e: dict) -> None:
        self.state.riichi_sticks += 1
        if "scores" in e:
            self.scores = list(e["scores"])
        else:
            self.scores[e["actor"]] -= 1000

    def _call(self, e: dict, kind: str) -> None:
        a, target = e["actor"], e["target"]
        ds = self.state.players[target].discards
        ds[-1] = Discard(ds[-1].tile, ds[-1].seq, ds[-1].tsumogiri, ds[-1].riichi, called=True)
        for pai in e["consumed"]:
            self._remove(a, pai)
        tiles = tuple(sorted(from_mjai(x)[0] for x in [e["pai"], *e["consumed"]]))
        red = any(x.endswith("r") for x in [e["pai"], *e["consumed"]])
        self.state.players[a].melds.append(Meld(kind, tiles, red))

    def _on_chi(self, e: dict) -> None:
        self._call(e, "chi")

    def _on_pon(self, e: dict) -> None:
        self._call(e, "pon")

    def _on_daiminkan(self, e: dict) -> None:
        self._call(e, "daiminkan")

    def _on_ankan(self, e: dict) -> None:
        a = e["actor"]
        for pai in e["consumed"]:
            self._remove(a, pai)
        t = from_mjai(e["consumed"][0])[0]
        red = any(x.endswith("r") for x in e["consumed"])
        self.state.players[a].melds.append(Meld("ankan", (t,) * 4, red))

    def _on_kakan(self, e: dict) -> None:
        a = e["actor"]
        self._remove(a, e["pai"])
        t, red = from_mjai(e["pai"])
        melds = self.state.players[a].melds
        i = next(i for i, m in enumerate(melds) if m.kind == "pon" and m.tiles[0] == t)
        melds[i] = Meld("kakan", (t,) * 4, melds[i].red or red)

    def _on_dora(self, e: dict) -> None:
        self.state.dora_indicators.append(from_mjai(e["dora_marker"])[0])

    def _end(self, e: dict) -> None:
        if "scores" in e:
            self.scores = list(e["scores"])
        elif "deltas" in e:
            self.scores = [s + d for s, d in zip(self.scores, e["deltas"])]
        self.round_over = True

    _on_hora = _end
    _on_ryukyoku = _end
