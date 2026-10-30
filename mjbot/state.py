"""Full-information round state. Observations and training labels are both derived from it."""

from dataclasses import dataclass, field

import numpy as np

from .tiles import N_TYPES


@dataclass(frozen=True)
class Discard:
    tile: int
    seq: int  # global discard order across all seats; resolves "passed after riichi" safety
    tsumogiri: bool = False
    riichi: bool = False  # this discard declared riichi
    called: bool = False  # taken by another player's chi/pon/kan; the tile then lives in that meld


@dataclass(frozen=True)
class Meld:
    kind: str  # chi | pon | daiminkan | kakan | ankan
    tiles: tuple[int, ...]


@dataclass
class Player:
    hand: np.ndarray  # concealed tile counts, meld tiles excluded
    melds: list[Meld] = field(default_factory=list)
    discards: list[Discard] = field(default_factory=list)

    @property
    def riichi_seq(self) -> int | None:
        return next((d.seq for d in self.discards if d.riichi), None)

    @property
    def is_open(self) -> bool:
        return any(m.kind != "ankan" for m in self.melds)


@dataclass
class RoundState:
    players: list[Player]  # absolute seats 0-3
    dora_indicators: list[int] = field(default_factory=list)

    def opponents(self, seat: int) -> list[int]:
        """Absolute seats of shimocha, toimen, kamicha (relative seats 1, 2, 3)."""
        return [(seat + k) % 4 for k in (1, 2, 3)]

    def visible_counts(self, seat: int) -> np.ndarray:
        """Copies of each tile type `seat` can see: own hand, all discards and melds, dora indicators."""
        v = self.players[seat].hand.astype(np.int8).copy()
        for p in self.players:
            for d in p.discards:
                if not d.called:  # called tiles are counted through the meld
                    v[d.tile] += 1
            for m in p.melds:
                for t in m.tiles:
                    v[t] += 1
        for t in self.dora_indicators:
            v[t] += 1
        assert (v <= 4).all() and v.shape == (N_TYPES,)
        return v
