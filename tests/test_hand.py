import numpy as np

from mjbot.hand import improving_tiles, shanten, ukeire, waits
from mjbot.tiles import parse


def test_ryanmen_waits():
    h = parse("123456789m11p45s")
    assert shanten(h) == 0
    assert waits(h) == [20, 23]  # 3s, 6s


def test_open_hand_waits():
    h = parse("456m11p45s")  # 7 tiles, two melds called
    assert waits(h, closed=False) == [20, 23]


def test_ukeire_on_adjusted_wall():
    h = parse("123456789m11p45s")
    naive = np.full(34, 4.0)
    naive[[20, 23]] = 3  # one 4s/5s each in hand doesn't matter; 3s and 6s each have one visible
    assert ukeire(h, naive) == 6
    adjusted = naive.copy()
    adjusted[20] = 0.5  # opponents believed to hold most 3s
    assert ukeire(h, adjusted) == 3.5


def test_improving_tiles_does_not_mutate():
    h = parse("13m58p2469s12355z")
    before = h.copy()
    improving_tiles(h)
    assert (h == before).all()
