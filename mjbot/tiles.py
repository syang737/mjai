"""34-tile indexing shared by every module.

0-8 = 1-9m, 9-17 = 1-9p, 18-26 = 1-9s, 27-33 = E S W N P F C (mjai honor order).
Red fives are tracked separately as a flag; they share the index of the plain five.
"""

import numpy as np

N_TYPES = 34
SUITS = "mps"
HONORS = ["E", "S", "W", "N", "P", "F", "C"]


def is_honor(t: int) -> bool:
    return t >= 27


def suit(t: int) -> int:
    """0/1/2 for m/p/s, 3 for honors."""
    return 3 if t >= 27 else t // 9


def number(t: int) -> int:
    """1-9 for number tiles, 0 for honors."""
    return 0 if t >= 27 else t % 9 + 1


def from_mjai(s: str) -> tuple[int, bool]:
    """'5mr' -> (4, True), 'E' -> (27, False)."""
    if s in HONORS:
        return 27 + HONORS.index(s), False
    aka = s.endswith("r")
    return SUITS.index(s[1]) * 9 + int(s[0]) - 1, aka


def to_mjai(t: int, aka: bool = False) -> str:
    if is_honor(t):
        return HONORS[t - 27]
    return f"{number(t)}{SUITS[suit(t)]}{'r' if aka else ''}"


def parse(hand: str) -> np.ndarray:
    """Tenhou shorthand -> 34-count array: '123m456p77z' (z: 1-7 = E S W N P F C, 0 = red five)."""
    counts = np.zeros(N_TYPES, dtype=np.int8)
    digits: list[int] = []
    for ch in hand:
        if ch.isdigit():
            digits.append(int(ch))
            continue
        base = {"m": 0, "p": 9, "s": 18, "z": 27}[ch]
        for d in digits:
            counts[base + (d if ch == "z" else (d or 5)) - 1] += 1
        digits = []
    return counts


def to_list(counts: np.ndarray) -> list[int]:
    return [t for t in range(N_TYPES) for _ in range(counts[t])]
