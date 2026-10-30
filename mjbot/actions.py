"""Fixed 46-way action space over riichienv's legal `Action` lists (same layout as Mortal)."""

import numpy as np
from riichienv import Action, ActionType

from .tiles import RED_FIVES, is_red_id

N_ACTIONS = 46
RED_DISCARD = {t: 34 + i for i, t in enumerate(RED_FIVES)}  # 34-36: discard a red five
RIICHI, CHI_LOW, CHI_MID, CHI_HIGH, PON, KAN, AGARI, RYUKYOKU, PASS = range(37, 46)

_SIMPLE = {
    ActionType.RIICHI: RIICHI,
    ActionType.PON: PON,
    ActionType.DAIMINKAN: KAN,
    ActionType.ANKAN: KAN,
    ActionType.KAKAN: KAN,
    ActionType.TSUMO: AGARI,
    ActionType.RON: AGARI,
    ActionType.KYUSHU_KYUHAI: RYUKYOKU,
    ActionType.PASS: PASS,
}


def action_index(a: Action) -> int:
    if a.action_type == ActionType.DISCARD:
        t = a.tile // 4
        return RED_DISCARD[t] if is_red_id(a.tile) else t
    if a.action_type == ActionType.CHI:
        called = a.tile // 4
        others = sorted(i // 4 for i in a.consume_tiles)
        return CHI_LOW if called < others[0] else CHI_HIGH if called > others[1] else CHI_MID
    return _SIMPLE[a.action_type]


def _reds(a: Action) -> int:
    return sum(is_red_id(i) for i in a.consume_tiles)


def legal_map(actions: list[Action]) -> tuple[np.ndarray, dict[int, Action]]:
    """Mask over the 46 actions plus the concrete Action for each legal index. When several
    concrete actions share an index (pon with or without the red five), keep the one using more reds."""
    chosen: dict[int, Action] = {}
    for a in actions:
        i = action_index(a)
        if i not in chosen or _reds(a) > _reds(chosen[i]):
            chosen[i] = a
    mask = np.zeros(N_ACTIONS, bool)
    mask[list(chosen)] = True
    return mask, chosen
