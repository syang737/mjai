"""Model inputs.

`planes`: what the acting seat can see. riichienv's 74x34 encoding, our public belief
features (heuristic danger, genbutsu per opponent, visible copies), and own-hand efficiency
(shanten / ukeire after discarding each held tile, or the tiles that would improve a 3n+1 hand).
`oracle_planes`: hidden information for the critic only (asymmetric actor-critic / oracle guiding).
"""

import numpy as np

from .belief import belief_labels, genbutsu, heuristic_danger
from .hand import discard_table, improving_tiles, shanten
from .runner import Decision

N_ENGINE = 74
N_PLANES = N_ENGINE + 10
N_ORACLE = 6


def planes(d: Decision) -> np.ndarray:
    x = np.empty((N_PLANES, 34), np.float32)
    x[:N_ENGINE] = np.frombuffer(d.obs.encode(), np.float32).reshape(N_ENGINE, 34)
    st = d.state
    x[N_ENGINE : N_ENGINE + 3] = heuristic_danger(st, d.seat)
    for j, opp in enumerate(st.opponents(d.seat)):
        x[N_ENGINE + 3 + j] = genbutsu(st, d.seat, opp)
    visible = st.visible_counts(d.seat)
    x[N_ENGINE + 6] = visible / 4
    x[N_ENGINE + 7 :] = 0
    hand = st.players[d.seat].hand
    remaining = np.clip(4 - visible, 0, None)
    if hand.sum() % 3 == 2:
        for t, (sh, uk) in discard_table(hand, remaining).items():
            x[N_ENGINE + 7, t] = 1 - (sh + 1) / 8
            x[N_ENGINE + 8, t] = uk / 40
    elif shanten(hand) >= 0:
        x[N_ENGINE + 9, improving_tiles(hand)] = 1
    return x


def oracle_planes(labels) -> np.ndarray:
    """Opponents' concealed hands and ronnable tiles."""
    return np.concatenate([labels.holdings / 4, labels.ron]).astype(np.float32)


def labels(d: Decision):
    return belief_labels(d.full_state(), d.seat)
