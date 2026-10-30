"""Model inputs.

`planes`: what the acting seat can see. riichienv's 74x34 encoding plus our public belief
features (heuristic danger, genbutsu per opponent, visible copies).
`oracle_planes`: hidden information for the critic only (asymmetric actor-critic / oracle guiding).
"""

import numpy as np

from .belief import belief_labels, genbutsu, heuristic_danger
from .runner import Decision

N_ENGINE = 74
N_PLANES = N_ENGINE + 7
N_ORACLE = 6


def planes(d: Decision) -> np.ndarray:
    x = np.empty((N_PLANES, 34), np.float32)
    x[:N_ENGINE] = np.frombuffer(d.obs.encode(), np.float32).reshape(N_ENGINE, 34)
    st = d.state
    x[N_ENGINE : N_ENGINE + 3] = heuristic_danger(st, d.seat)
    for j, opp in enumerate(st.opponents(d.seat)):
        x[N_ENGINE + 3 + j] = genbutsu(st, d.seat, opp)
    x[N_ENGINE + 6] = st.visible_counts(d.seat) / 4
    return x


def oracle_planes(labels) -> np.ndarray:
    """Opponents' concealed hands and ronnable tiles."""
    return np.concatenate([labels.holdings / 4, labels.ron]).astype(np.float32)


def labels(d: Decision):
    return belief_labels(d.full_state(), d.seat)
