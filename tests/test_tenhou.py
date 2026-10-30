import gzip
from pathlib import Path

import numpy as np

from mjbot.replay import decisions, final_ranks, leftover_deposits, outcome
from mjbot.tenhou import decode_meld, mjlog_to_mjai
from mjbot.tracker import Tracker

FIXTURE = Path(__file__).parent / "fixtures" / "houou_2026103000gm-00a9-0000-74fd942e.xml.gz"


def events():
    return mjlog_to_mjai(gzip.decompress(FIXTURE.read_bytes()))


def test_decode_meld_types():
    assert decode_meld(3, 59879) == dict(type="chi", actor=3, target=2, pai="7s", consumed=["6s", "8s"])
    assert decode_meld(3, 49194) == dict(type="pon", actor=3, target=1, pai="F", consumed=["F", "F"])
    assert decode_meld(2, 33792) == dict(type="ankan", actor=2, consumed=["C", "C", "C", "C"])


def test_converted_game_structure():
    ev = events()
    types = [e["type"] for e in ev]
    assert types[0] == "start_game" and types[-1] == "end_game"
    assert types.count("start_kyoku") == types.count("end_kyoku") == 9
    assert sum(ev[-1]["scores"]) == 100000


def test_replay_accepts_every_logged_action_and_scores_add_up():
    ev = events()
    n = sum(1 for _ in decisions(ev))  # raises ReplayError on any illegal logged action
    assert n > 500
    scores, ranks, deltas = outcome(ev)
    start = next(e["scores"] for e in ev if e["type"] == "start_kyoku")
    got = np.array(start) + np.array(deltas).sum(0)
    got[ranks.index(1)] += leftover_deposits(ev)
    assert got.tolist() == scores
    assert sorted(ranks) == [1, 2, 3, 4]


def test_replay_hidden_hands_match_full_log():
    ev = events()
    full = Tracker()
    fed = 0
    for d, _ in decisions(ev):
        st = d.full_state()
        # advance the full-information tracker to the same point in the round
        while full.state is None or full.state.tiles_left != d.state.tiles_left or \
                [len(p.discards) for p in full.state.players] != [len(p.discards) for p in d.state.players] or \
                [len(p.melds) for p in full.state.players] != [len(p.melds) for p in d.state.players]:
            full.feed(ev[fed])
            fed += 1
        for s in range(4):
            assert (st.players[s].hand == full.state.players[s].hand).all()


def test_final_ranks_tie_break():
    assert final_ranks([30000, 30000, 20000, 20000], first_dealer=1) == [2, 1, 3, 4]
