import random

import numpy as np
from riichienv import ActionType, GameRule, RiichiEnv

from mjbot.actions import legal_map
from mjbot.tiles import ids_to_counts, is_red_id
from mjbot.tracker import Tracker

PRIORITY = (ActionType.TSUMO, ActionType.RON, ActionType.RIICHI, ActionType.PON, ActionType.CHI, ActionType.ANKAN)


def play(seed: int, check) -> RiichiEnv:
    env = RiichiEnv(game_mode="4p-red-east", rule=GameRule.default_tenhou(), seed=seed)
    rng = random.Random(seed)
    trackers = [Tracker() for _ in range(4)]
    obs = env.reset()
    while not env.done():
        actions = {}
        for seat, o in obs.items():
            for ev in o.new_events():
                trackers[seat].feed(ev)
            check(env, seat, o, trackers[seat])
            legal = o.legal_actions()
            pick = next((a for p in PRIORITY for a in legal if a.action_type == p and rng.random() < 0.7), None)
            actions[seat] = pick or rng.choice(legal)
        obs = env.step(actions)
    return env


def test_tracker_matches_engine_hands():
    calls = []

    def check(env, seat, o, tr):
        p = tr.state.players[seat]
        assert (p.hand == ids_to_counts(o.hand)).all()
        assert p.red == {i // 4 for i in o.hand if is_red_id(i)}
        assert len(p.melds) == len(env.melds[seat])
        for other in range(4):
            assert len(tr.state.players[other].discards) == len(env.discards[other])
        assert tr.state.tiles_left == len(env.wall) - 14  # engine's wall includes the dead wall
        assert len(tr.state.dora_indicators) == len(env.dora_indicators)
        if any(m.kind != "ankan" for m in p.melds):
            calls.append(seat)

    for seed in range(20):
        play(seed, check)
    assert calls, "no calls exercised"


def test_legal_map_roundtrip():
    def check(env, seat, o, tr):
        mask, chosen = legal_map(o.legal_actions())
        assert mask.sum() == len(chosen) > 0
        assert set(map(str, chosen.values())) <= set(map(str, o.legal_actions()))

    play(0, check)
