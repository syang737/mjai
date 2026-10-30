import numpy as np
import torch

from mjbot.agents import NetAgent
from mjbot.arena import duplicate_match
from mjbot.data import Recorder
from mjbot.features import N_PLANES
from mjbot.model import Net
from mjbot.runner import run_games
from mjbot.teacher import Teacher
from mjbot.train_rl import advantages, step_rewards
from mjbot.train_sl import auc


def test_teacher_games_and_recorder():
    t = Teacher()
    rec = Recorder()
    results = run_games([0, 1], lambda i: [t] * 4, on_decision=rec)
    assert all(sorted(r.ranks) == [1, 2, 3, 4] and sum(r.scores) <= 100000 for r in results)
    arr = rec.arrays(results)
    n = len(arr["action"])
    assert arr["x"].shape == (n, N_PLANES, 34) and arr["ron"].shape == (n, 3, 34)
    assert arr["mask"][np.arange(n), arr["action"]].all()
    assert (arr["ron"] <= (arr["tenpai"][:, :, None])).all()  # ron implies tenpai
    assert arr["ron"].any() and arr["opp_riichi"].any()


def test_net_agent_plays_legal_moves():
    torch.manual_seed(0)
    m = duplicate_match(NetAgent(Net(ch=16, blocks=1), greedy=False), Teacher(), [0], game_mode="4p-red-east")
    assert m["games"] == 4 and 1 <= m["avg_rank"] <= 4


def test_advantages_terminal_reward():
    keys = [(0, 0)] * 3 + [(0, 1)] * 2
    values = np.zeros(5, np.float32)
    adv, ret = advantages(keys, [0, 0, 1, 0, 1], values, {(0, 0): 1.0, (0, 1): -1.0}, {}, gamma=1.0, lam=1.0)
    assert np.allclose(ret, [1, 1, 1, -1, -1])


def test_step_rewards_credit_rounds():
    # decisions in rounds 0, 0, 2 of a 4-round game; round 1 had no decision, round 3 neither
    r = step_rewards([0, 0, 2], final=1.0, round_rewards=[0.1, 0.2, 0.3, 0.4])
    assert np.allclose(r, [0, 0.1, 0.2 + 0.3 + 0.4 + 1.0])


def test_auc():
    assert auc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
