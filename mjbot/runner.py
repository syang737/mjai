"""Run many riichienv games in lockstep so neural agents can batch their forward passes."""

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
from riichienv import Action, GameRule, Observation, RiichiEnv

from .actions import legal_map
from .state import Player, RoundState
from .tiles import ids_to_counts, is_red_id
from .tracker import Tracker

# Tenhou-style rank points (houou), scaled so 1st = +1
RANK_REWARD = np.array([90, 45, 0, -135]) / 90


@dataclass
class Decision:
    game: int
    seat: int
    obs: Observation
    state: RoundState  # this seat's view: opponents' concealed hands are empty
    mask: np.ndarray
    legal: dict[int, Action]
    env: RiichiEnv
    round_index: int = 0  # which round of the game this decision is in

    def full_state(self) -> RoundState:
        """Same round with every concealed hand filled in from the engine (for labels / oracle features)."""
        players = []
        for s, p in enumerate(self.state.players):
            ids = self.env.hands[s]
            players.append(Player(ids_to_counts(ids), {i // 4 for i in ids if is_red_id(i)}, p.melds, p.discards))
        st = self.state
        return RoundState(players, st.dora_indicators, st.round_wind, st.dealer, st.honba, st.riichi_sticks, st.tiles_left)


class Agent(Protocol):
    def act(self, decisions: list[Decision]) -> list[int]: ...


@dataclass
class GameResult:
    game: int
    seed: int
    agents: list  # agent per seat
    scores: list[int]
    ranks: list[int]  # 1-4
    stats: list[dict] = field(default_factory=list)  # per seat: wins, deal_ins, riichi, calls, rounds
    round_deltas: list[list[int]] = field(default_factory=list)  # per round: score change per seat


def _round_deltas(log_events: list[dict]) -> list[list[int]]:
    """Score change per seat for each round, including riichi deposits paid."""
    out: list[list[int]] = []
    for e in log_events:
        if e["type"] == "start_kyoku":
            out.append([0, 0, 0, 0])
        elif e["type"] in ("hora", "ryukyoku") and "deltas" in e:
            out[-1] = [a + b for a, b in zip(out[-1], e["deltas"])]
        elif e["type"] == "reach_accepted":
            out[-1][e["actor"]] -= 1000
    return out


def _stats(log_events: list[dict]) -> list[dict]:
    st = [dict(wins=0, deal_ins=0, riichi=0, calls=0, rounds=0) for _ in range(4)]
    for e in log_events:
        t = e["type"]
        if t == "start_kyoku":
            for s in st:
                s["rounds"] += 1
        elif t == "hora":
            st[e["actor"]]["wins"] += 1
            if e["target"] != e["actor"]:
                st[e["target"]]["deal_ins"] += 1
        elif t == "reach_accepted":
            st[e["actor"]]["riichi"] += 1
        elif t in ("chi", "pon", "daiminkan"):
            st[e["actor"]]["calls"] += 1
    return st


def run_games(
    seeds: list[int],
    seat_agents: Callable[[int], list[Agent]],
    n_parallel: int = 32,
    game_mode: str = "4p-red-east",
    on_decision: Callable[[Decision, int], None] | None = None,
) -> list[GameResult]:
    """Play one game per seed. `seat_agents(i)` gives the 4 agents for game i; `on_decision` sees
    every decision and the chosen action index (for recording training samples)."""
    results: list[GameResult] = []
    queue = list(enumerate(seeds))
    live: dict[int, dict] = {}

    def start(i: int, seed: int) -> None:
        env = RiichiEnv(game_mode=game_mode, rule=GameRule.default_tenhou(), seed=seed)
        live[i] = dict(env=env, seed=seed, agents=seat_agents(i), trackers=[Tracker() for _ in range(4)],
                       obs=env.reset())

    while queue or live:
        while queue and len(live) < n_parallel:
            start(*queue.pop(0))
        pending: dict[int, list[Decision]] = {}  # id(agent) -> decisions
        owners: dict[int, Agent] = {}
        for i, g in live.items():
            for seat, o in g["obs"].items():
                for ev in o.new_events():
                    g["trackers"][seat].feed(ev)
                mask, legal = legal_map(o.legal_actions())
                tr = g["trackers"][seat]
                d = Decision(i, seat, o, tr.state, mask, legal, g["env"], tr.round_index)
                agent = g["agents"][seat]
                pending.setdefault(id(agent), []).append(d)
                owners[id(agent)] = agent
        chosen: dict[int, dict[int, Action]] = {}
        for key, ds in pending.items():
            for d, a in zip(ds, owners[key].act(ds)):
                assert d.mask[a], f"illegal action {a}"
                if on_decision:
                    on_decision(d, a)
                chosen.setdefault(d.game, {})[d.seat] = d.legal[a]
        for i in list(live):
            g = live[i]
            g["obs"] = g["env"].step(chosen.get(i, {}))
            if g["env"].done():
                env = g["env"]
                log = env.mjai_log  # O(len) to fetch, so only once per game
                results.append(GameResult(i, g["seed"], g["agents"], env.scores(), env.ranks(), _stats(log),
                                          _round_deltas(log)))
                del live[i]
    return sorted(results, key=lambda r: r.game)
