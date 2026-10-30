"""Turn full mjai game logs (Tenhou converted, or self-play) into training decisions.

Each seat gets its own riichienv engine fed the log through `observe_event`, which returns an
Observation (with legal actions) whenever that seat has a decision. The logged response is the
seat's next action; a seat offered a call or ron that it didn't take gets PASS. Hidden hands come
from the engine, so `Decision.full_state()` and the belief labels work as in self-play.

    python -m mjbot.replay --src data/tenhou/mjai --check 1000   # verify the converter
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from riichienv import ActionType, GameRule, RiichiEnv

from .actions import action_index, legal_map
from .runner import Decision, _round_deltas
from .tracker import Tracker

_KANS = {"ankan", "kakan", "daiminkan"}
_RESPONSES = {"dahai", "reach", "chi", "pon", "daiminkan", "ankan", "kakan", "hora", "ryukyoku"}


class ReplayError(Exception):
    pass


def load_events(path) -> list[dict]:
    """mjai .jsonl, or a raw Tenhou mjlog (.xml.gz), converted on the fly."""
    path = Path(path)
    if path.name.endswith(".xml.gz"):
        from .tenhou import mjlog_to_mjai

        return mjlog_to_mjai(gzip.decompress(path.read_bytes()))
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _mask(e: dict, seat: int) -> dict:
    """What `seat` sees of an event: other players' haipai and draws are hidden."""
    if e["type"] == "start_kyoku":
        return {**e, "tehais": [t if i == seat else ["?"] * len(t) for i, t in enumerate(e["tehais"])]}
    if e["type"] == "tsumo" and e["actor"] != seat:
        return {**e, "pai": "?"}
    return e


def _response(events: list[dict], i: int, seat: int) -> dict | None:
    """The seat's logged reaction to the decision point right after events[i], or None if it passed."""
    for e in events[i + 1 :]:
        t, actor = e["type"], e.get("actor")
        if actor == seat and t in _RESPONSES:
            return e
        if t in ("tsumo", "end_kyoku", "start_kyoku") or (t in ("dahai", "chi", "pon", "daiminkan") and actor != seat):
            return None
    return None


def final_ranks(scores: list[int], first_dealer: int = 0) -> list[int]:
    """1-4; ties go to the seat closer to the first dealer (Tenhou rule)."""
    order = sorted(range(4), key=lambda s: (-scores[s], (s - first_dealer) % 4))
    ranks = [0] * 4
    for r, s in enumerate(order):
        ranks[s] = r + 1
    return ranks


def decisions(events: list[dict], game: int = 0) -> Iterator[tuple[Decision, int]]:
    """Yield (Decision, logged action index) for every decision with more than one legal action."""
    envs = [RiichiEnv(game_mode="4p-red-half", rule=GameRule.default_tenhou()) for _ in range(4)]
    trackers = [Tracker() for _ in range(4)]
    for i, e in enumerate(events):
        for seat in range(4):
            trackers[seat].feed(_mask(e, seat))
            obs = envs[seat].observe_event(e, seat)
            if obs is None or (e["type"] in _KANS and e["actor"] == seat):
                continue  # after our own kan the real decision comes with the rinshan draw
            legal = obs.legal_actions()
            if len(legal) <= 1:
                continue
            resp = _response(events, i, seat)
            if resp is None:
                action = next((a for a in legal if a.action_type == ActionType.PASS), None)
            else:
                action = obs.select_action_from_mjai(resp)
            if action is None:
                raise ReplayError(f"event {i}: seat {seat} logged {resp} not among {[a.to_mjai() for a in legal]}")
            mask, chosen = legal_map(legal)
            tr = trackers[seat]
            yield Decision(game, seat, obs, tr.state, mask, chosen, envs[seat], tr.round_index), action_index(action)


def outcome(events: list[dict]) -> tuple[list[int], list[int], list[list[int]]]:
    """Final scores, ranks and per-round score changes of a logged game."""
    start = next(e for e in events if e["type"] == "start_kyoku")
    deltas = _round_deltas(events)
    end = next((e for e in reversed(events) if e["type"] == "end_game"), {})
    scores = end.get("scores") or (np.array(start["scores"]) + np.array(deltas).sum(0)).tolist()
    return scores, final_ranks(scores, start["oya"]), deltas


def leftover_deposits(events: list[dict]) -> int:
    """Riichi deposits still on the table when the game ends (Tenhou gives them to first place)."""
    sticks = 0
    for e in events:
        if e["type"] == "start_kyoku":
            sticks = e["kyotaku"]
        elif e["type"] == "reach_accepted":
            sticks += 1
        elif e["type"] == "hora":
            sticks = 0
    return sticks * 1000


def check(paths: list[Path]) -> None:
    """Replay games end to end: every logged action must be legal and the per-round score changes
    (plus leftover deposits to first place) must add up to the logged final scores."""
    ok = n_dec = 0
    for p in paths:
        try:
            events = load_events(p)
            n_dec += sum(1 for _ in decisions(events))
            scores, ranks, deltas = outcome(events)
            start = next(e["scores"] for e in events if e["type"] == "start_kyoku")
            got = np.array(start) + np.array(deltas).sum(0)
            got[ranks.index(1)] += leftover_deposits(events)
            if got.tolist() != scores:
                raise ReplayError(f"scores {got.tolist()} != logged {scores}")
            ok += 1
        except Exception as ex:
            print(f"{p.name}: {type(ex).__name__}: {str(ex)[:300]}", flush=True)
    print(f"{ok}/{len(paths)} games replayed cleanly, {n_dec} decisions", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/tenhou/mjai")
    ap.add_argument("--check", type=int, default=100, help="number of games to verify")
    a = ap.parse_args()
    src = Path(a.src)
    check(sorted([*src.glob("*.jsonl"), *src.glob("*.xml.gz")])[: a.check])


if __name__ == "__main__":
    main()
