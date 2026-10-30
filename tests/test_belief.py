import numpy as np

from mjbot.belief import adjusted_remaining, belief_labels, furiten_tiles, heuristic_danger
from mjbot.state import Discard, Meld, Player, RoundState
from mjbot.tiles import parse

S3, S6, M9 = 20, 23, 8


def make_state(**overrides) -> RoundState:
    players = [
        Player(parse("1199m2468p13579s")),  # seat 0: us, 14 tiles, about to discard
        Player(parse("123456789m11p45s")),  # shimocha: tenpai on 3s/6s
        Player(parse("1357m2468p13579s")),  # toimen: far from tenpai
        Player(parse("223456m55p77s"), melds=[Meld("pon", (31, 31, 31))]),  # kamicha: open
    ]
    for seat, ds in overrides.items():
        players[int(seat[1])].discards = ds
    return RoundState(players)


def test_labels_tenpai_and_waits():
    lab = belief_labels(make_state(), seat=0)
    assert lab.tenpai.tolist() == [True, False, False]
    assert np.flatnonzero(lab.waits[0]).tolist() == [S3, S6]
    assert (lab.ron == lab.waits).all()
    assert (lab.holdings[0] == parse("123456789m11p45s")).all()


def test_discard_furiten_blocks_ron():
    st = make_state(p1=[Discard(S6, seq=0)])
    lab = belief_labels(st, seat=0)
    assert np.flatnonzero(lab.ron[0]).tolist() == [S3]


def test_passed_after_riichi_is_furiten_and_genbutsu():
    st = make_state(p1=[Discard(M9, seq=0, riichi=True)], p2=[Discard(S3, seq=1)])
    assert S3 in furiten_tiles(st, 1)
    assert not belief_labels(st, seat=0).ron[0, S3]
    assert heuristic_danger(st, seat=0)[0, S3] == 0


def test_riichi_blocks_calls_and_chi_only_for_shimocha():
    st = make_state(p1=[Discard(M9, seq=0, riichi=True)])
    lab = belief_labels(st, seat=0)
    assert not lab.pon[0].any() and not lab.chi[0].any()
    assert lab.pon[2, 1] and not lab.chi[2].any()  # kamicha holds 22m but can't chi our discard


def test_suji_lowers_danger():
    st = make_state(p1=[Discard(S3, seq=0)])
    d = heuristic_danger(st, seat=0)
    assert d[0, S3] == 0  # genbutsu
    assert d[0, S6] < d[0, S6 - 1]  # 6s is suji of 3s, 5s is not


def test_adjusted_remaining():
    visible = np.zeros(34, int)
    visible[0] = 3
    held = np.zeros((3, 34))
    held[1, 0] = 0.6
    r = adjusted_remaining(visible, held)
    assert r[0] == 0.4 and r[1] == 4


S1, S4 = 18, 21
HAKU = 31


def shimocha_state(hand: str, melds=(), riichi=False, **kw) -> RoundState:
    players = [Player(parse("1199m2468p13579s")), Player(parse(hand), melds=list(melds)),
               Player(parse("1357m2468p13579s")), Player(parse("1357m2468p13579s"))]
    if riichi:
        players[1].discards = [Discard(M9, seq=0, riichi=True)]
    return RoundState(players, **kw)


def ron_tiles(st: RoundState) -> list[int]:
    return np.flatnonzero(belief_labels(st, seat=0).ron[0]).tolist()


def test_open_tanyao_one_sided_yaku():
    st = shimocha_state("234m567p23s55s", melds=[Meld("pon", (7, 7, 7))])
    assert np.flatnonzero(belief_labels(st, seat=0).waits[0]).tolist() == [S1, S4]
    assert ron_tiles(st) == [S4]  # 1s breaks tanyao


def test_closed_dama_without_yaku_cannot_ron():
    assert ron_tiles(shimocha_state("123m456p789s23s55z")) == []


def test_riichi_gives_yaku_and_payment():
    st = shimocha_state("123m456p789s23s55z", riichi=True, honba=1)
    assert ron_tiles(st) == [S1, S4]
    assert belief_labels(st, seat=0).payment[0, S4] == 1300 + 300  # 1 han 40 fu, non-dealer


def test_yakuhai_pon_rons_on_all_waits():
    st = shimocha_state("234m567p23s99m", melds=[Meld("pon", (HAKU,) * 3)])
    assert ron_tiles(st) == [S1, S4]


def test_no_ron_when_all_copies_accounted_for():
    # riichi waiting on 1s/4s, but all four 1s are in its own closed kan
    st = shimocha_state("123m456p23s55z", melds=[Meld("ankan", (S1,) * 4)], riichi=True)
    lab = belief_labels(st, seat=0)
    assert lab.waits[0, S1] and not lab.ron[0, S1]
    assert lab.ron[0, S4]
