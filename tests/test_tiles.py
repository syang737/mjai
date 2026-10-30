from mjbot.tiles import from_mjai, parse, to_list, to_mjai


def test_mjai_roundtrip():
    for t in range(34):
        assert from_mjai(to_mjai(t)) == (t, False)
    assert from_mjai("5pr") == (13, True)
    assert to_mjai(27) == "E" and to_mjai(33) == "C"


def test_parse_red_five_and_honors():
    assert to_list(parse("0m1z")) == [4, 27]
    assert parse("123m456p789s1122z").sum() == 13


def test_dora_from_indicator():
    from mjbot.tiles import dora_from_indicator

    assert [dora_from_indicator(t) for t in (0, 8, 17, 26, 27, 30, 31, 33)] == [1, 0, 9, 18, 28, 27, 32, 31]
