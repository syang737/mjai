"""Tenhou Houou logs: index, download, and convert mjlog XML to mjai events.

    python -m mjbot.tenhou index                 # -> data/tenhou/ids.txt
    python -m mjbot.tenhou fetch --limit 20000   # -> data/tenhou/raw/<id>.xml.gz (resumable, 1 req/s)
    python -m mjbot.tenhou convert               # -> data/tenhou/mjai/<id>.jsonl

Only 4-player Houou hanchan with red fives and open tanyao (四鳳南喰赤) are kept, which is the
ruleset riichienv's "4p-red-half" / Tenhou rule plays.
"""

import argparse
import gzip
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from riichienv.convert import tid_to_mjai

BASE = "https://tenhou.net/sc/raw"
ROOT = Path("data/tenhou")
_GAME = re.compile(r"四鳳南喰赤.*?log=(\d{10}gm-00a9-0000-[0-9a-f]{8})")


def _get(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mjbot-research (log download)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))
    raise AssertionError


def _list(old: bool) -> list[str]:
    text = _get(f"{BASE}/list.cgi" + ("?old" if old else "")).decode()
    return re.findall(r"file:'([^']*scc[^']*)'", text)


def index(out: Path = ROOT / "ids.txt", delay: float = 1.0) -> int:
    """Collect 4p Houou hanchan log ids from the daily (archived) and hourly (current week) scc files."""
    files = _list(old=True) + _list(old=False)
    ids: list[str] = []
    for f in files:
        html = gzip.decompress(_get(f"{BASE}/dat/{f}")).decode("utf-8", "replace")
        ids += _GAME.findall(html)
        time.sleep(delay)
    ids = sorted(set(ids))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(ids) + "\n")
    return len(ids)


def fetch(ids_path: Path = ROOT / "ids.txt", out: Path = ROOT / "raw", limit: int = 0, delay: float = 1.0,
          newest_first: bool = True) -> None:
    ids = ids_path.read_text().split()
    if newest_first:
        ids = ids[::-1]
    if limit:
        ids = ids[:limit]
    out.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, log_id in enumerate(ids):
        path = out / f"{log_id}.xml.gz"
        if path.exists():
            continue
        try:
            xml = _get(f"https://tenhou.net/0/log/?{log_id}")
        except Exception as e:  # keep going; a rerun retries missing files
            print(f"{log_id}: {e}", flush=True)
            continue
        if not xml.startswith(b"<mjloggm"):
            print(f"{log_id}: unexpected response", flush=True)
            continue
        path.write_bytes(gzip.compress(xml))
        done += 1
        if done % 500 == 0:
            print(f"{i + 1}/{len(ids)} ({done} new)", flush=True)
        time.sleep(delay)
    print(f"done: {done} new files", flush=True)


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",")] if s else []


def decode_meld(who: int, m: int) -> dict:
    """Tenhou meld bitfield -> mjai call event."""
    kui = m & 3  # 0 = own (ankan); else relative seat the tile came from
    target = (who + kui) % 4
    if m & 0x4:  # chi
        t = (m & 0xFC00) >> 10
        called_idx, t = t % 3, t // 3
        base = (t // 7 * 9 + t % 7) * 4
        tiles = [base + (m >> 3 & 3), base + 4 + (m >> 5 & 3), base + 8 + (m >> 7 & 3)]
        called = tiles.pop(called_idx)
        return dict(type="chi", actor=who, target=target, pai=tid_to_mjai(called), consumed=[tid_to_mjai(x) for x in tiles])
    if m & 0x18:  # pon (0x8) or kakan (0x10)
        t = (m & 0xFE00) >> 9
        called_idx, base = t % 3, t // 3 * 4
        unused = m >> 5 & 3
        tiles = [base + i for i in range(4) if i != unused]
        if m & 0x8:
            called = tiles.pop(called_idx)
            return dict(type="pon", actor=who, target=target, pai=tid_to_mjai(called),
                        consumed=[tid_to_mjai(x) for x in tiles])
        return dict(type="kakan", actor=who, pai=tid_to_mjai(base + unused), consumed=[tid_to_mjai(x) for x in tiles])
    if m & 0x20:
        raise ValueError("nukidora (3-player) not supported")
    hai = (m & 0xFF00) >> 8
    base = hai // 4 * 4
    tiles = [base + i for i in range(4)]
    if kui == 0:
        return dict(type="ankan", actor=who, consumed=[tid_to_mjai(x) for x in tiles])
    tiles.remove(hai)
    return dict(type="daiminkan", actor=who, target=target, pai=tid_to_mjai(hai), consumed=[tid_to_mjai(x) for x in tiles])


_DRAW = {c: i for i, c in enumerate("TUVW")}
_DISCARD = {c: i for i, c in enumerate("DEFG")}


def mjlog_to_mjai(xml: str | bytes) -> list[dict]:
    """Convert one Tenhou mjlog game to a list of mjai events (riichienv's conventions)."""
    root = ET.fromstring(xml)
    events: list[dict] = [dict(type="start_game")]
    last_draw = [None] * 4
    last_drawer = None
    for el in root:
        tag, at = el.tag, el.attrib
        if tag == "INIT":
            seed = _ints(at["seed"])
            events.append(dict(
                type="start_kyoku", bakaze="ESWN"[seed[0] // 4], kyoku=seed[0] % 4 + 1, honba=seed[1],
                kyotaku=seed[2], oya=int(at["oya"]), dora_marker=tid_to_mjai(seed[5]),
                scores=[x * 100 for x in _ints(at["ten"])],
                tehais=[[tid_to_mjai(x) for x in _ints(at[f"hai{i}"])] for i in range(4)],
            ))
            last_draw = [None] * 4
        elif tag[0] in _DRAW and tag[1:].isdigit():
            seat, tile = _DRAW[tag[0]], int(tag[1:])
            last_draw[seat] = tile
            last_drawer = seat
            events.append(dict(type="tsumo", actor=seat, pai=tid_to_mjai(tile)))
        elif tag[0] in _DISCARD and tag[1:].isdigit():
            seat, tile = _DISCARD[tag[0]], int(tag[1:])
            events.append(dict(type="dahai", actor=seat, pai=tid_to_mjai(tile), tsumogiri=tile == last_draw[seat]))
            last_draw[seat] = None
        elif tag == "N":
            events.append(decode_meld(int(at["who"]), int(at["m"])))
            last_draw = [None] * 4
        elif tag == "REACH":
            kind = "reach" if at["step"] == "1" else "reach_accepted"
            events.append(dict(type=kind, actor=int(at["who"])))
        elif tag == "DORA":
            events.append(dict(type="dora", dora_marker=tid_to_mjai(int(at["hai"]))))
        elif tag in ("AGARI", "RYUUKYOKU"):
            deltas = [x * 100 for x in _ints(at["sc"])[1::2]]
            if tag == "AGARI":
                who, from_who = int(at["who"]), int(at["fromWho"])
                ev = dict(type="hora", actor=who, target=from_who, deltas=deltas,
                          ura_markers=[tid_to_mjai(x) for x in _ints(at.get("doraHaiUra", ""))])
                if who == from_who:
                    ev["tsumo"] = True
            else:
                ev = dict(type="ryukyoku", deltas=deltas, reason=at.get("type", "exhaustive_draw"))
                if at.get("type") == "yao9":  # nine-terminal abort is declared by the player who just drew
                    ev["actor"] = last_drawer
            # double ron: a second AGARI follows the first without end_kyoku in between
            if events[-1]["type"] == "end_kyoku" and tag == "AGARI" and events[-2]["type"] == "hora":
                events.insert(len(events) - 1, ev)
            else:
                events += [ev, dict(type="end_kyoku")]
            if "owari" in at:
                events.append(dict(type="end_game", scores=[int(x) * 100 for x in at["owari"].split(",")[0::2]]))
    return events


def convert_all(raw: Path = ROOT / "raw", out: Path = ROOT / "mjai") -> None:
    out.mkdir(parents=True, exist_ok=True)
    n = bad = 0
    for path in sorted(raw.glob("*.xml.gz")):
        dst = out / path.name.replace(".xml.gz", ".jsonl")
        if dst.exists():
            continue
        try:
            events = mjlog_to_mjai(gzip.decompress(path.read_bytes()))
        except Exception as e:
            bad += 1
            print(f"{path.name}: {e}", flush=True)
            continue
        dst.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n")
        n += 1
    print(f"converted {n}, failed {bad}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["index", "fetch", "convert"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    a = ap.parse_args()
    if a.cmd == "index":
        print(f"{index(delay=a.delay)} game ids")
    elif a.cmd == "fetch":
        fetch(limit=a.limit, delay=a.delay)
    else:
        convert_all()


if __name__ == "__main__":
    main()
