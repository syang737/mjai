"""Training samples built on the fly from mjai game logs.

Dense planes for 100k Houou games would be hundreds of GB, so logs are replayed in DataLoader
workers and samples pass through a shuffle buffer (decisions from one game are highly correlated).

    loader = DataLoader(LogStream(paths), batch_size=512, num_workers=8)
"""

import random
from pathlib import Path

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .data import FIELDS, sample
from .replay import ReplayError, decisions, load_events, outcome
from .runner import RANK_REWARD


def game_samples(path, game: int = 0) -> list[dict]:
    """All samples of one logged game, with the rank reward filled in."""
    events = load_events(path)
    _, ranks, _ = outcome(events)
    out = []
    for d, a in decisions(events, game):
        s = sample(d, a)
        s["reward"] = np.float32(RANK_REWARD[ranks[d.seat] - 1])
        out.append(s)
    return out


def load_games(paths) -> dict[str, np.ndarray]:
    """Materialize a few games in memory (validation set), same layout as `data.load`."""
    rows = [s for i, p in enumerate(paths) for s in _safe_samples(p, i)]
    return {k: np.stack([r[k] for r in rows]) for k in (*FIELDS, "reward")}


def _safe_samples(path, game: int) -> list[dict]:
    try:
        return game_samples(path, game)
    except (ReplayError, StopIteration, KeyError) as e:  # skip the rare game the engine disagrees with
        print(f"skip {Path(path).name}: {type(e).__name__}", flush=True)
        return []


def log_files(src) -> list[Path]:
    """Game logs in a directory: mjai .jsonl or raw Tenhou .xml.gz."""
    src = Path(src)
    return sorted([*src.glob("*.jsonl"), *src.glob("*.xml.gz")])


class LogStream(IterableDataset):
    """`source` is a list of paths, or a directory that is re-scanned every pass (so a download
    still in progress keeps adding games); `exclude` holds file names kept out (validation)."""

    def __init__(self, source, buffer: int = 20000, seed: int = 0, loop: bool = True, exclude=()):
        self.source = source if isinstance(source, (str, Path)) else [str(p) for p in source]
        self.exclude = set(exclude)
        self.buffer = buffer
        self.seed = seed
        self.loop = loop

    def _paths(self) -> list[str]:
        paths = log_files(self.source) if isinstance(self.source, (str, Path)) else self.source
        return [str(p) for p in paths if Path(p).name not in self.exclude]

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1000 + wid)
        buf: list[dict] = []
        epoch = 0
        while True:
            order = self._paths()[wid::nw]
            rng.shuffle(order)
            for p in order:
                for s in _safe_samples(p, game=hash((epoch, p)) & 0x7FFFFFFF):
                    if len(buf) < self.buffer:
                        buf.append(s)
                        continue
                    j = rng.randrange(len(buf))
                    buf[j], s = s, buf[j]
                    yield s
            epoch += 1
            if not self.loop:
                break
        rng.shuffle(buf)
        yield from buf
