# mjbot

Riichi mahjong AI research. The core idea is to make a **tile belief tensor** the central state
representation. It stores per-opponent, per-tile-type summaries of the hidden state (copies held,
tenpai, waits, ron-ability after furiten, call-ability) instead of a distribution over all ~1e32
possible opponent hands. The plan is to train it from Tenhou logs, which record every hand, so
ground-truth labels are free.

| module | contents |
|---|---|
| `mjbot/tiles.py` | 34-type indexing, mjai / Tenhou-shorthand conversion |
| `mjbot/state.py` | full-information round state; `visible_counts` for a seat |
| `mjbot/hand.py` | shanten, waits, ukeire on an arbitrary (e.g. belief-adjusted) wall |
| `mjbot/belief.py` | `belief_labels` (training targets), `heuristic_danger` (genbutsu/suji/kabe baseline), `adjusted_remaining` |

The package is named `mjbot` so it doesn't shadow the `mjai` protocol package.

```bash
uv venv && uv pip install -e '.[dev]' && .venv/bin/pytest
```

Known gaps: ron labels are shape-only (no yaku check, so an open yakuless hand counts as able to ron).
Payment-on-deal-in labels and the Tenhou log → `RoundState` pipeline come next.
