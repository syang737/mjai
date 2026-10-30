# mjbot

Riichi mahjong AI research. The core idea is to make a **tile belief tensor** the central state
representation. It stores per-opponent, per-tile-type summaries of the hidden state (copies held,
tenpai, waits, ron-ability after furiten and yaku, deal-in payment, call-ability) instead of a
distribution over all ~1e32 possible opponent hands. Labels come from full-information game
records (simulator or Tenhou logs), so they're free.

## Pipeline

```
riichienv (Rust rules engine, Tenhou rules)
  └─ runner.run_games      lockstep games, batched agent calls
       ├─ tracker.Tracker  mjai events -> RoundState (per-seat view)
       ├─ teacher.Teacher  rule-based bot: efficiency + heuristic-danger folding
       ├─ data             record samples (planes, mask, action, belief labels, rank reward)
       ├─ train_sl         imitation + belief heads + oracle critic
       ├─ train_rl         PPO self-play, league opponents, anchor KL
       └─ arena            duplicate-seat evaluation
```

| module | contents |
|---|---|
| `tiles.py` | 34-type indexing, mjai / Tenhou shorthand, 136-id allocation (red fives) |
| `state.py` | round state; `visible_counts` for a seat |
| `tracker.py` | mjai event stream → `RoundState` |
| `hand.py` | shanten, waits, ukeire on an arbitrary (e.g. belief-adjusted) wall |
| `belief.py` | `belief_labels` (training targets), `heuristic_danger` (genbutsu/suji/kabe baseline), `adjusted_remaining` |
| `actions.py` | 46-way action space over engine actions |
| `features.py` | engine planes (74×34) + public belief planes; oracle planes for the critic |
| `model.py` | 1-D ResNet; policy, belief (ron / holdings / tenpai) and oracle-critic heads |

The package is named `mjbot` so it doesn't shadow the `mjai` protocol package.

## Usage

```bash
uv venv && uv pip install -e '.[dev]' && .venv/bin/pytest
python -m mjbot.data --games 1600 --out data/teacher --workers 4
python -m mjbot.train_sl --data data/teacher --out checkpoints/sl_b2p.pt --epochs 2 --belief-to-policy
python -m mjbot.train_rl --init checkpoints/sl_b2p.pt --out checkpoints/rl.pt --iters 40
python -m mjbot.arena --a checkpoints/rl.pt --b teacher --seeds 100
```

## Known gaps

- Training data is self-play by the rule-based teacher: Tenhou logs (tenhou.net) and Mortal
  (GitHub) were not reachable from the development environment. `Tracker` already consumes mjai
  logs, so converted Tenhou logs can replace the teacher data; legal masks for logged decisions
  would come from replaying through riichienv.
- Payments ignore ura dora and ippatsu (riichi hands are underestimated).
- Not built yet: event-sequence transformer encoder, belief-sampled search at inference time,
  learned global reward predictor (the oracle critic stands in for it).
