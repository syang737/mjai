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

## Results so far (CPU only, 4 cores)

Data: 2,400 east-only games of teacher self-play (≈815k decisions). Model: 96-channel, 5-block
1-D ResNet (~0.4M params), beliefs fed into the policy head.

Supervised, held-out games:

| metric | value |
|---|---|
| move agreement with teacher | 97.1% (92% before adding shanten/ukeire-after-discard planes) |
| deal-in AUC, all opponents: learned / heuristic | 0.939 / 0.556 |
| deal-in AUC, opponents in riichi: learned / heuristic | 0.692 / 0.656 |

The all-opponents gap is mostly the learned head knowing *whether* an opponent is tenpai; against
riichi (tenpai known) the learned head still beats genbutsu/suji/kabe.

Duplicate arena, 1 candidate vs 3 teachers, east-only, 200 seeds × 4 seat rotations (800 games),
± is one standard error over seeds; 2.5 = parity:

| candidate | avg rank | rank pts | win | deal-in |
|---|---|---|---|---|
| SL | 2.454 ± 0.032 | +3.0 ± 2.5 | 19.4% | 10.3% |
| SL + 60 PPO iters, final-rank reward only | 2.565 ± 0.032 | −6.1 ± 2.4 | 19.4% | 11.7% |

| SL + 60 PPO iters, rank + per-round score reward (`--round-reward 0.5`) | 2.561 ± 0.035 | −4.1 ± 2.7 | 18.4% | 9.5% |

Same per-round RL model against 3 copies of the SL model: avg rank 2.436 ± 0.034 (rank pts
+4.3 ± 2.6), i.e. it beats its own starting point but not the teacher. Small-scale self-play
(~3,800 games, opponents mostly the learner and its snapshots) appears to learn to exploit
SL-like opponents rather than to get generally stronger. PPO with only the final-rank reward
made the policy worse outright: one rank outcome per game is too noisy a signal at this scale.

## Known gaps

- Training data is self-play by the rule-based teacher: Tenhou logs (tenhou.net) and Mortal
  (GitHub) were not reachable from the development environment. `Tracker` already consumes mjai
  logs, so converted Tenhou logs can replace the teacher data; legal masks for logged decisions
  would come from replaying through riichienv.
- Payments ignore ura dora and ippatsu (riichi hands are underestimated).
- Not built yet: event-sequence transformer encoder, belief-sampled search at inference time,
  learned global reward predictor (the oracle critic stands in for it).
