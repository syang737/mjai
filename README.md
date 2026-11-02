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
| `tenhou.py` | Houou log index + rate-limited download, mjlog XML → mjai |
| `replay.py` | replay a full mjai log through the engine: legal mask + logged action per decision |
| `stream.py` | samples built on the fly from logs (`IterableDataset` with a shuffle buffer) |

The package is named `mjbot` so it doesn't shadow the `mjai` protocol package.

## Usage

```bash
uv venv && uv pip install -e '.[dev]' && .venv/bin/pytest
python -m mjbot.data --games 1600 --out data/teacher --workers 4
python -m mjbot.train_sl --data data/teacher --out checkpoints/sl_b2p.pt --epochs 2 --belief-to-policy
python -m mjbot.train_rl --init checkpoints/sl_b2p.pt --out checkpoints/rl.pt --iters 40
python -m mjbot.arena --a checkpoints/rl.pt --b teacher --seeds 100
```

Human expert data (Tenhou Houou, 4-player hanchan):

```bash
python -m mjbot.tenhou index && python -m mjbot.tenhou fetch --limit 20000 && python -m mjbot.tenhou convert
python -m mjbot.replay --check 1000          # every logged action legal, scores add up
python -m mjbot.train_sl --logs data/tenhou/mjai --steps 20000 --belief-to-policy --out checkpoints/houou.pt
```

On a GPU machine: `bash scripts/setup_gpu.sh` (CUDA is picked up automatically; bf16 autocast).

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

### Houou logs (human expert play)

SL on Tenhou Houou 4-player hanchan, streamed from raw logs while they downloaded (≈1.5k → 3.8k
games seen; 7,000 steps × 512 on CPU, same 96×5 model), 100 held-out games (61k decisions):

| metric | value |
|---|---|
| move agreement with Houou players | 74.6% |
| deal-in AUC, all opponents: learned / heuristic | 0.927 / 0.580 |
| deal-in AUC, opponents in riichi: learned / heuristic | 0.713 / 0.691 |

Duplicate arena, 800 games each (east-only):

| candidate | vs 3× | avg rank | rank pts | win | deal-in |
|---|---|---|---|---|---|
| Houou SL | teacher | **2.185 ± 0.041** | **+23.3 ± 2.8** | 23.3% | 8.4% |
| Houou SL | teacher-trained SL | 2.214 ± 0.040 | +18.8 ± 3.0 | 22.8% | 9.2% |

A few thousand human games beat everything trained on the teacher's self-play by a wide margin
(~7 standard errors vs the teacher). Against riichi the learned deal-in head beats
genbutsu/suji/kabe on human opponents too, but the gap is modest; human riichi waits are much
harder to read than the teacher's.

**20k games.** The model above was fine-tuned on all 20,000 downloaded games (19,900 train,
the 100 oldest held out; none of those were seen by either model). It ran two 7,000-step CPU runs
(lr 1e-3, then 3e-4, chained with `--init`), ≈7.2M samples, still under one pass over the ≈12M decisions:

| metric (100 held-out games, 60k decisions) | `houou.pt` | `houou20k.pt` |
|---|---|---|
| move agreement | 75.1% | **76.0%** |
| policy loss | 0.672 | 0.639 |
| deal-in AUC, all opponents (heuristic 0.564) | 0.928 | 0.934 |
| deal-in AUC, opponents in riichi (heuristic 0.675) | 0.699 | 0.705 |

| candidate | vs 3× | avg rank | rank pts | win | deal-in |
|---|---|---|---|---|---|
| Houou 20k SL | Houou SL (`houou.pt`) | **2.415 ± 0.036** | +5.5 ± 2.7 | 22.9% | 10.3% |
| Houou 20k SL | teacher | 2.228 ± 0.038 | +18.2 ± 2.7 | 23.9% | 9.2% |

It beats its predecessor head-to-head (~2.4 SE), but against the teacher it is no better than
`houou.pt` (2.185 ± 0.041; the difference is within noise). The teacher seems too weak an
opponent to separate two human-imitation models. Gains in imitation accuracy are now small per
added game; the ~0.4M-param network is the likely limit, and a larger model needs the GPU box.

## Known gaps

- Mortal (GitHub) is still unreachable from the development environment, so the strongest
  baseline is our own models.
- Payments ignore ura dora and ippatsu (riichi hands are underestimated).
- riichienv's replay mode rejects a rinshan kaihou tsumo after kakan when rinshan is the only yaku;
  such games (≈1 in 1,500) are skipped.
- Not built yet: event-sequence transformer encoder, belief-sampled search at inference time,
  learned global reward predictor (the oracle critic stands in for it).
