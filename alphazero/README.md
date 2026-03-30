# AlphaZero-Lite

A from-scratch implementation of [AlphaZero](https://arxiv.org/abs/1712.01815) that learns to play Connect 4 through pure self-play. No human game data, no hand-crafted heuristics — just a neural network, Monte Carlo Tree Search, and games against itself.

## How it works

```
repeat:
  1. Play games against yourself using MCTS + neural net
  2. Record every (position, MCTS policy, game outcome)
  3. Train the neural net to predict MCTS policy + game outcome
  4. The net gets better → MCTS gets better → training data gets better
```

Three ingredients:

- **Neural network** — takes a board position, outputs a policy (which moves look good) and a value (who's winning). Starts random, improves through training.
- **MCTS** — before each move, simulates hundreds of future game trajectories using the neural net to guide the search. Balances exploring new moves vs going deeper on promising ones.
- **Self-play** — the agent plays against itself, generating its own training data. No human games needed.

## Setup

```bash
pip install -r requirements.txt  # torch, numpy, pytest
```

## Train

```bash
python train.py
```

Trains for 20 iterations with 50 self-play games each. Checkpoints saved to `checkpoints/` after each iteration. Evaluates against baseline agents every 5 iterations:

```
--- Iteration 5/20 ---
Generating 50 self-play games...
  Self-play: 24.3s | Games: 50 | Examples: 2,146 | Buffer: 10,730
  Train loss: 1.834 (policy: 1.421, value: 0.413) | LR: 0.001000
  Evaluating...
  vs       Random: W40 L0 D0 (100.0%)
  vs    Lookahead: W35 L3 D2 (90.0%)
  vs   Minimax(4): W12 L22 D6 (37.5%)
```

For a stronger agent, edit `config.py`:

```python
Config(num_iterations=40, games_per_iteration=100, num_simulations=200)
```

## Play

```bash
python play.py checkpoints/iter_020.pt           # you go first
python play.py checkpoints/iter_020.pt --second   # AI goes first
```

```
  1   2   3   4   5   6   7
| .   .   .   .   .   .   . |
| .   .   .   .   .   .   . |
| .   .   .   .   .   .   . |
| .   .   X   .   .   .   . |
| .   .   O   X   .   .   . |
| .   X   O   O   X   .   . |
  ---------------------

AI plays column 4 (confidence: 87%, eval: +0.34)
```

## Test

```bash
python -m pytest tests/ -v  # 53 tests
```

## Architecture

```
Input (3, 6, 7) → Conv 3→64 + BN + ReLU
  → 5x ResBlock (Conv+BN+ReLU+Conv+BN + skip)
  ├→ Policy head → softmax over 7 columns
  └→ Value head → tanh → [-1, 1]
```

376K parameters. The 3 input channels encode: current player's pieces, opponent's pieces, and a color plane (whose turn). Board is always encoded from the current player's perspective so one network plays both sides.

## File structure

| File | What it does |
|------|-------------|
| `config.py` | All hyperparameters as a dataclass |
| `game.py` | Connect 4 environment (board, moves, win detection, encoding) |
| `model.py` | ResNet with policy + value heads |
| `mcts.py` | Monte Carlo Tree Search with UCB selection |
| `self_play.py` | Self-play data generation with multiprocessing |
| `train.py` | Training loop (replay buffer, loss, checkpointing) |
| `evaluate.py` | Baseline agents (Random, Lookahead, Minimax) + ELO |
| `play.py` | Human vs AI terminal UI |

## Key RL concepts

| Concept | Where to find it |
|---------|-----------------|
| Policy learning | `model.py` — policy head outputs move probabilities |
| Value estimation | `model.py` — value head estimates who's winning |
| Exploration vs exploitation | `mcts.py` — UCB formula balances both |
| Self-play | `self_play.py` — agent generates its own training data |
| Credit assignment | `self_play.py` — game outcome propagated to every move |

## References

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815)
- [Mastering the game of Go without human knowledge](https://www.nature.com/articles/nature24270)
