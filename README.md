# Reinforcement Learning from Scratch

Two RL algorithms implemented from scratch in PyTorch. No external RL libraries — the point is to learn by building.

## Projects

### [AlphaZero](alphazero/) — Learn a board game through self-play

Neural network + Monte Carlo Tree Search learns Connect 4 with zero human knowledge.

```bash
cd alphazero && pip install -r requirements.txt
python train.py                                  # ~15 min on GPU
python play.py checkpoints/iter_020.pt           # play against it
```

53 tests | 376K parameters | Beats minimax depth-4 after ~12 iterations of self-play

**You'll learn:** MCTS, policy/value networks, self-play, exploration vs exploitation, board encoding

---

### [PPO](ppo/) — Teach a simulated creature to walk

Proximal Policy Optimization trains simulated robots to move through continuous control. This is the algorithm behind RLHF/ChatGPT.

```bash
cd ppo && pip install -r requirements.txt
python train.py cartpole                         # sanity check (~30s)
python train.py halfcheetah                      # main target (~15 min on GPU)
```

25 tests | ~9K parameters | CartPole → Pendulum → HalfCheetah → Ant progression

**You'll learn:** Policy gradients, GAE (advantage estimation), clipped surrogate objective, continuous action spaces, vectorized environments

---

## How they compare

| | AlphaZero | PPO |
|---|---|---|
| Action space | Discrete (7 columns) | Continuous (joint torques) |
| Planning | MCTS search at each move | Direct policy output, no search |
| Learning signal | Self-play game outcomes | Trial-and-error rewards |
| Environment model | Known (game rules given) | Unknown (learn from interaction) |
| Network | CNN (spatial board patterns) | MLP (observation vector) |
| Paradigm | Model-based RL | Model-free RL |

## References

- [Mastering Chess and Shogi by Self-Play](https://arxiv.org/abs/1712.01815) — AlphaZero paper
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — PPO paper
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE paper
