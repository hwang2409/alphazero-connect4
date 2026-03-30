# PPO — Proximal Policy Optimization

A from-scratch implementation of [PPO](https://arxiv.org/abs/1707.06347) for continuous control. Trains simulated robots to walk, run, and balance using the same algorithm behind RLHF/ChatGPT.

## How it works

```
repeat:
  1. Collect rollouts: run current policy in parallel environments
  2. Compute advantages using GAE (how much better was each action than expected?)
  3. Update policy with clipped objective (improve, but not too much per step)
```

The key insight of PPO: **clip the policy update** so it never changes too much in one step. This prevents catastrophic policy collapse — a common failure mode in policy gradient methods.

```
ratio = new_policy(action) / old_policy(action)
loss  = -min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
```

If the policy tries to change too aggressively (ratio far from 1), the clipping kicks in and the gradient vanishes. This creates a stable "trust region" around the old policy.

## Setup

```bash
pip install -r requirements.txt  # torch, numpy, gymnasium[mujoco], pytest
```

## Train

```bash
python train.py cartpole      # discrete sanity check (~30s)
python train.py halfcheetah   # continuous control (~15 min on GPU)
python train.py               # defaults to HalfCheetah
```

Environment progression (easy → hard):

| Environment | Action space | Steps | Expected reward | Time (GPU) |
|------------|-------------|-------|----------------|------------|
| CartPole-v1 | Discrete (2) | 100K | 500 (max) | ~30s |
| Pendulum-v1 | Continuous (1) | 200K | -200 to -150 | ~2 min |
| HalfCheetah-v5 | Continuous (6) | 1M | 1000-2000 | ~15 min |
| Ant-v5 | Continuous (8) | 2M | 1000-2000 | ~30 min |

Training output:

```
--- Update 10/61 (163,840 steps) ---
  Episodes: 42 | Avg return: 342.7 | Last: 456.0
  PPO: policy=0.018 value=45.2 entropy=3.89 kl=0.005 clip=0.08
  Eval: 456.2 +/- 34.1
  LR: 0.000251
```

Edit presets in `config.py` or create your own:

```python
from config import PPOConfig
config = PPOConfig(env_name="Ant-v5", total_timesteps=2_000_000, num_envs=16)
```

## Test

```bash
python -m pytest tests/ -v  # 25 tests
```

## Architecture

**Actor** (policy) — outputs a Gaussian distribution over continuous actions:

```
obs → Linear(64) → Tanh → Linear(64) → Tanh → Linear(act_dim) → mean
                                                  + learnable log_std → std
action ~ Normal(mean, std)
```

**Critic** (value function) — estimates expected return from a state:

```
obs → Linear(64) → Tanh → Linear(64) → Tanh → Linear(1) → V(s)
```

Separate networks (no shared trunk) — prevents gradient interference between the policy and value objectives.

## File structure

| File | What it does |
|------|-------------|
| `config.py` | Hyperparameters + presets per environment |
| `network.py` | Actor-Critic networks (continuous + discrete) |
| `rollout_buffer.py` | Trajectory storage + GAE advantage computation |
| `ppo.py` | Core PPO algorithm (clipped objective, value loss, entropy) |
| `train.py` | Training loop with vectorized environments |
| `evaluate.py` | Deterministic evaluation + video recording |

## Key RL concepts

| Concept | Where to find it |
|---------|-----------------|
| Policy gradients | `ppo.py` — ratio * advantage objective |
| Clipped surrogate | `ppo.py` — prevents destructive policy updates |
| GAE (advantage estimation) | `rollout_buffer.py` — bias-variance tradeoff via lambda |
| Continuous actions | `network.py` — Gaussian policy with learned mean + std |
| Vectorized environments | `train.py` — parallel rollout collection |
| Value function | `network.py` — critic estimates expected return |
| Entropy bonus | `ppo.py` — encourages exploration |

## Hyperparameter guide

The most impactful hyperparameters to tune:

- **`learning_rate`** — start with 3e-4. Too high → unstable, too low → slow
- **`clip_epsilon`** — 0.2 is standard. Lower = more conservative updates
- **`gae_lambda`** — 0.95 is standard. Lower = more bias, higher = more variance
- **`num_epochs`** — 10 for continuous, 4 for discrete. More epochs = more data reuse
- **`num_envs`** — more = faster data collection, but diminishing returns past 16

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — the PPO paper
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE
- [Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — why PPO details matter
