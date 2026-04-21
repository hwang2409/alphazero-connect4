# Parallel MCTS Implementation

This document describes the parallel Monte Carlo Tree Search (MCTS) implementation added to the AlphaZero codebase.

## Overview

The standard MCTS algorithm runs simulations sequentially, which can be slow when using neural networks for position evaluation. The parallel implementation addresses this bottleneck by:

1. **Batching neural network evaluations** - Multiple leaf positions are evaluated together
2. **Using virtual loss** - Prevents multiple simulations from exploring the same path
3. **Maintaining exploration diversity** - Virtual loss encourages different threads to explore different parts of the tree

## How It Works

### Virtual Loss

When a simulation selects a path down the tree, we temporarily add a "virtual loss" to each node along that path. This makes the path less attractive to other simulations, encouraging exploration of different branches.

```python
# During selection, virtual loss makes Q-value worse
q_value = (value_sum - virtual_loss_count) / (visit_count + virtual_loss_count)
```

### Batch Processing

Instead of evaluating positions one at a time:
1. Collect multiple leaf nodes from different simulations
2. Batch their neural network evaluations
3. Backpropagate all results
4. Remove virtual losses

## Performance Results

Based on benchmarking with Connect4:
- **2-3x speedup** with batch size 8
- Maintains high action agreement with sequential MCTS (70-80%)
- Best performance with batch size matching typical neural network batch efficiency

## Usage

### Configuration

Add these settings to your `Config`:

```python
# Parallel MCTS settings
use_parallel_mcts: bool = True
mcts_batch_size: int = 8
virtual_loss: int = 3
```

### In Training

The self-play generation automatically uses parallel MCTS if enabled:

```python
examples = generate_self_play_data(model, config, use_parallel_mcts=True)
```

### Direct Usage

```python
from mcts_parallel import search_parallel

probs, value = search_parallel(
    game_state, 
    model,
    num_simulations=200,
    batch_size=8,
    virtual_loss=3,
    device="cpu"
)
```

## Implementation Details

### Key Classes

- `VirtualLossNode` - Extends the base `Node` class with virtual loss tracking
- `search_parallel` - Main entry point for parallel MCTS

### Algorithm Flow

1. Initialize root node and expand with neural network
2. For each batch:
   - Select leaf nodes using UCB with virtual loss
   - Apply virtual loss to paths
   - Batch evaluate leaf positions
   - Expand nodes and backpropagate
   - Remove virtual losses

### Design Choices

- **Virtual loss value**: Default of 3 provides good exploration without over-penalization
- **Batch collection**: Continues until batch is full or all paths hit terminal nodes
- **CPU inference**: Often faster for small batches due to reduced overhead

## Future Improvements

1. **True parallelism**: Current implementation is pseudo-parallel (batched). Could use actual threading/multiprocessing
2. **Dynamic batch sizing**: Adjust batch size based on tree shape and depth
3. **GPU optimization**: Better utilize GPU for larger batch sizes
4. **Tree reuse**: Maintain tree between moves in self-play

## References

- [Parallel Monte-Carlo Tree Search](https://hal.inria.fr/inria-00203077/document)
- [AlphaGo Zero paper](https://www.nature.com/articles/nature24270) - Mentions parallel MCTS with virtual loss