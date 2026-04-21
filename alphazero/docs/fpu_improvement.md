# First Play Urgency (FPU) Improvement for MCTS

## Overview

This improvement adds First Play Urgency (FPU) to the Monte Carlo Tree Search implementation. FPU is a technique that assigns an initial value estimate to unvisited nodes, improving the exploration-exploitation trade-off.

## The Problem

In the original implementation, unvisited nodes have a Q-value of 0.0 (neutral). This can lead to suboptimal behavior:

- If the current position is winning (+1), unvisited nodes look artificially bad compared to visited winning nodes
- If the current position is losing (-1), unvisited nodes look artificially good compared to visited losing nodes
- The algorithm may over-explore or under-explore depending on the position

## The Solution

FPU assigns a configurable initial value to unvisited nodes. The recommended default is slightly negative (-0.1), which:

1. **Encourages exploitation**: The algorithm prefers to deepen search in already-visited branches before trying new ones
2. **Improves tactical play**: In positions with forced sequences (like preventing opponent wins), the algorithm focuses on critical lines
3. **Reduces noise**: By slightly discouraging exploration of all possible moves, the algorithm produces more stable policies

## Implementation Details

- Added `fpu_value` parameter to `_ucb_score()` function
- Modified `_select_child()` to pass FPU value through
- Added `fpu_value` parameter to `search()` with default -0.1
- Added configuration option in `Config` class

## Usage

```python
# Use default FPU value of -0.1
probs, value = search(game, model, num_simulations=800)

# Or specify custom FPU value
probs, value = search(game, model, num_simulations=800, fpu_value=-0.2)
```

## Tuning Guidelines

- **Negative FPU (-0.3 to -0.05)**: More exploitation, better for tactical positions
- **Zero FPU (0.0)**: Original behavior
- **Positive FPU (0.05 to 0.3)**: More exploration, better for strategic positions

The optimal value depends on the game and position characteristics. For Connect 4, slight negative values (-0.1 to -0.2) work well.