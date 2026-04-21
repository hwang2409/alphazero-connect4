"""Benchmark script to compare sequential vs parallel MCTS performance."""
import time
import numpy as np

from config import Config
from game import Connect4
from model import AlphaZeroNet
from mcts import search
from mcts_parallel import search_parallel


def benchmark_mcts(num_positions: int = 10, num_simulations: int = 200):
    """Compare sequential and parallel MCTS performance."""
    print(f"\nBenchmarking MCTS with {num_simulations} simulations per position")
    print(f"Testing on {num_positions} different game positions\n")
    
    # Create model and test positions
    model = AlphaZeroNet()
    model.eval()
    
    # Generate random positions by playing random moves
    positions = []
    for _ in range(num_positions):
        g = Connect4()
        num_moves = np.random.randint(5, 15)
        for _ in range(num_moves):
            if g.is_terminal()[0]:
                break
            valid = np.where(g.get_valid_moves())[0]
            move = np.random.choice(valid)
            g = g.make_move(move)
        if not g.is_terminal()[0]:
            positions.append(g)
    
    print(f"Generated {len(positions)} non-terminal positions")
    
    # Benchmark sequential MCTS
    print("\n--- Sequential MCTS ---")
    seq_start = time.time()
    seq_results = []
    for pos in positions:
        probs, value = search(pos, model, num_simulations=num_simulations, add_noise=False)
        seq_results.append((probs, value))
    seq_time = time.time() - seq_start
    
    print(f"Total time: {seq_time:.2f}s")
    print(f"Time per position: {seq_time/len(positions):.3f}s")
    print(f"Simulations per second: {(num_simulations * len(positions)) / seq_time:.1f}")
    
    # Benchmark parallel MCTS with different batch sizes
    for batch_size in [4, 8, 16]:
        print(f"\n--- Parallel MCTS (batch_size={batch_size}) ---")
        par_start = time.time()
        par_results = []
        for pos in positions:
            probs, value = search_parallel(pos, model, num_simulations=num_simulations,
                                         batch_size=batch_size, add_noise=False)
            par_results.append((probs, value))
        par_time = time.time() - par_start
        
        print(f"Total time: {par_time:.2f}s")
        print(f"Time per position: {par_time/len(positions):.3f}s")
        print(f"Simulations per second: {(num_simulations * len(positions)) / par_time:.1f}")
        print(f"Speedup vs sequential: {seq_time/par_time:.2f}x")
        
        # Check result similarity
        diffs = []
        for (sp, sv), (pp, pv) in zip(seq_results, par_results):
            # Compare top action
            if np.argmax(sp) == np.argmax(pp):
                diffs.append(1.0)
            else:
                # Check if top actions have similar probabilities
                diffs.append(0.5 if abs(sp[np.argmax(sp)] - pp[np.argmax(pp)]) < 0.1 else 0.0)
        
        print(f"Top action agreement: {np.mean(diffs)*100:.1f}%")


if __name__ == "__main__":
    # Quick benchmark
    benchmark_mcts(num_positions=5, num_simulations=100)
    
    print("\n" + "="*60)
    
    # More thorough benchmark
    benchmark_mcts(num_positions=20, num_simulations=200)