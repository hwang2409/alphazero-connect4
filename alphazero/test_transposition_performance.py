#!/usr/bin/env python3
"""Test script to demonstrate transposition table performance improvement."""

import time
import numpy as np

from game import Connect4
from model import AlphaZeroNet
from mcts import search


def measure_search_performance(num_simulations=100, num_trials=5):
    """Measure search performance with and without transposition table."""
    net = AlphaZeroNet()
    
    # Test position with many transpositions possible
    g = Connect4().make_move(3).make_move(3).make_move(2).make_move(4)
    
    # Without transposition table
    times_without = []
    for _ in range(num_trials):
        start = time.time()
        probs, val = search(g, net, num_simulations=num_simulations, 
                           add_noise=False, transposition_table=None)
        times_without.append(time.time() - start)
    
    # With transposition table
    times_with = []
    unique_positions = []
    for _ in range(num_trials):
        transposition_table = {}
        start = time.time()
        probs, val = search(g, net, num_simulations=num_simulations,
                           add_noise=False, transposition_table=transposition_table)
        times_with.append(time.time() - start)
        unique_positions.append(len(transposition_table))
    
    avg_time_without = np.mean(times_without)
    avg_time_with = np.mean(times_with)
    avg_unique_positions = np.mean(unique_positions)
    
    print(f"MCTS Performance Test ({num_simulations} simulations per search)")
    print(f"=" * 50)
    print(f"Without transposition table: {avg_time_without:.4f}s average")
    print(f"With transposition table:    {avg_time_with:.4f}s average")
    print(f"Speed improvement:           {avg_time_without/avg_time_with:.2f}x")
    print(f"Unique positions cached:     {avg_unique_positions:.0f} average")
    print()
    
    return avg_time_without, avg_time_with, avg_unique_positions


def test_transposition_detection():
    """Test that transpositions are correctly detected."""
    print("Transposition Detection Test")
    print("=" * 50)
    
    # Create several transpositions of the same position
    moves_list = [
        [0, 1, 2, 3],  # X at 0,2; O at 1,3
        [2, 3, 0, 1],  # Same position, different order
        [0, 3, 2, 1],  # Same position, another order
        [2, 1, 0, 3],  # Same position, yet another order
    ]
    
    games = []
    for moves in moves_list:
        g = Connect4()
        for move in moves:
            g = g.make_move(move)
        games.append(g)
    
    # Check all have same hash
    hashes = [hash(g) for g in games]
    print(f"Created {len(games)} game states via different move orders")
    print(f"Unique hashes: {len(set(hashes))}")
    print(f"All equal: {all(g == games[0] for g in games)}")
    
    # Run MCTS with transposition table
    net = AlphaZeroNet()
    transposition_table = {}
    
    for i, g in enumerate(games):
        probs, val = search(g, net, num_simulations=50,
                           add_noise=False, transposition_table=transposition_table)
        print(f"After search {i+1}: {len(transposition_table)} positions cached")
    
    print()


if __name__ == "__main__":
    print("Testing MCTS with Transposition Table")
    print("=" * 50)
    print()
    
    # Test transposition detection
    test_transposition_detection()
    
    # Test performance improvement
    measure_search_performance(num_simulations=100, num_trials=3)
    
    # Test with more simulations
    print("Testing with more simulations...")
    measure_search_performance(num_simulations=200, num_trials=3)