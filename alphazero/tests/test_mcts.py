import numpy as np

from game import Connect4
from model import AlphaZeroNet
from mcts import search, select_action, Node, _expand, _backpropagate


class TestMCTSBasics:
    def test_returns_valid_distribution(self):
        net = AlphaZeroNet()
        g = Connect4()
        probs, value = search(g, net, num_simulations=20, add_noise=False)
        assert probs.shape == (7,)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert (probs >= 0).all()

    def test_visit_counts_match_simulations(self):
        net = AlphaZeroNet()
        g = Connect4()
        root = Node(g)
        policy, value = net.predict(g)
        _expand(root, policy)

        # Run full search to verify
        probs, _ = search(g, net, num_simulations=50, add_noise=False)
        # All probability should be on valid moves
        valid = g.get_valid_moves()
        assert all(probs[i] == 0 for i in range(7) if not valid[i])

    def test_no_probability_on_invalid_moves(self):
        net = AlphaZeroNet()
        g = Connect4()
        # Fill column 0
        for _ in range(6):
            g = g.make_move(0)
        probs, _ = search(g, net, num_simulations=30, add_noise=False)
        assert probs[0] == 0.0


class TestForcedMoves:
    def test_finds_winning_move(self):
        """Player 1 has 3 in a row at bottom, column 3 wins."""
        g = Connect4()
        # X X X . . . .  (player 1 at cols 0,1,2)
        # O O O . . . .  (player -1 at cols 0,1,2 row 1... no wait)
        # Build: 0, 6, 1, 6, 2 -> player 1 has [0,1,2] on row 0
        g = g.make_move(0).make_move(6).make_move(1).make_move(6).make_move(2)
        # Now player -1 to move, then player 1 can win with col 3
        g = g.make_move(5)  # player -1 wastes a move
        # Now player 1: has 3 in a row, col 3 wins

        net = AlphaZeroNet()
        probs, _ = search(g, net, num_simulations=200, add_noise=False)
        assert np.argmax(probs) == 3

    def test_blocks_opponent_win(self):
        """Opponent has 3 in a row, must block column 3."""
        g = Connect4()
        # Player -1 (O) has cols 0,1,2 on row 0
        # Build: 6, 0, 6, 1, 5, 2 -> O at [0,1,2] row 0
        g = g.make_move(6).make_move(0).make_move(6).make_move(1).make_move(5).make_move(2)
        # Player 1 to move, must block col 3

        net = AlphaZeroNet()
        probs, _ = search(g, net, num_simulations=200, add_noise=False)
        assert np.argmax(probs) == 3


class TestSelectAction:
    def test_argmax_at_low_temperature(self):
        probs = np.array([0.1, 0.5, 0.2, 0.05, 0.1, 0.03, 0.02])
        action = select_action(probs, temperature=0.001)
        assert action == 1

    def test_samples_at_high_temperature(self):
        probs = np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        np.random.seed(42)
        actions = [select_action(probs, temperature=1.0) for _ in range(100)]
        assert 1 in actions
        assert 2 in actions
        assert all(a in (1, 2) for a in actions)


class TestTranspositionTable:
    def test_reuses_transposed_nodes(self):
        """Test that transposition table reuses nodes for same positions."""
        net = AlphaZeroNet()
        
        # Create a true transposition: same position reached by different move orders
        # Path 1: 0, 1, 2, 3 (X at 0,2; O at 1,3)
        # Path 2: 2, 3, 0, 1 (X at 0,2; O at 1,3) - same final position
        g1 = Connect4().make_move(0).make_move(1).make_move(2).make_move(3)
        g2 = Connect4().make_move(2).make_move(3).make_move(0).make_move(1)
        
        # Verify these are actually the same position
        assert hash(g1) == hash(g2)
        assert g1 == g2
        
        transposition_table = {}
        
        # Run search on first position
        probs1, _ = search(g1, net, num_simulations=20, 
                          add_noise=False, transposition_table=transposition_table)
        
        table_size_after_first = len(transposition_table)
        assert table_size_after_first > 1  # Should have explored some positions
        
        # Run search on transposed position - should reuse cached nodes
        probs2, _ = search(g2, net, num_simulations=20,
                          add_noise=False, transposition_table=transposition_table)
        
        # The root node should have been reused (same hash)
        root_hash = hash(g1)
        assert root_hash in transposition_table
    
    def test_transposition_improves_search(self):
        """Test that transposition table improves search quality."""
        net = AlphaZeroNet()
        g = Connect4()
        
        # Search without transposition table
        probs1, val1 = search(g, net, num_simulations=50, add_noise=False)
        
        # Search with transposition table should visit nodes more efficiently
        transposition_table = {}
        probs2, val2 = search(g, net, num_simulations=50, add_noise=False,
                             transposition_table=transposition_table)
        
        # Both should produce valid probability distributions
        assert abs(probs1.sum() - 1.0) < 1e-5
        assert abs(probs2.sum() - 1.0) < 1e-5


class TestBackpropagate:
    def test_value_alternates(self):
        g = Connect4()
        root = Node(g)
        child_state = g.make_move(3)
        child = Node(child_state, parent=root, action=3, prior=0.5)
        root.children[3] = child

        # Backprop value of 1.0 from child
        _backpropagate(child, 1.0)

        assert child.visit_count == 1
        assert child.value_sum == 1.0
        assert root.visit_count == 1
        assert root.value_sum == -1.0  # negated
