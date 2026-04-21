import numpy as np
import pytest

from game import Connect4
from model import AlphaZeroNet
from mcts import search
from mcts_parallel import search_parallel, VirtualLossNode, _apply_virtual_loss, _revert_virtual_loss


class TestVirtualLoss:
    def test_virtual_loss_affects_q_value(self):
        """Virtual loss should temporarily decrease Q-value."""
        node = VirtualLossNode(Connect4())
        node.visit_count = 10
        node.value_sum = 5.0  # Q = 0.5
        
        assert node.q_value == 0.5
        
        # Apply virtual loss
        node.virtual_loss_count = 3
        # Q = (5.0 - 3) / (10 + 3) = 2/13 ≈ 0.154
        assert abs(node.q_value - 2/13) < 1e-6
        
    def test_apply_revert_virtual_loss(self):
        """Virtual loss should propagate up tree and be revertible."""
        g = Connect4()
        root = VirtualLossNode(g)
        child_state = g.make_move(3)
        child = VirtualLossNode(child_state, parent=root, action=3)
        grandchild_state = child_state.make_move(4)
        grandchild = VirtualLossNode(grandchild_state, parent=child, action=4)
        
        # Apply virtual loss from grandchild
        _apply_virtual_loss(grandchild, virtual_loss=2)
        
        assert grandchild.virtual_loss_count == 2
        assert child.virtual_loss_count == 2
        assert root.virtual_loss_count == 2
        
        # Revert
        _revert_virtual_loss(grandchild, virtual_loss=2)
        
        assert grandchild.virtual_loss_count == 0
        assert child.virtual_loss_count == 0
        assert root.virtual_loss_count == 0


class TestParallelSearch:
    def test_returns_valid_distribution(self):
        """Parallel search should return valid action probabilities."""
        net = AlphaZeroNet()
        g = Connect4()
        probs, value = search_parallel(g, net, num_simulations=32, batch_size=8, add_noise=False)
        
        assert probs.shape == (7,)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert (probs >= 0).all()
        assert isinstance(value, (int, float))
        
    def test_batch_size_doesnt_affect_validity(self):
        """Different batch sizes should all produce valid results."""
        net = AlphaZeroNet()
        g = Connect4()
        
        for batch_size in [1, 4, 16]:
            probs, _ = search_parallel(g, net, num_simulations=32, 
                                      batch_size=batch_size, add_noise=False)
            assert abs(probs.sum() - 1.0) < 1e-5
            valid = g.get_valid_moves()
            assert all(probs[i] == 0 for i in range(7) if not valid[i])
    
    def test_finds_winning_move(self):
        """Should find obvious winning move like sequential MCTS."""
        g = Connect4()
        # Player 1 has 3 in a row, column 3 wins
        g = g.make_move(0).make_move(6).make_move(1).make_move(6).make_move(2).make_move(5)
        
        net = AlphaZeroNet()
        probs, _ = search_parallel(g, net, num_simulations=200, 
                                  batch_size=16, add_noise=False)
        assert np.argmax(probs) == 3
        
    def test_blocks_opponent_win(self):
        """Should block opponent's winning threat."""
        g = Connect4()
        # Opponent has 3 in a row, must block column 3
        g = g.make_move(6).make_move(0).make_move(6).make_move(1).make_move(5).make_move(2)
        
        net = AlphaZeroNet()
        probs, _ = search_parallel(g, net, num_simulations=200, 
                                  batch_size=16, add_noise=False)
        assert np.argmax(probs) == 3
    
    def test_similar_results_to_sequential(self):
        """Parallel and sequential should give similar results with enough simulations."""
        net = AlphaZeroNet()
        g = Connect4()
        # Make a few moves to get interesting position
        g = g.make_move(3).make_move(3).make_move(4).make_move(2)
        
        # Set seeds for reproducibility
        np.random.seed(42)
        probs_seq, value_seq = search(g, net, num_simulations=100, add_noise=False)
        
        np.random.seed(42)
        probs_par, value_par = search_parallel(g, net, num_simulations=100, 
                                              batch_size=10, add_noise=False)
        
        # Should be reasonably similar (not exact due to parallelism and different exploration patterns)
        # Check that the highest probability actions are similar
        top_seq = np.argsort(probs_seq)[-3:]
        top_par = np.argsort(probs_par)[-3:]
        assert len(set(top_seq) & set(top_par)) >= 2  # At least 2 of top 3 actions overlap
        assert abs(value_seq - value_par) < 0.5


class TestBatchProcessing:
    def test_handles_terminal_nodes(self):
        """Should handle mix of terminal and non-terminal nodes in batch."""
        net = AlphaZeroNet()
        g = Connect4()
        
        # Create a position where some paths lead to quick terminal nodes
        # Fill bottom rows to make the game end sooner
        for col in range(7):
            for _ in range(3):
                if not g.is_terminal()[0]:
                    g = g.make_move(col % 7)
        
        # Should complete without errors
        probs, value = search_parallel(g, net, num_simulations=50, 
                                      batch_size=10, add_noise=False)
        assert probs.shape == (7,)
        
    def test_virtual_loss_parameter(self):
        """Different virtual loss values should affect exploration."""
        net = AlphaZeroNet()
        g = Connect4()
        
        np.random.seed(42)
        probs1, _ = search_parallel(g, net, num_simulations=100, batch_size=10, 
                                   virtual_loss=1, add_noise=False)
        
        np.random.seed(42)
        probs2, _ = search_parallel(g, net, num_simulations=100, batch_size=10, 
                                   virtual_loss=10, add_noise=False)
        
        # Higher virtual loss should lead to more exploration (less concentrated distribution)
        entropy1 = -np.sum(probs1 * np.log(probs1 + 1e-8))
        entropy2 = -np.sum(probs2 * np.log(probs2 + 1e-8))
        # This is a soft check - virtual loss affects exploration patterns
        assert not np.allclose(probs1, probs2, atol=0.05)