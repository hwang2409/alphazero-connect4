import numpy as np
import pytest

from game import Connect4
from model import AlphaZeroNet
from mcts import Node, _expand, _expand_progressive, search


class TestProgressiveWidening:
    def test_progressive_expansion_initial(self):
        """Test that progressive widening starts with fewer children."""
        g = Connect4()
        node = Node(g)
        
        # Create a dummy policy
        policy = np.ones(7) / 7  # uniform policy
        
        # Standard expansion should create all 7 children
        _expand(node, policy)
        assert len(node.children) == 7
        assert node.is_expanded == True
        
        # Progressive expansion with visit_count=0 should create fewer
        node2 = Node(g)
        _expand_progressive(node2, policy, c_pw=1.5, alpha_pw=0.5)
        assert len(node2.children) == 1  # floor(1.5 * 1^0.5) = 1
        assert node2.is_expanded == False
    
    def test_progressive_expansion_grows_with_visits(self):
        """Test that more children are added as visit count increases."""
        g = Connect4()
        node = Node(g)
        policy = np.ones(7) / 7
        
        # Initial expansion
        _expand_progressive(node, policy, c_pw=1.5, alpha_pw=0.5)
        assert len(node.children) == 1
        
        # Simulate visits and re-expand
        node.visit_count = 4
        _expand_progressive(node, policy, c_pw=1.5, alpha_pw=0.5)
        assert len(node.children) == 3  # floor(1.5 * 4^0.5) = 3
        
        node.visit_count = 9
        _expand_progressive(node, policy, c_pw=1.5, alpha_pw=0.5)
        assert len(node.children) == 4  # floor(1.5 * 9^0.5) = 4
        
        node.visit_count = 16
        _expand_progressive(node, policy, c_pw=1.5, alpha_pw=0.5)
        assert len(node.children) == 6  # floor(1.5 * 16^0.5) = 6
        
        node.visit_count = 25
        _expand_progressive(node, policy, c_pw=1.5, alpha_pw=0.5)
        assert len(node.children) == 7  # floor(1.5 * 25^0.5) = 7, capped at 7
        assert node.is_expanded == True
    
    def test_progressive_expansion_respects_policy_order(self):
        """Test that children are added in order of policy probability."""
        g = Connect4()
        node = Node(g)
        
        # Non-uniform policy with clear preferences
        policy = np.array([0.05, 0.3, 0.05, 0.4, 0.05, 0.1, 0.05])
        
        # First expansion should add highest probability move
        _expand_progressive(node, policy, c_pw=1.0, alpha_pw=0.5)
        assert 3 in node.children  # column 3 has highest prob (0.4)
        
        # Next expansion should add second highest
        node.visit_count = 4
        _expand_progressive(node, policy, c_pw=1.0, alpha_pw=0.5)
        assert 1 in node.children  # column 1 has second highest (0.3)
    
    def test_progressive_widening_with_invalid_moves(self):
        """Test progressive widening when some moves are invalid."""
        g = Connect4()
        # Fill column 0 completely
        for _ in range(6):
            g = g.make_move(0)
        
        node = Node(g)
        policy = np.ones(7) / 7
        
        # Should only consider valid moves (columns 1-6)
        _expand_progressive(node, policy, c_pw=1.5, alpha_pw=0.5)
        assert 0 not in node.children
        assert len(node.children) == 1
        
        # Increase visits to expand more
        node.visit_count = 16
        _expand_progressive(node, policy, c_pw=1.5, alpha_pw=0.5)
        assert 0 not in node.children
        assert len(node.children) == 6  # All valid moves
        assert node.is_expanded == True
    
    def test_search_with_progressive_widening(self):
        """Test full MCTS search with progressive widening enabled."""
        net = AlphaZeroNet()
        g = Connect4()
        
        # Run search with progressive widening
        probs_pw, value_pw = search(
            g, net, 
            num_simulations=50, 
            add_noise=False,
            use_progressive_widening=True,
            c_pw=1.5,
            alpha_pw=0.5
        )
        
        # Run search without progressive widening
        probs_std, value_std = search(
            g, net, 
            num_simulations=50, 
            add_noise=False,
            use_progressive_widening=False
        )
        
        # Both should return valid probability distributions
        assert probs_pw.shape == (7,)
        assert abs(probs_pw.sum() - 1.0) < 1e-5
        assert probs_std.shape == (7,)
        assert abs(probs_std.sum() - 1.0) < 1e-5
        
        # Results might differ slightly but should be reasonable
        assert isinstance(value_pw, float)
        assert isinstance(value_std, float)
    
    def test_progressive_widening_parameters(self):
        """Test different progressive widening parameters."""
        g = Connect4()
        node = Node(g)
        policy = np.ones(7) / 7
        
        # Test with different c_pw values
        node1 = Node(g)
        node1.visit_count = 4
        _expand_progressive(node1, policy, c_pw=1.0, alpha_pw=0.5)
        
        node2 = Node(g)
        node2.visit_count = 4
        _expand_progressive(node2, policy, c_pw=2.0, alpha_pw=0.5)
        
        assert len(node2.children) >= len(node1.children)
        
        # Test with different alpha_pw values
        node3 = Node(g)
        node3.visit_count = 16
        _expand_progressive(node3, policy, c_pw=1.0, alpha_pw=0.3)
        
        node4 = Node(g)
        node4.visit_count = 16
        _expand_progressive(node4, policy, c_pw=1.0, alpha_pw=0.7)
        
        # Higher alpha means faster growth
        assert len(node4.children) >= len(node3.children)