import numpy as np
import torch

from game import Connect4
from model import AlphaZeroNet
from mcts import search, Node, _expand, _ucb_score, _select_child


class MockModel(AlphaZeroNet):
    """Mock model that returns controllable policy/value for testing."""
    def __init__(self, policy=None, value=0.0):
        super().__init__()
        self.mock_policy = policy
        self.mock_value = value
    
    def predict(self, state, device="cpu"):
        if self.mock_policy is None:
            # Uniform policy
            policy = np.ones(state.cols) / state.cols
        else:
            policy = self.mock_policy
        return policy, self.mock_value


class TestFPU:
    def test_fpu_affects_unvisited_nodes(self):
        """Test that FPU value is used for unvisited nodes."""
        g = Connect4()
        parent = Node(g)
        
        # Create children with priors
        policy = np.array([0.1, 0.2, 0.3, 0.15, 0.15, 0.05, 0.05])
        _expand(parent, policy)
        
        # With FPU = 0 (default), unvisited nodes have effective Q = 0
        score_default = _ucb_score(parent, parent.children[0], c_puct=1.0, fpu_value=0.0)
        
        # With negative FPU, unvisited nodes are penalized
        score_negative = _ucb_score(parent, parent.children[0], c_puct=1.0, fpu_value=-0.5)
        assert score_negative < score_default
        
        # With positive FPU, unvisited nodes are favored
        score_positive = _ucb_score(parent, parent.children[0], c_puct=1.0, fpu_value=0.5)
        assert score_positive > score_default
    
    def test_fpu_ignored_for_visited_nodes(self):
        """Test that FPU is not used once a node has been visited."""
        g = Connect4()
        parent = Node(g)
        parent.visit_count = 10
        
        child = Node(g.make_move(0), parent=parent, action=0, prior=0.2)
        child.visit_count = 5
        child.value_sum = 2.5  # Q = 0.5
        
        # FPU should not affect visited nodes
        score1 = _ucb_score(parent, child, c_puct=1.0, fpu_value=-1.0)
        score2 = _ucb_score(parent, child, c_puct=1.0, fpu_value=1.0)
        
        # Only exploration term differs slightly due to visit count
        assert abs(score1 - score2) < 1e-6
    
    def test_fpu_affects_exploration(self):
        """Test that FPU value affects exploration behavior."""
        g = Connect4()
        
        # Create a model with skewed policy to make the effect more visible
        skewed_policy = np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025])
        model = MockModel(policy=skewed_policy, value=0.0)
        
        # Run search with negative FPU - discourages exploring unvisited nodes
        probs_negative, _ = search(g, model, num_simulations=50, 
                                   fpu_value=-0.5, add_noise=False, c_puct=2.0)
        
        # Run search with positive FPU - encourages exploring unvisited nodes
        probs_positive, _ = search(g, model, num_simulations=50,
                                   fpu_value=0.5, add_noise=False, c_puct=2.0)
        
        # With positive FPU, unvisited nodes are more attractive, so the algorithm
        # should spread visits more evenly rather than focusing on high-prior moves
        # This means the dominant move (index 0) should get fewer visits
        assert probs_positive[0] < probs_negative[0]
        
        # And with negative FPU, visits should be more concentrated
        max_prob_negative = np.max(probs_negative)
        max_prob_positive = np.max(probs_positive)
        assert max_prob_negative > max_prob_positive
    
    def test_fpu_in_select_child(self):
        """Test that select_child passes FPU value correctly."""
        g = Connect4()
        parent = Node(g)
        parent.visit_count = 20
        
        # Create children
        policy = np.array([0.4, 0.3, 0.2, 0.05, 0.03, 0.01, 0.01])
        _expand(parent, policy)
        
        # Visit some children
        parent.children[0].visit_count = 10
        parent.children[0].value_sum = 3.0  # Q = 0.3
        parent.children[1].visit_count = 5
        parent.children[1].value_sum = -1.0  # Q = -0.2
        
        # With very negative FPU, should prefer visited nodes
        action1, child1 = _select_child(parent, c_puct=1.0, fpu_value=-2.0)
        assert action1 in [0, 1]  # Should pick a visited node
        
        # With very positive FPU, might pick unvisited high-prior node
        action2, child2 = _select_child(parent, c_puct=1.0, fpu_value=2.0)
        # Could pick action 2 (prior=0.2, unvisited) due to high FPU
    
    def test_search_accepts_fpu_parameter(self):
        """Test that search function accepts and uses FPU parameter."""
        g = Connect4()
        model = AlphaZeroNet()
        
        # Should run without error with custom FPU
        probs, value = search(g, model, num_simulations=50, fpu_value=-0.15)
        assert probs.shape == (7,)
        assert isinstance(value, float)