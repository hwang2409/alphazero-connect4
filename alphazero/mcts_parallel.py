import math
from typing import List, Tuple

import numpy as np
import torch

from game import Connect4
from model import AlphaZeroNet
from mcts import Node, _ucb_score, _expand


class VirtualLossNode(Node):
    """Node class extended with virtual loss for parallel MCTS."""
    __slots__ = Node.__slots__ + ("virtual_loss_count",)
    
    def __init__(self, state: Connect4, parent: "VirtualLossNode | None" = None,
                 action: int | None = None, prior: float = 0.0):
        super().__init__(state, parent, action, prior)
        self.virtual_loss_count = 0
    
    @property
    def q_value(self) -> float:
        # Include virtual losses as actual losses (value = -1)
        total_visits = self.visit_count + self.virtual_loss_count
        if total_visits == 0:
            return 0.0
        total_value = self.value_sum - self.virtual_loss_count
        return total_value / total_visits


def _apply_virtual_loss(node: VirtualLossNode, virtual_loss: int = 1) -> None:
    """Apply virtual loss along the path to discourage other threads."""
    while node is not None:
        node.virtual_loss_count += virtual_loss
        node = node.parent


def _revert_virtual_loss(node: VirtualLossNode, virtual_loss: int = 1) -> None:
    """Remove virtual loss after evaluation completes."""
    while node is not None:
        node.virtual_loss_count -= virtual_loss
        node = node.parent


def _select_child_virtual(node: VirtualLossNode, c_puct: float) -> Tuple[int, VirtualLossNode]:
    """Select child considering virtual losses."""
    best_score = -float("inf")
    best_action = -1
    best_child = None
    for action, child in node.children.items():
        score = _ucb_score(node, child, c_puct)
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    return best_action, best_child


def _expand_virtual(node: VirtualLossNode, policy: np.ndarray) -> None:
    """Create VirtualLossNode children for all valid moves."""
    valid = node.state.get_valid_moves()
    
    # Mask invalid moves and renormalize
    policy = policy * valid
    policy_sum = policy.sum()
    if policy_sum > 0:
        policy = policy / policy_sum
    else:
        # Fallback: uniform over valid moves
        policy = valid.astype(np.float32) / valid.sum()
    
    for action in range(len(valid)):
        if valid[action]:
            child_state = node.state.make_move(action)
            node.children[action] = VirtualLossNode(
                state=child_state,
                parent=node,
                action=action,
                prior=policy[action],
            )
    node.is_expanded = True


def _backpropagate_virtual(node: VirtualLossNode, value: float) -> None:
    """Propagate value up the tree, same as regular backprop."""
    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        value = -value  # flip perspective for parent
        node = node.parent


def search_parallel(state: Connect4, model: AlphaZeroNet, num_simulations: int,
                    batch_size: int = 8, c_puct: float = 1.5, 
                    dirichlet_alpha: float = 1.0, dirichlet_epsilon: float = 0.25,
                    add_noise: bool = True, device: str = "cpu", 
                    virtual_loss: int = 3) -> Tuple[np.ndarray, float]:
    """Parallel MCTS with virtual loss and batched neural network evaluation.
    
    Args:
        state: Root game state
        model: Neural network for position evaluation
        num_simulations: Total number of simulations to run
        batch_size: Number of positions to evaluate in parallel
        c_puct: Exploration constant
        dirichlet_alpha: Dirichlet noise parameter
        dirichlet_epsilon: Noise mixing parameter
        add_noise: Whether to add Dirichlet noise at root
        device: Device for neural network inference
        virtual_loss: Virtual loss value to apply during selection
        
    Returns:
        action_probs: Visit count distribution over actions
        root_value: Estimated value of root position
    """
    root = VirtualLossNode(state)
    
    # Expand root with neural net
    policy, value = model.predict(state, device=device)
    _expand_virtual(root, policy)
    
    # Add Dirichlet noise to root for exploration
    if add_noise:
        valid_actions = list(root.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(valid_actions))
        for i, action in enumerate(valid_actions):
            child = root.children[action]
            child.prior = (1 - dirichlet_epsilon) * child.prior + dirichlet_epsilon * noise[i]
    
    # Run simulations in batches
    num_batches = (num_simulations + batch_size - 1) // batch_size
    
    for _ in range(num_batches):
        # Collect batch of leaf nodes
        leaf_nodes = []
        paths = []
        
        for _ in range(min(batch_size, num_simulations - len(leaf_nodes))):
            node = root
            path = []
            
            # SELECT: walk down tree with virtual loss
            while node.is_expanded and node.children:
                _, child = _select_child_virtual(node, c_puct)
                path.append(child)
                node = child
            
            # Check if terminal
            terminal, terminal_reward = node.state.is_terminal()
            if terminal:
                # Apply virtual loss to path
                for n in path:
                    _apply_virtual_loss(n, virtual_loss)
                
                # Immediately backpropagate and revert virtual loss
                _backpropagate_virtual(node, terminal_reward)
                for n in path:
                    _revert_virtual_loss(n, virtual_loss)
            else:
                # Non-terminal leaf - add to batch
                leaf_nodes.append(node)
                paths.append(path)
                # Apply virtual loss to discourage other threads
                for n in path:
                    _apply_virtual_loss(n, virtual_loss)
        
        if not leaf_nodes:
            continue
            
        # BATCH EVALUATE: Evaluate all leaf positions together
        states = [node.state for node in leaf_nodes]
        
        # Stack encoded states for batch processing
        encoded_states = np.stack([s.encode() for s in states])
        encoded_tensor = torch.from_numpy(encoded_states).to(device)
        
        with torch.no_grad():
            model.eval()
            policies, values = model(encoded_tensor)
            policies = policies.cpu().numpy()
            values = values.cpu().numpy().squeeze(-1)
        
        # EXPAND and BACKPROPAGATE for each leaf
        for i, (node, path) in enumerate(zip(leaf_nodes, paths)):
            # Expand with policy
            _expand_virtual(node, policies[i])
            
            # Backpropagate value (negate because value is from current player's perspective)
            _backpropagate_virtual(node, -values[i])
            
            # Revert virtual loss
            for n in path:
                _revert_virtual_loss(n, virtual_loss)
    
    # Extract action probabilities from visit counts
    cols = state.cols
    action_probs = np.zeros(cols, dtype=np.float32)
    for action, child in root.children.items():
        action_probs[action] = child.visit_count
    
    # Normalize
    total = action_probs.sum()
    if total > 0:
        action_probs /= total
    
    root_value = root.q_value
    return action_probs, root_value