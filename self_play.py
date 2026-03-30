from dataclasses import dataclass

import numpy as np
import torch

from config import Config
from game import Connect4
from model import AlphaZeroNet
from mcts import search, select_action


@dataclass
class TrainingExample:
    state: np.ndarray       # (3, rows, cols)
    policy: np.ndarray      # (cols,)
    value: float            # from current player's perspective


def play_game(model: AlphaZeroNet, config: Config, device: str = "cpu") -> list[TrainingExample]:
    """Play a single self-play game and return training examples."""
    game = Connect4(config.rows, config.cols, config.win_length)
    history: list[tuple[np.ndarray, np.ndarray, int]] = []

    while True:
        terminal, _ = game.is_terminal()
        if terminal:
            break

        temp = 1.0 if game.move_count < config.temperature_threshold else 0.01
        action_probs, _ = search(
            game, model,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            add_noise=True,
            device=device,
        )

        history.append((game.encode(), action_probs, game.current_player))
        action = select_action(action_probs, temperature=temp)
        game = game.make_move(action)

    # Determine game outcome
    _, result = game.is_terminal()
    # result is 1.0 if last mover won, 0.0 for draw
    # last mover is -game.current_player (since current_player flipped after last move)
    last_mover = -game.current_player

    examples = []
    for state, policy, player in history:
        # Value from this position's current player perspective
        if result == 0.0:
            value = 0.0
        elif player == last_mover:
            value = result   # last mover won with result=1.0
        else:
            value = -result  # opponent of last mover lost
        examples.append(TrainingExample(state, policy, value))

    # Data augmentation: horizontal flip
    augmented = []
    for ex in examples:
        augmented.append(ex)
        augmented.append(TrainingExample(
            state=np.flip(ex.state, axis=2).copy(),
            policy=np.flip(ex.policy).copy(),
            value=ex.value,
        ))

    return augmented


def generate_self_play_data(model: AlphaZeroNet, config: Config,
                            device: str = "cpu") -> list[TrainingExample]:
    """Generate self-play data for one iteration.

    Uses sequential play (multiprocessing complicates CUDA).
    """
    all_examples = []
    model.eval()

    for _ in range(config.games_per_iteration):
        examples = play_game(model, config, device=device)
        all_examples.extend(examples)

    return all_examples
