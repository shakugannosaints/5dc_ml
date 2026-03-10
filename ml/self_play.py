# ml/self_play.py
"""
Self-Play data generation for 5D Chess ML training.

Plays games using the current agent (or random policy for bootstrapping),
collecting (state, action, outcome) trajectories for training.
"""

import sys
import os
import time
import random
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import torch

# Engine import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

from .config import TrainingConfig


@dataclass
class StepRecord:
    """A single step in a game trajectory."""
    state: object                     # engine.state copy (actual game state at this step)
    per_tl_moves: list                # live per_timeline_moves objects from engine
    chosen_indices: list[int]         # which move was chosen per timeline
    log_prob: float                   # log probability of the action
    value_est: float                  # value estimate at this state
    player: int                       # 0=white, 1=black


@dataclass
class GameRecord:
    """A complete game trajectory."""
    steps: list[StepRecord] = field(default_factory=list)
    outcome: float = 0.0             # +1 = white wins, -1 = black wins, 0 = draw
    num_moves: int = 0
    termination: str = "unknown"     # "checkmate", "softmate", "stalemate", "draw", "max_length"


class SelfPlayWorker:
    """
    Generates training data via self-play.
    
    Uses the Agent to play both sides, collecting trajectories.
    Supports mixed mode: agent moves + random exploration.
    """

    def __init__(self, agent, cfg: TrainingConfig, use_agent: bool = True,
                 epsilon: float = 0.0):
        """
        Args:
            agent:     Agent model (or None for pure random play)
            cfg:       training config
            use_agent: if False, play random moves (for bootstrapping)
            epsilon:   probability of taking a random move (exploration)
        """
        self.agent = agent
        self.cfg = cfg
        self.use_agent = use_agent and agent is not None
        self.epsilon = epsilon

    def play_game(self, temperature: float = 1.0) -> GameRecord:
        """
        Play a single self-play game.
        
        Returns:
            GameRecord with trajectory and outcome
        """
        record = GameRecord()

        # Create initial state from variant PGN
        state = engine.create_state_from_pgn(self.cfg.variant_pgn)
        if state is None:
            print(f"[SelfPlay] ERROR: Failed to create state from PGN: {self.cfg.variant_pgn}")
            record.termination = "error"
            return record

        for move_num in range(self.cfg.max_game_length):
            # Check game status
            match_status = state.get_match_status()
            # match_status: PLAYING=0, WHITE_WINS, BLACK_WINS, STALEMATE
            if match_status != engine.match_status_t.PLAYING and match_status != engine.match_status_t.STALEMATE:
                if match_status == engine.match_status_t.WHITE_WINS:
                    record.outcome = 1.0
                    record.termination = "white_wins"
                elif match_status == engine.match_status_t.BLACK_WINS:
                    record.outcome = -1.0
                    record.termination = "black_wins"
                else:
                    record.outcome = 0.0
                    record.termination = str(match_status)
                break

            # Get per-timeline moves
            per_tl_moves = engine.get_per_timeline_moves(state)
            if len(per_tl_moves) == 0:
                # No moves available
                record.outcome = 0.0
                record.termination = "no_moves"
                break

            # Select action (with epsilon-greedy exploration)
            use_random = (not self.use_agent) or (random.random() < self.epsilon)

            if not use_random and self.use_agent:
                action, log_prob, entropy, value_est = self.agent.select_action(
                    state, engine, temperature
                )
                if action is None:
                    record.outcome = 0.0
                    record.termination = "no_legal_action"
                    break

                chosen_indices = self._action_to_indices(action, per_tl_moves)
                lp = log_prob.item()
                ve = value_est.item()
            else:
                # Random play (for bootstrapping or epsilon-greedy)
                action = engine.random_action(state)
                if action is None:
                    record.outcome = 0.0
                    record.termination = "no_legal_action"
                    break
                chosen_indices = self._action_to_indices(action, per_tl_moves)
                lp = 0.0
                ve = 0.0

            # Record step — store a COPY of the actual state and live per_tl_moves
            present = state.get_present()  # (turn, is_black)
            current_player = 1 if present[1] else 0  # 0=white, 1=black
            step = StepRecord(
                state=engine.state(state),  # copy constructor
                per_tl_moves=per_tl_moves,  # live engine objects
                chosen_indices=chosen_indices,
                log_prob=lp,
                value_est=ve,
                player=current_player,
            )
            record.steps.append(step)

            # Apply action
            new_state = engine.apply_action(state, action)
            if new_state is None:
                record.outcome = 0.0
                record.termination = "invalid_action"
                break
            state = new_state

        else:
            # Max length reached
            record.outcome = 0.0
            record.termination = "max_length"

        record.num_moves = len(record.steps)
        return record

    def _action_to_indices(self, action, per_tl_moves) -> list[int]:
        """
        Map an action (list of ext_moves) back to per-timeline move indices.
        """
        indices = []
        action_moves = action.get_moves()  # list of ext_move

        # Build lookup: line_idx -> (from_x, from_y, to_x, to_y)
        action_by_line = {}
        for mv in action_moves:
            line = mv.get_from().l()
            action_by_line[line] = (mv.get_from().x(), mv.get_from().y(), mv.get_to().x(), mv.get_to().y())

        for tl in per_tl_moves:
            line_idx = tl.line_idx
            if line_idx in action_by_line:
                ax, ay, bx, by = action_by_line[line_idx]
                found = -1
                for i, (from_v4, to_v4) in enumerate(tl.moves):
                    if (from_v4.x() == ax and from_v4.y() == ay and
                        to_v4.x() == bx and to_v4.y() == by):
                        found = i
                        break
                indices.append(found if found >= 0 else 0)
            else:
                indices.append(0)  # default to first move

        return indices

    def _serialize_tl_moves(self, per_tl_moves) -> list:
        """Serialize per_timeline_moves for storage."""
        result = []
        for tl in per_tl_moves:
            moves_data = []
            for from_v4, to_v4 in tl.moves:
                moves_data.append({
                    'from': (from_v4.x(), from_v4.y(), from_v4.t(), from_v4.l()),
                    'to': (to_v4.x(), to_v4.y(), to_v4.t(), to_v4.l()),
                })
            result.append({
                'line_idx': tl.line_idx,
                'is_mandatory': tl.is_mandatory,
                'moves': moves_data,
            })
        return result

    def generate_training_data(self, num_games: int, temperature: float = 1.0,
                                discount: float = 1.0) -> list[tuple]:
        """
        Play multiple games and produce training samples.
        
        Returns:
            list of (engine_state, (per_tl_moves, chosen_indices), return_value)
            where return_value is the discounted outcome from this player's perspective.
        """
        all_samples = []

        for game_idx in range(num_games):
            t0 = time.time()
            record = self.play_game(temperature)
            elapsed = time.time() - t0
            
            print(f"  Game {game_idx+1}/{num_games}: "
                  f"{record.num_moves} moves, "
                  f"outcome={record.outcome:+.0f}, "
                  f"term={record.termination}, "
                  f"time={elapsed:.1f}s")

            # Convert game record to training samples
            # For each step, the return is the game outcome from that player's perspective
            for step in record.steps:
                # From white's perspective: outcome is as-is
                # From black's perspective: negate
                if step.player == 0:
                    ret = record.outcome
                else:
                    ret = -record.outcome

                # We store the actual state copy + live per_tl_moves for training
                all_samples.append({
                    'state': step.state,           # actual engine state copy
                    'per_tl_moves': step.per_tl_moves,  # live per_tl_moves
                    'chosen_indices': step.chosen_indices,
                    'return': ret,
                    'player': step.player,
                })

        return all_samples
