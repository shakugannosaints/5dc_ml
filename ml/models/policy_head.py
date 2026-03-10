# ml/models/policy_head.py
"""
Factored Policy Head: Autoregressive per-timeline move selection.

In 5D chess, an action consists of one move per "movable" timeline.
This policy factorizes the action probability as:
  P(action) = Π_i P(move_i | timeline_i, global_state, previous_moves)

For each timeline with legal moves, we:
  1. Embed each candidate move (from_sq, to_sq) using the source/target board embeddings
  2. Score candidates using cross-attention with the global state
  3. Select a move via softmax

This avoids the combinatorial explosion of enumerating all action combinations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ..config import PolicyConfig


class MoveEmbedder(nn.Module):
    """
    Embeds a single move (from_square, to_square) given board context.
    
    Each square is identified by (x, y) on the board.
    We learn embeddings for positions and combine with piece-type info.
    """

    def __init__(self, board_size: int, embed_dim: int):
        super().__init__()
        self.board_size = board_size
        # Position embedding for from/to squares (flatten 2D -> 1D index)
        self.pos_embed = nn.Embedding(board_size * board_size, embed_dim)
        # Combine from + to
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
        )

    def forward(self, from_xy: torch.Tensor, to_xy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            from_xy: [M, 2] int tensor of (x, y) source squares
            to_xy:   [M, 2] int tensor of (x, y) destination squares
        Returns:
            [M, embed_dim] move embeddings
        """
        from_idx = from_xy[:, 0] * self.board_size + from_xy[:, 1]  # [M]
        to_idx = to_xy[:, 0] * self.board_size + to_xy[:, 1]        # [M]
        from_emb = self.pos_embed(from_idx.clamp(0, self.board_size * self.board_size - 1))  # [M, embed_dim]
        to_emb = self.pos_embed(to_idx.clamp(0, self.board_size * self.board_size - 1))      # [M, embed_dim]
        return self.combine(torch.cat([from_emb, to_emb], dim=-1))  # [M, embed_dim]


class TimelinePolicyHead(nn.Module):
    """
    Scores candidate moves for a single timeline, conditioned on:
      - The global state embedding
      - The board embedding for this timeline's current board
      - Context from other timelines (via a context vector)
    """

    def __init__(self, state_dim: int, move_embed_dim: int, hidden_dim: int):
        super().__init__()
        # Query: state context → move scoring
        self.context_proj = nn.Linear(state_dim, hidden_dim)
        self.move_proj = nn.Linear(move_embed_dim, hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: torch.Tensor, move_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context:     [hidden_dim] single context vector for this timeline
            move_embeds: [M, move_embed_dim] embeddings of M candidate moves
        Returns:
            logits: [M] scores for each candidate move
        """
        ctx = self.context_proj(context).unsqueeze(0).expand(move_embeds.size(0), -1)  # [M, hidden]
        mv = self.move_proj(move_embeds)  # [M, hidden]
        combined = torch.cat([ctx, mv], dim=-1)  # [M, hidden*2]
        return self.score(combined).squeeze(-1)  # [M]


class FactoredPolicyHead(nn.Module):
    """
    Full factored policy: scores moves independently per timeline.
    
    For self-play / training, we:
      1. Get per-timeline move lists from the engine
      2. Score each timeline's moves independently
      3. Sample/argmax one move per timeline
      4. Combine into a full action
      
    The log-probability of the full action is the sum of per-timeline log-probs.
    """

    def __init__(self, cfg: PolicyConfig | None = None, board_size: int = 8):
        super().__init__()
        if cfg is None:
            cfg = PolicyConfig()
        self.cfg = cfg

        self.move_embedder = MoveEmbedder(board_size, cfg.move_embed_dim)
        self.timeline_head = TimelinePolicyHead(cfg.state_dim, cfg.move_embed_dim, cfg.hidden_dim)

        # Cross-timeline context: combine global embed + per-node embed for the active board
        self.context_mlp = nn.Sequential(
            nn.Linear(cfg.state_dim + cfg.board_embed_dim, cfg.state_dim),
            nn.ReLU(),
        )

    def score_timeline_moves(self, global_embed: torch.Tensor, board_embed: torch.Tensor,
                             from_squares: torch.Tensor, to_squares: torch.Tensor
                             ) -> torch.Tensor:
        """
        Score candidate moves for one timeline.
        
        Args:
            global_embed: [global_dim] global state embedding
            board_embed:  [board_embed_dim] embedding of the active board on this timeline
            from_squares: [M, 2] (x, y) of source squares
            to_squares:   [M, 2] (x, y) of destination squares
        Returns:
            logits: [M] unnormalized scores
        """
        # Build context from global + board
        context = self.context_mlp(torch.cat([global_embed, board_embed], dim=-1))  # [state_dim]
        move_embs = self.move_embedder(from_squares, to_squares)  # [M, move_embed_dim]
        return self.timeline_head(context, move_embs)  # [M]

    def compute_action_logprob(self, global_embed: torch.Tensor,
                                node_embeds_dict: dict[tuple, torch.Tensor],
                                per_timeline_moves: list,
                                chosen_move_indices: list[int],
                                device: torch.device) -> torch.Tensor:
        """
        Compute log-probability of a chosen action (one move per timeline).
        
        Args:
            global_embed:       [global_dim]
            node_embeds_dict:   {(l,t,c): [embed_dim]} per-board embeddings from GNN
            per_timeline_moves: list of timeline_moves from engine (line_idx, is_mandatory, moves)
            chosen_move_indices: list of ints, one per timeline (index into that timeline's move list)
            device: torch device
        Returns:
            total_log_prob: scalar tensor (sum of per-timeline log probs)
        """
        total_log_prob = torch.tensor(0.0, device=device)

        for tl_idx, tl_moves in enumerate(per_timeline_moves):
            moves = tl_moves.moves  # list of (from_vec4, to_vec4)
            if len(moves) == 0:
                continue

            # Extract (x, y) from vec4 for from/to
            from_xy = torch.tensor([[m[0].x(), m[0].y()] for m in moves], dtype=torch.long, device=device)
            to_xy = torch.tensor([[m[1].x(), m[1].y()] for m in moves], dtype=torch.long, device=device)

            # Find the board embedding for this timeline's active board
            # The active board is at (l, t, color) where l = tl_moves.line_idx
            # We use the first move's from-square to identify (t, l) via vec4
            first_from = moves[0][0]
            board_key = (first_from.l(), first_from.t(), 0)  # approximate; we'll search node_embeds_dict
            board_embed = None
            for k, v in node_embeds_dict.items():
                if k[0] == first_from.l():
                    board_embed = v
                    break
            if board_embed is None:
                # Fallback: zero embedding
                board_embed = torch.zeros(self.cfg.board_embed_dim, device=device)

            logits = self.score_timeline_moves(global_embed, board_embed, from_xy, to_xy)
            # Clamp logits for numerical stability
            logits = logits.clamp(-10.0, 10.0)
            log_probs = F.log_softmax(logits, dim=0)

            chosen_idx = chosen_move_indices[tl_idx]
            if 0 <= chosen_idx < len(log_probs):
                total_log_prob = total_log_prob + log_probs[chosen_idx]

        return total_log_prob

    def sample_action(self, global_embed: torch.Tensor,
                      node_embeds_dict: dict[tuple, torch.Tensor],
                      per_timeline_moves: list,
                      temperature: float,
                      device: torch.device) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """
        Sample one move per timeline, return indices + log_prob + entropy.
        
        Returns:
            chosen_indices: list of int (one per timeline)
            total_log_prob: scalar
            total_entropy:  scalar
        """
        chosen_indices = []
        total_log_prob = torch.tensor(0.0, device=device)
        total_entropy = torch.tensor(0.0, device=device)

        for tl_moves in per_timeline_moves:
            moves = tl_moves.moves
            if len(moves) == 0:
                chosen_indices.append(-1)
                continue

            from_xy = torch.tensor([[m[0].x(), m[0].y()] for m in moves], dtype=torch.long, device=device)
            to_xy = torch.tensor([[m[1].x(), m[1].y()] for m in moves], dtype=torch.long, device=device)

            # Find board embedding
            first_from = moves[0][0]
            board_embed = torch.zeros(self.cfg.board_embed_dim, device=device)
            for k, v in node_embeds_dict.items():
                if k[0] == first_from.l():
                    board_embed = v
                    break

            logits = self.score_timeline_moves(global_embed, board_embed, from_xy, to_xy)
            # Clamp logits for numerical stability
            logits = logits.clamp(-10.0, 10.0)
            logits = logits / max(temperature, 1e-8)

            # Add Dirichlet noise for exploration (like AlphaZero)
            if temperature > 0.1 and len(moves) > 1:
                alpha = 0.3  # Dirichlet concentration parameter
                noise = torch.tensor(
                    np.random.dirichlet([alpha] * len(moves)),
                    dtype=torch.float32, device=device
                )
                noise_weight = 0.25  # mix ratio
                probs_raw = F.softmax(logits, dim=0)
                probs = (1.0 - noise_weight) * probs_raw + noise_weight * noise
                # Ensure minimum probability to prevent collapse
                probs = probs.clamp(min=0.01)
                probs = probs / probs.sum()
                log_probs = probs.log()
            else:
                probs = F.softmax(logits, dim=0)
                log_probs = F.log_softmax(logits, dim=0)

            # Sample
            idx = torch.multinomial(probs, 1).item()
            chosen_indices.append(idx)
            total_log_prob = total_log_prob + log_probs[idx]

            # Entropy: -Σ p log p
            entropy = -(probs * log_probs).sum()
            total_entropy = total_entropy + entropy

        return chosen_indices, total_log_prob, total_entropy
