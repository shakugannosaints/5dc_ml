# ml/models/agent.py
"""
Agent: Combines all components into a unified 5D Chess agent.

Architecture:
  BoardEncoder  → per-board embeddings
  MultiverseEncoder (GNN) → node embeddings + global embedding
  FactoredPolicyHead → per-timeline move scoring
  ValueHead → state value estimation

The Agent provides high-level methods for:
  - evaluate(state): get value estimate
  - select_action(state, temperature): sample an action
  - compute_loss(batch): training loss (policy + value)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from ..config import TrainingConfig
from .board_encoder import BoardEncoder
from .multiverse_encoder import MultiverseEncoder
from .policy_head import FactoredPolicyHead
from .value_head import ValueHead


class Agent(nn.Module):
    """Full 5D Chess agent combining all network components."""

    def __init__(self, cfg: TrainingConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = TrainingConfig()
        self.cfg = cfg

        self.board_encoder = BoardEncoder(cfg.board_encoder)
        self.multiverse_encoder = MultiverseEncoder(cfg.multiverse_encoder)
        
        # Policy head's board_embed_dim must match GNN's output (gnn_hidden_dim),
        # since we feed GNN node embeddings into the policy.
        policy_cfg = cfg.policy
        if policy_cfg.board_embed_dim != cfg.multiverse_encoder.gnn_hidden_dim:
            # Auto-correct
            from dataclasses import replace
            policy_cfg = replace(policy_cfg, board_embed_dim=cfg.multiverse_encoder.gnn_hidden_dim)
        if policy_cfg.state_dim != cfg.multiverse_encoder.global_dim:
            from dataclasses import replace
            policy_cfg = replace(policy_cfg, state_dim=cfg.multiverse_encoder.global_dim)
        
        self.policy_head = FactoredPolicyHead(policy_cfg, board_size=max(cfg.board_size_x, cfg.board_size_y))
        
        # Value head's global_dim must match MultiverseEncoder's global_dim
        value_cfg = cfg.value
        if value_cfg.global_dim != cfg.multiverse_encoder.global_dim:
            from dataclasses import replace
            value_cfg = replace(value_cfg, global_dim=cfg.multiverse_encoder.global_dim)
        
        self.value_head = ValueHead(value_cfg)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def load_state_dict_transfer(self, source_state_dict: dict, strict: bool = False):
        """
        Load weights from a checkpoint trained with a different board_size.
        
        Handles the MoveEmbedder.pos_embed size mismatch:
          - If source has smaller pos_embed (e.g. 16 from 4x4), copy those rows
            into the corresponding positions of the larger embedding (e.g. 64 from 8x8).
          - All other weights are loaded normally.
          - Unmatched keys get random initialization (default).
        """
        target_state_dict = self.state_dict()
        adapted = {}
        skipped = []

        for key, src_tensor in source_state_dict.items():
            if key not in target_state_dict:
                skipped.append((key, 'not in target model'))
                continue

            tgt_tensor = target_state_dict[key]

            if src_tensor.shape == tgt_tensor.shape:
                # Direct copy
                adapted[key] = src_tensor
            elif 'pos_embed' in key and src_tensor.dim() == 2 and tgt_tensor.dim() == 2:
                # Position embedding expansion: copy learned embeddings for shared positions
                src_num, embed_dim = src_tensor.shape
                tgt_num, tgt_embed_dim = tgt_tensor.shape
                if embed_dim != tgt_embed_dim:
                    skipped.append((key, f'embed_dim mismatch {embed_dim} vs {tgt_embed_dim}'))
                    continue

                src_bs = int(src_num ** 0.5)  # source board_size
                tgt_bs = int(tgt_num ** 0.5)  # target board_size

                # Start from random (already initialized in target)
                new_embed = tgt_tensor.clone()
                # Copy learned rows for positions that exist in both
                for sy in range(src_bs):
                    for sx in range(src_bs):
                        src_idx = sx * src_bs + sy
                        tgt_idx = sx * tgt_bs + sy  # same (x,y) position in larger board
                        new_embed[tgt_idx] = src_tensor[src_idx]

                adapted[key] = new_embed
                print(f"  [Transfer] Expanded {key}: ({src_num},{embed_dim}) → ({tgt_num},{embed_dim}) "
                      f"  ({src_bs}x{src_bs} → {tgt_bs}x{tgt_bs})")
            else:
                skipped.append((key, f'shape mismatch {src_tensor.shape} vs {tgt_tensor.shape}'))

        # Load adapted weights
        missing = set(target_state_dict.keys()) - set(adapted.keys())
        self.load_state_dict(adapted, strict=False)

        if skipped:
            print(f"  [Transfer] Skipped {len(skipped)} keys:")
            for k, reason in skipped:
                print(f"    {k}: {reason}")
        if missing:
            print(f"  [Transfer] Random-initialized {len(missing)} keys (new/resized layers)")

        return adapted, skipped, missing

    def encode_state(self, engine_state) -> tuple[dict, torch.Tensor, dict, torch.Tensor]:
        """
        Encode an engine state into embeddings.
        
        Args:
            engine_state: engine.state object from C++ bindings
        
        Returns:
            board_embeds_dict: {(l,t,c): [embed_dim] tensor}
            global_embed:      [global_dim] tensor
            node_embeds_dict:  {(l,t,c): [gnn_hidden_dim] tensor}  
            node_embeds_flat:  [N, gnn_hidden_dim] tensor
        """
        device = self.device

        # 1. Get all board tensors from engine  → {(l,t,c): numpy[C,H,W]}
        board_tensors = engine_state.get_all_board_tensors()
        
        # 2. Encode each board with the BoardEncoder
        board_keys, board_embeds_tensor = self.board_encoder.encode_boards(board_tensors, device)
        
        # Build dict: key → embed
        board_embeds_dict = {}
        for i, key in enumerate(board_keys):
            board_embeds_dict[tuple(key)] = board_embeds_tensor[i]

        # 3. Get graph structure from engine
        graph_struct = engine_state.get_graph_structure()

        # 4. Build GNN inputs
        (gnn_board_embeds, node_scalars, edge_index, edge_type, node_keys
         ) = self.multiverse_encoder.build_graph_from_engine(graph_struct, board_embeds_dict, device)

        # 5. Run GNN
        if len(node_keys) == 0:
            # Edge case: empty graph
            global_embed = torch.zeros(1, self.cfg.multiverse_encoder.global_dim, device=device)
            return board_embeds_dict, global_embed.squeeze(0), {}, torch.zeros(0, self.cfg.multiverse_encoder.gnn_hidden_dim, device=device)

        node_embeds_flat, global_embed = self.multiverse_encoder(
            gnn_board_embeds, node_scalars, edge_index, edge_type
        )
        global_embed = global_embed.squeeze(0)  # [global_dim]

        # Build node embeds dict
        node_embeds_dict = {}
        for i, key in enumerate(node_keys):
            k = tuple(key) if not isinstance(key, tuple) else key
            node_embeds_dict[k] = node_embeds_flat[i]

        return board_embeds_dict, global_embed, node_embeds_dict, node_embeds_flat

    def evaluate(self, engine_state) -> float:
        """
        Get value estimate for a state.
        
        Returns:
            float in [-1, 1]
        """
        with torch.no_grad():
            _, global_embed, _, _ = self.encode_state(engine_state)
            value = self.value_head(global_embed.unsqueeze(0))  # [1, 1]
            return value.item()

    def select_action(self, engine_state, engine_module, temperature: float = 1.0):
        """
        Select an action by:
          1. Encoding the state
          2. Getting per-timeline moves from engine
          3. Scoring with policy head
          4. Sampling per-timeline moves
          5. Reconstructing a full action
        
        Args:
            engine_state:  engine.state object
            engine_module: engine Python module (with get_per_timeline_moves etc.)
            temperature:   exploration temperature
        
        Returns:
            action:       engine action object (or None if no legal moves)
            log_prob:     scalar tensor
            entropy:      scalar tensor
            value:        scalar tensor
        """
        device = self.device

        # Encode
        board_embeds_dict, global_embed, node_embeds_dict, _ = self.encode_state(engine_state)

        # Value
        value = self.value_head(global_embed.unsqueeze(0)).squeeze()  # scalar

        # Get per-timeline moves from engine
        per_tl_moves = engine_module.get_per_timeline_moves(engine_state)

        if len(per_tl_moves) == 0:
            return None, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), value

        # Sample with policy head
        chosen_indices, log_prob, entropy = self.policy_head.sample_action(
            global_embed, node_embeds_dict, per_tl_moves, temperature, device
        )

        # Reconstruct action: find the matching legal action from the engine
        action = self._reconstruct_action(engine_state, engine_module, per_tl_moves, chosen_indices)

        return action, log_prob, entropy, value

    def _reconstruct_action(self, engine_state, engine_module, per_tl_moves, chosen_indices):
        """
        Given chosen move indices per timeline, find a matching legal action.
        
        Strategy: enumerate legal actions and find the one that matches our per-timeline choices.
        If no exact match (due to interdependencies), pick the closest.
        """
        # Build the target move set from our choices
        target_moves = {}  # line_idx -> (from_vec4, to_vec4)
        for tl_idx, tl_moves in enumerate(per_tl_moves):
            idx = chosen_indices[tl_idx]
            if idx >= 0 and idx < len(tl_moves.moves):
                from_v4, to_v4 = tl_moves.moves[idx]
                target_moves[tl_moves.line_idx] = (from_v4, to_v4)

        # Enumerate legal actions (limit to avoid explosion)
        legal_actions = engine_module.enumerate_legal_actions(engine_state, 200)

        if len(legal_actions) == 0:
            return None

        # Score each legal action by how many of our target moves it contains
        best_action = None
        best_score = -1

        for action in legal_actions:
            score = 0
            moves = action.get_moves()  # list of ext_move
            for mv in moves:
                from_v4 = mv.get_from()
                to_v4 = mv.get_to()
                line = from_v4.l()
                if line in target_moves:
                    tf, tt = target_moves[line]
                    if (from_v4.x() == tf.x() and from_v4.y() == tf.y() and
                        to_v4.x() == tt.x() and to_v4.y() == tt.y()):
                        score += 1
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def compute_loss(self, states_data: list, actions_data: list,
                     returns: torch.Tensor, engine_module) -> dict[str, torch.Tensor]:
        """
        Compute combined policy + value loss for a batch of (state, action, return) tuples.
        
        Args:
            states_data:  list of engine.state objects
            actions_data: list of (per_tl_moves, chosen_indices) tuples
            returns:      [B] tensor of discounted returns
            engine_module: engine module
        
        Returns:
            dict with 'total_loss', 'policy_loss', 'value_loss', 'entropy'
        """
        device = self.device
        cfg = self.cfg

        total_policy_loss = torch.tensor(0.0, device=device)
        total_value_loss = torch.tensor(0.0, device=device)
        total_entropy = torch.tensor(0.0, device=device)
        count = 0

        for i, (engine_state, (per_tl_moves, chosen_indices)) in enumerate(zip(states_data, actions_data)):
            # Encode state
            board_embeds_dict, global_embed, node_embeds_dict, _ = self.encode_state(engine_state)

            # Value prediction
            value_pred = self.value_head(global_embed.unsqueeze(0)).squeeze()  # scalar

            # Policy log-prob of chosen action
            log_prob = self.policy_head.compute_action_logprob(
                global_embed, node_embeds_dict, per_tl_moves, chosen_indices, device
            )

            # Advantage = return - value (detached for policy gradient)
            advantage = returns[i] - value_pred.detach()

            # Policy loss (REINFORCE with baseline)
            policy_loss = -(log_prob * advantage)
            total_policy_loss = total_policy_loss + policy_loss

            # Value loss (MSE)
            value_loss = (value_pred - returns[i]) ** 2
            total_value_loss = total_value_loss + value_loss

            # Entropy (for regularization, recompute)
            # We approximate: just use the timeline move structure
            _, _, entropy = self.policy_head.sample_action(
                global_embed.detach(), node_embeds_dict, per_tl_moves, 1.0, device
            )
            total_entropy = total_entropy + entropy

            count += 1

        if count == 0:
            count = 1

        avg_policy_loss = total_policy_loss / count
        avg_value_loss = total_value_loss / count
        avg_entropy = total_entropy / count

        total_loss = (cfg.policy_loss_weight * avg_policy_loss +
                      cfg.value_loss_weight * avg_value_loss -
                      cfg.entropy_bonus * avg_entropy)

        return {
            'total_loss': total_loss,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
        }
