# alphazero/network.py
"""
Transformer network for semimove-level AlphaZero.

State encoder outputs:
  - value: scalar in [-1, 1] from current player's perspective
  - submit_logit: scalar prior logit for submit action
  - raw_logits: per-board source-square logits (auxiliary)

Policy training/inference uses score_legal_actions(), which produces logits
exactly aligned to the legal action list used by MCTS/self-play.
"""

import math
import torch
import torch.nn as nn

from .config import NetworkConfig


class SinusoidalPositionEncoding(nn.Module):
    """
    Encode (L, T) coordinates of each small board into d_model dims.
    Uses standard sinusoidal encoding on two separate axes (L and T),
    concatenated and projected to d_model.
    """

    def __init__(self, d_model: int, max_l: int = 10, max_t: int = 50):
        super().__init__()
        self.d_model = d_model
        half = d_model // 2
        self.proj = nn.Linear(half * 2, d_model)
        div_term = torch.exp(
            torch.arange(0, half, 2, dtype=torch.float32) * (-math.log(10000.0) / half)
        )
        self.register_buffer("div_term", div_term)
        self.half = half

    def _encode_scalar(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().unsqueeze(-1)  # [N, 1]
        pe = torch.zeros(x.size(0), self.half, device=x.device)
        pe[:, 0::2] = torch.sin(x * self.div_term)
        pe[:, 1::2] = torch.cos(x * self.div_term)
        return pe

    def forward(self, l_coords: torch.Tensor, t_coords: torch.Tensor) -> torch.Tensor:
        pe_l = self._encode_scalar(l_coords)
        pe_t = self._encode_scalar(t_coords)
        return self.proj(torch.cat([pe_l, pe_t], dim=-1))


class BoardTokenizer(nn.Module):
    """Convert one board's tensor representation into a d_model embedding."""

    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        input_dim = cfg.piece_channels * cfg.board_squares + cfg.board_squares
        self.proj = nn.Sequential(
            nn.Linear(input_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )

    def forward(self, board_planes: torch.Tensor, last_move_marker: torch.Tensor) -> torch.Tensor:
        flat = board_planes.reshape(board_planes.size(0), -1)
        combined = torch.cat([flat, last_move_marker], dim=-1)
        return self.proj(combined)


class AlphaZeroNetwork(nn.Module):
    """Transformer-based network for semimove-level 5D chess."""

    def __init__(self, cfg: NetworkConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = NetworkConfig()
        self.cfg = cfg

        self.board_tokenizer = BoardTokenizer(cfg)
        self.pos_encoder = SinusoidalPositionEncoding(cfg.d_model, cfg.max_timelines, cfg.max_turns)

        self.urgency_embed = nn.Sequential(
            nn.Linear(1, cfg.d_model),
            nn.GELU(),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )

        # Value and submit heads from [CLS]
        self.value_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
            nn.Tanh(),
        )
        self.submit_head = nn.Linear(cfg.d_model, 1)

        # Auxiliary per-board source-square logits (kept for diagnostics)
        self.policy_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.board_squares),
        )

        # Destination-aware legal-action scorer
        move_dim = max(32, cfg.d_model // 4)
        self.from_square_embed = nn.Embedding(cfg.board_squares, move_dim)
        self.to_square_embed = nn.Embedding(cfg.board_squares, move_dim)
        self.move_feat_proj = nn.Sequential(
            nn.Linear(4, move_dim),
            nn.GELU(),
            nn.Linear(move_dim, move_dim),
            nn.GELU(),
        )
        self.move_scorer = nn.Sequential(
            nn.Linear(cfg.d_model + move_dim * 3, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(
        self,
        board_planes: torch.Tensor,
        last_move_markers: torch.Tensor,
        l_coords: torch.Tensor,
        t_coords: torch.Tensor,
        urgency: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_latent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """
        Args:
            board_planes: [B, N, 14, 16]
            last_move_markers: [B, N, 16]
            l_coords: [B, N] int
            t_coords: [B, N] int
            urgency: [B, 1] float
            padding_mask: [B, N] bool (True = padding)
        Returns:
            value: [B, 1]
            submit_logit: [B, 1]
            raw_logits: [B, N, 16]
            if return_latent: also (board_out [B, N, d_model], cls_out [B, d_model])
        """
        B, N = board_planes.shape[:2]
        device = board_planes.device

        bp_flat = board_planes.reshape(B * N, self.cfg.piece_channels, self.cfg.board_squares)
        lm_flat = last_move_markers.reshape(B * N, self.cfg.board_squares)
        tokens = self.board_tokenizer(bp_flat, lm_flat).reshape(B, N, self.cfg.d_model)

        l_flat = l_coords.reshape(B * N)
        t_flat = t_coords.reshape(B * N)
        pos_enc = self.pos_encoder(l_flat, t_flat).reshape(B, N, self.cfg.d_model)
        tokens = tokens + pos_enc

        urg_emb = self.urgency_embed(urgency)
        tokens = tokens + urg_emb.unsqueeze(1)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        out = self.transformer(tokens, src_key_padding_mask=padding_mask)

        cls_out = out[:, 0, :]
        board_out = out[:, 1:, :]

        value = self.value_head(cls_out)
        submit_logit = self.submit_head(cls_out)
        raw_logits = self.policy_head(board_out)

        if return_latent:
            return value, submit_logit, raw_logits, board_out, cls_out
        return value, submit_logit, raw_logits

    def score_legal_actions(
        self,
        board_out: torch.Tensor,
        submit_logit: torch.Tensor,
        action_board_indices: torch.Tensor,
        action_from_squares: torch.Tensor,
        action_to_squares: torch.Tensor,
        action_delta_t: torch.Tensor,
        action_delta_l: torch.Tensor,
        action_is_submit: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score explicit legal actions for one state.

        Args:
            board_out: [N, d_model] contextual board embeddings
            submit_logit: scalar or [1]
            action_*: [A] aligned tensors (A = number of legal actions)
        Returns:
            logits: [A] aligned legal-action logits
        """
        device = board_out.device
        A = action_board_indices.shape[0]
        logits = torch.full((A,), -20.0, dtype=board_out.dtype, device=device)
        submit_mask = action_is_submit.bool()

        submit_values = submit_logit.reshape(()).expand_as(logits)
        logits = torch.where(submit_mask, submit_values, logits)

        idx = action_board_indices.long()
        valid = (~submit_mask) & (idx >= 0) & (idx < board_out.shape[0])
        zeros = torch.zeros_like(idx)
        upper = torch.full_like(idx, board_out.shape[0] - 1)
        safe_idx = torch.minimum(torch.maximum(idx, zeros), upper)
        board_ctx = board_out[safe_idx]

        from_sq = action_from_squares.long().clamp(0, self.cfg.board_squares - 1)
        to_sq = action_to_squares.long().clamp(0, self.cfg.board_squares - 1)
        from_emb = self.from_square_embed(from_sq)
        to_emb = self.to_square_embed(to_sq)

        dt = action_delta_t.float() / 8.0
        dl = action_delta_l.float() / 8.0
        is_jump = ((action_delta_t != 0) | (action_delta_l != 0)).float()
        is_tl_change = (action_delta_l != 0).float()
        move_feats = torch.stack([dt, dl, is_jump, is_tl_change], dim=-1)
        feat_emb = self.move_feat_proj(move_feats)

        move_repr = torch.cat([board_ctx, from_emb, to_emb, feat_emb], dim=-1)
        move_logits = self.move_scorer(move_repr).squeeze(-1)
        logits = torch.where(valid, move_logits, logits)

        return logits

    def score_legal_actions_batched_flat(
        self,
        board_out: torch.Tensor,
        submit_logit: torch.Tensor,
        action_state_indices: torch.Tensor,
        action_board_indices: torch.Tensor,
        action_from_squares: torch.Tensor,
        action_to_squares: torch.Tensor,
        action_delta_t: torch.Tensor,
        action_delta_l: torch.Tensor,
        action_is_submit: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score explicit legal actions for a batch of states with flattened actions.

        Args:
            board_out: [B, N, d_model]
            submit_logit: [B, 1] or [B]
            action_state_indices: [A] which state each action belongs to
            action_*: [A] flattened aligned action tensors
        Returns:
            logits: [A] flattened legal-action logits
        """
        device = board_out.device
        A = action_board_indices.shape[0]
        logits = torch.full((A,), -20.0, dtype=board_out.dtype, device=device)
        if A == 0:
            return logits

        B = board_out.shape[0]
        submit_mask = action_is_submit.bool()

        state_idx = action_state_indices.long()
        zeros = torch.zeros_like(state_idx)
        state_upper = torch.full_like(state_idx, B - 1)
        safe_state_idx = torch.minimum(torch.maximum(state_idx, zeros), state_upper)

        submit_values = submit_logit.reshape(-1)[safe_state_idx]
        logits = torch.where(submit_mask, submit_values, logits)

        idx = action_board_indices.long()
        valid = (
            (~submit_mask)
            & (state_idx >= 0)
            & (state_idx < B)
            & (idx >= 0)
            & (idx < board_out.shape[1])
        )
        board_zeros = torch.zeros_like(idx)
        board_upper = torch.full_like(idx, board_out.shape[1] - 1)
        safe_idx = torch.minimum(torch.maximum(idx, board_zeros), board_upper)
        board_ctx = board_out[safe_state_idx, safe_idx]

        from_sq = action_from_squares.long().clamp(0, self.cfg.board_squares - 1)
        to_sq = action_to_squares.long().clamp(0, self.cfg.board_squares - 1)
        from_emb = self.from_square_embed(from_sq)
        to_emb = self.to_square_embed(to_sq)

        dt = action_delta_t.float() / 8.0
        dl = action_delta_l.float() / 8.0
        is_jump = ((action_delta_t != 0) | (action_delta_l != 0)).float()
        is_tl_change = (action_delta_l != 0).float()
        move_feats = torch.stack([dt, dl, is_jump, is_tl_change], dim=-1)
        feat_emb = self.move_feat_proj(move_feats)

        move_repr = torch.cat([board_ctx, from_emb, to_emb, feat_emb], dim=-1)
        move_logits = self.move_scorer(move_repr).squeeze(-1)
        logits = torch.where(valid, move_logits, logits)
        return logits

    def predict(
        self,
        board_planes: torch.Tensor,
        last_move_markers: torch.Tensor,
        l_coords: torch.Tensor,
        t_coords: torch.Tensor,
        urgency: torch.Tensor,
    ) -> tuple[float, float, torch.Tensor]:
        """Single-sample inference convenience."""
        with torch.no_grad():
            v, sl, rl = self.forward(
                board_planes.unsqueeze(0),
                last_move_markers.unsqueeze(0),
                l_coords.unsqueeze(0),
                t_coords.unsqueeze(0),
                urgency.unsqueeze(0) if urgency.dim() == 1 else urgency.unsqueeze(0),
            )
            return v.item(), sl.item(), rl.squeeze(0)

    def predict_actions(
        self,
        board_planes: torch.Tensor,
        last_move_markers: torch.Tensor,
        l_coords: torch.Tensor,
        t_coords: torch.Tensor,
        urgency: torch.Tensor,
        action_board_indices: torch.Tensor,
        action_from_squares: torch.Tensor,
        action_to_squares: torch.Tensor,
        action_delta_t: torch.Tensor,
        action_delta_l: torch.Tensor,
        action_is_submit: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
        """
        Single-sample inference for explicit legal-action logits.

        Returns:
            value: float from current player's perspective
            action_logits: [A] aligned legal-action logits
        """
        with torch.no_grad():
            v, submit_logit, _, board_out, _ = self.forward(
                board_planes.unsqueeze(0),
                last_move_markers.unsqueeze(0),
                l_coords.unsqueeze(0),
                t_coords.unsqueeze(0),
                urgency.unsqueeze(0) if urgency.dim() == 1 else urgency.unsqueeze(0),
                return_latent=True,
            )
            logits = self.score_legal_actions(
                board_out=board_out.squeeze(0),
                submit_logit=submit_logit.squeeze(0),
                action_board_indices=action_board_indices,
                action_from_squares=action_from_squares,
                action_to_squares=action_to_squares,
                action_delta_t=action_delta_t,
                action_delta_l=action_delta_l,
                action_is_submit=action_is_submit,
            )
            return v.item(), logits
