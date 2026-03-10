# ml/models/value_head.py
"""
Value Head: Estimates the expected outcome from a given state.

Takes the global embedding from the MultiverseEncoder and outputs:
  - A scalar value in [-1, 1] representing expected outcome for the current player
    (1 = win, -1 = loss, 0 = draw)
"""

import torch
import torch.nn as nn

from ..config import ValueConfig


class ValueHead(nn.Module):
    """
    Simple MLP value head.
    global_embed → hidden → hidden → tanh(scalar)
    """

    def __init__(self, cfg: ValueConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = ValueConfig()
        self.cfg = cfg

        self.mlp = nn.Sequential(
            nn.Linear(cfg.global_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, global_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_embed: [B, global_dim] from MultiverseEncoder
        Returns:
            value: [B, 1] estimated value in [-1, 1]
        """
        return self.mlp(global_embed)
