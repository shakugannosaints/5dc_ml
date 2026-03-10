# ml/models/board_encoder.py
"""
Board Encoder: CNN/ResNet for encoding a single 8x8 (or smaller) board.
Input: [B, C=27, H, W] tensor  (C = piece channel planes from C++ engine)
Output: [B, embed_dim] vector per board
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import BoardEncoderConfig


class ResBlock(nn.Module):
    """Pre-activation residual block with batch norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + residual


class BoardEncoder(nn.Module):
    """
    Encodes a single board (tensor of piece planes) into a fixed-dim embedding.
    
    Architecture:
        - Initial Conv: C_in -> inner_channels
        - N ResBlocks at inner_channels
        - Global average pooling
        - Linear projection -> embed_dim
    
    Supports variable board sizes (4x4, 5x5, 8x8 etc.) via global pooling.
    """

    def __init__(self, cfg: BoardEncoderConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = BoardEncoderConfig()
        self.cfg = cfg

        # Initial convolution: map piece planes to internal channels
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.num_piece_channels, cfg.inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.inner_channels),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.res_tower = nn.Sequential(
            *[ResBlock(cfg.inner_channels) for _ in range(cfg.num_res_blocks)]
        )

        # Final batch norm + relu before pooling
        self.final_bn = nn.BatchNorm2d(cfg.inner_channels)

        # Global average pooling → flatten → linear → embed_dim
        self.fc = nn.Linear(cfg.inner_channels, cfg.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]  where C = num_piece_channels (27)
        Returns:
            [B, embed_dim] board embeddings
        """
        out = self.stem(x)              # [B, inner_ch, H, W]
        out = self.res_tower(out)       # [B, inner_ch, H, W]
        out = F.relu(self.final_bn(out))
        out = out.mean(dim=(-2, -1))    # [B, inner_ch]  global avg pool
        out = self.fc(out)              # [B, embed_dim]
        return out

    def encode_boards(self, board_tensors: dict, device: torch.device) -> tuple[list, torch.Tensor]:
        """
        Convenience: encode a dict of {(l,t,c): numpy_array} from engine.
        
        Returns:
            keys:    list of (l,t,c) tuples, in order
            embeds:  [N, embed_dim] tensor
        """
        if len(board_tensors) == 0:
            return [], torch.zeros(0, self.cfg.embed_dim, device=device)

        keys = sorted(board_tensors.keys())
        # Stack into batch: each value is numpy [C, H, W]
        batch = torch.stack([
            torch.from_numpy(board_tensors[k]).float()
            for k in keys
        ]).to(device)  # [N, C, H, W]

        embeds = self.forward(batch)    # [N, embed_dim]
        return keys, embeds
