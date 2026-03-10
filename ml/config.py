# ml/config.py
# Hyperparameters and configuration for 5D Chess ML

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class BoardEncoderConfig:
    """Configuration for single-board CNN encoder."""
    num_piece_channels: int = 27       # input channels from C++ (13w + 13b + unmoved + wall + occupied)
    board_size: int = 8                 # 8x8 board
    embed_dim: int = 128                # output embedding dimension
    num_res_blocks: int = 4             # number of residual blocks
    inner_channels: int = 64            # channels in residual blocks

@dataclass
class MultiverseEncoderConfig:
    """Configuration for GNN multiverse encoder."""
    node_scalar_dim: int = 6            # scalar features per node (is_active, is_mandatory, is_player, norm_l, norm_t, color)
    board_embed_dim: int = 128          # from BoardEncoder
    gnn_hidden_dim: int = 256           # hidden dimension in GNN layers
    gnn_num_layers: int = 3             # number of GAT layers
    gnn_num_heads: int = 4              # attention heads in GAT
    global_dim: int = 256               # global state embedding dimension
    num_edge_types: int = 2             # temporal, branch

@dataclass
class PolicyConfig:
    """Configuration for factored policy network."""
    state_dim: int = 256                # from MultiverseEncoder global
    board_embed_dim: int = 256          # from MultiverseEncoder node (gnn_hidden_dim)
    hidden_dim: int = 256               # hidden dimension
    num_cross_attn_heads: int = 4       # cross-attention heads between timelines
    move_embed_dim: int = 64            # embedding dim for from/to square

@dataclass
class ValueConfig:
    """Configuration for value network."""
    global_dim: int = 256               # from MultiverseEncoder
    hidden_dim: int = 128               # hidden layers
    num_heads: int = 1                  # output: scalar value

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Game variant
    variant: str = "Very Small - Open"  # start with small variant
    variant_pgn: str = '[Board "Very Small - Open"]\n[Mode "5D"]\n'
    board_size_x: int = 4
    board_size_y: int = 4
    
    # Self-play
    num_games_per_epoch: int = 8        # games per training epoch
    max_game_length: int = 100          # max moves per game before draw
    num_action_samples: int = 64        # candidate actions to sample from HC search
    temperature_start: float = 1.0      # exploration temperature
    temperature_end: float = 0.5
    temperature_decay_epochs: int = 200
    
    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    entropy_bonus: float = 0.1          # encourage exploration (was 0.01, too low)
    num_epochs: int = 500
    
    # Architecture
    board_encoder: BoardEncoderConfig = field(default_factory=BoardEncoderConfig)
    multiverse_encoder: MultiverseEncoderConfig = field(default_factory=MultiverseEncoderConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    value: ValueConfig = field(default_factory=ValueConfig)
    
    # System
    device: str = "cuda"                # or "cpu"
    log_interval: int = 1
    save_interval: int = 10
    checkpoint_dir: str = "checkpoints"
    log_file: str = "logs/ml_training.jsonl"
    
    # Engine path
    engine_path: str = ""               # will be set at runtime


# Convenience: small variant config for fast iteration
SMALL_CONFIG = TrainingConfig(
    variant="Very Small - Open",
    variant_pgn='[Board "Very Small - Open"]\n[Mode "5D"]\n',
    board_size_x=4,
    board_size_y=4,
    num_games_per_epoch=8,
    max_game_length=75,
    num_action_samples=32,
    num_epochs=200,
)

# Standard variant config
STANDARD_CONFIG = TrainingConfig(
    variant="Standard",
    variant_pgn='[Board "Standard"]\n[Mode "5D"]\n',
    board_size_x=8,
    board_size_y=8,
    num_games_per_epoch=4,
    max_game_length=200,
    num_action_samples=64,
    num_epochs=1000,
)
