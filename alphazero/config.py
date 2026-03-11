# alphazero/config.py
"""
Configuration for semimove-level AlphaZero on 5D Chess.
"""

from dataclasses import dataclass, field

from .variants import VERY_SMALL_PROFILE, get_variant_profile


@dataclass
class NetworkConfig:
    """Transformer network configuration."""
    # Board representation. The active variant preset rewrites these fields.
    num_piece_types: int = VERY_SMALL_PROFILE.num_piece_types
    board_squares: int = VERY_SMALL_PROFILE.board_squares
    board_side: int = VERY_SMALL_PROFILE.board_side

    # Input channels per board: 6 white + 6 black + unmoved + occupied = 14 planes.
    piece_channels: int = VERY_SMALL_PROFILE.piece_channels

    # Transformer
    d_model: int = 256                  # model dimension
    n_heads: int = 8                    # attention heads
    n_layers: int = 6                   # transformer layers
    d_ff: int = 512                     # feed-forward inner dim
    dropout: float = 0.1

    # Positional encoding
    max_timelines: int = 10             # max L index range (for sinusoidal PE)
    max_turns: int = 50                 # max T index range

    # Output heads
    # raw_logits: (n, 16) for each board (4x4 squares)
    # submit_logit: scalar
    # value: scalar in [-1, 1]


@dataclass
class MCTSConfig:
    """MCTS search configuration."""
    num_simulations: int = 200          # MCTS simulations per move
    c_puct: float = 2.0                 # exploration constant
    dirichlet_alpha: float = 0.3        # root noise
    dirichlet_epsilon: float = 0.25     # root noise weight
    temperature_start: float = 1.0      # exploration temperature
    temperature_threshold: int = 15     # after this many full moves, use temp=0
    # Submit prior: base probability assigned to submit action
    submit_prior_weight: float = 0.1
    # Reuse state expansions within one semimove search via a transposition table.
    use_transposition_table: bool = True
    # Reuse full-action suffixes between semimove nodes inside one unsubmitted turn.
    reuse_semimove_suffixes: bool = True


@dataclass
class SelfPlayConfig:
    """Self-play data generation configuration."""
    # Self-play backend. "cpp_onnx" runs game generation inside the C++ runner.
    self_play_backend: str = "cpp_onnx"
    # Training rules. "capture_king" skips check legality and ends on royal capture.
    rules_mode: str = "capture_king"
    # Board count limit for forced termination (material scoring)
    min_board_limit: int = 15            # minimum board count before forced end
    max_board_limit: int = 25           # maximum board count for forced end
    # Material scoring
    material_scale: float = 2.0         # tanh scaling for material difference
    # Optional game-length safety fuse. <= 0 means disabled.
    max_game_length: int = 0            # max semimoves before forced end (disabled by default)
    # Number of games per iteration
    num_games: int = 64
    # Save one importable PGN snapshot every N completed games. <= 0 disables it.
    pgn_snapshot_interval: int = 100
    # Per-game legal-action LRU cache size in semimove env
    legal_cache_max_entries: int = 4000
    # Parallel workers for self-play game generation (CPU only)
    num_workers: int = 1
    # Games per submitted worker task. Larger chunks amortize C++ process/session startup.
    worker_task_games: int = 4
    # Restart worker process after this many tasks to return RSS to the OS.
    worker_max_tasks_per_child: int = 8
    # Emit per-task worker timing / RSS logs from the parent process.
    log_worker_task_stats: bool = True
    # C++ ONNX self-play backend settings.
    cpp_selfplay_executable: str = "build_onnx_selfplay/az_selfplay_onnx.exe"
    cpp_onnx_model_path: str = "alphazero/checkpoints/selfplay_fp16.onnx"
    cpp_onnx_model_precision: str = "fp16"
    cpp_onnx_opset: int = 18
    cpp_onnx_provider: str = "cpu"
    cpp_onnx_cuda_device_id: int = 0
    cpp_onnx_ort_threads: int = 1
    # Legacy compatibility knob. Dynamic-board ONNX export no longer uses it.
    cpp_onnx_max_boards: int = 32
    # Temperature schedule
    temperature: float = 1.0            # exploration temperature (early game)
    temperature_final: float = 0.1      # exploitation temperature (late game)
    temp_threshold: int = 30            # semimove count to switch temperature


@dataclass
class TrainConfig:
    """Training loop configuration."""
    # Variant
    variant_name: str = VERY_SMALL_PROFILE.name
    variant_pgn: str = VERY_SMALL_PROFILE.variant_pgn
    board_size_x: int = VERY_SMALL_PROFILE.board_side
    board_size_y: int = VERY_SMALL_PROFILE.board_side

    # Training
    batch_size: int = 256
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    num_iterations: int = 100           # AlphaZero iterations (self-play + train)
    epochs_per_iteration: int = 10      # gradient steps per iteration
    replay_buffer_size: int = 50000     # max samples in replay buffer
    min_replay_size: int = 512          # min samples before training starts

    # System
    device: str = "cuda"
    checkpoint_dir: str = "alphazero/checkpoints"
    log_dir: str = "alphazero/logs"
    save_interval: int = 5             # save checkpoint every N iterations

    # Sub-configs
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)

    def __post_init__(self):
        self.apply_variant(self.variant_name)

    def apply_variant(self, variant_name: str):
        profile = get_variant_profile(variant_name)
        self.variant_name = profile.name
        self.variant_pgn = profile.variant_pgn
        self.board_size_x = profile.board_side
        self.board_size_y = profile.board_side
        self.network.num_piece_types = profile.num_piece_types
        self.network.board_side = profile.board_side
        self.network.board_squares = profile.board_squares
        self.network.piece_channels = profile.piece_channels
