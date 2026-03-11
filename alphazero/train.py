# alphazero/train.py
"""
Full AlphaZero training loop for semimove-level 5D Chess.

Pipeline per iteration:
  1. Self-play: generate games 鈫?fill replay buffer
  2. Train: sample batches 鈫?optimize network (policy CE + value MSE)
  3. Checkpoint: save network periodically

Usage:
  python -m alphazero.train [--cpu] [--resume PATH] [--resume-weights-only] [--no-resume] [--iterations N]
"""

import os
import sys
import json
import time
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .config import TrainConfig
from .network import AlphaZeroNetwork
from .self_play import CppOnnxSelfPlayWorker, SelfPlayWorker, ReplayBuffer, collate_samples
from .mcts import SUBMIT_ACTION
from .variants import VARIANT_PROFILES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("alphazero.train")


def setup_logging(log_dir: str):
    """Configure logging to file + console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    root = logging.getLogger("alphazero")
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------
def compute_loss(network: AlphaZeroNetwork, batch: dict,
                 cfg: TrainConfig) -> tuple[torch.Tensor, dict]:
    """
    Compute combined loss on a batch.

    Policy loss is exact cross-entropy on legal actions aligned with MCTS
    action ordering (no surrogate/top-k approximation).
    """
    value, submit_logit, _, board_out, _ = network(
        batch['board_planes'],
        batch['last_move_markers'],
        batch['l_coords'],
        batch['t_coords'],
        batch['urgency'],
        batch['padding_mask'],
        return_latent=True,
    )

    value_target = batch['value_target']
    value_loss = nn.functional.mse_loss(value, value_target)

    policy_targets = batch['policy_targets']
    action_board_indices = batch['action_board_indices']
    action_from_squares = batch['action_from_squares']
    action_to_squares = batch['action_to_squares']
    action_delta_t = batch['action_delta_t']
    action_delta_l = batch['action_delta_l']
    action_is_submit = batch['action_is_submit']

    policy_loss = torch.tensor(0.0, device=value.device)
    valid_policy_count = 0

    B = value.shape[0]
    for i in range(B):
        pt = policy_targets[i]
        if len(pt) == 0:
            continue

        n_valid = int((~batch['padding_mask'][i]).sum().item())
        if n_valid <= 0:
            continue

        abi = torch.from_numpy(action_board_indices[i]).to(value.device, dtype=torch.long)
        afs = torch.from_numpy(action_from_squares[i]).to(value.device, dtype=torch.long)
        ats = torch.from_numpy(action_to_squares[i]).to(value.device, dtype=torch.long)
        adt = torch.from_numpy(action_delta_t[i]).to(value.device, dtype=torch.float32)
        adl = torch.from_numpy(action_delta_l[i]).to(value.device, dtype=torch.float32)
        ais = torch.from_numpy(action_is_submit[i]).to(value.device, dtype=torch.bool)

        if abi.numel() == 0 or abi.numel() != len(pt):
            continue

        logits_i = network.score_legal_actions(
            board_out=board_out[i, :n_valid],
            submit_logit=submit_logit[i],
            action_board_indices=abi,
            action_from_squares=afs,
            action_to_squares=ats,
            action_delta_t=adt,
            action_delta_l=adl,
            action_is_submit=ais,
        )

        pt_tensor = torch.from_numpy(pt).float().to(value.device)
        pt_sum = pt_tensor.sum()
        if pt_sum <= 0:
            continue
        pt_tensor = pt_tensor / pt_sum

        log_probs = nn.functional.log_softmax(logits_i, dim=0)
        loss_i = -torch.sum(pt_tensor * log_probs)

        policy_loss = policy_loss + loss_i
        valid_policy_count += 1

    if valid_policy_count > 0:
        policy_loss = policy_loss / valid_policy_count

    total_loss = (cfg.value_loss_weight * value_loss +
                  cfg.policy_loss_weight * policy_loss)

    metrics = {
        'total_loss': total_loss.item(),
        'value_loss': value_loss.item(),
        'policy_loss': policy_loss.item(),
    }
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
class Trainer:
    """Manages the full AlphaZero training loop."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg

        # Device
        if cfg.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        logger.info(f"Using device: {self.device}")

        # Network
        self.network = AlphaZeroNetwork(cfg.network).to(self.device)
        param_count = sum(p.numel() for p in self.network.parameters())
        logger.info(f"Network parameters: {param_count:,}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.network.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # LR scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.num_iterations * cfg.epochs_per_iteration,
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(cfg.replay_buffer_size)

        # Self-play backend
        backend = getattr(cfg.self_play, "self_play_backend", "python")
        if backend == "cpp_onnx":
            self.worker = CppOnnxSelfPlayWorker(
                network=self.network,
                mcts_cfg=cfg.mcts,
                sp_cfg=cfg.self_play,
                device=self.device,
                train_cfg=cfg,
            )
        elif backend == "python":
            self.worker = SelfPlayWorker(
                network=self.network,
                mcts_cfg=cfg.mcts,
                sp_cfg=cfg.self_play,
                device=self.device,
                variant_pgn=cfg.variant_pgn,
            )
        else:
            raise ValueError(f"Unsupported self-play backend: {backend}")
        logger.info(f"Self-play backend: {backend}")

        # Tracking
        self.iteration = 0
        self.total_games = 0
        self.total_samples = 0
        self.metrics_history = []

        # Directories
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)

    def train(self, num_iterations: int = None):
        """Run the full training loop."""
        n_iter = num_iterations or self.cfg.num_iterations
        start_iter = self.iteration
        end_iter = start_iter + n_iter
        logger.info(f"Starting AlphaZero training: iterations {start_iter+1}..{end_iter}")
        logger.info(f"Config: {self.cfg.self_play.num_games} games/iter, "
                     f"{self.cfg.epochs_per_iteration} epochs/iter, "
                     f"batch_size={self.cfg.batch_size}")
        if start_iter > 0:
            logger.info(f"Resuming from iteration {start_iter} "
                         f"(buffer={len(self.replay_buffer)})")

        for iteration in range(start_iter, end_iter):
            self.iteration = iteration
            iter_start = time.time()
            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration + 1}/{end_iter}")
            logger.info(f"{'='*60}")

            # --- Phase 1: Self-play ---
            sp_start = time.time()
            self._self_play_phase()
            sp_time = time.time() - sp_start
            logger.info(f"Self-play: {sp_time:.1f}s, "
                         f"buffer size: {len(self.replay_buffer)}")

            # --- Phase 2: Training ---
            if len(self.replay_buffer) >= self.cfg.min_replay_size:
                train_start = time.time()
                train_metrics = self._training_phase()
                train_time = time.time() - train_start
                logger.info(f"Training: {train_time:.1f}s, "
                             f"loss={train_metrics['total_loss']:.4f}, "
                             f"value_loss={train_metrics['value_loss']:.4f}, "
                             f"policy_loss={train_metrics['policy_loss']:.4f}")
            else:
                logger.info(f"Skipping training (need {self.cfg.min_replay_size} "
                             f"samples, have {len(self.replay_buffer)})")
                train_metrics = {}

            # --- Phase 3: Checkpoint ---
            if (iteration + 1) % self.cfg.save_interval == 0:
                self._save_checkpoint(iteration + 1)

            iter_time = time.time() - iter_start
            logger.info(f"Iteration time: {iter_time:.1f}s")

            # Log to JSONL
            self._log_metrics(iteration + 1, train_metrics, iter_time)

        # Final save
        self._save_checkpoint(end_iter, final=True)
        logger.info("Training complete!")

    def _self_play_phase(self):
        """Generate self-play games and add to replay buffer."""
        self.network.eval()

        with torch.no_grad():
            games = self.worker.generate_games()

        snapshot_games = []
        snapshot_interval = int(getattr(self.cfg.self_play, "pgn_snapshot_interval", 100))
        for game in games:
            self.replay_buffer.push_game(game)
            self.total_games += 1
            self.total_samples += len(game.samples)
            if snapshot_interval > 0 and self.total_games % snapshot_interval == 0:
                snapshot_games.append((self.total_games, game))

        # Log game stats
        outcomes = [g.outcome for g in games]
        lengths = [g.total_semimoves for g in games]
        reasons = {}
        for g in games:
            reasons[g.terminal_reason] = reasons.get(g.terminal_reason, 0) + 1

        logger.info(f"  Games: {len(games)}, "
                     f"avg_length: {np.mean(lengths):.1f}, "
                     f"white_wins: {sum(1 for o in outcomes if o > 0)}, "
                     f"black_wins: {sum(1 for o in outcomes if o < 0)}, "
                     f"draws: {sum(1 for o in outcomes if o == 0)}")
        logger.info(f"  Termination reasons: {reasons}")

        # Persist importable PGN snapshots at the configured game interval.
        self._log_sample_games(games, snapshot_games)

    def _training_phase(self) -> dict:
        """Train the network on replay buffer samples."""
        self.network.train()

        total_metrics = {'total_loss': 0.0, 'value_loss': 0.0, 'policy_loss': 0.0}
        n_steps = 0

        for epoch in range(self.cfg.epochs_per_iteration):
            # Sample batch
            samples = self.replay_buffer.sample(self.cfg.batch_size)
            batch = collate_samples(samples, self.device)

            # Forward + backward
            self.optimizer.zero_grad()
            loss, metrics = compute_loss(self.network, batch, self.cfg)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            for k in total_metrics:
                total_metrics[k] += metrics[k]
            n_steps += 1

        # Average metrics
        for k in total_metrics:
            total_metrics[k] /= max(n_steps, 1)

        return total_metrics

    def _save_checkpoint(self, iteration: int, final: bool = False):
        """Save model checkpoint with full resumable state."""
        tag = "final" if final else f"iter_{iteration:04d}"
        path = os.path.join(self.cfg.checkpoint_dir, f"agent_{tag}.pt")

        # Serialize replay buffer (positions + values only, compact)
        buffer_data = None
        try:
            buffer_data = {
                'buffer': self.replay_buffer.buffer,
                'position': self.replay_buffer.position,
                'capacity': self.replay_buffer.capacity,
            }
        except Exception as e:
            logger.warning(f"Could not serialize replay buffer: {e}")

        torch.save({
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_games': self.total_games,
            'total_samples': self.total_samples,
            'replay_buffer': buffer_data,
            'config': {
                'variant_name': self.cfg.variant_name,
                'variant_pgn': self.cfg.variant_pgn,
                'board_size_x': self.cfg.board_size_x,
                'board_size_y': self.cfg.board_size_y,
                'network': self.cfg.network.__dict__,
                'mcts': self.cfg.mcts.__dict__,
                'self_play': self.cfg.self_play.__dict__,
            },
        }, path)
        logger.info(f"Saved checkpoint: {path} "
                     f"(buffer={len(self.replay_buffer)} samples)")

        # Also maintain a 'latest' symlink/copy pointer
        latest_path = os.path.join(self.cfg.checkpoint_dir, "latest.pt")
        try:
            if os.path.exists(latest_path):
                os.remove(latest_path)
            # On Windows, copy instead of symlink for reliability
            import shutil
            shutil.copy2(path, latest_path)
        except Exception as e:
            logger.warning(f"Could not create latest checkpoint pointer: {e}")

    def load_checkpoint(self, path: str, weights_only: bool = False):
        """Load model from checkpoint, optionally restoring weights only."""
        # Resolve 'latest' keyword
        if path.lower() == 'latest':
            path = os.path.join(self.cfg.checkpoint_dir, "latest.pt")
            if not os.path.exists(path):
                # Fallback: find highest-numbered checkpoint
                path = self._find_latest_checkpoint()
            if path is None:
                logger.warning("No checkpoint found to resume from.")
                return

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.network.load_state_dict(ckpt['model_state_dict'])
        self.iteration = ckpt.get('iteration', 0)
        self.total_games = ckpt.get('total_games', 0)
        self.total_samples = ckpt.get('total_samples', 0)

        if not weights_only:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])

            # Restore replay buffer
            buf_data = ckpt.get('replay_buffer')
            if buf_data and isinstance(buf_data, dict):
                try:
                    self.replay_buffer.buffer = buf_data['buffer']
                    self.replay_buffer.position = buf_data['position']
                    logger.info(f"Restored replay buffer: {len(self.replay_buffer)} samples")
                except Exception as e:
                    logger.warning(f"Could not restore replay buffer: {e}")

            logger.info(f"Loaded checkpoint from {path} (iteration {self.iteration}, "
                         f"games={self.total_games}, samples={self.total_samples})")
            return

        logger.info(
            f"Loaded model weights only from {path} (iteration {self.iteration}, "
            f"games={self.total_games}, samples={self.total_samples}); "
            f"optimizer, scheduler, and replay buffer were reset."
        )

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the highest-numbered checkpoint in checkpoint_dir."""
        ckpt_dir = self.cfg.checkpoint_dir
        if not os.path.isdir(ckpt_dir):
            return None
        candidates = []
        for f in os.listdir(ckpt_dir):
            if f.startswith("agent_iter_") and f.endswith(".pt"):
                try:
                    num = int(f.replace("agent_iter_", "").replace(".pt", ""))
                    candidates.append((num, os.path.join(ckpt_dir, f)))
                except ValueError:
                    pass
        if not candidates:
            # Try agent_final.pt
            final = os.path.join(ckpt_dir, "agent_final.pt")
            return final if os.path.exists(final) else None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _log_sample_games(self, games: list, snapshot_games: list[tuple[int, object]]):
        """Persist periodic PGN snapshots and log one brief human-readable sample."""
        games_dir = os.path.join(self.cfg.log_dir, "games")
        os.makedirs(games_dir, exist_ok=True)

        saved_paths = []
        for game_id, game in snapshot_games:
            if not getattr(game, "pgn", ""):
                logger.warning("Skipping saved game #%s because no PGN was recorded.", game_id)
                continue
            game_path = os.path.join(
                games_dir,
                f"iter{self.iteration+1:04d}_game{game_id:06d}.pgn",
            )
            with open(game_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(game.pgn.rstrip())
                f.write("\n")
            saved_paths.append(game_path)

        if saved_paths:
            logger.info("  Saved PGN snapshots: %s", ", ".join(os.path.basename(p) for p in saved_paths))

        if not games:
            return

        # Also log a brief summary of the most interesting game to main log
        # Pick the longest decisive game
        decisive = [g for g in games if g.outcome != 0.0]
        if decisive:
            show_game = max(decisive, key=lambda g: g.total_semimoves)
        else:
            show_game = max(games, key=lambda g: g.total_semimoves)

        logger.info(f"  Sample game (longest decisive):")
        for entry in show_game.move_history:
            player_str = "W" if entry.player == 0 else "B"
            if entry.action_type == 'submit':
                logger.info(f"    [{player_str}] SUBMIT  "
                             f"(boards={entry.board_count}, v={entry.mcts_value:+.3f})")
            else:
                logger.info(f"    [{player_str}] {entry.ext_move_str}  "
                             f"(boards={entry.board_count}, v={entry.mcts_value:+.3f})")
        outcome_str = {1.0: "White wins", -1.0: "Black wins", 0.0: "Draw"}
        logger.info(f"    Result: {outcome_str.get(show_game.outcome, f'{show_game.outcome:.3f}')} "
                     f"({show_game.terminal_reason}, "
                     f"board_limit={show_game.board_limit})")

    def _log_metrics(self, iteration: int, metrics: dict, iter_time: float):
        """Append metrics to JSONL log file."""
        log_file = os.path.join(self.cfg.log_dir, "training_log.jsonl")
        entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'total_games': self.total_games,
            'total_samples': self.total_samples,
            'buffer_size': len(self.replay_buffer),
            'iter_time': round(iter_time, 2),
            'lr': self.optimizer.param_groups[0]['lr'],
            **metrics,
        }
        with open(log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AlphaZero training for 5D Chess")
    parser.add_argument('--cpu', action='store_true', help="Force CPU")
    parser.add_argument('--variant', type=str, choices=sorted(VARIANT_PROFILES.keys()), default=None,
                        help="Board variant preset")
    parser.add_argument('--resume', type=str, default=None,
                        help="Resume from checkpoint path, or 'latest' for auto-discovery")
    parser.add_argument('--resume-weights-only', action='store_true',
                        help="Load model weights but keep fresh optimizer, scheduler, and replay buffer")
    parser.add_argument('--no-resume', action='store_true',
                        help="Start from scratch even if a latest checkpoint exists")
    parser.add_argument('--iterations', type=int, default=None,
                        help="Number of NEW iterations to run (added on top of resumed progress)")
    parser.add_argument('--games', type=int, default=None, help="Games per iteration")
    parser.add_argument('--pgn-snapshot-interval', type=int, default=None,
                        help="Save one PGN snapshot every N completed games; <=0 disables snapshots")
    parser.add_argument('--sims', type=int, default=None, help="MCTS simulations per move")
    parser.add_argument('--leaf-batch-size', type=int, default=None,
                        help="Batched leaf evaluations per MCTS search wave")
    parser.add_argument('--min-board-limit', type=int, default=None,
                        help="Minimum board-count limit for self-play termination")
    parser.add_argument('--max-board-limit', type=int, default=None,
                        help="Maximum board-count limit for self-play termination")
    parser.add_argument('--rules-mode', type=str, choices=["capture_king", "strict"], default=None,
                        help="Training rules mode for self-play")
    parser.add_argument('--selfplay-backend', type=str, choices=["python", "cpp_onnx"], default=None,
                        help="Self-play backend implementation")
    parser.add_argument('--sp-workers', type=int, default=None,
                        help="Parallel self-play worker processes (CPU only)")
    parser.add_argument('--sp-task-games', type=int, default=None,
                        help="Games per submitted self-play worker task")
    parser.add_argument('--sp-max-tasks-per-child', type=int, default=None,
                        help="Restart each self-play worker after this many tasks")
    parser.add_argument('--max-game-length', type=int, default=None,
                        help="Optional semimove cutoff for self-play; <=0 disables it")
    parser.add_argument('--legal-cache-max', type=int, default=None,
                        help="Per-game legal-action cache entry cap for semimove env")
    parser.add_argument('--cpp-selfplay-exe', type=str, default=None,
                        help="Path to the az_selfplay_onnx executable")
    parser.add_argument('--cpp-onnx-model', type=str, default=None,
                        help="Path for the exported self-play ONNX model")
    parser.add_argument('--cpp-onnx-precision', type=str, choices=["auto", "fp16", "fp32"], default=None,
                        help="Precision used for exported self-play ONNX")
    parser.add_argument('--cpp-onnx-provider', type=str, choices=["cpu", "cuda"], default=None,
                        help="Execution provider used by the C++ ONNX self-play runner")
    parser.add_argument('--cpp-onnx-device-id', type=int, default=None,
                        help="CUDA device id used by the C++ ONNX self-play runner")
    parser.add_argument('--cpp-onnx-ort-threads', type=int, default=None,
                        help="ORT intra-op threads used inside the C++ self-play runner; 0 means auto")
    parser.add_argument('--cpp-onnx-max-boards', type=int, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.variant:
        cfg.apply_variant(args.variant)
        cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, cfg.variant_name)
        cfg.log_dir = os.path.join(cfg.log_dir, cfg.variant_name)
        cfg.self_play.cpp_onnx_model_path = os.path.join(cfg.checkpoint_dir, "selfplay_fp16.onnx")
    if args.cpu:
        cfg.device = "cpu"
    if args.games:
        cfg.self_play.num_games = args.games
    if args.pgn_snapshot_interval is not None:
        cfg.self_play.pgn_snapshot_interval = int(args.pgn_snapshot_interval)
    if args.sims:
        cfg.mcts.num_simulations = args.sims
    if args.leaf_batch_size:
        cfg.mcts.leaf_batch_size = max(1, args.leaf_batch_size)
    if args.min_board_limit:
        cfg.self_play.min_board_limit = max(1, args.min_board_limit)
    if args.max_board_limit:
        cfg.self_play.max_board_limit = max(cfg.self_play.min_board_limit, args.max_board_limit)
    if args.rules_mode:
        cfg.self_play.rules_mode = args.rules_mode
    if args.selfplay_backend:
        cfg.self_play.self_play_backend = args.selfplay_backend
    if args.sp_workers:
        cfg.self_play.num_workers = max(1, args.sp_workers)
    if args.sp_task_games:
        cfg.self_play.worker_task_games = max(1, args.sp_task_games)
    if args.sp_max_tasks_per_child:
        cfg.self_play.worker_max_tasks_per_child = max(1, args.sp_max_tasks_per_child)
    if args.max_game_length is not None:
        cfg.self_play.max_game_length = int(args.max_game_length)
    if args.legal_cache_max:
        cfg.self_play.legal_cache_max_entries = max(128, args.legal_cache_max)
    if args.cpp_selfplay_exe:
        cfg.self_play.cpp_selfplay_executable = args.cpp_selfplay_exe
    if args.cpp_onnx_model:
        cfg.self_play.cpp_onnx_model_path = args.cpp_onnx_model
    if args.cpp_onnx_precision:
        cfg.self_play.cpp_onnx_model_precision = args.cpp_onnx_precision
    if args.cpp_onnx_provider:
        cfg.self_play.cpp_onnx_provider = args.cpp_onnx_provider
    if args.cpp_onnx_device_id is not None:
        cfg.self_play.cpp_onnx_cuda_device_id = max(0, int(args.cpp_onnx_device_id))
    if args.cpp_onnx_ort_threads is not None:
        cfg.self_play.cpp_onnx_ort_threads = max(0, int(args.cpp_onnx_ort_threads))
    if args.cpp_onnx_max_boards is not None:
        logger.warning("--cpp-onnx-max-boards is deprecated and ignored; ONNX self-play now uses a dynamic board axis.")

    setup_logging(cfg.log_dir)

    trainer = Trainer(cfg)
    resume_path = args.resume
    if args.no_resume:
        resume_path = None
    elif resume_path is None:
        latest_path = os.path.join(cfg.checkpoint_dir, "latest.pt")
        if os.path.exists(latest_path) or trainer._find_latest_checkpoint() is not None:
            resume_path = "latest"

    if resume_path:
        if args.resume is None and not args.no_resume:
            logger.info("Auto-resuming from latest checkpoint.")
        trainer.load_checkpoint(resume_path, weights_only=args.resume_weights_only)
    else:
        logger.info("Starting training from scratch.")

    trainer.train(num_iterations=args.iterations)


if __name__ == "__main__":
    main()

