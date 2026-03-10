# ml/trainer.py
"""
Trainer: Manages the training loop for the 5D Chess agent.

Training approach: Policy Gradient (REINFORCE) with value baseline.
  1. Self-play generates game trajectories
  2. Compute returns (game outcome from each player's perspective)
  3. Update policy via REINFORCE with advantage = return - value_baseline
  4. Update value head via MSE on returns
  5. Entropy bonus for exploration

This is an AlphaZero-lite approach adapted for factored policy.
"""

import sys
import os
import time
import json
import random
from pathlib import Path

import torch
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

from .config import TrainingConfig
from .models.agent import Agent
from .self_play import SelfPlayWorker


class Trainer:
    """
    Main training loop:
    
    For each epoch:
      1. Self-play N games with current agent
      2. Collect all (state, action, return) samples
      3. Shuffle and train on mini-batches
      4. Log metrics
      5. Periodically save checkpoints
    """

    def __init__(self, cfg: TrainingConfig | None = None):
        if cfg is None:
            cfg = TrainingConfig()
        self.cfg = cfg

        # Device
        if cfg.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"[Trainer] Using device: {self.device}")

        # Build agent
        self.agent = Agent(cfg).to(self.device)
        param_count = sum(p.numel() for p in self.agent.parameters())
        print(f"[Trainer] Agent parameters: {param_count:,}")

        # Optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Scheduler: cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.num_epochs, eta_min=1e-5
        )

        # Logging
        self.log_path = Path(cfg.log_file)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Checkpoints
        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Metrics
        self.epoch = 0
        self.total_games = 0
        self.total_samples = 0

    def find_latest_checkpoint(self) -> Path | None:
        """Find the latest checkpoint in the checkpoint directory."""
        if not self.ckpt_dir.exists():
            return None
        # Look for epoch checkpoints (agent_epoch_XXXX.pt), pick highest epoch
        ckpts = sorted(self.ckpt_dir.glob("agent_epoch_*.pt"))
        if ckpts:
            return ckpts[-1]  # sorted lexicographically = highest epoch
        # Fallback: look for final checkpoint
        final = self.ckpt_dir / "agent_final.pt"
        if final.exists():
            return final
        return None

    def auto_resume(self) -> bool:
        """Automatically find and load the latest checkpoint. Returns True if resumed."""
        latest = self.find_latest_checkpoint()
        if latest is not None:
            print(f"[Trainer] Auto-resume: found checkpoint {latest}")
            self.load_checkpoint(str(latest))
            return True
        return False

    def get_temperature(self) -> float:
        """Compute exploration temperature for current epoch."""
        cfg = self.cfg
        if self.epoch >= cfg.temperature_decay_epochs:
            return cfg.temperature_end
        frac = self.epoch / cfg.temperature_decay_epochs
        return cfg.temperature_start + frac * (cfg.temperature_end - cfg.temperature_start)

    def train(self):
        """Main training loop."""
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f"  5D Chess ML Training")
        print(f"  Variant: {cfg.variant}")
        print(f"  Board: {cfg.board_size_x}x{cfg.board_size_y}")
        print(f"  Epochs: {cfg.num_epochs}")
        print(f"  Games/epoch: {cfg.num_games_per_epoch}")
        print(f"{'='*60}\n")

        start_epoch = self.epoch
        for epoch in range(start_epoch, cfg.num_epochs):
            self.epoch = epoch
            t0 = time.time()

            temperature = self.get_temperature()

            # ---- Phase 1: Self-play ----
            print(f"\n--- Epoch {epoch+1}/{cfg.num_epochs} (T={temperature:.2f}) ---")
            print("[Self-play]")
            
            # Always use the agent (even epoch 0), with epsilon-greedy exploration
            # Early epochs use high epsilon for exploration, decaying over time
            epsilon = max(0.05, 0.5 * (1.0 - epoch / max(cfg.num_epochs * 0.5, 1)))
            worker = SelfPlayWorker(self.agent, cfg, use_agent=True, epsilon=epsilon)
            samples = worker.generate_training_data(
                num_games=cfg.num_games_per_epoch,
                temperature=temperature,
            )

            self.total_games += cfg.num_games_per_epoch
            self.total_samples += len(samples)

            # Compute win-rate statistics from samples
            white_wins = sum(1 for s in samples if s['return'] > 0 and s['player'] == 0)
            black_wins = sum(1 for s in samples if s['return'] > 0 and s['player'] == 1)
            draws = sum(1 for s in samples if s['return'] == 0) // max(1, len(set(id(s['state']) for s in samples)))
            total_steps = len(samples)
            unique_games = cfg.num_games_per_epoch

            if len(samples) == 0:
                print("  WARNING: No training samples generated!")
                continue

            # ---- Phase 2: Training ----
            print(f"[Training] {len(samples)} samples")
            metrics = self._train_on_samples(samples)

            # LR schedule step
            self.scheduler.step()

            elapsed = time.time() - t0

            # ---- Phase 3: Logging ----
            log_entry = {
                'epoch': epoch + 1,
                'temperature': temperature,
                'epsilon': round(epsilon, 3),
                'num_samples': len(samples),
                'total_games': self.total_games,
                'lr': self.optimizer.param_groups[0]['lr'],
                'elapsed_sec': round(elapsed, 1),
                **{k: round(v, 6) for k, v in metrics.items()},
            }

            if (epoch + 1) % cfg.log_interval == 0:
                print(f"  Loss: {metrics['total_loss']:.4f} "
                      f"(policy={metrics['policy_loss']:.4f}, "
                      f"value={metrics['value_loss']:.4f}, "
                      f"entropy={metrics['entropy']:.4f}) "
                      f"eps={epsilon:.2f} "
                      f"[{elapsed:.1f}s]")

            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            # ---- Phase 4: Checkpoint ----
            if (epoch + 1) % cfg.save_interval == 0:
                self._save_checkpoint(epoch + 1)

        print(f"\n{'='*60}")
        print(f"  Training complete! {cfg.num_epochs} epochs, {self.total_games} games")
        print(f"{'='*60}")

        # Final save
        self._save_checkpoint(cfg.num_epochs, final=True)

    def _train_on_samples(self, samples: list[dict]) -> dict[str, float]:
        """
        Train on collected self-play samples.
        
        Uses the actual game states stored during self-play (not reconstructed from PGN).
        Each sample contains: state (engine.state copy), per_tl_moves, chosen_indices, return.
        
        Returns:
            dict of average metrics
        """
        self.agent.train()
        cfg = self.cfg
        device = self.device

        # Shuffle samples
        random.shuffle(samples)

        # Process in mini-batches (state-by-state, since graph sizes vary)
        total_metrics = {'total_loss': 0, 'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
        num_processed = 0
        max_grad_norm_seen = 0.0

        # Accumulate gradients over mini-batch
        self.optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)
        batch_count = 0

        for sample_idx, sample in enumerate(samples):
            # Use the actual game state stored during self-play
            state = sample['state']           # actual engine.state copy
            per_tl_moves = sample['per_tl_moves']  # live per_tl_moves from that step

            if per_tl_moves is None or len(per_tl_moves) == 0:
                continue

            # Encode the ACTUAL game state
            try:
                board_embeds_dict, global_embed, node_embeds_dict, _ = self.agent.encode_state(state)
            except Exception as e:
                if sample_idx < 3:  # only warn for first few
                    print(f"  WARNING: encode_state failed: {e}")
                continue

            # Value prediction + loss
            value_pred = self.agent.value_head(global_embed.unsqueeze(0)).squeeze()
            target_return = torch.tensor(sample['return'], dtype=torch.float32, device=device)
            value_loss = (value_pred - target_return) ** 2

            # Policy: use stored chosen_indices with the ACTUAL per_tl_moves from that step
            chosen_indices = sample['chosen_indices']
            
            # Clamp indices to valid range (should already be correct, but safety check)
            adj_indices = []
            for tl_idx, tl in enumerate(per_tl_moves):
                if tl_idx < len(chosen_indices):
                    idx = chosen_indices[tl_idx]
                    idx = max(0, min(idx, len(tl.moves) - 1)) if len(tl.moves) > 0 else -1
                    adj_indices.append(idx)
                else:
                    adj_indices.append(0 if len(tl.moves) > 0 else -1)

            log_prob = self.agent.policy_head.compute_action_logprob(
                global_embed, node_embeds_dict, per_tl_moves, adj_indices, device
            )

            # Clamp log_prob to prevent extreme values (stabilization)
            log_prob = log_prob.clamp(min=-20.0, max=0.0)

            advantage = target_return - value_pred.detach()
            # Clamp advantage to prevent extreme policy gradients
            advantage = advantage.clamp(-2.0, 2.0)
            policy_loss = -(log_prob * advantage)

            # Entropy (recompute for regularization)
            with torch.no_grad():
                _, _, entropy = self.agent.policy_head.sample_action(
                    global_embed.detach(), node_embeds_dict, per_tl_moves, 1.0, device
                )

            # Combined loss for this sample
            loss = (cfg.policy_loss_weight * policy_loss +
                    cfg.value_loss_weight * value_loss -
                    cfg.entropy_bonus * entropy)

            batch_loss = batch_loss + loss
            batch_count += 1

            # Record metrics
            total_metrics['total_loss'] += loss.item()
            total_metrics['policy_loss'] += policy_loss.item()
            total_metrics['value_loss'] += value_loss.item()
            total_metrics['entropy'] += entropy.item()
            num_processed += 1

            # Step optimizer every batch_size samples
            if batch_count >= cfg.batch_size:
                (batch_loss / batch_count).backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
                max_grad_norm_seen = max(max_grad_norm_seen, grad_norm.item())
                self.optimizer.step()
                self.optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=device)
                batch_count = 0

        # Final partial batch
        if batch_count > 0:
            (batch_loss / batch_count).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
            max_grad_norm_seen = max(max_grad_norm_seen, grad_norm.item())
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Average metrics
        if num_processed > 0:
            for k in total_metrics:
                total_metrics[k] /= num_processed
        
        total_metrics['grad_norm'] = max_grad_norm_seen
        return total_metrics

    def _save_checkpoint(self, epoch: int, final: bool = False):
        """Save model checkpoint."""
        tag = "final" if final else f"epoch_{epoch:04d}"
        path = self.ckpt_dir / f"agent_{tag}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.cfg,
            'total_games': self.total_games,
            'total_samples': self.total_samples,
        }, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.epoch = ckpt['epoch']  # train() will start from this epoch
        self.total_games = ckpt.get('total_games', 0)
        self.total_samples = ckpt.get('total_samples', 0)
        print(f"  Loaded checkpoint: {path} (epoch {self.epoch})")

    def load_checkpoint_transfer(self, path: str):
        """
        Load model weights from a checkpoint trained with a different variant.
        
        Only loads model weights (with automatic adaptation for board_size changes).
        Optimizer and scheduler are NOT loaded (fresh training state).
        Epoch counter resets to 0.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        src_cfg = ckpt.get('config', None)

        src_variant = src_cfg.variant if src_cfg else 'unknown'
        src_board = f"{src_cfg.board_size_x}x{src_cfg.board_size_y}" if src_cfg else '?'
        tgt_board = f"{self.cfg.board_size_x}x{self.cfg.board_size_y}"

        print(f"  [Transfer] Source: {src_variant} ({src_board})")
        print(f"  [Transfer] Target: {self.cfg.variant} ({tgt_board})")

        adapted, skipped, missing = self.agent.load_state_dict_transfer(
            ckpt['model_state_dict']
        )

        # Reset training state (fresh optimizer for new variant)
        self.epoch = 0
        self.total_games = 0
        self.total_samples = 0
        print(f"  [Transfer] Model loaded from {path} — epoch reset to 0")
