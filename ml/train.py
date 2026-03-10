# ml/train.py
"""
Entry point for training the 5D Chess ML agent.

Usage:
    cd ml/
    python train.py                    # default: Very Small variant
    python train.py --variant standard # use standard chess variant
    python train.py --resume checkpoints/agent_epoch_0010.pt
    python train.py --cpu              # force CPU
    python train.py --epochs 100 --games 4 --lr 0.001
"""

import sys
import os
import argparse

# Ensure ml/ is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.config import TrainingConfig, SMALL_CONFIG, STANDARD_CONFIG
from ml.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="5D Chess ML Training")
    parser.add_argument("--variant", type=str, default="small",
                        choices=["small", "standard"],
                        help="Game variant (default: small)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--games", type=int, default=None,
                        help="Number of self-play games per epoch")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU device")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (auto-detects latest if omitted)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh training, ignore existing checkpoints")
    parser.add_argument("--transfer", type=str, default=None,
                        help="Transfer learning: load weights from a different variant's checkpoint "
                             "(e.g. small→standard). Only model weights are loaded, optimizer resets.")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Fixed exploration temperature (overrides schedule)")
    parser.add_argument("--action-samples", type=int, default=None,
                        help="Max number of actions to sample from HC search")
    return parser.parse_args()


def main():
    args = parse_args()

    # Select base config
    if args.variant == "standard":
        cfg = STANDARD_CONFIG
    else:
        cfg = SMALL_CONFIG

    # Apply overrides
    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.games is not None:
        cfg.num_games_per_epoch = args.games
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.cpu:
        cfg.device = "cpu"
    if args.action_samples is not None:
        cfg.num_action_samples = args.action_samples
    if args.temperature is not None:
        cfg.temperature_start = args.temperature
        cfg.temperature_end = args.temperature

    # Create trainer
    trainer = Trainer(cfg)

    # Resume: explicit path > transfer > auto-detect > fresh start
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif args.transfer:
        trainer.load_checkpoint_transfer(args.transfer)
    elif not args.no_resume:
        if trainer.auto_resume():
            print(f"[Train] Resumed from epoch {trainer.epoch}")
        else:
            print("[Train] No checkpoint found, starting fresh.")

    # Train!
    trainer.train()


if __name__ == "__main__":
    main()
