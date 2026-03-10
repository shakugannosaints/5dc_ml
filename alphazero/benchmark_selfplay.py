"""
Benchmark C++ ONNX self-play throughput across providers and task settings.

Usage example:
  python -m alphazero.benchmark_selfplay --variant very_small --games 16 --cases cpu:1:1 cuda:1:1 cuda:1:8
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig
from .network import AlphaZeroNetwork
from .self_play import CppOnnxSelfPlayWorker


def _parse_case(spec: str) -> dict:
    parts = spec.split(":")
    if len(parts) not in {3, 4}:
        raise ValueError(f"Invalid case '{spec}'. Expected provider:workers:task_games[:ort_threads].")
    provider = parts[0].strip().lower()
    if provider not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported provider in case '{spec}'.")
    workers = max(1, int(parts[1]))
    task_games = max(1, int(parts[2]))
    ort_threads = max(1, int(parts[3])) if len(parts) == 4 else 1
    return {
        "provider": provider,
        "workers": workers,
        "task_games": task_games,
        "ort_threads": ort_threads,
        "label": f"{provider}-w{workers}-tg{task_games}-ort{ort_threads}",
    }


def _load_network(cfg: TrainConfig, checkpoint: str | None) -> AlphaZeroNetwork:
    device = torch.device("cpu")
    network = AlphaZeroNetwork(cfg.network).to(device)
    if checkpoint:
        if checkpoint == "latest":
            latest = Path(cfg.checkpoint_dir) / "latest.pt"
            if latest.exists():
                checkpoint = str(latest)
            else:
                checkpoint = None
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            network.load_state_dict(ckpt["model_state_dict"])
    network.eval()
    return network


def run_case(base_cfg: TrainConfig, network: AlphaZeroNetwork, case: dict, num_games: int, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)

    cfg = replace(base_cfg)
    cfg.self_play = replace(base_cfg.self_play)
    cfg.mcts = replace(base_cfg.mcts)
    cfg.self_play.self_play_backend = "cpp_onnx"
    cfg.self_play.cpp_onnx_provider = case["provider"]
    cfg.self_play.num_workers = case["workers"]
    cfg.self_play.worker_task_games = case["task_games"]
    cfg.self_play.cpp_onnx_ort_threads = case["ort_threads"]
    cfg.self_play.num_games = num_games

    worker = CppOnnxSelfPlayWorker(
        network=network,
        mcts_cfg=cfg.mcts,
        sp_cfg=cfg.self_play,
        device=torch.device("cpu"),
        train_cfg=cfg,
    )

    t0 = time.perf_counter()
    games = worker.generate_games(num_games=num_games)
    elapsed = time.perf_counter() - t0

    total_semimoves = sum(g.total_semimoves for g in games)
    total_samples = sum(len(g.samples) for g in games)
    return {
        "label": case["label"],
        "provider": case["provider"],
        "workers": case["workers"],
        "task_games": case["task_games"],
        "ort_threads": case["ort_threads"],
        "games": len(games),
        "elapsed_sec": round(elapsed, 3),
        "games_per_sec": round(len(games) / max(elapsed, 1e-9), 3),
        "semimoves": total_semimoves,
        "semimoves_per_sec": round(total_semimoves / max(elapsed, 1e-9), 3),
        "samples": total_samples,
        "samples_per_sec": round(total_samples / max(elapsed, 1e-9), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark C++ ONNX self-play throughput")
    parser.add_argument("--variant", type=str, default="very_small")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--sims", type=int, default=32)
    parser.add_argument("--min-board-limit", type=int, default=None)
    parser.add_argument("--max-board-limit", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint path or 'latest'")
    parser.add_argument("--seed", type=int, default=12345,
                        help="Base RNG seed for reproducible case comparisons")
    parser.add_argument("--cases", nargs="+", required=True,
                        help="Cases like cpu:1:1 or cuda:1:8:1")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.apply_variant(args.variant)
    cfg.self_play.self_play_backend = "cpp_onnx"
    cfg.self_play.log_worker_task_stats = False
    cfg.mcts.num_simulations = args.sims
    if args.min_board_limit is not None:
        cfg.self_play.min_board_limit = max(1, int(args.min_board_limit))
    if args.max_board_limit is not None:
        cfg.self_play.max_board_limit = max(cfg.self_play.min_board_limit, int(args.max_board_limit))

    if args.variant:
        cfg.checkpoint_dir = str(Path(cfg.checkpoint_dir) / cfg.variant_name)
        cfg.log_dir = str(Path(cfg.log_dir) / cfg.variant_name)
        cfg.self_play.cpp_onnx_model_path = str(Path(cfg.checkpoint_dir) / "selfplay_fp16.onnx")

    network = _load_network(cfg, args.checkpoint)
    results = []
    for idx, spec in enumerate(args.cases):
        case = _parse_case(spec)
        result = run_case(cfg, network, case, args.games, args.seed + idx * 9973)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
