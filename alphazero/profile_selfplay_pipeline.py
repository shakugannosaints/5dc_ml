"""
Detailed profiler for the current self-play pipeline.

This script is meant to answer "where is self-play spending time?" for the
current C++ ONNX backend by combining:

1. End-to-end timing of generate_games()
2. Inclusive per-method timing inside generate_games()
3. Focused microbenchmarks for cold-start tasks, persistent workers, and
   binary deserialization

Example:
  python -m alphazero.profile_selfplay_pipeline ^
      --variant standard_turn_zero ^
      --games 8 ^
      --sims 256 ^
      --provider cpu ^
      --workers 8 ^
      --task-games 2 ^
      --repeat 3 ^
      --output alphazero/logs/selfplay_profile.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import statistics
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch

from .config import TrainConfig
from .network import AlphaZeroNetwork
from .self_play import CppOnnxSelfPlayWorker, GameRecord


def _load_network(cfg: TrainConfig, checkpoint: str | None) -> AlphaZeroNetwork:
    device = torch.device("cpu")
    network = AlphaZeroNetwork(cfg.network).to(device)
    if checkpoint:
        if checkpoint.lower() in {"none", "random"}:
            checkpoint = None
        if checkpoint is None:
            network.eval()
            return network
        ckpt_path = checkpoint
        if checkpoint == "latest":
            latest = Path(cfg.checkpoint_dir) / "latest.pt"
            if latest.exists():
                ckpt_path = str(latest)
            else:
                ckpt_path = None
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            network.load_state_dict(ckpt["model_state_dict"])
    network.eval()
    return network


def _game_metrics(games: list[GameRecord], elapsed_sec: float) -> dict[str, Any]:
    total_semimoves = sum(g.total_semimoves for g in games)
    total_samples = sum(len(g.samples) for g in games)
    reasons = Counter(g.terminal_reason for g in games)
    return {
        "games": len(games),
        "elapsed_sec": round(elapsed_sec, 6),
        "games_per_sec": round(len(games) / max(elapsed_sec, 1e-9), 6),
        "semimoves": total_semimoves,
        "semimoves_per_sec": round(total_semimoves / max(elapsed_sec, 1e-9), 6),
        "samples": total_samples,
        "samples_per_sec": round(total_samples / max(elapsed_sec, 1e-9), 6),
        "avg_semimoves": round(total_semimoves / max(len(games), 1), 6),
        "termination_reasons": dict(sorted(reasons.items())),
    }


def _series_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"calls": 0, "total_sec": 0.0, "avg_sec": 0.0, "min_sec": 0.0, "max_sec": 0.0}
    return {
        "calls": len(values),
        "total_sec": round(sum(values), 6),
        "avg_sec": round(statistics.fmean(values), 6),
        "min_sec": round(min(values), 6),
        "max_sec": round(max(values), 6),
    }


class ProfilingCppOnnxSelfPlayWorker(CppOnnxSelfPlayWorker):
    """CppOnnxSelfPlayWorker with per-method inclusive timings."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_times: dict[str, list[float]] = {}

    def _record(self, name: str, elapsed_sec: float) -> None:
        self.phase_times.setdefault(name, []).append(elapsed_sec)

    def phase_summary(self) -> dict[str, dict[str, float]]:
        return {name: _series_stats(values) for name, values in sorted(self.phase_times.items())}

    def _prepare_runtime_binaries(self, exe_path: Path) -> None:
        t0 = time.perf_counter()
        try:
            return super()._prepare_runtime_binaries(exe_path)
        finally:
            self._record("prepare_runtime_binaries", time.perf_counter() - t0)

    def _export_model(self) -> tuple[Path, str]:
        t0 = time.perf_counter()
        try:
            return super()._export_model()
        finally:
            self._record("export_model", time.perf_counter() - t0)

    def _resolve_provider(
        self,
        exe_path: Path,
        model_path: Path,
        requested_provider: str,
        ort_threads: int,
        cuda_device_id: int,
    ) -> str:
        t0 = time.perf_counter()
        try:
            return super()._resolve_provider(
                exe_path=exe_path,
                model_path=model_path,
                requested_provider=requested_provider,
                ort_threads=ort_threads,
                cuda_device_id=cuda_device_id,
            )
        finally:
            self._record("resolve_provider", time.perf_counter() - t0)

    def _run_task(self, *args, **kwargs) -> list[GameRecord]:
        t0 = time.perf_counter()
        try:
            return super()._run_task(*args, **kwargs)
        finally:
            self._record("run_task", time.perf_counter() - t0)

    def _run_task_bucket(self, *args, **kwargs) -> list[GameRecord]:
        t0 = time.perf_counter()
        try:
            return super()._run_task_bucket(*args, **kwargs)
        finally:
            self._record("run_task_bucket", time.perf_counter() - t0)

    def _start_persistent_worker(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return super()._start_persistent_worker(*args, **kwargs)
        finally:
            self._record("start_persistent_worker", time.perf_counter() - t0)

    def _run_persistent_task(self, *args, **kwargs) -> list[GameRecord]:
        t0 = time.perf_counter()
        try:
            return super()._run_persistent_task(*args, **kwargs)
        finally:
            self._record("run_persistent_task", time.perf_counter() - t0)

    def _stop_persistent_worker(self, process) -> None:
        t0 = time.perf_counter()
        try:
            return super()._stop_persistent_worker(process)
        finally:
            self._record("stop_persistent_worker", time.perf_counter() - t0)

    def _load_games_from_binary(self, path: Path) -> list[GameRecord]:
        t0 = time.perf_counter()
        try:
            return super()._load_games_from_binary(path)
        finally:
            self._record("load_games_from_binary", time.perf_counter() - t0)


def _build_cfg(args, temp_model_path: Path) -> TrainConfig:
    cfg = TrainConfig()
    cfg.apply_variant(args.variant)
    cfg.device = "cpu"
    cfg.self_play.self_play_backend = "cpp_onnx"
    cfg.self_play.cpp_onnx_provider = args.provider
    cfg.self_play.num_workers = max(1, int(args.workers))
    cfg.self_play.worker_task_games = max(1, int(args.task_games))
    cfg.self_play.cpp_onnx_ort_threads = max(0, int(args.ort_threads))
    cfg.self_play.cpp_onnx_cuda_device_id = max(0, int(args.cuda_device_id))
    cfg.self_play.cpp_onnx_model_precision = args.precision
    cfg.self_play.cpp_onnx_model_path = str(temp_model_path)
    cfg.self_play.log_worker_task_stats = False
    cfg.self_play.num_games = max(1, int(args.games))
    cfg.self_play.min_board_limit = int(args.min_board_limit)
    cfg.self_play.max_board_limit = int(args.max_board_limit)
    cfg.self_play.max_game_length = int(args.max_game_length)
    cfg.mcts.num_simulations = int(args.sims)
    cfg.mcts.leaf_batch_size = max(1, int(args.leaf_batch_size))
    if args.variant:
        cfg.checkpoint_dir = str(Path(cfg.checkpoint_dir) / cfg.variant_name)
        cfg.log_dir = str(Path(cfg.log_dir) / cfg.variant_name)
    return cfg


def _make_worker(cfg: TrainConfig, network: AlphaZeroNetwork) -> ProfilingCppOnnxSelfPlayWorker:
    return ProfilingCppOnnxSelfPlayWorker(
        network=network,
        mcts_cfg=cfg.mcts,
        sp_cfg=cfg.self_play,
        device=torch.device("cpu"),
        train_cfg=cfg,
    )


def _microbench(
    worker: ProfilingCppOnnxSelfPlayWorker,
    cfg: TrainConfig,
    exe_path: Path,
    repeat: int,
    micro_games: int,
) -> dict[str, Any]:
    worker.phase_times = {}
    worker._prepare_runtime_binaries(exe_path)
    model_path, _ = worker._export_model()
    requested_provider = cfg.self_play.cpp_onnx_provider
    probe_threads = worker._resolve_ort_threads(requested_provider, cfg.self_play.num_workers)
    provider = worker._resolve_provider(
        exe_path=exe_path,
        model_path=model_path,
        requested_provider=requested_provider,
        ort_threads=probe_threads,
        cuda_device_id=cfg.self_play.cpp_onnx_cuda_device_id,
    )
    ort_threads = worker._resolve_ort_threads(provider, cfg.self_play.num_workers)

    cold_runs: list[dict[str, Any]] = []
    persistent_runs: list[dict[str, Any]] = []
    deserialize_runs: list[float] = []

    with TemporaryDirectory(prefix="selfplay_profile_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        for idx in range(repeat):
            output_path = tmp_dir / f"cold_{idx:02d}.bin"
            t0 = time.perf_counter()
            games = worker._run_task(
                exe_path=exe_path,
                model_path=model_path,
                output_path=output_path,
                num_games=micro_games,
                seed=1000 + idx,
                ort_threads=ort_threads,
                provider=provider,
                cuda_device_id=cfg.self_play.cpp_onnx_cuda_device_id,
                log_task_stats=False,
            )
            elapsed = time.perf_counter() - t0
            cold_runs.append(_game_metrics(games, elapsed))

            t0 = time.perf_counter()
            _ = worker._load_games_from_binary(output_path)
            deserialize_runs.append(time.perf_counter() - t0)

        startup_times: list[float] = []
        shutdown_times: list[float] = []
        process = None
        try:
            for idx in range(repeat):
                t0 = time.perf_counter()
                process = worker._start_persistent_worker(
                    exe_path=exe_path,
                    model_path=model_path,
                    ort_threads=ort_threads,
                    provider=provider,
                    cuda_device_id=cfg.self_play.cpp_onnx_cuda_device_id,
                )
                startup_times.append(time.perf_counter() - t0)

                output_path = tmp_dir / f"persistent_{idx:02d}.bin"
                t0 = time.perf_counter()
                games = worker._run_persistent_task(
                    process=process,
                    output_path=output_path,
                    num_games=micro_games,
                    seed=2000 + idx,
                    provider=provider,
                    log_task_stats=False,
                    exe_name=exe_path.name,
                )
                elapsed = time.perf_counter() - t0
                persistent_runs.append(_game_metrics(games, elapsed))

                t0 = time.perf_counter()
                worker._stop_persistent_worker(process)
                shutdown_times.append(time.perf_counter() - t0)
                process = None
        finally:
            if process is not None:
                worker._stop_persistent_worker(process)

    cold_elapsed = [run["elapsed_sec"] for run in cold_runs]
    persistent_elapsed = [run["elapsed_sec"] for run in persistent_runs]
    result = {
        "provider_after_probe": provider,
        "ort_threads_after_resolve": ort_threads,
        "micro_games": micro_games,
        "cold_task": {
            "runs": cold_runs,
            "timing": _series_stats(cold_elapsed),
        },
        "persistent_worker": {
            "startup_timing": _series_stats(startup_times),
            "task_runs": persistent_runs,
            "task_timing": _series_stats(persistent_elapsed),
            "shutdown_timing": _series_stats(shutdown_times),
        },
        "deserialize_only": _series_stats(deserialize_runs),
    }
    if cold_elapsed and persistent_elapsed:
        result["estimated_cold_start_overhead_sec"] = round(
            max(0.0, statistics.fmean(cold_elapsed) - statistics.fmean(persistent_elapsed)),
            6,
        )
    return result


def _cpp_internal_profile(
    worker: ProfilingCppOnnxSelfPlayWorker,
    cfg: TrainConfig,
    exe_path: Path,
    num_games: int,
) -> dict[str, Any]:
    worker.phase_times = {}
    worker._prepare_runtime_binaries(exe_path)
    model_path, _ = worker._export_model()
    requested_provider = cfg.self_play.cpp_onnx_provider
    probe_threads = worker._resolve_ort_threads(requested_provider, cfg.self_play.num_workers)
    provider = worker._resolve_provider(
        exe_path=exe_path,
        model_path=model_path,
        requested_provider=requested_provider,
        ort_threads=probe_threads,
        cuda_device_id=cfg.self_play.cpp_onnx_cuda_device_id,
    )
    ort_threads = worker._resolve_ort_threads(provider, cfg.self_play.num_workers)

    with TemporaryDirectory(prefix="cpp_internal_profile_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        output_path = tmp_dir / "profile_games.bin"
        profile_path = tmp_dir / "profile.json"
        cmd = [
            str(exe_path),
            "--model", str(model_path),
            "--variant", str(cfg.variant_name),
            "--games", str(num_games),
            "--sims", str(cfg.mcts.num_simulations),
            "--leaf-batch-size", str(max(1, int(getattr(cfg.mcts, "leaf_batch_size", 1)))),
            "--min-board-limit", str(cfg.self_play.min_board_limit),
            "--max-board-limit", str(cfg.self_play.max_board_limit),
            "--material-scale", str(cfg.self_play.material_scale),
            "--max-game-length", str(cfg.self_play.max_game_length),
            "--temperature", str(cfg.self_play.temperature),
            "--temperature-final", str(cfg.self_play.temperature_final),
            "--temperature-threshold", str(cfg.self_play.temp_threshold),
            "--c-puct", str(cfg.mcts.c_puct),
            "--dirichlet-alpha", str(cfg.mcts.dirichlet_alpha),
            "--dirichlet-epsilon", str(cfg.mcts.dirichlet_epsilon),
            "--provider", provider,
            "--cuda-device-id", str(cfg.self_play.cpp_onnx_cuda_device_id),
            "--ort-threads", str(ort_threads),
            "--seed", "424242",
            "--output-data", str(output_path),
            "--profile-json", str(profile_path),
            "--quiet",
        ]
        t0 = time.perf_counter()
        completed = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            env=worker._build_subprocess_env(provider),
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        if completed.returncode != 0:
            raise RuntimeError(
                f"C++ internal profiling run failed (exit={completed.returncode}).\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        data["wall_clock_elapsed_sec"] = round(elapsed, 6)
        data["provider_after_probe"] = provider
        data["ort_threads_after_resolve"] = ort_threads
        return data


def _end_to_end_profile(
    worker: ProfilingCppOnnxSelfPlayWorker,
    num_games: int,
) -> dict[str, Any]:
    worker.phase_times = {}
    t0 = time.perf_counter()
    games = worker.generate_games(num_games=num_games)
    elapsed = time.perf_counter() - t0
    return {
        "metrics": _game_metrics(games, elapsed),
        "inclusive_phase_timing": worker.phase_summary(),
    }


def _print_console_summary(result: dict[str, Any]) -> None:
    end_to_end = result["end_to_end"]
    print("\n=== End-to-End ===")
    print(json.dumps(end_to_end["metrics"], indent=2, ensure_ascii=False))

    print("\n=== Inclusive Phase Timing ===")
    for name, stats in sorted(
        end_to_end["inclusive_phase_timing"].items(),
        key=lambda item: item[1]["total_sec"],
        reverse=True,
    ):
        print(
            f"{name:28s} calls={stats['calls']:2d} "
            f"total={stats['total_sec']:.3f}s avg={stats['avg_sec']:.3f}s"
        )

    micro = result["microbench"]
    print("\n=== Microbench Summary ===")
    print(
        f"provider_after_probe={micro['provider_after_probe']} "
        f"ort_threads_after_resolve={micro.get('ort_threads_after_resolve', result['config'].get('ort_threads', 0))} "
        f"cold_avg={micro['cold_task']['timing']['avg_sec']:.3f}s "
        f"persistent_start_avg={micro['persistent_worker']['startup_timing']['avg_sec']:.3f}s "
        f"persistent_task_avg={micro['persistent_worker']['task_timing']['avg_sec']:.3f}s "
        f"deserialize_avg={micro['deserialize_only']['avg_sec']:.3f}s"
    )
    if "estimated_cold_start_overhead_sec" in micro:
        print(
            "estimated_cold_start_overhead_sec="
            f"{micro['estimated_cold_start_overhead_sec']:.3f}s"
        )

    cpp_internal = result["cpp_internal_profile"]
    print("\n=== C++ Internal Timing ===")
    ordered = sorted(
        cpp_internal["timing"].items(),
        key=lambda item: item[1]["total_sec"],
        reverse=True,
    )
    for name, stats in ordered[:10]:
        print(
            f"{name:28s} calls={stats['calls']:6d} "
            f"total={stats['total_sec']:.3f}s avg={stats['avg_sec']:.6f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the detailed self-play pipeline")
    parser.add_argument("--variant", type=str, default="standard_turn_zero")
    parser.add_argument("--checkpoint", type=str, default="latest",
                        help="Checkpoint path, 'latest', or 'none' for random weights")
    parser.add_argument("--games", type=int, default=8,
                        help="Games for the end-to-end generate_games profile")
    parser.add_argument("--micro-games", type=int, default=None,
                        help="Games per microbench task; defaults to min(games, task_games)")
    parser.add_argument("--sims", type=int, default=128)
    parser.add_argument("--leaf-batch-size", type=int, default=4)
    parser.add_argument("--provider", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "fp16", "fp32"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--task-games", type=int, default=2)
    parser.add_argument("--ort-threads", type=int, default=0,
                        help="ORT intra-op threads for C++ self-play; 0 means auto")
    parser.add_argument("--cuda-device-id", type=int, default=0)
    parser.add_argument("--min-board-limit", type=int, default=80)
    parser.add_argument("--max-board-limit", type=int, default=120)
    parser.add_argument("--max-game-length", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=3,
                        help="How many times to repeat the microbench probes")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON output path")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)

    with TemporaryDirectory(prefix="selfplay_profile_model_") as tmp_model_dir_str:
        tmp_model_path = Path(tmp_model_dir_str) / f"selfplay_profile_{args.precision}.onnx"
        cfg = _build_cfg(args, tmp_model_path)
        network = _load_network(cfg, args.checkpoint)

        worker = _make_worker(cfg, network)
        exe_path = Path(cfg.self_play.cpp_selfplay_executable)
        if not exe_path.is_absolute():
            exe_path = Path.cwd() / exe_path
        if not exe_path.exists():
            raise FileNotFoundError(
                f"C++ self-play executable not found: {exe_path}. "
                "Build az_selfplay_onnx first."
            )

        micro_games = args.micro_games
        if micro_games is None:
            micro_games = min(max(1, args.games), max(1, args.task_games))

        microbench = _microbench(
            worker=worker,
            cfg=cfg,
            exe_path=exe_path,
            repeat=max(1, int(args.repeat)),
            micro_games=max(1, int(micro_games)),
        )
        cpp_internal_profile = _cpp_internal_profile(
            worker=worker,
            cfg=cfg,
            exe_path=exe_path,
            num_games=max(1, int(args.games)),
        )
        end_to_end = _end_to_end_profile(worker, num_games=max(1, int(args.games)))

        result = {
            "config": {
                "variant": cfg.variant_name,
                "checkpoint": args.checkpoint,
                "games": int(args.games),
                "micro_games": int(micro_games),
                "sims": int(args.sims),
                "leaf_batch_size": int(args.leaf_batch_size),
                "provider_requested": args.provider,
                "precision": args.precision,
                "workers": int(args.workers),
                "task_games": int(args.task_games),
                "ort_threads": int(args.ort_threads),
                "cuda_device_id": int(args.cuda_device_id),
                "min_board_limit": int(args.min_board_limit),
                "max_board_limit": int(args.max_board_limit),
                "max_game_length": int(args.max_game_length),
                "repeat": int(args.repeat),
                "executable": str(exe_path),
            },
            "microbench": microbench,
            "cpp_internal_profile": cpp_internal_profile,
            "end_to_end": end_to_end,
        }

        _print_console_summary(result)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
