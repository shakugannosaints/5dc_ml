# alphazero/self_play.py
"""
Self-play module for semimove-level AlphaZero training.

Each self-play game:
  1. Rolls a random board_limit in [min_board_limit, max_board_limit]
  2. Plays semimoves using MCTS until terminal
  3. Records (state_encoding, aligned_policy_target, value_target)
"""

import math
import os
import random
import time
import logging
import gc
import shutil
import struct
import subprocess
import sys
import numpy as np
import torch
import multiprocessing as mp
import concurrent.futures as cf
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from dataclasses import dataclass, field

from .config import MCTSConfig, SelfPlayConfig
from .env import SemimoveEnv, Semimove
from .mcts import MCTS, SUBMIT_ACTION

logger = logging.getLogger("alphazero.self_play")
_TORCH_WORKER_THREADS_CONFIGURED = False


@dataclass
class WorkerTaskStats:
    """Telemetry for one worker task."""

    pid: int
    num_games: int
    elapsed_sec: float
    rss_mb_before: Optional[float]
    rss_mb_after: Optional[float]
    rss_mb_peak: Optional[float]
    total_semimoves: int
    total_samples: int


@dataclass
class SemimoveRecord:
    """Single training sample from self-play."""

    board_planes: np.ndarray      # [N, 14, 16]
    last_move_markers: np.ndarray # [N, 16]
    l_coords: np.ndarray          # [N]
    t_coords: np.ndarray          # [N]
    urgency: float                # scalar
    padding_mask: np.ndarray      # [N] bool

    # Policy target aligned to explicit legal action list
    policy_target: np.ndarray         # [A]
    action_board_indices: np.ndarray  # [A], -1 for submit
    action_from_squares: np.ndarray   # [A]
    action_to_squares: np.ndarray     # [A]
    action_delta_t: np.ndarray        # [A]
    action_delta_l: np.ndarray        # [A]
    action_is_submit: np.ndarray      # [A] bool

    # Value target from current player's perspective
    value_target: float


@dataclass
class MoveEntry:
    """One recorded action in a game."""

    player: int
    action_type: str
    semimove: Optional[Semimove] = None
    ext_move_str: str = ""
    mcts_value: float = 0.0
    board_count: int = 0


@dataclass
class GameRecord:
    """A complete self-play game."""

    samples: list[SemimoveRecord] = field(default_factory=list)
    move_history: list[MoveEntry] = field(default_factory=list)
    outcome: float = 0.0
    total_semimoves: int = 0
    board_limit: int = 0
    terminal_reason: str = ""
    final_board_str: str = ""
    final_fen: str = ""
    pgn: str = ""


def _play_games_worker(payload) -> tuple[list[GameRecord], WorkerTaskStats]:
    """
    Process entrypoint for parallel self-play generation.
    """
    from .network import AlphaZeroNetwork
    from .config import NetworkConfig, MCTSConfig, SelfPlayConfig

    (state_dict_cpu, network_cfg_dict, mcts_cfg_dict, sp_cfg_dict, variant_pgn, num_games, seed, torch_threads) = payload

    global _TORCH_WORKER_THREADS_CONFIGURED

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(torch_threads)))
    if not _TORCH_WORKER_THREADS_CONFIGURED:
        torch.set_num_interop_threads(1)
        _TORCH_WORKER_THREADS_CONFIGURED = True

    device = torch.device("cpu")
    net = AlphaZeroNetwork(NetworkConfig(**network_cfg_dict)).to(device)
    net.load_state_dict(state_dict_cpu)
    net.eval()

    sp_cfg_local = SelfPlayConfig(**sp_cfg_dict)
    sp_cfg_local.num_workers = 1

    process = None
    try:
        import psutil
        process = psutil.Process(os.getpid())
    except Exception:
        process = None

    def rss_mb() -> Optional[float]:
        if process is None:
            return None
        try:
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return None

    worker = SelfPlayWorker(
        network=net,
        mcts_cfg=MCTSConfig(**mcts_cfg_dict),
        sp_cfg=sp_cfg_local,
        device=device,
        variant_pgn=variant_pgn,
    )

    games = []
    rss_before = rss_mb()
    rss_peak = rss_before
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_games):
            game = worker.play_game()
            games.append(game)
            gc.collect()
            current_rss = rss_mb()
            if current_rss is not None:
                if rss_peak is None:
                    rss_peak = current_rss
                else:
                    rss_peak = max(rss_peak, current_rss)

    stats = WorkerTaskStats(
        pid=os.getpid(),
        num_games=num_games,
        elapsed_sec=time.perf_counter() - t0,
        rss_mb_before=rss_before,
        rss_mb_after=rss_mb(),
        rss_mb_peak=rss_peak,
        total_semimoves=sum(g.total_semimoves for g in games),
        total_samples=sum(len(g.samples) for g in games),
    )
    return games, stats


class SelfPlayWorker:
    """Generates training data through self-play games."""

    URGENCY_ALPHA = 0.2  # exp(-alpha * boards_remaining)

    def __init__(
        self,
        network,
        mcts_cfg: MCTSConfig,
        sp_cfg: SelfPlayConfig,
        device: torch.device,
        variant_pgn: str,
    ):
        self.network = network
        self.mcts_cfg = mcts_cfg
        self.sp_cfg = sp_cfg
        self.device = device
        self.variant_pgn = variant_pgn

    @staticmethod
    def _entries_to_arrays(action_entries: list[dict]) -> dict[str, np.ndarray]:
        if not action_entries:
            return {
                "board_idx": np.zeros((0,), dtype=np.int64),
                "from_sq": np.zeros((0,), dtype=np.int64),
                "to_sq": np.zeros((0,), dtype=np.int64),
                "delta_t": np.zeros((0,), dtype=np.float32),
                "delta_l": np.zeros((0,), dtype=np.float32),
                "is_submit": np.zeros((0,), dtype=bool),
            }

        return {
            "board_idx": np.array([e["board_idx"] for e in action_entries], dtype=np.int64),
            "from_sq": np.array([e["from_sq"] for e in action_entries], dtype=np.int64),
            "to_sq": np.array([e["to_sq"] for e in action_entries], dtype=np.int64),
            "delta_t": np.array([e["delta_t"] for e in action_entries], dtype=np.float32),
            "delta_l": np.array([e["delta_l"] for e in action_entries], dtype=np.float32),
            "is_submit": np.array([e["is_submit"] for e in action_entries], dtype=bool),
        }

    def play_game(self) -> GameRecord:
        """Play a single self-play game and collect training samples."""
        record = GameRecord()

        board_limit = random.randint(self.sp_cfg.min_board_limit, self.sp_cfg.max_board_limit)
        record.board_limit = board_limit

        env = SemimoveEnv(
            self.variant_pgn,
            board_limit=board_limit,
            legal_cache_max_entries=self.sp_cfg.legal_cache_max_entries,
            rules_mode=self.sp_cfg.rules_mode,
        )
        env.reset()

        mcts = MCTS(self.network, self.mcts_cfg, self.device)

        player_at_step = []
        samples = []

        max_game_length = int(getattr(self.sp_cfg, "max_game_length", 0))
        use_max_game_length = max_game_length > 0
        terminated_by_no_action = False
        while not env.done and (not use_max_game_length or env.total_semimoves < max_game_length):
            boards_remaining = max(0, board_limit - self._count_boards(env))
            urgency = math.exp(-self.URGENCY_ALPHA * boards_remaining)

            if env.total_semimoves < self.sp_cfg.temp_threshold:
                temperature = self.sp_cfg.temperature
            else:
                temperature = self.sp_cfg.temperature_final

            action, policy_probs, root_value, action_entries = mcts.select_action(
                env, urgency=urgency, temperature=temperature
            )

            if action is None:
                terminated_by_no_action = True
                break

            encoded = env.encode_state(urgency=urgency)
            action_arrays = self._entries_to_arrays(action_entries)

            sample = SemimoveRecord(
                board_planes=encoded["board_planes"],
                last_move_markers=encoded["last_move_markers"],
                l_coords=encoded["l_coords"],
                t_coords=encoded["t_coords"],
                urgency=urgency,
                padding_mask=np.zeros(len(encoded["board_keys"]), dtype=bool),
                policy_target=np.array(policy_probs, dtype=np.float32),
                action_board_indices=action_arrays["board_idx"],
                action_from_squares=action_arrays["from_sq"],
                action_to_squares=action_arrays["to_sq"],
                action_delta_t=action_arrays["delta_t"],
                action_delta_l=action_arrays["delta_l"],
                action_is_submit=action_arrays["is_submit"],
                value_target=0.0,
            )
            samples.append(sample)
            player_at_step.append(env.current_player)

            player = env.current_player
            if action == SUBMIT_ACTION:
                record.move_history.append(
                    MoveEntry(
                        player=player,
                        action_type="submit",
                        mcts_value=root_value,
                        board_count=env.board_count,
                    )
                )
                outcome = env.submit_turn(assume_legal=True)
                if outcome is not None:
                    break
            else:
                em = action.to_ext_move()
                record.move_history.append(
                    MoveEntry(
                        player=player,
                        action_type="semimove",
                        semimove=action,
                        ext_move_str=em.to_string(),
                        mcts_value=root_value,
                        board_count=env.board_count,
                    )
                )
                env.apply_semimove(action)

        try:
            record.final_board_str = env.state.to_string()
            record.final_fen = env.state.show_fen()
        except Exception:
            pass
        try:
            record.pgn = env.game.show_pgn(engine.SHOW_CAPTURE | engine.SHOW_PROMOTION)
        except Exception:
            record.pgn = ""

        if env.done and env.outcome is not None:
            game_outcome = env.outcome  # white perspective
            if self.sp_cfg.rules_mode == "capture_king":
                record.terminal_reason = "capture_king_or_material"
            else:
                record.terminal_reason = "checkmate_or_material"
        elif terminated_by_no_action:
            game_outcome = 0.0
            record.terminal_reason = "no_legal_action"
        elif use_max_game_length and env.total_semimoves >= max_game_length:
            game_outcome = 0.0
            record.terminal_reason = "max_game_length"
        else:
            game_outcome = 0.0
            record.terminal_reason = "aborted"

        record.outcome = game_outcome
        record.total_semimoves = env.total_semimoves

        for i, sample in enumerate(samples):
            player = player_at_step[i]
            sample.value_target = game_outcome if player == 0 else -game_outcome

        record.samples = samples
        return record

    def generate_games(self, num_games: Optional[int] = None) -> list[GameRecord]:
        n = num_games or self.sp_cfg.num_games
        workers = max(1, int(getattr(self.sp_cfg, "num_workers", 1)))
        task_games = max(1, int(getattr(self.sp_cfg, "worker_task_games", 1)))
        max_tasks_per_child = max(1, int(getattr(self.sp_cfg, "worker_max_tasks_per_child", 8)))
        log_task_stats = bool(getattr(self.sp_cfg, "log_worker_task_stats", True))

        # Parallel self-play is process-based and runs worker inference on CPU.
        # It is valid even when the trainer/learner device is CUDA.
        if workers <= 1 or n <= 1:
            games = []
            for _ in range(n):
                games.append(self.play_game())
            return games

        workers = min(workers, n)

        state_dict_cpu = {k: v.detach().cpu() for k, v in self.network.state_dict().items()}
        network_cfg_dict = dict(self.network.cfg.__dict__)
        mcts_cfg_dict = dict(self.mcts_cfg.__dict__)
        sp_cfg_dict = dict(self.sp_cfg.__dict__)
        sp_cfg_dict["num_workers"] = 1

        ctx = mp.get_context("spawn")
        payloads = []
        base_seed = random.randint(0, 2**31 - 1)
        parent_torch_threads = torch.get_num_threads()
        threads_per_worker = max(1, parent_torch_threads // workers)
        remaining = n
        task_idx = 0
        while remaining > 0:
            c = min(task_games, remaining)
            payloads.append((
                state_dict_cpu,
                network_cfg_dict,
                mcts_cfg_dict,
                sp_cfg_dict,
                self.variant_pgn,
                c,
                base_seed + task_idx * 9973,
                threads_per_worker,
            ))
            remaining -= c
            task_idx += 1

        games: list[GameRecord] = []
        with cf.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            max_tasks_per_child=max_tasks_per_child,
        ) as ex:
            futures = [ex.submit(_play_games_worker, p) for p in payloads]
            for fut in cf.as_completed(futures):
                task_games_out, stats = fut.result()
                games.extend(task_games_out)
                if log_task_stats:
                    rss_before = "n/a" if stats.rss_mb_before is None else f"{stats.rss_mb_before:.1f} MB"
                    rss_after = "n/a" if stats.rss_mb_after is None else f"{stats.rss_mb_after:.1f} MB"
                    rss_peak = "n/a" if stats.rss_mb_peak is None else f"{stats.rss_mb_peak:.1f} MB"
                    logger.info(
                        "Self-play task pid=%s games=%s samples=%s semimoves=%s "
                        "time=%.2fs rss(before/after/peak)=%s/%s/%s",
                        stats.pid,
                        stats.num_games,
                        stats.total_samples,
                        stats.total_semimoves,
                        stats.elapsed_sec,
                        rss_before,
                        rss_after,
                        rss_peak,
                    )
        return games

    def _count_boards(self, env: SemimoveEnv) -> int:
        try:
            return len(env.state.get_boards())
        except Exception:
            return 0


def _read_exact(handle, num_bytes: int) -> bytes:
    data = handle.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Expected {num_bytes} bytes, got {len(data)}")
    return data


def _read_struct(handle, fmt: str):
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))


def _read_string(handle) -> str:
    (size,) = _read_struct(handle, "<I")
    if size == 0:
        return ""
    return _read_exact(handle, size).decode("utf-8")


class CppOnnxSelfPlayWorker:
    """Runs self-play in the C++ ONNX backend and rebuilds python training records."""

    DATA_MAGIC = 0x50535A41
    DATA_VERSION = 2

    def __init__(self, network, mcts_cfg: MCTSConfig, sp_cfg: SelfPlayConfig, device: torch.device, train_cfg):
        self.network = network
        self.mcts_cfg = mcts_cfg
        self.sp_cfg = sp_cfg
        self.device = device
        self.train_cfg = train_cfg

    def generate_games(self, num_games: Optional[int] = None) -> list[GameRecord]:
        if self.sp_cfg.rules_mode != "capture_king":
            raise ValueError("cpp_onnx self-play backend currently supports capture_king only.")

        n = num_games or self.sp_cfg.num_games
        if n <= 0:
            return []

        exe_path = Path(getattr(self.sp_cfg, "cpp_selfplay_executable", "build_onnx_selfplay/az_selfplay_onnx.exe"))
        if not exe_path.is_absolute():
            exe_path = Path.cwd() / exe_path
        if not exe_path.exists():
            raise FileNotFoundError(
                f"C++ self-play executable not found: {exe_path}. "
                "Build target az_selfplay_onnx first or override cpp_selfplay_executable."
            )
        self._prepare_runtime_binaries(exe_path)

        model_path = self._export_model()
        workers = max(1, int(getattr(self.sp_cfg, "num_workers", 1)))
        task_games = max(1, int(getattr(self.sp_cfg, "worker_task_games", 1)))
        ort_threads = max(1, int(getattr(self.sp_cfg, "cpp_onnx_ort_threads", 1)))
        requested_provider = str(getattr(self.sp_cfg, "cpp_onnx_provider", "cpu")).lower()
        cuda_device_id = max(0, int(getattr(self.sp_cfg, "cpp_onnx_cuda_device_id", 0)))
        provider = self._resolve_provider(
            exe_path=exe_path,
            model_path=model_path,
            requested_provider=requested_provider,
            ort_threads=ort_threads,
            cuda_device_id=cuda_device_id,
        )
        log_task_stats = bool(getattr(self.sp_cfg, "log_worker_task_stats", True))

        tasks: list[tuple[int, int]] = []
        base_seed = random.randint(0, 2**31 - 1)
        if workers <= 1:
            # With a single worker there is no load-balancing benefit to tiny tasks.
            # Keeping the whole batch in one C++ process avoids repeated ORT session startup.
            tasks.append((n, base_seed))
        else:
            remaining = n
            task_idx = 0
            while remaining > 0:
                c = min(task_games, remaining)
                tasks.append((c, base_seed + task_idx * 9973))
                remaining -= c
                task_idx += 1

        if provider == "cuda" and task_games < 4:
            logger.warning(
                "cpp_onnx CUDA self-play is using worker_task_games=%s; small task sizes tend to underutilize the GPU.",
                task_games,
            )

        games: list[GameRecord] = []
        with TemporaryDirectory(prefix="cpp_selfplay_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            if len(tasks) == 1:
                task_n, seed = tasks[0]
                return self._run_task(
                    exe_path,
                    model_path,
                    tmp_dir / "task_0000.bin",
                    task_n,
                    seed,
                    ort_threads,
                    provider,
                    cuda_device_id,
                    log_task_stats,
                )

            max_workers = min(workers, len(tasks))
            if provider == "cpu":
                task_buckets: list[list[tuple[int, int, Path]]] = [[] for _ in range(max_workers)]
                for idx, (task_n, seed) in enumerate(tasks):
                    bucket_idx = idx % max_workers
                    task_buckets[bucket_idx].append((task_n, seed, tmp_dir / f"task_{idx:04d}.bin"))

                with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futures = [
                        ex.submit(
                            self._run_task_bucket,
                            exe_path,
                            model_path,
                            bucket,
                            ort_threads,
                            provider,
                            cuda_device_id,
                            log_task_stats,
                        )
                        for bucket in task_buckets
                        if bucket
                    ]
                    for fut in cf.as_completed(futures):
                        games.extend(fut.result())
                    return games

            with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [
                    ex.submit(
                        self._run_task,
                        exe_path,
                        model_path,
                        tmp_dir / f"task_{idx:04d}.bin",
                        task_n,
                        seed,
                        ort_threads,
                        provider,
                        cuda_device_id,
                        log_task_stats,
                    )
                    for idx, (task_n, seed) in enumerate(tasks)
                ]
                for fut in cf.as_completed(futures):
                    games.extend(fut.result())
                return games
        return games

    def _resolve_provider(
        self,
        exe_path: Path,
        model_path: Path,
        requested_provider: str,
        ort_threads: int,
        cuda_device_id: int,
    ) -> str:
        provider = requested_provider.lower()
        if provider != "cuda":
            return provider

        try:
            with TemporaryDirectory(prefix="cpp_selfplay_probe_") as tmp_dir_str:
                output_path = Path(tmp_dir_str) / "probe.bin"
                cmd = [
                    str(exe_path),
                    "--model", str(model_path),
                    "--variant", str(self.train_cfg.variant_name),
                    "--games", "1",
                    "--sims", str(max(1, min(4, int(self.mcts_cfg.num_simulations)))),
                    "--min-board-limit", str(self.sp_cfg.min_board_limit),
                    "--max-board-limit", str(self.sp_cfg.min_board_limit),
                    "--max-game-length", str(self.sp_cfg.max_game_length),
                    "--temperature", str(self.sp_cfg.temperature),
                    "--temperature-final", str(self.sp_cfg.temperature_final),
                    "--temperature-threshold", str(self.sp_cfg.temp_threshold),
                    "--c-puct", str(self.mcts_cfg.c_puct),
                    "--dirichlet-alpha", str(self.mcts_cfg.dirichlet_alpha),
                    "--dirichlet-epsilon", str(self.mcts_cfg.dirichlet_epsilon),
                    "--provider", "cuda",
                    "--cuda-device-id", str(cuda_device_id),
                    "--ort-threads", str(ort_threads),
                    "--seed", "1",
                    "--output-data", str(output_path),
                    "--quiet",
                ]
                completed = subprocess.run(
                    cmd,
                    cwd=Path.cwd(),
                    env=self._build_subprocess_env("cuda"),
                    capture_output=True,
                    text=True,
                    check=False,
                )
            if completed.returncode == 0:
                return "cuda"

            stderr_text = (completed.stderr or "").strip()
            stdout_text = (completed.stdout or "").strip()
            logger.warning(
                "cpp_onnx CUDA self-play was requested but the C++ ONNX runner failed its startup probe "
                "(exit=%s, stdout=%r, stderr=%r). Falling back to CPU self-play.",
                completed.returncode,
                stdout_text[-400:],
                stderr_text[-400:],
            )
            return "cpu"
        except Exception as exc:
            logger.warning(
                "cpp_onnx CUDA self-play was requested but its startup probe failed: %s. "
                "Falling back to CPU self-play.",
                exc,
            )
            return "cpu"

    def _export_model(self) -> Path:
        from .export_onnx import export_live_network

        precision = str(getattr(self.sp_cfg, "cpp_onnx_model_precision", "fp16")).lower()
        if precision not in {"fp16", "fp32"}:
            raise ValueError(f"Unsupported cpp_onnx_model_precision: {precision}")

        model_path = Path(getattr(self.sp_cfg, "cpp_onnx_model_path", "alphazero/checkpoints/selfplay_fp16.onnx"))
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        metadata_extra = {
            "iteration": getattr(self.train_cfg, "iteration", None),
            "backend": "cpp_onnx",
        }
        export_live_network(
            network=self.network,
            cfg=self.train_cfg,
            output_path=str(model_path),
            device_name="cpu",
            fp16_output=(precision == "fp16"),
            opset=int(getattr(self.sp_cfg, "cpp_onnx_opset", 17)),
            metadata_extra=metadata_extra,
        )
        return model_path

    def _run_task(
        self,
        exe_path: Path,
        model_path: Path,
        output_path: Path,
        num_games: int,
        seed: int,
        ort_threads: int,
        provider: str,
        cuda_device_id: int,
        log_task_stats: bool,
    ) -> list[GameRecord]:
        cmd = [
            str(exe_path),
            "--model", str(model_path),
            "--variant", str(self.train_cfg.variant_name),
            "--games", str(num_games),
            "--sims", str(self.mcts_cfg.num_simulations),
            "--min-board-limit", str(self.sp_cfg.min_board_limit),
            "--max-board-limit", str(self.sp_cfg.max_board_limit),
            "--max-game-length", str(self.sp_cfg.max_game_length),
            "--temperature", str(self.sp_cfg.temperature),
            "--temperature-final", str(self.sp_cfg.temperature_final),
            "--temperature-threshold", str(self.sp_cfg.temp_threshold),
            "--c-puct", str(self.mcts_cfg.c_puct),
            "--dirichlet-alpha", str(self.mcts_cfg.dirichlet_alpha),
            "--dirichlet-epsilon", str(self.mcts_cfg.dirichlet_epsilon),
            "--provider", provider,
            "--cuda-device-id", str(cuda_device_id),
            "--ort-threads", str(ort_threads),
            "--seed", str(seed),
            "--output-data", str(output_path),
            "--quiet",
        ]

        env = self._build_subprocess_env(provider)

        t0 = time.perf_counter()
        completed = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        if completed.returncode != 0:
            raise RuntimeError(
                f"C++ self-play failed (exit={completed.returncode}).\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        games = self._load_games_from_binary(output_path)
        if log_task_stats:
            total_samples = sum(len(g.samples) for g in games)
            total_semimoves = sum(g.total_semimoves for g in games)
            logger.info(
                "Cpp self-play task games=%s samples=%s semimoves=%s time=%.2fs exe=%s provider=%s",
                len(games),
                total_samples,
                total_semimoves,
                elapsed,
                exe_path.name,
                provider,
            )
        return games

    def _run_task_bucket(
        self,
        exe_path: Path,
        model_path: Path,
        task_bucket: list[tuple[int, int, Path]],
        ort_threads: int,
        provider: str,
        cuda_device_id: int,
        log_task_stats: bool,
    ) -> list[GameRecord]:
        process = self._start_persistent_worker(
            exe_path=exe_path,
            model_path=model_path,
            ort_threads=ort_threads,
            provider=provider,
            cuda_device_id=cuda_device_id,
        )
        try:
            games: list[GameRecord] = []
            for num_games, seed, output_path in task_bucket:
                games.extend(
                    self._run_persistent_task(
                        process=process,
                        output_path=output_path,
                        num_games=num_games,
                        seed=seed,
                        provider=provider,
                        log_task_stats=log_task_stats,
                        exe_name=exe_path.name,
                    )
                )
            return games
        finally:
            self._stop_persistent_worker(process)

    def _start_persistent_worker(
        self,
        exe_path: Path,
        model_path: Path,
        ort_threads: int,
        provider: str,
        cuda_device_id: int,
    ) -> subprocess.Popen:
        cmd = [
            str(exe_path),
            "--model", str(model_path),
            "--variant", str(self.train_cfg.variant_name),
            "--sims", str(self.mcts_cfg.num_simulations),
            "--min-board-limit", str(self.sp_cfg.min_board_limit),
            "--max-board-limit", str(self.sp_cfg.max_board_limit),
            "--max-game-length", str(self.sp_cfg.max_game_length),
            "--temperature", str(self.sp_cfg.temperature),
            "--temperature-final", str(self.sp_cfg.temperature_final),
            "--temperature-threshold", str(self.sp_cfg.temp_threshold),
            "--c-puct", str(self.mcts_cfg.c_puct),
            "--dirichlet-alpha", str(self.mcts_cfg.dirichlet_alpha),
            "--dirichlet-epsilon", str(self.mcts_cfg.dirichlet_epsilon),
            "--provider", provider,
            "--cuda-device-id", str(cuda_device_id),
            "--ort-threads", str(ort_threads),
            "--serve",
            "--quiet",
        ]
        return subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            env=self._build_subprocess_env(provider),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def _run_persistent_task(
        self,
        process: subprocess.Popen,
        output_path: Path,
        num_games: int,
        seed: int,
        provider: str,
        log_task_stats: bool,
        exe_name: str,
    ) -> list[GameRecord]:
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Persistent C++ self-play worker missing stdin/stdout pipes.")

        command = f"RUN\t{num_games}\t{seed}\t{output_path}\n"
        t0 = time.perf_counter()
        process.stdin.write(command)
        process.stdin.flush()
        response = process.stdout.readline()
        elapsed = time.perf_counter() - t0

        if not response:
            stderr_text = ""
            if process.stderr is not None:
                try:
                    stderr_text = process.stderr.read()
                except Exception:
                    stderr_text = ""
            raise RuntimeError(
                f"Persistent C++ self-play worker exited unexpectedly.\nSTDERR:\n{stderr_text}"
            )

        response = response.rstrip("\r\n")
        if not response.startswith("OK\t"):
            stderr_text = ""
            if process.stderr is not None:
                try:
                    stderr_text = process.stderr.read()
                except Exception:
                    stderr_text = ""
            raise RuntimeError(
                f"Persistent C++ self-play worker returned error: {response}\nSTDERR:\n{stderr_text}"
            )

        games = self._load_games_from_binary(output_path)
        if log_task_stats:
            total_samples = sum(len(g.samples) for g in games)
            total_semimoves = sum(g.total_semimoves for g in games)
            logger.info(
                "Cpp self-play task games=%s samples=%s semimoves=%s time=%.2fs exe=%s provider=%s mode=persistent",
                len(games),
                total_samples,
                total_semimoves,
                elapsed,
                exe_name,
                provider,
            )
        return games

    @staticmethod
    def _stop_persistent_worker(process: subprocess.Popen) -> None:
        try:
            if process.stdin is not None:
                process.stdin.write("QUIT\n")
                process.stdin.flush()
        except Exception:
            pass
        try:
            process.communicate(timeout=5)
        except Exception:
            process.kill()
            try:
                process.communicate(timeout=2)
            except Exception:
                pass

    @staticmethod
    def _build_subprocess_env(provider: str) -> dict[str, str]:
        env = dict(os.environ)
        if provider != "cuda":
            return env

        extra_paths: list[str] = []
        try:
            import onnxruntime
            ort_dir = Path(onnxruntime.__file__).resolve().parent / "capi"
            if ort_dir.exists():
                extra_paths.append(str(ort_dir))
        except Exception:
            pass

        try:
            import torch as torch_module
            torch_lib_dir = Path(torch_module.__file__).resolve().parent / "lib"
            if torch_lib_dir.exists():
                extra_paths.append(str(torch_lib_dir))
        except Exception:
            pass

        extra_paths.append(sys.executable.rsplit("\\", 1)[0])

        path_entries = [p for p in extra_paths if p]
        if path_entries:
            env["PATH"] = os.pathsep.join(path_entries + [env.get("PATH", "")])
        return env

    def _prepare_runtime_binaries(self, exe_path: Path) -> None:
        try:
            import onnxruntime
        except Exception as exc:
            provider = str(getattr(self.sp_cfg, "cpp_onnx_provider", "cpu")).lower()
            if provider == "cuda":
                raise RuntimeError("CUDA self-play requires onnxruntime-gpu to be importable.") from exc
            return

        ort_capi_dir = Path(onnxruntime.__file__).resolve().parent / "capi"
        provider = str(getattr(self.sp_cfg, "cpp_onnx_provider", "cpu")).lower()
        required = [
            "onnxruntime.dll",
            "onnxruntime_providers_shared.dll",
        ]
        if provider == "cuda":
            required.append("onnxruntime_providers_cuda.dll")
        for name in required:
            src = ort_capi_dir / name
            if not src.exists():
                raise FileNotFoundError(f"Required ONNX Runtime DLL not found: {src}")
            dst = exe_path.parent / name
            if dst.exists():
                src_stat = src.stat()
                dst_stat = dst.stat()
                if src_stat.st_size == dst_stat.st_size and int(src_stat.st_mtime) == int(dst_stat.st_mtime):
                    continue
            shutil.copy2(src, dst)

    def _load_games_from_binary(self, path: Path) -> list[GameRecord]:
        games: list[GameRecord] = []
        with path.open("rb") as handle:
            magic, version, num_games = _read_struct(handle, "<III")
            if magic != self.DATA_MAGIC:
                raise ValueError(f"Unexpected self-play binary magic: {magic:#x}")
            if version != self.DATA_VERSION:
                raise ValueError(f"Unsupported self-play binary version: {version}")

            for _ in range(num_games):
                outcome, total_semimoves, board_limit = _read_struct(handle, "<fii")
                terminal_reason = _read_string(handle)
                pgn = _read_string(handle)
                game = GameRecord(
                    outcome=float(outcome),
                    total_semimoves=int(total_semimoves),
                    board_limit=int(board_limit),
                    terminal_reason=terminal_reason,
                    pgn=pgn,
                )

                (num_moves,) = _read_struct(handle, "<I")
                for _ in range(num_moves):
                    player, is_submit = _read_struct(handle, "<bB")
                    root_value, board_count = _read_struct(handle, "<fi")
                    move_text = _read_string(handle)
                    game.move_history.append(
                        MoveEntry(
                            player=int(player),
                            action_type="submit" if is_submit else "semimove",
                            ext_move_str=move_text,
                            mcts_value=float(root_value),
                            board_count=int(board_count),
                        )
                    )

                (num_samples,) = _read_struct(handle, "<I")
                for _ in range(num_samples):
                    (_player,) = _read_struct(handle, "<b")
                    urgency, value_target = _read_struct(handle, "<ff")
                    num_boards, num_actions = _read_struct(handle, "<iI")
                    piece_channels = int(self.train_cfg.network.piece_channels)
                    board_squares = int(self.train_cfg.network.board_squares)

                    board_planes = np.frombuffer(
                        _read_exact(handle, num_boards * piece_channels * board_squares),
                        dtype=np.uint8,
                    ).astype(np.float32, copy=False).reshape(num_boards, piece_channels, board_squares)
                    last_move_markers = np.frombuffer(
                        _read_exact(handle, num_boards * board_squares),
                        dtype=np.int8,
                    ).astype(np.float32, copy=False).reshape(num_boards, board_squares)
                    l_coords = np.frombuffer(
                        _read_exact(handle, num_boards * 4),
                        dtype=np.int32,
                    ).astype(np.int64, copy=False)
                    t_coords = np.frombuffer(
                        _read_exact(handle, num_boards * 4),
                        dtype=np.int32,
                    ).astype(np.int64, copy=False)
                    policy_target = np.frombuffer(
                        _read_exact(handle, num_actions * 4),
                        dtype=np.float32,
                    ).copy()
                    action_board_indices = np.frombuffer(
                        _read_exact(handle, num_actions * 4),
                        dtype=np.int32,
                    ).astype(np.int64, copy=False)
                    action_from_squares = np.frombuffer(
                        _read_exact(handle, num_actions * 4),
                        dtype=np.int32,
                    ).astype(np.int64, copy=False)
                    action_to_squares = np.frombuffer(
                        _read_exact(handle, num_actions * 4),
                        dtype=np.int32,
                    ).astype(np.int64, copy=False)
                    action_delta_t = np.frombuffer(
                        _read_exact(handle, num_actions * 4),
                        dtype=np.float32,
                    ).copy()
                    action_delta_l = np.frombuffer(
                        _read_exact(handle, num_actions * 4),
                        dtype=np.float32,
                    ).copy()
                    action_is_submit = np.frombuffer(
                        _read_exact(handle, num_actions),
                        dtype=np.uint8,
                    ).astype(bool, copy=False)

                    game.samples.append(
                        SemimoveRecord(
                            board_planes=board_planes,
                            last_move_markers=last_move_markers,
                            l_coords=l_coords,
                            t_coords=t_coords,
                            urgency=float(urgency),
                            padding_mask=np.zeros(num_boards, dtype=bool),
                            policy_target=policy_target,
                            action_board_indices=action_board_indices,
                            action_from_squares=action_from_squares,
                            action_to_squares=action_to_squares,
                            action_delta_t=action_delta_t,
                            action_delta_l=action_delta_l,
                            action_is_submit=action_is_submit,
                            value_target=float(value_target),
                        )
                    )
                games.append(game)
        return games


class ReplayBuffer:
    """Circular replay buffer for semimove records."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list[SemimoveRecord] = []
        self.position = 0

    def push(self, samples: list[SemimoveRecord]):
        for sample in samples:
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.position] = sample
            self.position = (self.position + 1) % self.capacity

    def push_game(self, game: GameRecord):
        self.push(game.samples)

    def sample(self, batch_size: int) -> list[SemimoveRecord]:
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


def collate_samples(samples: list[SemimoveRecord], device: torch.device, max_boards: int = 32) -> dict:
    """
    Collate a batch of SemimoveRecords into padded tensors.

    Returns dict with tensors for state/value and python lists for variable-length
    policy targets + action metadata.
    """
    batch_size = len(samples)
    max_n = min(max(s.board_planes.shape[0] for s in samples), max_boards)
    piece_channels = samples[0].board_planes.shape[1]
    board_squares = samples[0].board_planes.shape[2]

    bp = np.zeros((batch_size, max_n, piece_channels, board_squares), dtype=np.float32)
    lm = np.zeros((batch_size, max_n, board_squares), dtype=np.float32)
    lc = np.zeros((batch_size, max_n), dtype=np.int64)
    tc = np.zeros((batch_size, max_n), dtype=np.int64)
    urg = np.zeros((batch_size, 1), dtype=np.float32)
    mask = np.ones((batch_size, max_n), dtype=bool)
    vt = np.zeros((batch_size, 1), dtype=np.float32)

    for i, s in enumerate(samples):
        n = min(s.board_planes.shape[0], max_n)
        bp[i, :n] = s.board_planes[:n]
        lm[i, :n] = s.last_move_markers[:n]
        lc[i, :n] = s.l_coords[:n]
        tc[i, :n] = s.t_coords[:n]
        urg[i, 0] = s.urgency
        mask[i, :n] = False
        vt[i, 0] = s.value_target

    return {
        "board_planes": torch.from_numpy(bp).to(device),
        "last_move_markers": torch.from_numpy(lm).to(device),
        "l_coords": torch.from_numpy(lc).to(device),
        "t_coords": torch.from_numpy(tc).to(device),
        "urgency": torch.from_numpy(urg).to(device),
        "padding_mask": torch.from_numpy(mask).to(device),
        "value_target": torch.from_numpy(vt).to(device),
        "policy_targets": [s.policy_target for s in samples],
        "action_board_indices": [s.action_board_indices for s in samples],
        "action_from_squares": [s.action_from_squares for s in samples],
        "action_to_squares": [s.action_to_squares for s in samples],
        "action_delta_t": [s.action_delta_t for s in samples],
        "action_delta_l": [s.action_delta_l for s in samples],
        "action_is_submit": [s.action_is_submit for s in samples],
    }
