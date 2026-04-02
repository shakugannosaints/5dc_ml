"""
Evaluate a trained C++ ONNX self-play model under low-noise settings.

This is meant to answer questions like:
  - Does the current model still show first-player advantage on `very_small`?
  - What happens if we remove temperature / Dirichlet noise and raise MCTS sims?

Typical usage:
  python -m alphazero.evaluate_selfplay --variant very_small --games 32 --sims 800

Recommended for "true strength" checks:
  --temperature 0.0 --temperature-final 0.0 --temperature-threshold 0 --dirichlet-epsilon 0.0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import TrainConfig


@dataclass
class MoveEntry:
    player: int
    is_submit: bool
    move_text: str
    root_value: float
    board_count: int


@dataclass
class GameSummary:
    outcome: float
    total_semimoves: int
    board_limit: int
    terminal_reason: str
    pgn: str
    move_history: list[MoveEntry] = field(default_factory=list)


def _runtime_env(provider: str) -> dict[str, str]:
    env = dict(os.environ)
    if provider != "cuda":
        return env

    import onnxruntime
    import torch

    extra_paths: list[str] = []
    ort_dir = Path(onnxruntime.__file__).resolve().parent / "capi"
    if ort_dir.exists():
        extra_paths.append(str(ort_dir))
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib_dir.exists():
        extra_paths.append(str(torch_lib_dir))
    extra_paths.append(sys.executable.rsplit("\\", 1)[0])
    env["PATH"] = os.pathsep.join(extra_paths + [env.get("PATH", "")])
    return env


def _sync_runtime_dlls(exe_path: Path, provider: str) -> None:
    try:
        import onnxruntime
    except Exception:
        return

    ort_capi_dir = Path(onnxruntime.__file__).resolve().parent / "capi"
    names = ["onnxruntime.dll", "onnxruntime_providers_shared.dll"]
    if provider == "cuda":
        names.append("onnxruntime_providers_cuda.dll")

    for name in names:
        src = ort_capi_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Required ONNX Runtime DLL not found: {src}")
        shutil.copy2(src, exe_path.parent / name)


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {len(data)}")
    return data


def _read_struct(handle, fmt: str):
    return struct.unpack(fmt, _read_exact(handle, struct.calcsize(fmt)))


def _read_string(handle) -> str:
    (size,) = _read_struct(handle, "<I")
    if size == 0:
        return ""
    return _read_exact(handle, size).decode("utf-8")


def _load_game_summaries(path: Path, piece_channels: int, board_squares: int) -> list[GameSummary]:
    data_magic = 0x50535A41
    data_version = 2
    games: list[GameSummary] = []

    with path.open("rb") as handle:
        magic, version, num_games = _read_struct(handle, "<III")
        if magic != data_magic:
            raise ValueError(f"Unexpected binary magic: {magic:#x}")
        if version != data_version:
            raise ValueError(f"Unsupported binary version: {version}")

        for _ in range(num_games):
            outcome, total_semimoves, board_limit = _read_struct(handle, "<fii")
            terminal_reason = _read_string(handle)
            pgn = _read_string(handle)
            game = GameSummary(
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
                        is_submit=bool(is_submit),
                        move_text=move_text,
                        root_value=float(root_value),
                        board_count=int(board_count),
                    )
                )

            # Skip sample payloads; we only need summary and move history.
            (num_samples,) = _read_struct(handle, "<I")
            for _ in range(num_samples):
                _player, = _read_struct(handle, "<b")
                _urgency, _value_target = _read_struct(handle, "<ff")
                num_boards, num_actions = _read_struct(handle, "<iI")

                board_planes_bytes = num_boards * piece_channels * board_squares
                last_markers_bytes = num_boards * board_squares
                coord_bytes = num_boards * 4
                action_i32_bytes = num_actions * 4
                action_bool_bytes = num_actions

                _read_exact(handle, board_planes_bytes)
                _read_exact(handle, last_markers_bytes)
                _read_exact(handle, coord_bytes)  # l_coords
                _read_exact(handle, coord_bytes)  # t_coords
                _read_exact(handle, num_actions * 4)  # policy_target float32
                _read_exact(handle, action_i32_bytes)  # action_board_indices
                _read_exact(handle, action_i32_bytes)  # action_from_squares
                _read_exact(handle, action_i32_bytes)  # action_to_squares
                _read_exact(handle, num_actions * 4)  # action_delta_t float32
                _read_exact(handle, num_actions * 4)  # action_delta_l float32
                _read_exact(handle, action_bool_bytes)  # action_is_submit

            games.append(game)

    return games


def _resolve_default_model(cfg: TrainConfig) -> Path:
    checkpoint_dir = Path(cfg.checkpoint_dir) / cfg.variant_name
    model_path = checkpoint_dir / "selfplay_fp16.onnx"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Default ONNX model not found: {model_path}. "
            "Pass --model explicitly if you exported it elsewhere."
        )
    return model_path


def _run_eval(args, cfg: TrainConfig, model_path: Path, exe_path: Path) -> tuple[list[GameSummary], float]:
    _sync_runtime_dlls(exe_path, args.provider)

    with tempfile.TemporaryDirectory(prefix="az_eval_") as tmp_dir_str:
        output_path = Path(tmp_dir_str) / "eval.bin"
        cmd = [
            str(exe_path),
            "--model", str(model_path),
            "--variant", str(cfg.variant_name),
            "--games", str(args.games),
            "--sims", str(args.sims),
            "--leaf-batch-size", str(args.leaf_batch_size),
            "--min-board-limit", str(args.min_board_limit),
            "--max-board-limit", str(args.max_board_limit),
            "--material-scale", str(args.material_scale),
            "--max-game-length", str(args.max_game_length),
            "--temperature", str(args.temperature),
            "--temperature-final", str(args.temperature_final),
            "--temperature-threshold", str(args.temperature_threshold),
            "--c-puct", str(args.c_puct),
            "--dirichlet-alpha", str(args.dirichlet_alpha),
            "--dirichlet-epsilon", str(args.dirichlet_epsilon),
            "--provider", args.provider,
            "--cuda-device-id", str(args.cuda_device_id),
            "--ort-threads", str(args.ort_threads),
            "--seed", str(args.seed),
            "--output-data", str(output_path),
        ]
        if args.quiet_runner:
            cmd.append("--quiet")

        t0 = time.perf_counter()
        completed = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            env=_runtime_env(args.provider),
            capture_output=bool(args.quiet_runner),
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        if completed.returncode != 0:
            raise RuntimeError(
                f"C++ self-play eval failed (exit={completed.returncode}).\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )

        games = _load_game_summaries(
            output_path,
            piece_channels=cfg.network.piece_channels,
            board_squares=cfg.network.board_squares,
        )
        return games, elapsed


def _summarize(games: list[GameSummary], elapsed_sec: float) -> dict:
    white_wins = sum(1 for g in games if g.outcome > 0)
    black_wins = sum(1 for g in games if g.outcome < 0)
    draws = sum(1 for g in games if g.outcome == 0)
    reasons = Counter(g.terminal_reason for g in games)
    semimoves = [g.total_semimoves for g in games]
    return {
        "games": len(games),
        "elapsed_sec": round(elapsed_sec, 3),
        "games_per_sec": round(len(games) / max(elapsed_sec, 1e-9), 3),
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "white_rate": round(white_wins / max(len(games), 1), 4),
        "black_rate": round(black_wins / max(len(games), 1), 4),
        "draw_rate": round(draws / max(len(games), 1), 4),
        "avg_semimoves": round(sum(semimoves) / max(len(semimoves), 1), 2),
        "termination_reasons": dict(reasons),
    }


def _print_sample_games(games: list[GameSummary], count: int) -> None:
    if count <= 0 or not games:
        return

    def game_sort_key(game: GameSummary):
        return (game.outcome == 0.0, -game.total_semimoves)

    chosen = sorted(games, key=game_sort_key)[:count]
    for idx, game in enumerate(chosen, start=1):
        outcome_text = "White wins" if game.outcome > 0 else ("Black wins" if game.outcome < 0 else "Draw")
        print(f"\nSample Game #{idx}: {outcome_text}, semimoves={game.total_semimoves}, reason={game.terminal_reason}")
        for entry in game.move_history:
            player = "W" if entry.player == 0 else "B"
            move_text = "SUBMIT" if entry.is_submit else entry.move_text
            print(f"  [{player}] {move_text:20} boards={entry.board_count:2d} value={entry.root_value:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained ONNX self-play model with low-noise settings.")
    parser.add_argument("--variant", choices=["very_small", "standard", "standard_turn_zero"], default="very_small")
    parser.add_argument("--model", default=None, help="Path to an exported ONNX model; defaults to checkpoints/<variant>/selfplay_fp16.onnx")
    parser.add_argument("--exe", default="build_onnx_selfplay/az_selfplay_onnx.exe", help="Path to az_selfplay_onnx executable")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cuda-device-id", type=int, default=0)
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--sims", type=int, default=800)
    parser.add_argument("--leaf-batch-size", type=int, default=4)
    parser.add_argument("--min-board-limit", type=int, default=30)
    parser.add_argument("--max-board-limit", type=int, default=50)
    parser.add_argument("--material-scale", type=float, default=2.0)
    parser.add_argument("--max-game-length", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--temperature-final", type=float, default=0.0)
    parser.add_argument("--temperature-threshold", type=int, default=0)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.0)
    parser.add_argument("--ort-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--show-games", type=int, default=2, help="Print up to N sample main lines")
    parser.add_argument("--json-output", default=None, help="Optional path to write full JSON summary")
    parser.add_argument("--quiet-runner", action="store_true", help="Suppress raw az_selfplay_onnx game dumps")
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.apply_variant(args.variant)

    model_path = Path(args.model) if args.model else _resolve_default_model(cfg)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    exe_path = Path(args.exe)
    if not exe_path.is_absolute():
        exe_path = Path.cwd() / exe_path
    if not exe_path.exists():
        raise FileNotFoundError(f"C++ self-play executable not found: {exe_path}")

    games, elapsed_sec = _run_eval(args, cfg, model_path, exe_path)
    summary = _summarize(games, elapsed_sec)
    summary.update(
        {
            "variant": args.variant,
            "model": str(model_path),
            "exe": str(exe_path),
            "provider": args.provider,
            "sims": args.sims,
            "temperature": args.temperature,
            "temperature_final": args.temperature_final,
            "temperature_threshold": args.temperature_threshold,
            "dirichlet_epsilon": args.dirichlet_epsilon,
            "seed": args.seed,
        }
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    _print_sample_games(games, args.show_games)

    if args.json_output:
        output_path = Path(args.json_output)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "games": [
                {
                    "outcome": game.outcome,
                    "total_semimoves": game.total_semimoves,
                    "board_limit": game.board_limit,
                    "terminal_reason": game.terminal_reason,
                    "pgn": game.pgn,
                    "move_history": [
                        {
                            "player": entry.player,
                            "is_submit": entry.is_submit,
                            "move_text": entry.move_text,
                            "root_value": entry.root_value,
                            "board_count": entry.board_count,
                        }
                        for entry in game.move_history
                    ],
                }
                for game in games
            ],
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
