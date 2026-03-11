"""
Export the semimove AlphaZero policy/value network to ONNX for C++ inference.

The exported graph matches the explicit legal-action scoring path used by MCTS:
inputs are one encoded state plus one aligned legal action list, outputs are the
current-player value estimate and per-action logits.

Usage:
  python -m alphazero.export_onnx --checkpoint latest --output alphazero/checkpoints/latest_fp16.onnx
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import warnings
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from torch.export import Dim
from onnxruntime.transformers.onnx_model import OnnxModel
from onnxruntime.transformers.float16 import convert_float_to_float16

from .config import NetworkConfig, TrainConfig
from .network import AlphaZeroNetwork


class OnnxActionWrapper(nn.Module):
    """Batched ONNX wrapper around AlphaZeroNetwork legal-action scoring."""

    def __init__(self, network: AlphaZeroNetwork):
        super().__init__()
        self.network = network

    def forward(
        self,
        board_planes: torch.Tensor,
        last_move_markers: torch.Tensor,
        l_coords: torch.Tensor,
        t_coords: torch.Tensor,
        used_board_counts: torch.Tensor,
        urgency: torch.Tensor,
        action_state_indices: torch.Tensor,
        action_board_indices: torch.Tensor,
        action_from_squares: torch.Tensor,
        action_to_squares: torch.Tensor,
        action_delta_t: torch.Tensor,
        action_delta_l: torch.Tensor,
        action_is_submit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = board_planes.shape[0]
        num_boards = board_planes.shape[1]
        board_idx = torch.arange(num_boards, device=board_planes.device)
        padding_mask = board_idx.unsqueeze(0) >= used_board_counts.reshape(-1, 1)

        value, submit_logit, _, board_out, _ = self.network(
            board_planes,
            last_move_markers,
            l_coords,
            t_coords,
            urgency.reshape(batch_size, 1),
            padding_mask=padding_mask,
            return_latent=True,
        )
        logits = self.network.score_legal_actions_batched_flat(
            board_out=board_out,
            submit_logit=submit_logit,
            action_state_indices=action_state_indices,
            action_board_indices=action_board_indices,
            action_from_squares=action_from_squares,
            action_to_squares=action_to_squares,
            action_delta_t=action_delta_t,
            action_delta_l=action_delta_l,
            action_is_submit=action_is_submit,
        )
        return value.reshape(batch_size), logits


def _resolve_checkpoint_path(cfg: TrainConfig, checkpoint: str) -> Path:
    if checkpoint.lower() == "latest":
        latest = Path(cfg.checkpoint_dir) / "latest.pt"
        if latest.exists():
            return latest

        ckpts = sorted(Path(cfg.checkpoint_dir).glob("agent_iter_*.pt"))
        if ckpts:
            return ckpts[-1]

        final = Path(cfg.checkpoint_dir) / "agent_final.pt"
        if final.exists():
            return final

        raise FileNotFoundError("No checkpoint found for 'latest'")
    return Path(checkpoint)


def _load_network(cfg: TrainConfig, checkpoint_path: Path, device: torch.device) -> tuple[AlphaZeroNetwork, dict, TrainConfig]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    variant_name = checkpoint_cfg.get("variant_name")
    if variant_name:
        cfg.apply_variant(variant_name)
    if "network" in checkpoint_cfg:
        cfg.network = NetworkConfig(**checkpoint_cfg["network"])
    if "self_play" in checkpoint_cfg:
        for key, value in checkpoint_cfg["self_play"].items():
            setattr(cfg.self_play, key, value)
    cfg.variant_pgn = checkpoint_cfg.get("variant_pgn", cfg.variant_pgn)
    cfg.board_size_x = checkpoint_cfg.get("board_size_x", cfg.board_size_x)
    cfg.board_size_y = checkpoint_cfg.get("board_size_y", cfg.board_size_y)

    network = AlphaZeroNetwork(cfg.network).to(device)
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    return network, checkpoint, cfg


def _build_dummy_inputs(
    device: torch.device,
    batch_size: int,
    num_boards: int,
    piece_channels: int,
    board_squares: int,
) -> tuple[torch.Tensor, ...]:
    board_planes = torch.zeros((batch_size, num_boards, piece_channels, board_squares), dtype=torch.float32, device=device)
    last_move_markers = torch.zeros((batch_size, num_boards, board_squares), dtype=torch.float32, device=device)
    l_coords = torch.zeros((batch_size, num_boards), dtype=torch.long, device=device)
    t_coords = torch.ones((batch_size, num_boards), dtype=torch.long, device=device)
    used_board_counts = torch.full((batch_size,), num_boards, dtype=torch.long, device=device)
    urgency = torch.zeros((batch_size,), dtype=torch.float32, device=device)
    action_state_indices = torch.zeros((1,), dtype=torch.long, device=device)
    action_board_indices = torch.zeros((1,), dtype=torch.long, device=device)
    action_from_squares = torch.zeros((1,), dtype=torch.long, device=device)
    action_to_squares = torch.zeros((1,), dtype=torch.long, device=device)
    action_delta_t = torch.zeros((1,), dtype=torch.float32, device=device)
    action_delta_l = torch.zeros((1,), dtype=torch.float32, device=device)
    action_is_submit = torch.zeros((1,), dtype=torch.long, device=device)
    return (
        board_planes,
        last_move_markers,
        l_coords,
        t_coords,
        used_board_counts,
        urgency,
        action_state_indices,
        action_board_indices,
        action_from_squares,
        action_to_squares,
        action_delta_t,
        action_delta_l,
        action_is_submit,
    )


def _export_wrapper_to_onnx(
    wrapper: OnnxActionWrapper,
    cfg: TrainConfig,
    output: Path,
    metadata: dict,
    device: torch.device,
    fp16_output: bool,
    opset: int,
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output if not fp16_output else output.with_suffix(".fp32.onnx")
    effective_opset = max(int(opset), 18)

    input_names = [
        "board_planes",
        "last_move_markers",
        "l_coords",
        "t_coords",
        "used_board_counts",
        "urgency",
        "action_state_indices",
        "action_board_indices",
        "action_from_squares",
        "action_to_squares",
        "action_delta_t",
        "action_delta_l",
        "action_is_submit",
    ]
    output_names = ["value", "action_logits"]
    num_boards_dim = Dim("num_boards", min=1)
    num_actions_dim = Dim("num_actions", min=1)
    batch_size = max(1, int(getattr(cfg.mcts, "leaf_batch_size", 1)))
    dynamic_shapes = {
        "board_planes": {1: num_boards_dim},
        "last_move_markers": {1: num_boards_dim},
        "l_coords": {1: num_boards_dim},
        "t_coords": {1: num_boards_dim},
        "used_board_counts": None,
        "urgency": None,
        "action_state_indices": {0: num_actions_dim},
        "action_board_indices": {0: num_actions_dim},
        "action_from_squares": {0: num_actions_dim},
        "action_to_squares": {0: num_actions_dim},
        "action_delta_t": {0: num_actions_dim},
        "action_delta_l": {0: num_actions_dim},
        "action_is_submit": {0: num_actions_dim},
    }

    dummy_inputs = _build_dummy_inputs(
        device,
        batch_size,
        4,
        wrapper.network.cfg.piece_channels,
        wrapper.network.cfg.board_squares,
    )
    with torch.no_grad():
        prev_fastpath = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        try:
            # Suppress exporter progress glyphs so Windows code pages do not break export.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                torch.onnx.export(
                    wrapper,
                    dummy_inputs,
                    temp_output,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_shapes=dynamic_shapes,
                    opset_version=effective_opset,
                    export_params=True,
                    do_constant_folding=True,
                    dynamo=True,
                )
        finally:
            torch.backends.mha.set_fastpath_enabled(prev_fastpath)

    fp32_model = onnx.load(temp_output)
    OnnxModel(fp32_model).topological_sort()
    try:
        onnx.checker.check_model(fp32_model)
    except Exception as exc:
        warnings.warn(f"ONNX checker failed on fp32 export; continuing because ORT may still accept the model: {exc}")
    onnx.save(fp32_model, temp_output)

    if fp16_output:
        model = onnx.load(temp_output)
        fp16_model = convert_float_to_float16(model, keep_io_types=True)
        OnnxModel(fp16_model).topological_sort()
        try:
            onnx.checker.check_model(fp16_model)
        except Exception as exc:
            warnings.warn(f"ONNX checker failed on fp16 export; continuing because ORT may still accept the model: {exc}")
        onnx.save(fp16_model, output)
        temp_output.unlink(missing_ok=True)

    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def export_live_network(
    network: AlphaZeroNetwork,
    cfg: TrainConfig,
    output_path: str,
    device_name: str = "cpu",
    fp16_output: bool = True,
    opset: int = 17,
    metadata_extra: dict | None = None,
) -> Path:
    device = torch.device(device_name)
    batch_slots = max(1, int(getattr(cfg.mcts, "leaf_batch_size", 1)))
    network_copy = AlphaZeroNetwork(cfg.network).to(device)
    state_dict = {
        k: v.detach().to(device=device, dtype=v.dtype).clone()
        for k, v in network.state_dict().items()
    }
    network_copy.load_state_dict(state_dict)
    network_copy.eval()
    wrapper = OnnxActionWrapper(network_copy).to(device)
    wrapper.eval()

    output = Path(output_path)
    metadata = {
        "checkpoint": None,
        "iteration": None,
        "variant_name": cfg.variant_name,
        "rules_mode": cfg.self_play.rules_mode,
        "variant_pgn": cfg.variant_pgn,
        "board_size_x": cfg.board_size_x,
        "board_size_y": cfg.board_size_y,
        "piece_channels": cfg.network.piece_channels,
        "board_squares": cfg.network.board_squares,
        "board_side": cfg.network.board_side,
        "board_axis": "dynamic",
        "batch_slots": batch_slots,
        "model_precision": "fp16" if fp16_output else "fp32",
        "io_precision": "fp32",
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return _export_wrapper_to_onnx(wrapper, cfg, output, metadata, device, fp16_output, opset)


def export_onnx(
    checkpoint: str,
    output_path: str,
    device_name: str = "cpu",
    fp16_output: bool = True,
    opset: int = 17,
) -> Path:
    cfg = TrainConfig()
    device = torch.device(device_name)
    checkpoint_path = _resolve_checkpoint_path(cfg, checkpoint)
    network, checkpoint_obj, cfg = _load_network(cfg, checkpoint_path, device)
    metadata = {
        "checkpoint": str(checkpoint_path),
        "iteration": checkpoint_obj.get("iteration"),
        "variant_name": cfg.variant_name,
        "rules_mode": cfg.self_play.rules_mode,
        "variant_pgn": cfg.variant_pgn,
        "board_size_x": cfg.board_size_x,
        "board_size_y": cfg.board_size_y,
        "piece_channels": cfg.network.piece_channels,
        "board_squares": cfg.network.board_squares,
        "board_side": cfg.network.board_side,
        "board_axis": "dynamic",
        "model_precision": "fp16" if fp16_output else "fp32",
        "io_precision": "fp32",
    }
    wrapper = OnnxActionWrapper(network).to(device)
    wrapper.eval()
    return _export_wrapper_to_onnx(wrapper, cfg, Path(output_path), metadata, device, fp16_output, opset)


def main():
    parser = argparse.ArgumentParser(description="Export semimove AlphaZero network to ONNX")
    parser.add_argument("--checkpoint", type=str, default="latest",
                        help="Checkpoint path, or 'latest'")
    parser.add_argument("--output", type=str, required=True,
                        help="Output ONNX path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device used for loading/export")
    parser.add_argument("--fp32", action="store_true",
                        help="Keep model weights in fp32 instead of converting to fp16")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    args = parser.parse_args()

    output = export_onnx(
        checkpoint=args.checkpoint,
        output_path=args.output,
        device_name=args.device,
        fp16_output=not args.fp32,
        opset=args.opset,
    )
    print(output)


if __name__ == "__main__":
    main()
