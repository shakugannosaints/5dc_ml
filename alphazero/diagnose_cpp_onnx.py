"""Diagnose the C++ ONNX self-play runtime before large-scale training."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from .config import TrainConfig
from .export_onnx import export_live_network
from .network import AlphaZeroNetwork


def _runtime_env(provider: str) -> dict[str, str]:
    env = dict(os.environ)
    if provider != "cuda":
        return env

    import onnxruntime

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


def _sync_runtime_dlls(exe_path: Path, provider: str) -> list[dict]:
    try:
        import onnxruntime
    except Exception:
        return []

    ort_capi_dir = Path(onnxruntime.__file__).resolve().parent / "capi"
    names = ["onnxruntime.dll", "onnxruntime_providers_shared.dll"]
    if provider == "cuda":
        names.append("onnxruntime_providers_cuda.dll")

    copied: list[dict] = []
    for name in names:
        src = ort_capi_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Required ONNX Runtime DLL not found: {src}")
        dst = exe_path.parent / name
        shutil.copy2(src, dst)
        copied.append(
            {
                "name": name,
                "src": str(src),
                "dst": str(dst),
                "size": src.stat().st_size,
            }
        )
    return copied


def _run_probe(
    exe_path: Path,
    model_path: Path,
    cfg: TrainConfig,
    provider: str,
    ort_threads: int,
    cuda_device_id: int,
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"az_diag_{provider}_") as tmp_dir_str:
        output_path = Path(tmp_dir_str) / "probe.bin"
        cmd = [
            str(exe_path),
            "--model", str(model_path),
            "--variant", str(cfg.variant_name),
            "--games", "1",
            "--sims", "4",
            "--min-board-limit", "3",
            "--max-board-limit", "3",
            "--max-game-length", "0",
            "--temperature", "0.0",
            "--temperature-final", "0.0",
            "--temperature-threshold", "0",
            "--c-puct", "2.0",
            "--dirichlet-alpha", "0.3",
            "--dirichlet-epsilon", "0.0",
            "--provider", provider,
            "--cuda-device-id", str(cuda_device_id),
            "--ort-threads", str(ort_threads),
            "--seed", "1",
            "--output-data", str(output_path),
            "--quiet",
        ]
        t0 = time.perf_counter()
        completed = subprocess.run(
            cmd,
            cwd=Path.cwd(),
            env=_runtime_env(provider),
            capture_output=True,
            text=True,
            check=False,
        )
        elapsed = time.perf_counter() - t0
        return {
            "provider": provider,
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "elapsed_sec": round(elapsed, 3),
            "stdout_tail": (completed.stdout or "")[-400:],
            "stderr_tail": (completed.stderr or "")[-400:],
            "output_exists": output_path.exists(),
            "output_size": output_path.stat().st_size if output_path.exists() else 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose C++ ONNX self-play runtime availability.")
    parser.add_argument("--variant", choices=["very_small", "standard", "standard_turn_zero"], default="very_small")
    parser.add_argument("--exe", default="build_onnx_selfplay/az_selfplay_onnx.exe")
    parser.add_argument("--provider", choices=["cpu", "cuda", "both"], default="both")
    parser.add_argument("--ort-threads", type=int, default=1)
    parser.add_argument("--cuda-device-id", type=int, default=0)
    parser.add_argument("--keep-model", action="store_true")
    args = parser.parse_args()

    exe_path = Path(args.exe)
    if not exe_path.is_absolute():
        exe_path = Path.cwd() / exe_path
    if not exe_path.exists():
        raise FileNotFoundError(f"C++ self-play executable not found: {exe_path}")

    cfg = TrainConfig()
    cfg.apply_variant(args.variant)
    network = AlphaZeroNetwork(cfg.network)
    network.eval()

    with tempfile.TemporaryDirectory(prefix="az_cpp_onnx_diag_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        model_path = tmp_dir / "probe_fp16.onnx"
        export_live_network(
            network=network,
            cfg=cfg,
            output_path=str(model_path),
            device_name="cpu",
            fp16_output=True,
            opset=18,
            metadata_extra={"diagnose_cpp_onnx": True},
        )
        if args.keep_model:
            kept = Path.cwd() / "alphazero" / "checkpoints" / args.variant / "diagnose_probe_fp16.onnx"
            kept.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_path, kept)

        providers = ["cpu", "cuda"] if args.provider == "both" else [args.provider]
        results: list[dict] = []
        for provider in providers:
            copied = _sync_runtime_dlls(exe_path, provider)
            probe = _run_probe(
                exe_path=exe_path,
                model_path=model_path,
                cfg=cfg,
                provider=provider,
                ort_threads=max(1, args.ort_threads),
                cuda_device_id=max(0, args.cuda_device_id),
            )
            probe["dlls"] = copied
            results.append(probe)

        print(json.dumps({"variant": args.variant, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
