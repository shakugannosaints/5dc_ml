# 5D Chess Engine – Copilot Instructions

## Architecture Overview

This is a **5D chess** (multi-timeline chess variant) engine with three compilation targets from a single C++ core:

```
src/core/  →  C++23 bitboard engine (game, state, multiverse, board)
    ├── pymodule.cpp  →  pybind11 Python module (`engine.pyd`)
    ├── emmodule.cpp  →  Emscripten WASM module (embind)
    └── cli.cpp       →  Command-line interface
ml/                   →  PyTorch ML agent (AlphaZero-lite approach)
extern/client/        →  Flask + SocketIO web UI (legacy)
ui/                   →  Static WASM-based web UI (current)
```

**Key domain concepts:** A `game` holds a tree of `state` objects. Each `state` owns a `multiverse` (2D array of `board` shared_ptrs indexed by timeline L and turn T). Boards are 8×8 max, stored as bitboards (64 bytes). Two coordinate systems exist: **LTCXY** (for moves, L can be negative) and **UVXY** (for board indexing, U ≥ 0). See `docs/index.md` for diagrams.

## Build Targets (CMake)

All builds require **C++23** (`CMAKE_CXX_STANDARD 23`). Use `-DCMAKE_BUILD_TYPE=Release` — unoptimized builds are 6-7× slower.

| Target | CMake Flag | Output | Dependencies |
|--------|-----------|--------|-------------|
| Tests + CLI | `-DTEST=on` | `build/cli`, ctest executables | None (C++ only) |
| Python module | `-DPYMODULE=on` | `engine.*.pyd` in build dir | pybind11 submodule |
| WASM | `-DEMMODULE=on` (via `emcmake`) | `build_wasm/ui/wasm/engine.js` | Emscripten SDK |

**Critical:** The pybind11 submodule at `extern/pybind11/` must be cloned (`git clone --recurse-submodules`).

## Python Engine Bindings (`pymodule.cpp`)

The `engine` module is a compiled `.pyd`—**not a Python file**. It exposes: `game`, `state`, `vec4`, `ext_move`, `action`, `match_status_t`, `create_state_from_pgn()`, plus ML helpers: `state.get_all_board_tensors()`, `state.get_graph_data()`, `state.get_per_timeline_moves()`, `enumerate_legal_actions()`, `random_action()`.

Board tensors are 27-channel planes: 12 white piece types + 12 black + unmoved flag + wall mask + occupied mask (see `board_to_planes()` in `pymodule.cpp`).

## ML Pipeline (`ml/`)

**Training:** `python ml/train.py [--variant small|standard] [--resume PATH] [--cpu]`

Architecture: `BoardEncoder` (CNN/ResNet per board) → `MultiverseEncoder` (GATv2 GNN over timeline DAG) → `FactoredPolicyHead` (autoregressive per-timeline) + `ValueHead` (MLP → tanh). Config in `ml/config.py` as dataclasses (`TrainingConfig`, `SMALL_CONFIG`, `STANDARD_CONFIG`).

- **Self-play** (`self_play.py`): generates `(state, action, outcome)` trajectories via REINFORCE with value baseline
- **Factored policy**: actions are Π of per-timeline moves — avoids combinatorial explosion
- **HC search**: `HC_info::build_HC(state)` in C++ enumerates legal actions via coroutine generators
- **AI server** (`ml/ai_server.py`): Flask HTTP API at port 8080 for WASM UI integration (`/api/move`, `/api/status`)
- **Smoke test**: `python ml/smoke_test.py` validates the full pipeline end-to-end
- **Virtualenv**: `ml_venv/` with PyTorch, torch_geometric, flask, flask_cors

## Conventions & Patterns

- **PGN format**: 5dpgn notation with headers like `[Board "Very Small - Open"]`, `[Mode "5D"]`. See `docs/pgn-bnf.txt` for full BNF grammar. Moves use `(LT)` coordinate prefixes, e.g., `(0T13)b6b5`.
- **Two checkmate algorithms**: `hc` (coroutine-based, better worst-case) and `naive` (DFS, better average-case). Referenced in CLI as `fast`/`naive`.
- **Transfer learning**: `Agent.load_state_dict_transfer()` handles board-size mismatches (e.g., 4×4 → 8×8) by copying compatible embedding rows.
- **Game variants**: Configured via PGN headers + `TrainingConfig.variant_pgn`. "Very Small - Open" (4×4) for fast iteration, "Standard" (8×8) for full games.
- **Error handling in WASM**: `emmodule.cpp` wraps all calls returning `{success, error, message}` JS objects.

## Testing

- **C++ tests**: `cd build && ctest` — each `.cpp` in `test/` auto-registers as a ctest target
- **Python integration**: `python test.py` (loads engine from `build/`), `python test_api.py` (HTTP tests for AI server)
- **ML smoke test**: `python ml/smoke_test.py` — tests each model component then runs minimal training
- **Perftests**: CLI-based — `cat test/pgn/*.5dpgn | cli perftest` (registered as ctests)

## Key Files for Understanding

- `src/core/state.h` — central game state: owns multiverse, handles `can_apply()`, move generation
- `src/core/multiverse_base.h` — timeline/board management, L↔U / TC↔V coordinate conversion
- `pymodule.cpp` — full Python binding surface including ML tensor helpers (lines 1–200)
- `ml/models/agent.py` — ties all ML components together; `select_action()` is the inference entry point
- `ml/config.py` — all hyperparameters as dataclasses with sensible defaults
