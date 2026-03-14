# Python 模块编译指南（自对弈训练用）

本文介绍如何编译 `engine.pyd`（Python 扩展模块），这是运行 AlphaZero 自对弈训练的前提。

> **与 WASM 的区别**：WASM 编译结果只能在浏览器中运行。训练脚本直接 `import engine`，需要的是 pybind11 编译的 `.pyd` 文件，两者不可互换。

---

## 第一步：安装依赖

### 必须安装

| 工具 | 版本要求 | 说明 |
|------|----------|------|
| **Git** | 任意版本 | https://git-scm.com |
| **CMake** | ≥ 3.20 | https://cmake.org/download |
| **Visual Studio 2022** | Community 版即可 | 需勾选 **"使用 C++ 的桌面开发"** 工作负载 |
| **Python** | ≥ 3.10（推荐 3.12） | https://python.org，安装时勾选"Add to PATH" |

> **为什么需要 Visual Studio？** pybind11 在 Windows 上需要 MSVC 编译器，MinGW 不支持生成 `.pyd`。

### Python 依赖包

训练还需要 PyTorch、torch-geometric 等包。项目根目录已有完整的 `requirements.txt`：

```powershell
# 建议先创建虚拟环境
python -m venv ml_venv
.\ml_venv\Scripts\Activate.ps1

# 安装所有依赖
pip install -r requirements.txt
```

> **GPU 加速**：`requirements.txt` 中已包含 `torch==2.x+cu128`（CUDA 12.8）。
> 如果你没有 NVIDIA 显卡，需要手动替换为 CPU 版本：
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 第二步：克隆仓库（包含子模块）

```powershell
git clone --recurse-submodules https://github.com/ftxi/5dchess_engine.git
cd 5dchess_engine
```

> **关键**：`--recurse-submodules` 会同时拉取 `extern/pybind11/`，缺少它无法编译。
>
> 如果已经克隆但忘记了这个参数，补执行：
> ```powershell
> git submodule update --init --recursive
> ```

---

## 第三步：编译 Python 模块

用 **x64 Native Tools Command Prompt for VS 2022**（或普通 PowerShell，CMake 会自动找 MSVC）：

```powershell
mkdir build_py_ml
cd build_py_ml
cmake .. -DPYMODULE=on -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

编译成功后，`build_py_ml/` 目录下会出现：

```
build_py_ml/
└── engine.cp312-win_amd64.pyd   ← Python 扩展模块（文件名含 Python 版本）
```

> `alphazero/__init__.py` 会按优先级依次搜索 `build_py_ml/`、`build_py/`、`build/`，找到 `engine.pyd` 后自动加入 `sys.path`，无需手动配置。

---

## 第四步：验证模块可以导入

激活虚拟环境后，在项目根目录运行：

```powershell
python -c "import sys; sys.path.insert(0, 'build_py_ml'); import engine; print(engine.__version__)"
```

或者运行完整的冒烟测试：

```powershell
python -m alphazero.smoke_test
```

全部打印 ✅ 即表示引擎绑定正常。

---

## 第五步：开始训练

### 快速启动（默认配置，Very Small 4×4 棋盘）

```powershell
python -m alphazero.train
```

### 常用参数

```powershell
# 指定棋盘变体
python -m alphazero.train --variant very_small    # 4×4，快速迭代（默认）
python -m alphazero.train --variant standard      # 8×8，完整训练

# 强制使用 CPU（没有 NVIDIA 显卡时）
python -m alphazero.train --cpu

# 从上次中断的检查点继续
python -m alphazero.train --resume latest

# 只加载权重，重置优化器和回放缓冲区
python -m alphazero.train --resume latest --resume-weights-only

# 指定本次运行的迭代次数
python -m alphazero.train --iterations 50

# 调整每轮自对弈局数和 MCTS 模拟次数
python -m alphazero.train --games 32 --sims 100
```

### 自对弈后端说明

训练默认使用 **`cpp_onnx`** 后端（C++ 进程 + ONNX Runtime 推理），速度最快，但需要额外编译 `az_selfplay_onnx.exe`（见下方）。

如果只想用纯 Python 后端先跑通流程：

```powershell
python -m alphazero.train --selfplay-backend python
```

---

## 附：编译 C++ ONNX 自对弈加速器（可选）

如果想使用速度更快的 `cpp_onnx` 后端，需要额外编译：

```powershell
mkdir build_onnx_selfplay
cd build_onnx_selfplay
cmake .. -DONNX_SELFPLAY=on -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

编译完成后生成 `build_onnx_selfplay/az_selfplay_onnx.exe`。  
还需要导出 ONNX 模型供 C++ 推理：

```powershell
python -m alphazero.export_onnx --checkpoint latest --output alphazero/checkpoints/selfplay_fp16.onnx
```

---

## 训练产物说明

| 路径 | 内容 |
|------|------|
| `alphazero/checkpoints/<variant>/latest.pt` | 最新检查点（可用于 `--resume latest`） |
| `alphazero/checkpoints/<variant>/agent_iter_NNNN.pt` | 每 N 轮保存一次的检查点 |
| `alphazero/logs/<variant>/training_log.jsonl` | 每轮训练指标（loss、局数等）|
| `alphazero/logs/<variant>/games/` | 按间隔保存的 PGN 对局快照 |

---

## 常见问题

### ❓ `Cannot import compiled 'engine' module`

- 确认 `build_py_ml/` 目录下存在 `engine.*.pyd`
- 确认 Python 版本与 `.pyd` 文件名中的版本一致（如 `cp312` 对应 Python 3.12）
- 如果版本不符，用对应版本 Python 重新编译

### ❓ CMake 找不到 Python

CMake 有时找不到虚拟环境中的 Python，可以显式指定：

```powershell
cmake .. -DPYMODULE=on -DCMAKE_BUILD_TYPE=Release `
    -DPython3_EXECUTABLE="$PWD\ml_venv\Scripts\python.exe"
```

### ❓ pybind11 报错 / 找不到头文件

`extern/pybind11/` 目录为空，说明子模块未拉取：

```powershell
git submodule update --init --recursive
```

### ❓ 训练时 `CUDA out of memory`

减小每批大小或每局自对弈局数：

```powershell
python -m alphazero.train --games 16
```

或修改 `alphazero/config.py` 中 `TrainConfig.batch_size`。

---

## 速查命令

```powershell
# 激活虚拟环境
.\ml_venv\Scripts\Activate.ps1

# 编译引擎
mkdir build_py_ml ; cd build_py_ml
cmake .. -DPYMODULE=on -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cd ..

# 冒烟测试
python -m alphazero.smoke_test

# 开始训练
python -m alphazero.train --variant very_small

# 继续训练
python -m alphazero.train --resume latest
```
