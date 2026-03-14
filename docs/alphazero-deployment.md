# AlphaZero 部署指南

本文只覆盖 `alphazero/` 训练链路，不涉及旧的 `ml/` 目录。

目标是让用户在只有源码、没有任何现成环境和编译产物的前提下，从零完成：

1. 准备编译和 Python 环境
2. 编译 `engine` Python 扩展
3. 安装 AlphaZero 训练依赖
4. 跑通一次最小训练
5. 可选启用 `cpp_onnx` 自对弈加速

## 1. 先说明推荐路径

仓库里的 AlphaZero 训练有两种自对弈后端：

- `python`：纯 Python + `engine` 扩展，最容易首次跑通，推荐作为第一次部署验收路径
- `cpp_onnx`：C++ 可执行文件 + ONNX Runtime，自对弈速度更高，但部署步骤更多

建议按下面顺序执行：

1. 先完成 `engine` 编译
2. 先用 `python` backend 跑通一次最小训练
3. 再决定是否切到 `cpp_onnx`

## 2. 平台和工具要求

### Windows 推荐配置

这是当前最稳的完整部署环境：

- Windows 10/11 x64
- Git
- CMake 3.20 或更高
- Visual Studio 2022
- 安装工作负载：`Desktop development with C++`
- Python 3.12.x

之所以推荐 Python 3.12，是因为这个仓库当前的训练依赖和已有构建习惯都围绕 3.12。

### Linux / macOS 说明

Linux 和 macOS 也可以编译 Python 扩展并运行 `python` backend 训练，但当前 `ONNX_SELFPLAY` 目标在 CMake 中只支持 Windows，因此 `cpp_onnx` 加速路径默认按 Windows 写。

## 3. 获取源码

一定要带子模块克隆，否则 `pybind11` 不完整，`engine` 无法编译。

```powershell
git clone --recurse-submodules <你的仓库地址>
cd 5dchess_engine
```

如果已经克隆过但忘了子模块，补一次：

```powershell
git submodule update --init --recursive
```

## 4. 创建 Python 3.12 虚拟环境

在仓库根目录执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

如果机器上没有 `py` 启动器，也可以直接用 Python 3.12 的完整路径创建虚拟环境。

## 5. 安装 Python 依赖

`requirements.txt` 同时包含了 CPU/GPU 运行时相关包。为了降低首次安装失败率，建议先安装公共依赖，再按你的机器选择 CPU 或 GPU 版本。

### 5.1 安装公共依赖

在 PowerShell 中从 `requirements.txt` 过滤掉运行时相关包：

```powershell
Get-Content requirements.txt |
  Where-Object { $_ -notmatch '^(torch|torchvision|onnxruntime|onnxruntime-gpu)==' } |
  Set-Content requirements.base.txt

pip install -r requirements.base.txt
```

### 5.2 如果你使用 NVIDIA GPU

下面这一组对应仓库当前依赖版本：

```powershell
pip install --extra-index-url https://download.pytorch.org/whl/cu128 `
  torch==2.10.0+cu128 `
  torchvision==0.25.0+cu128

pip install onnxruntime-gpu==1.24.3
```

### 5.3 如果你只打算先用 CPU 跑通

```powershell
pip install --index-url https://download.pytorch.org/whl/cpu `
  torch==2.10.0+cpu `
  torchvision==0.25.0+cpu

pip install onnxruntime==1.24.3
```

### 5.4 可选检查

```powershell
python -m pip show torch
python -m pip show torch-geometric
python -m pip show onnxruntime
python -m pip show onnxruntime-gpu
```

说明：

- CPU 部署时，`onnxruntime` 存在而 `onnxruntime-gpu` 不存在是正常的
- GPU 部署时，`onnxruntime-gpu` 存在即可

## 6. 编译 AlphaZero 依赖的 `engine` Python 扩展

AlphaZero 代码在导入时会自动尝试从 `build_py_ml/`、`build_py/`、`build/` 查找已编译的 `engine` 模块。因此最推荐的输出目录就是 `build_py_ml/`。

### 6.1 配置和编译

在仓库根目录执行：

```powershell
cmake -S . -B build_py_ml `
  -DPYMODULE=on `
  -DCMAKE_BUILD_TYPE=Release `
  -DPython3_EXECUTABLE="$PWD\.venv\Scripts\python.exe"

cmake --build build_py_ml --config Release
```

成功后会在 `build_py_ml/` 下看到类似文件：

- Windows: `engine.cp312-win_amd64.pyd`
- Linux/macOS: `engine*.so`

### 6.2 为什么一定要显式指定 `Python3_EXECUTABLE`

因为机器上可能同时装了多个 Python。  
如果 CMake 错选到了别的解释器，最终编出来的 `engine` 会和当前虚拟环境不匹配，导入时报错。

## 7. 验证 `engine` 已经可用

先做最小导入检查：

```powershell
python -c "import alphazero, engine; print('engine import ok')"
```

再检查 AlphaZero 配置能否正常初始化：

```powershell
python -c "from alphazero.config import TrainConfig; cfg = TrainConfig(); print(cfg.variant_name, cfg.self_play.self_play_backend)"
```

如果这两步都通过，说明训练所需的 Python 层和 C++ 扩展层已经接通。

## 8. 首次训练建议先用 `python` backend 跑通

虽然代码默认的 `self_play_backend` 是 `cpp_onnx`，但首次部署建议先显式改成 `python`，因为它依赖更少、排障更直接。

### 8.1 最小可跑通命令

```powershell
python -m alphazero.train `
  --variant very_small `
  --cpu `
  --selfplay-backend python `
  --no-resume `
  --iterations 1 `
  --games 1 `
  --sims 8
```

这条命令的作用是：

- 用最小变体 `very_small`
- 强制 CPU，避免首次部署时叠加 CUDA 问题
- 只跑 1 个 iteration
- 只生成 1 局自对弈
- 每步只做 8 次 MCTS 模拟

只要这条命令能顺利结束，就说明“从源码开始训练”这条链路已经成立。

### 8.2 训练产物在哪

当你显式传入 `--variant very_small` 时，训练输出会放在：

- `alphazero/checkpoints/very_small/`
- `alphazero/logs/very_small/`

常见文件包括：

- `alphazero/checkpoints/very_small/latest.pt`
- `alphazero/checkpoints/very_small/agent_iter_0005.pt`
- `alphazero/checkpoints/very_small/agent_final.pt`
- `alphazero/logs/very_small/training.log`
- `alphazero/logs/very_small/training_log.jsonl`
- `alphazero/logs/very_small/games/`

## 9. 正式训练常用命令

### 9.1 继续训练最近一次 checkpoint

```powershell
python -m alphazero.train `
  --variant very_small `
  --selfplay-backend python `
  --resume latest
```

如果不传 `--no-resume`，脚本本身也会自动尝试恢复 `latest.pt`。

### 9.2 用 GPU 训练 `very_small`

```powershell
python -m alphazero.train `
  --variant very_small `
  --selfplay-backend python `
  --iterations 50 `
  --games 8 `
  --sims 100
```

### 9.3 训练 `standard`

```powershell
python -m alphazero.train `
  --variant standard `
  --selfplay-backend python `
  --iterations 50 `
  --games 4 `
  --sims 100
```

### 9.4 训练 `standard_turn_zero`

这个变体通常更适合明确指定棋盘数量终止范围：

```powershell
python -m alphazero.train `
  --variant standard_turn_zero `
  --selfplay-backend python `
  --iterations 50 `
  --games 4 `
  --sims 100 `
  --min-board-limit 80 `
  --max-board-limit 120
```

## 10. 可选：启用 `cpp_onnx` 自对弈加速

这一节只在你已经确认第 8 节能跑通之后再做。

### 10.1 什么时候值得启用

适合以下情况：

- 你已经能正常训练
- 自对弈成为主要瓶颈
- 你在 Windows 上部署
- 你希望通过 ONNX Runtime 提高 self-play 吞吐

### 10.2 编译 C++ 自对弈可执行文件

仓库已经自带 Windows 版 ONNX Runtime SDK 压缩包和解压目录，正常情况下不需要你再单独下载 SDK。

在仓库根目录执行：

```powershell
cmake -S . -B build_onnx_selfplay `
  -DONNX_SELFPLAY=on `
  -DCMAKE_BUILD_TYPE=Release

cmake --build build_onnx_selfplay --config Release
```

成功后应得到：

- `build_onnx_selfplay/az_selfplay_onnx.exe`

### 10.3 先做诊断，不要直接大规模开跑

先测 CPU provider：

```powershell
python -m alphazero.diagnose_cpp_onnx --variant very_small --provider cpu
```

如果你是 GPU 环境，再测 CUDA provider：

```powershell
python -m alphazero.diagnose_cpp_onnx --variant very_small --provider cuda
```

只有诊断通过，再开始用 `cpp_onnx` 训练。

### 10.4 用 `cpp_onnx` 开始训练

CPU 版示例：

```powershell
python -m alphazero.train `
  --variant very_small `
  --selfplay-backend cpp_onnx `
  --cpu `
  --iterations 20 `
  --games 16 `
  --sims 100 `
  --sp-workers 4 `
  --sp-task-games 4
```

GPU 版示例：

```powershell
python -m alphazero.train `
  --variant standard_turn_zero `
  --selfplay-backend cpp_onnx `
  --iterations 200 `
  --games 16 `
  --sims 800 `
  --sp-workers 8 `
  --sp-task-games 8 `
  --min-board-limit 80 `
  --max-board-limit 120 `
  --cpp-onnx-provider cuda
```

### 10.5 `cpp_onnx` 额外说明

- 该后端当前只支持 `capture_king` 规则模式，正好也是默认值
- 训练时会在 checkpoint 目录下额外生成 `selfplay_fp16.onnx`
- 如果 `cpp_onnx` 诊断失败，请先退回 `--selfplay-backend python`

## 11. 推荐验收流程

如果你要给最终用户一条最稳的部署验收路径，建议按这个顺序：

1. 克隆仓库并拉取子模块
2. 创建 Python 3.12 虚拟环境
3. 安装依赖
4. 编译 `build_py_ml/engine*.pyd`
5. 执行 `python -c "import alphazero, engine; print('ok')"`
6. 执行第 8.1 节的一次最小训练
7. 只有在第 6 步成功后，再考虑启用 `cpp_onnx`

## 12. 常见问题排查

### 12.1 `Cannot import compiled 'engine' module`

优先检查三件事：

1. `build_py_ml/` 下是否真的生成了 `engine*.pyd` 或 `engine*.so`
2. CMake 是否使用了当前虚拟环境的 Python
3. 当前运行训练的解释器是否和编译扩展用的是同一个 Python 版本

最常见修复方式就是重新执行：

```powershell
cmake -S . -B build_py_ml `
  -DPYMODULE=on `
  -DCMAKE_BUILD_TYPE=Release `
  -DPython3_EXECUTABLE="$PWD\.venv\Scripts\python.exe"

cmake --build build_py_ml --config Release
```

### 12.2 `extern/pybind11` 是空的

说明子模块没拉下来：

```powershell
git submodule update --init --recursive
```

### 12.3 `pip install -r requirements.txt` 失败

不要直接硬装整份 `requirements.txt`。  
这个仓库把 CPU/GPU 运行时依赖混在一起列了，更稳的做法是按第 5 节拆开安装。

### 12.4 `cpp_onnx` 启动时报缺少 DLL

先确认你已经：

- 编译了 `build_onnx_selfplay/az_selfplay_onnx.exe`
- 安装了正确的 ONNX Runtime 包
- 运行过 `python -m alphazero.diagnose_cpp_onnx`

在 GPU 模式下，还要确保当前虚拟环境里装的是 `onnxruntime-gpu`，不是只有 CPU 版。

### 12.5 Windows 终端里运行 `alphazero.smoke_test` 出现编码报错

当前仓库里的 `smoke_test.py` 输出中包含一些非 ASCII 符号，在中文 Windows 终端上可能触发编码问题。  
部署验收建议优先使用：

- `python -c "import alphazero, engine; print('ok')"`
- 第 8.1 节的一次最小训练
- `python -m alphazero.diagnose_cpp_onnx ...`

而不要把 `smoke_test.py` 当成唯一验收标准。

## 13. 一套可以直接照抄的最小流程

下面这组命令适合第一次在 Windows 上从纯源码跑通 AlphaZero：

```powershell
git clone --recurse-submodules <你的仓库地址>
cd 5dchess_engine

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

Get-Content requirements.txt |
  Where-Object { $_ -notmatch '^(torch|torchvision|onnxruntime|onnxruntime-gpu)==' } |
  Set-Content requirements.base.txt

pip install -r requirements.base.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0+cpu torchvision==0.25.0+cpu
pip install onnxruntime==1.24.3

cmake -S . -B build_py_ml `
  -DPYMODULE=on `
  -DCMAKE_BUILD_TYPE=Release `
  -DPython3_EXECUTABLE="$PWD\.venv\Scripts\python.exe"

cmake --build build_py_ml --config Release

python -c "import alphazero, engine; print('engine import ok')"

python -m alphazero.train `
  --variant very_small `
  --cpu `
  --selfplay-backend python `
  --no-resume `
  --iterations 1 `
  --games 1 `
  --sims 8
```

如果这套流程通过，再继续提高 `games`、`sims`、`iterations`，或者切换到 `cpp_onnx` 加速即可。
