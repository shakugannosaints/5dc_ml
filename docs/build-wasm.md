# WASM 编译指南（傻瓜版）

本文介绍如何将 5D Chess 引擎编译为 WebAssembly，并在本地浏览器中运行。

---

## 第一步：安装依赖

你需要以下工具（缺一不可）：

| 工具 | 版本要求 | 下载地址 |
|------|----------|---------|
| **Git** | 任意版本 | https://git-scm.com |
| **CMake** | ≥ 3.20 | https://cmake.org/download |
| **Python** | ≥ 3.8 | https://python.org（仅用于本地预览服务器） |
| **Emscripten SDK** | 最新版 | https://emscripten.org/docs/getting_started/downloads.html |

> **编译器说明**：WASM 编译 **不需要** 额外安装 MSVC / GCC / Clang，Emscripten SDK 自带 Clang 编译器。

---

## 第二步：安装 Emscripten SDK

在任意目录（例如 `C:\emsdk`）打开终端，执行：

```sh
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest
./emsdk activate latest
```

**Windows 用户**使用 PowerShell：

```powershell
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
.\emsdk install latest
.\emsdk activate latest
.\emsdk_env.ps1      # 激活环境变量（每次新开终端都要执行这一步）
```

> ✅ 安装成功后，`emcmake` 命令应可用。可用 `emcmake --version` 验证。

---

## 第三步：克隆本仓库

```sh
git clone --recurse-submodules https://github.com/ftxi/5dchess_engine.git
cd 5dchess_engine
```

> **注意 `--recurse-submodules`**：这会同时拉取 `extern/pybind11` 等子模块。
> 如果忘记加这个参数，可以补执行：
> ```sh
> git submodule update --init --recursive
> ```

---

## 第四步：编译 WASM

在项目根目录下，先激活 Emscripten 环境（如果还没激活），再执行：

```sh
mkdir build_wasm
cd build_wasm
emcmake cmake .. -DEMMODULE=on -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

> **为什么要用 `Release`？**  
> 不加 `-DCMAKE_BUILD_TYPE=Release` 时不开启 `-O3` 优化，运行速度会慢 6–7 倍。

编译成功后，生成物在 `build_wasm/ui/wasm/` 目录下：

```
build_wasm/
└── ui/
    ├── index.html          ← 网页入口
    └── wasm/
        ├── engine.js       ← JS 胶水代码
        └── engine.wasm     ← WebAssembly 二进制
```

---

## 第五步：本地预览

**不能直接双击 `index.html`！** 浏览器的 CORS 策略禁止 `file://` 协议加载 `.wasm` 文件。

项目已提供一个配置好了正确 HTTP 响应头的本地服务器脚本，在 **`build_wasm/` 目录**下运行：

```sh
cd build_wasm
python serve.py
```

然后在浏览器中访问：

```
http://localhost:9999
```

---

## 常见问题

### ❓ `emcmake` 命令找不到

每次新开终端都要重新激活 Emscripten 环境：

```powershell
# Windows（进入 emsdk 安装目录）
cd C:\emsdk
.\emsdk_env.ps1
```

```sh
# Linux / macOS
source ~/emsdk/emsdk_env.sh
```

---

### ❓ CMake 报错 `EMMODULE option requires compilation with Emscripten`

你用了普通 `cmake` 而不是 `emcmake cmake`。确保第一条命令是：

```sh
emcmake cmake .. -DEMMODULE=on -DCMAKE_BUILD_TYPE=Release
```

---

### ❓ 编译报 C++23 相关错误

Emscripten 要求使用其自带的 Clang，确认没有在 CMake 中手动指定其他编译器。
删除 `build_wasm/` 目录后重新执行第四步，让 `emcmake` 自动配置。

---

### ❓ 浏览器报 `SharedArrayBuffer is not defined`

`serve.py` 已经自动设置了所需的响应头 (`COOP`/`COEP`)，**必须通过它提供的服务器访问**，直接打开 `index.html` 文件无效。

---

### ❓ 想要更快的增量编译

第一次 `cmake --build .` 之后，修改源码再次编译只需：

```sh
cd build_wasm
cmake --build .
```

无需重新运行 `emcmake cmake`。

---

## 完整流程速查

```sh
# 1. 安装并激活 Emscripten（仅首次）
git clone https://github.com/emscripten-core/emsdk.git C:\emsdk
cd C:\emsdk && .\emsdk install latest && .\emsdk activate latest

# 2. 每次新开终端激活环境
cd C:\emsdk ; .\emsdk_env.ps1

# 3. 克隆项目（仅首次）
git clone --recurse-submodules https://github.com/ftxi/5dchess_engine.git
cd 5dchess_engine

# 4. 编译
mkdir build_wasm ; cd build_wasm
emcmake cmake .. -DEMMODULE=on -DCMAKE_BUILD_TYPE=Release
cmake --build .

# 5. 本地预览
python serve.py
# 访问 http://localhost:9999
```
