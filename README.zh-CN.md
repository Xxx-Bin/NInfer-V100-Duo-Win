# NInfer V100 Duo Windows 快速开始

NInfer 的 Windows 原生移植：在两张 Tesla V100-SXM2-16GB（TCC、NVLink）上以 TP2、CUDA 12.8、`sm_70`
运行，发布版媒体/视觉开启（`NINFER_STUB_MEDIA=OFF`），沿用与 Linux 版相同的 `.ninfer` 产物、
`ninfer` CLI 和 `ninfer-serve` 服务接口。

- 英文权威文档：[README.md](README.md)
- Windows 构建、验证与已知限制：[docs/windows.md](docs/windows.md)
- 可复现构建步骤：[REBUILD.md](REBUILD.md)
- 文本与媒体 fixture：[examples/cli/README.md](examples/cli/README.md)

## 当前版本

`v0.1.1-win.5`（2026-10-20）由 `.github/workflows/build-windows.yml` 构建，在 2 × V100、TP2、媒体
开启的配置下验收通过后转正。构建运行、归档清单与验收记录见
[docs/Releases/v0.1.1-win.5.md](docs/Releases/v0.1.1-win.5.md)。

| 资产 | 字节数 | sha256 |
|---|---:|---|
| `ninfer-v0.1.1-win.5-windows-x64-cuda12.8-sm70.zip` | 506,273,857 | `bea77fddd47edfbbcc8cbe7c1d24576ff56ca5eb7ab8fbd66083aaf01f0b2af5` |
| `ninfer-v0.1.1-win.5-windows-x64-cuda12.8-sm70.zip.sha256` | 115 | 归档的校验 sidecar |

压缩包共十个文件（未压缩 855,872,132 字节）：`ninfer.exe`、`ninfer-serve.exe` 与媒体构建所需的
八个 FFmpeg/libcurl DLL。下载后必须整体解压：媒体构建从 exe 所在目录加载 DLL，只拷 exe 会启动
失败。CUDA 运行时静态链接，不依赖 `cudart64_*` DLL。

模型是独立的 `.ninfer` 产物。官方 NVFP4 为 `qwen3_8_27b_nvfp4.ninfer`，23,719,715,844 字节，
sha256 `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82`，从
[Hugging Face](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) 下载；用
`Get-FileHash -Algorithm SHA256` 核对，不要使用源代码 checkpoint 或原始 GGUF。

## 常用命令

`ninfer.exe` 指解压目录内的可执行文件（把该目录加入 `PATH` 或用完整路径）；
`F:\models\ninfer\...` 换成实际产物路径。

```powershell
ninfer.exe --help
```

文本推理：

```powershell
ninfer.exe F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer ^
  --prompt "List the first five prime numbers, comma separated." ^
  --max-new 40 --no-thinking --greedy ^
  --max-context 4096 --kv-capacity 4096 --tp 2 --devices 0,1
```

预期 stdout 为 `2, 3, 5, 7, 11`；加载约 75 s，decode 约 40 tok/s。

视觉推理（在仓库根目录执行，fixture 中的媒体路径相对仓库根目录）：

```powershell
ninfer.exe F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer ^
  --messages examples\cli\messages\image_chart.json ^
  --vision --max-context 8192 --no-thinking --greedy --max-new 128 ^
  --tp 2 --devices 0,1
```

预期答出 `NIFER VISION 731`、三个红色圆形、蓝色方块在左侧，stderr 中出现 `vision` 计时行。从源码
构建时把 `ninfer.exe` 换成 `build-win-media\apps\ninfer.exe`，其余参数不变。`--messages` 接受
OpenAI 风格的 content parts，格式见 [REBUILD.md](REBUILD.md)。

## 下载后自检

```powershell
ninfer.exe inspect F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer
```

预期输出含 `model_id: qwen3.8-27b`、`weights_id: nvfp4`、`direct_io_check: ok`。同一 prompt 分别加
`--print-token-ids` 和 `--no-cuda-graph` 各跑一次，两次打印的 `generated ids` 必须相同，这是跨设备
decode graph 的一致性检查。

已知限制：源码的默认树（`build-win`）媒体为 stub，视觉需 `build-win-media` 树；133 个单元测试中
12 个失败、1 个未构建，均不涉及媒体路径；`SystemMonitor` 在 Windows 上无数据。完整列表见
[docs/windows.md](docs/windows.md#known-limitations)。

## 从源码构建

完整的工具链安装、测试与传输门见 [REBUILD.md](REBUILD.md)。媒体构建的关键命令：

```powershell
scripts\windows\fetch-media-deps.cmd
scripts\windows\configure-media.cmd
$env:NINFER_BUILD_DIR = "$PWD\build-win-media"
scripts\windows\build.cmd ninfer ninfer-serve
```

前置条件：Windows 11 x64、Visual Studio 2022 的 Desktop development with C++、Python 3.10+、约
30 GiB 磁盘空间；产物在 `build-win-media\apps\`。

## 发布流程

1. 推代码到 `master`。
2. 手动派发 `.github/workflows/build-windows.yml`：`tag` 填下一个未使用的 `vX.Y.Z-win.N`，`prefix`
   填 `test-`；运行发布 `test-<tag>` 预发布版，不会成为 Latest。
3. 在 V100 主机下载并解压到空目录，执行 [docs/windows.md](docs/windows.md#verification) 的验收项和
   `examples\cli\messages\*.json` 冒烟。
4. 通过则原地转正：`scripts\windows\promote-release.ps1 -Tag v0.1.1-win.6 -AcceptanceNote '...'`
   （先加 `-DryRun` 看计划）——改名 tag、清 prerelease、去 `[TEST]` 标题与横幅、重命名两个资产、
   重写 `.sha256`、删除残留 `test-` tag。不通过则删除 release 及其 tag，修代码后换下一个编号。
   验收结论写进 `docs/Releases/<tag>.md`。

`win.N` 只增不减。workflow **只响应手动派发**（不会因为推 `v*` tag 自动构建：转正本身会创建 tag，
旧触发器会因此多跑一次构建并撞上已存在的 release）。细节见
[Release flow](docs/windows.md#release-flow)。

英文文档是权威版本，本页只是入口（English docs stay authoritative; this page is an entry point）。
