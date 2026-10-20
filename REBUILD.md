# Rebuilding the Windows TP2 port

This file is the exact recipe for reproducing the native Windows build of NInfer on two V100
SXM2 cards. It assumes a clean checkout of this repository; all local toolchains live under
`_toolchain/` and `.tools/`, which are git-ignored.

A GPU-free version of these steps runs in CI: `.github/workflows/build-windows.yml` executes the same
`scripts\windows\*.cmd` on a `windows-2022` runner and publishes the executables. Use it to check the
recipe still works without a local machine; use this file when you need the resulting binaries to run
on the V100s.

## 0. Prerequisites

- Windows 11 x64.
- Visual Studio 2022 with **Desktop development with C++** (`vcvars64.bat` must exist).
- Python 3.10 or newer on `PATH`.
- Two Tesla V100-SXM2-16GB cards in TCC mode:
  `nvidia-smi --query-gpu=index,name,driver_model.current,memory.total --format=csv`.
- About 30 GiB free on the drive that holds the build tree and 30 GiB for the artifact.

## 1. Source and toolchain

```powershell
git clone https://github.com/Xxx-Bin/NInfer-V100-Duo-Win.git C:\src\NInfer-V100-Duo-win
cd C:\src\NInfer-V100-Duo-win
git checkout master

python -m venv .tools\venv
.\.tools\venv\Scripts\python.exe -m pip install cmake ninja

python scripts\windows\fetch_cuda_toolkit.py
git clone --depth 1 --branch v4.4.2 https://github.com/nvidia/cutlass.git _toolchain\cutlass
```

Expected: `.tools\venv\Scripts\cmake.exe --version` prints 3.31 or newer;
`_toolchain\cuda-12.8\bin\nvcc.exe --version` prints `release 12.8, V12.8.61`.

## 2. Configure

```powershell
scripts\windows\configure.cmd
```

Expected tail:

```text
-- Configuring done
-- Generating done
-- Build files have been written to: .../build-win
```

The script selects `CMAKE_CUDA_ARCHITECTURES=70`, points CMake at the portable toolkit, enables
`NINFER_STUB_MEDIA` (no FFmpeg/libcurl), and keeps compiler temporaries inside `build-win-tmp`.

## 3. Build

```powershell
scripts\windows\build.cmd ninfer ninfer-serve
```

Expected artifacts:

```text
build-win\apps\ninfer.exe
build-win\apps\ninfer-serve.exe
```

### 3b. Media-enabled (multimodal) build

The default configure script stubs media. For image/video input, build a second tree instead of
changing the first one:

```powershell
scripts\windows\fetch-media-deps.cmd        # FFmpeg 8.1 dev tree + libcurl 8.11.1 (Schannel)
scripts\windows\configure-media.cmd         # -DNINFER_STUB_MEDIA=OFF, build-win-media
$env:NINFER_BUILD_DIR = "$PWD\build-win-media"
scripts\windows\build.cmd ninfer ninfer-serve
```

Expected: `build-win-media\apps\ninfer.exe`, and the FFmpeg/`libcurl` DLLs staged next to it by the
post-build step. Reusing an existing `build-win` tree is possible — copy the directory, rewrite
`CMAKE_CACHEFILE_DIR` inside `CMakeCache.txt` to the new path, then run `configure-media.cmd` — but
the copied dependency log still points at the old tree, so expect a full recompile anyway.

Verify with one image prompt (the artifact ships vision weights):

```powershell
build-win-media\apps\ninfer.exe F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer ^
  --messages _scratch\media-test\image.json --vision --max-new 60 --no-thinking --greedy ^
  --max-context 4096 --kv-capacity 4096 --tp 2 --devices 0,1
```

`--messages` takes OpenAI-style content parts:
`{"type":"text","text":...}` plus `{"type":"image_url","image_url":{"url":"<path|http(s) URL|data: URI>"}}`
or `{"type":"video_url","video_url":{"url":...}}`. A working run reports a `vision` timing line and
prompt tokens well above the text-only count.

The media tree also builds the host test suite. Put `build-win-media\apps` on `PATH` (the tests
import the same FFmpeg/libcurl DLLs) and point `TMP`/`TEMP` at a writable directory before running
`ctest`; without a usable temp directory the tests that need one abort with `0xC0000409` and no
output. Current pass rate and the remaining failures: `docs\windows.md`.

The first build compiles roughly 430 translation units and takes 20-40 minutes on eight cores. If
`cicc.exe` crashes while several `nvcc` processes run in parallel, the machine is out of memory;
retry the same command (the failed units are recompiled) or cap the parallelism:

```powershell
set NINFER_BUILD_JOBS=4
scripts\windows\build.cmd ninfer ninfer-serve
```

Verify the statically linked CUDA runtime:

```powershell
dumpbin /dependents build-win\apps\ninfer.exe
```

The list must not contain `cudart64_*.dll`.

## 4. Transport gate

```powershell
scripts\windows\configure.cmd -DBUILD_TESTING=ON
scripts\windows\build.cmd ninfer_allreduce_test
powershell -ExecutionPolicy Bypass -File tools\tp2\win\run_stability.ps1
```

Expected: `peer access: enabled (direct P2P)`, `OK allreduce` on every round, and
`RESULT: GO (all rounds passed, no driver events)`.

## 5. Artifact

Download `qwen3_8_27b_nvfp4.ninfer` from
<https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer> (a mirror works when the primary host is
unreachable), then:

```powershell
Get-Item F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer | Select-Object Length
# 23719715844
(Get-FileHash F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer -Algorithm SHA256).Hash
# 74D2C57145E6FF11D1D2FAA79594477F9BC903A611AF1FB20218189FBBB77D82

build-win\apps\ninfer.exe inspect F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer
```

Expected `model_id: qwen3.8-27b`, `weights_id: nvfp4`, `direct_io_check: ok`.

## 6. Run

```powershell
build-win\apps\ninfer.exe F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer ^
  --prompt "List the first five prime numbers, comma separated." ^
  --max-new 40 --no-thinking --greedy ^
  --max-context 4096 --kv-capacity 4096 --tp 2 --devices 0,1
```

Expected output `2, 3, 5, 7, 11`, about 40 tok/s decode and 180-220 tok/s prefill after a ~75 s load.

Graph parity check: rerun with `--print-token-ids` once as above and once with `--no-cuda-graph`.
The `generated ids` lines must be identical.
