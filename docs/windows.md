# NInfer on Windows

Native Windows 11 x64 build of NInfer, targeting two Tesla V100-SXM2-16GB cards in **TCC** mode and
tensor-parallel execution (`--tp 2`). The port keeps the product surface unchanged: one `.ninfer`
artifact, the `ninfer` CLI and `ninfer-serve`, the same Op contracts and the same serving API. The
platform work is confined to file I/O, terminal detection, timestamps and build plumbing.

The reproducible step-by-step recipe, including the toolchain installation, is
[REBUILD.md](../REBUILD.md). This page is the guide: what the build needs, how to run it, how it is
verified, and what is intentionally missing.

## Why CUDA 12.8 and not 13

The V100 is `sm_70`. CUDA 13 removed offline compilation for Volta, so the build rejects a 13.x
`nvcc` on `CMAKE_CUDA_ARCHITECTURES=70` and this port pins 12.8. The toolkit is installed as a
portable tree under `_toolchain/cuda-12.8` (see below); no system-wide CUDA installation or
elevated installer is required.

## Requirements

- Windows 11 x64; NVIDIA driver supporting CUDA 12.8 (validated with 576.57).
- Two identical compute-capability-7.0 GPUs, both in TCC (`nvidia-smi --query-gpu=driver_model.current`).
- Visual Studio 2022 17.8 with the "Desktop development with C++" workload (MSVC 19.38 / 14.38).
  Other 2022 releases work; the build only needs `vcvars64.bat` and the x64 toolset.
- CMake 3.28 or newer and Ninja. The scripts use `.tools\venv\Scripts\{cmake,ninja}.exe` when that
  virtual environment exists and fall back to `PATH`.
- Python 3.10+ for the toolkit fetcher and for downloading the artifact.

## Toolchain setup

```powershell
python -m venv .tools\venv
.\.tools\venv\Scripts\python.exe -m pip install cmake ninja

python scripts\windows\fetch_cuda_toolkit.py
git clone --depth 1 --branch v4.4.2 https://github.com/nvidia/cutlass.git _toolchain\cutlass
```

`fetch_cuda_toolkit.py` downloads the four CUDA 12.8 redistributable components the build consumes
(`cuda_nvcc`, `cuda_cudart`, `cuda_cccl`, `cuda_nvtx`, about 130 MB) and merges them into
`_toolchain\cuda-12.8`, which CMake receives as `CUDAToolkit_ROOT`. The CUTLASS checkout is optional
but recommended: CMake's `FetchContent` runs `git submodule update` through MSYS `sh`, which fails in
restricted environments; the configure script passes the local checkout as
`FETCHCONTENT_SOURCE_DIR_CUTLASS` when `_toolchain\cutlass` exists.

## Configure and build

From a normal PowerShell window (the scripts call `vcvars64.bat` themselves):

```powershell
scripts\windows\configure.cmd
scripts\windows\build.cmd ninfer ninfer-serve
```

Outputs:

```text
build-win\apps\ninfer.exe
build-win\apps\ninfer-serve.exe
```

Both are self-contained apart from the MSVC runtime: the CUDA runtime is linked statically on
Windows, so no `cudart64_*.dll` is required at run time. Verify with
`dumpbin /dependents build-win\apps\ninfer.exe`.

For the test suites (including the transport gate):

```powershell
scripts\windows\configure.cmd -DBUILD_TESTING=ON
scripts\windows\build.cmd ninfer_allreduce_test
```

## Media is stubbed in this build

The Windows build defaults to `-DNINFER_STUB_MEDIA=ON`, which removes the FFmpeg and libcurl
dependencies while keeping every target and its link graph. Text, chat, tools and serving work
normally; image/video prompts and HTTP(S) media acquisition throw a clear runtime error instead of
returning partial data:

```text
media decode is unavailable in this build (decode_image): it was configured with
NINFER_STUB_MEDIA=ON, which omits the FFmpeg dependency
```

Build a media-capable Windows tree by providing FFmpeg and libcurl through vcpkg or a system
prefix and configuring with `-DNINFER_STUB_MEDIA=OFF`.

## Artifact

The port was validated with the official Qwen3.8-27B NVFP4 artifact:

```text
qwen3_8_27b_nvfp4.ninfer
23,719,715,844 bytes
sha256 74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82
```

Check the size and hash after downloading (`Get-FileHash -Algorithm SHA256`), then read the identity
without creating a CUDA context:

```powershell
build-win\apps\ninfer.exe inspect F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer
```

```text
model_id: qwen3.8-27b
weights_id: nvfp4
file_bytes: 23719715844
payload_offset: 393216
objects: 1190
direct_io_check: ok (65536 bytes)
```

`inspect` also cross-checks the mapped view against the aligned `ReadFile`/`OVERLAPPED` path on the
first object, which is how the Windows reader is validated on a real file without a GPU.

## Running TP2

```powershell
build-win\apps\ninfer.exe F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer ^
  --prompt "List the first five prime numbers, comma separated." ^
  --max-new 40 --no-thinking --greedy ^
  --max-context 4096 --kv-capacity 4096 ^
  --tp 2 --devices 0,1
```

Serving:

```powershell
build-win\apps\ninfer-serve.exe F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer ^
  --host 127.0.0.1 --port 18080 ^
  --max-context 4096 --kv-capacity 4096 --tp 2 --devices 0,1 --max-concurrency 1
```

`GET /health` returns `{"model":"qwen3.8-27b","status":"ok"}` once loading finishes;
`POST /v1/chat/completions` behaves as documented in [serving.md](serving.md).

## Verification

1. **Transport gate.** `tools\tp2\win\run_stability.ps1` runs `ninfer_allreduce_test` several times
   and checks the Windows System event log for NVLink/Xid failures. The test covers both transport
   routes (event/copy and the one-shot kernel), 500 host-drained iterations each, graph capture and
   rank skew, and prints `peer access: enabled (direct P2P)` when peer access qualified.
2. **Reader gate.** `infer.exe inspect <artifact>` must print the expected `model_id`/`weights_id`
   and `direct_io_check: ok`.
3. **Graph parity.** The same prompt with and without `--no-cuda-graph` must produce identical token
   ids (`--print-token-ids`). This is the check that a captured cross-device decode graph replays
   the same arithmetic as eager execution.
4. **Serving smoke.** `/health` returns 200 and a chat completion returns 200 with the expected
   content.

Measured on 2 x V100-SXM2-16GB (TCC, NVLink), CUDA 12.8.61, MSVC 19.38, driver 576.57:

| Quantity | Value |
|---|---|
| Artifact load, 20.16 GiB weights | 73.6 s (H2D 70.1 s) |
| Per-card weight residency | 10.08 GiB (reserved 10.79 GiB, 4.75 GiB free) |
| Prefill | 181-216 tok/s |
| Decode, batch 1 | 40.3 tok/s |
| All-reduce, 5120 BF16 | one-shot mean 28-32 us, copy mean 43-52 us (host-drained) |

## Known limitations

- Media decode and URL acquisition are unavailable in the default (stubbed) build.
- `SystemMonitor` reports no host/GPU readings on Windows yet: its implementation is `/proc`-based
  on Linux and returns empty rows elsewhere. Serving is unaffected; the monitor page simply has no
  data. Porting it to `GlobalMemoryStatusEx`/NVML is follow-up work, not a dependency.
- The one-shot all-reduce route is admitted for payloads up to 131,072 BF16 elements; larger
  prefill reductions use the event/copy route, exactly as on Linux.
- The Windows event-log check needs a readable System log. When the log is unavailable the script
  warns and relies on the test's own exit code.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `nvcc fatal: Could not open output file ...\Temp\tmpxft_*` | The session temp directory is not writable for the compiler. The scripts set `TMP`/`TEMP` to `build-win-tmp`; do the same when invoking `nvcc` manually. |
| CMake fails in `cutlass-populate` with `sh.exe: ... NtCreateDirectoryObject` | MSYS `sh` cannot create its shared-memory object. Pre-clone CUTLASS into `_toolchain\cutlass` (the configure script picks it up). |
| `LNK2019` for `SequencePlan`/`RequestPlan` move constructors | MSVC does not emit an out-of-line `template <> ... = default;` that the defining TU never uses. The definitions in `api_impl.h` are spelled out; keep them that way. |
| `error C3861: '_wgetenv'` from `nvtx3` | Include `<cstdlib>` before `<nvtx3/nvToolsExt.h>`; `src/core/nvtx.h` does this. |
| `cicc.exe` crash dump in `build-win-tmp` while several CUDA objects compile | Parallel `nvcc` processes exhausted host memory. Rerun, or cap parallelism with `set NINFER_BUILD_JOBS=4` before `scripts\windows\build.cmd`. |
| Ninja hangs during configure or build | The harness sandbox can block compiler child-process pipes; run the build in a normal shell. |
| `cudaErrorMemoryAllocation` from `cudaSetDevice` | Another process owns the GPUs. Check `nvidia-smi --query-compute-apps`. |
