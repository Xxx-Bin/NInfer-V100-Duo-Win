# Handoff: NInfer Windows TP2 port

## State

The port runs end to end on Windows 11 with two V100-SXM2-16GB cards in TCC:

- `ninfer.exe` and `ninfer-serve.exe` build from a clean tree with MSVC 14.38 and portable
  CUDA 12.8 (`sm_70`), media stubbed.
- `--tp 2 --devices 0,1` loads the 23.7 GB Qwen3.8-27B NVFP4 artifact, splits weights 10.08 GiB per
  card, captures the cross-device decode graph, and generates correct text at ~40 tok/s (batch 1).
  Prefill measures 944-980 tok/s on 5.6k-31.2k prompts; a 25-token prompt reports ~200 tok/s
  because fixed per-request cost dominates it.
- Direct P2P over NVLink qualifies and is used; the host-staged fallback also validates and is
  exercised by the test suite.
- Long context ramps to 32k (needle still retrieved at 31.2k prompt tokens) and a 10-minute soak on
  the Chinese-named artifact path at 32k context with concurrency 1 runs without drift or failure.
- Serving answers `/health` and `/v1/chat/completions` with the expected status codes and content.

This satisfies Phase 0 (runtime gate), Phase 5 (end-to-end TP2), and Phase 6/7 (long context and
soak). It does not yet include an independent numerical comparison against a Linux baseline.

## Evidence

| Check | Result |
|---|---|
| `cudaDeviceCanAccessPeer` both directions | 1 / 1, `tccDriver=1`, 6 NVLink links at 25.781 GB/s |
| Exact 16 KiB cross-device copy, both directions | pass |
| `ninfer_allreduce_test` (500 iterations per route, capture, skew) | pass, three consecutive runs, `peer access: enabled (direct P2P)` |
| Artifact size / SHA-256 | 23,719,715,844 B / `74d2c571...b77d82`, matches the published value |
| `ninfer.exe inspect` | `qwen3.8-27b` / `nvfp4`, 1190 objects, `direct_io_check: ok (65536 bytes)` |
| TP2 load | 20.16 GiB weights, 10.08 GiB per card, 73.6 s (70.1 s H2D) |
| Decode / prefill | 40.3 tok/s; 980/978/944 tok/s prefill at 5.6k/15.3k/31.2k prompt |
| Graph vs eager | identical token ids (`17 11 220 18 11 220 20 11 220 22 11 220 16 16 248046`) |
| Long context 8k/16k/32k | code retrieved at 5,650/15,310/31,170 prompt tokens; 4.56/4.31/3.81 GiB free per card |
| 10-minute soak (32k, Chinese path, concurrency 1) | 32/32 requests correct, concurrent pair ok, VRAM flat at 12,342 MiB per card |
| Chinese artifact path (hard link) | inspect and direct-I/O cross-check pass |
| Serve smoke | `/health` 200, chat completion 200, content `21` |

## Changes

- `CMakeLists.txt`, `src/CMakeLists.txt`: `NINFER_STUB_MEDIA`, static CUDA runtime on Windows,
  `UTF8PROC_STATIC`, media target selection; `tests/`/`bench/` use the configured runtime target.
- `src/media/decode/decode_stub_win.cpp`, `src/product/media_acquire/acquire_stub_win.cpp`: symbols
  preserved, runtime error on use.
- `src/artifact/reader.cpp`: Windows mapped-file and aligned-read implementation.
- `src/core/nvtx*.h`: `<cstdlib>` before NVTX v3; `src/targets/.../api_impl.h`: explicit move
  definitions for MSVC.
- `src/product/{load_progress,logging}`, `src/serve/{console_log,request_log,startup_log}`:
  `_isatty`, `localtime_s`, `_getpid`, console width.
- `apps/cli`: `inspect` subcommand (identity plus reader cross-check, no CUDA context).
- `scripts/windows/`: configure/build scripts and the portable toolkit fetcher.
- `tools/tp2/win/run_stability.ps1`: transport gate with event-log monitoring.
- `tools/tp2/win/make_prompt.py`: long-context prompt generator (needle at the start, question at the end).
- `tools/tp2/win/soak.py`: 10-minute soak harness (server lifecycle, 1 Hz VRAM sampling, concurrency
  check); Python avoids the PowerShell 5.1 UTF-8 script decoding problem.
- `docs/windows.md`, `REBUILD.md`: build and verification documentation.

## Open issues

1. **Independent numerical baseline.** Graph/eager equality and correct answers are strong
   self-consistency evidence, but Phase 5 L2 asks for a comparison against an independent reference
   implementation on the same artifact. No such reference output exists on this machine yet.
2. **Transient startup failure, not reproduced.** One `--max-context 4096` start exited silently
   during weight upload (exit -1, no application crash event). Three identical subsequent runs
   succeeded. Every later start in the 8k/16k/32k ladder and the soak loaded first time. A start
   issued while a previous server is still releasing GPUs fails cleanly with "only N bytes are free
   before loading weights"; `tools/tp2/win/soak.py` now refuses to start on a non-idle GPU instead.
   Keep an eye on it; capture a dump if it recurs.
3. **System monitor is empty on Windows.** The implementation is `/proc`-based and has no Windows
   branch. Serving is unaffected. Porting `cpu_model_name`, memory, CPU load and the NVML loader to
   Windows APIs is follow-up work.
4. **Media is stubbed.** Image/video input and URL acquisition require an FFmpeg/libcurl Windows
   toolchain and `-DNINFER_STUB_MEDIA=OFF`.

## Next steps

1. Produce the independent numerical comparison for Phase 5 L2/L3 (same artifact, same prompts,
   temperature 0, compare token ids; if only a Linux baseline is available, record its ids).
2. Port `SystemMonitor` to Windows (`GlobalMemoryStatusEx`, `GetSystemTimes`, NVML via
   `LoadLibraryW`).
3. Promote the measured numbers into `docs/performance.md` if the port becomes a supported target.

## Layout

```text
scripts/windows/configure.cmd     configure (portable CUDA, stub media, sm_70)
scripts/windows/build.cmd         build targets
scripts/windows/fetch_cuda_toolkit.py
tools/tp2/win/run_stability.ps1   transport gate + event log
tools/tp2/win/make_prompt.py      long-context needle prompt generator
tools/tp2/win/soak.py             long-context soak with VRAM sampling
docs/windows.md                   build/run/verify guide
REBUILD.md                        exact reproduction recipe
probe/ (local, untracked)         ad-hoc Phase 0 experiments; not part of the build
```
