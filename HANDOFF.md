# Handoff: NInfer Windows TP2 port

## State

The port runs end to end on Windows 11 with two V100-SXM2-16GB cards in TCC:

- `ninfer.exe` and `ninfer-serve.exe` build from a clean tree with MSVC 14.38 and portable
  CUDA 12.8 (`sm_70`), media stubbed.
- `--tp 2 --devices 0,1` loads the 23.7 GB Qwen3.8-27B NVFP4 artifact, splits weights 10.08 GiB per
  card, captures the cross-device decode graph, and generates correct text at ~40 tok/s (batch 1)
  with 180-220 tok/s prefill.
- Direct P2P over NVLink qualifies and is used; the host-staged fallback also validates and is
  exercised by the test suite.
- Serving answers `/health` and `/v1/chat/completions` with the expected status codes and content.

This satisfies Phase 0 (runtime gate) and the working half of Phase 5 (end-to-end TP2). It does not
yet include an independent numerical comparison against a Linux baseline, long-context runs, or a
10-minute soak.

## Evidence

| Check | Result |
|---|---|
| `cudaDeviceCanAccessPeer` both directions | 1 / 1, `tccDriver=1`, 6 NVLink links at 25.781 GB/s |
| Exact 16 KiB cross-device copy, both directions | pass |
| `ninfer_allreduce_test` (500 iterations per route, capture, skew) | pass, three consecutive runs, `peer access: enabled (direct P2P)` |
| Artifact size / SHA-256 | 23,719,715,844 B / `74d2c571...b77d82`, matches the published value |
| `infer.exe inspect` | `qwen3.8-27b` / `nvfp4`, 1190 objects, `direct_io_check: ok (65536 bytes)` |
| TP2 load | 20.16 GiB weights, 10.08 GiB per card, 73.6 s (70.1 s H2D) |
| Decode / prefill | 40.3 tok/s / 181-216 tok/s |
| Graph vs eager | identical token ids (`17 11 220 18 11 220 20 11 220 22 11 220 16 16 248046`) |
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
- `docs/windows.md`, `REBUILD.md`: build and verification documentation.

## Open issues

1. **Independent numerical baseline.** Graph/eager equality and correct answers are strong
   self-consistency evidence, but Phase 5 L2 asks for a comparison against an independent reference
   implementation on the same artifact. No such reference output exists on this machine yet.
2. **Long context and soak.** 8k/32k context, KV-cache growth and a 10-minute stability run have
   not been executed on Windows. Startup reserves 10.79 GiB per card and leaves 4.75 GiB free at
   `--max-context 4096`; long context has to be ramped, not jumped to 131072.
3. **Transient startup failure, not reproduced.** One `--max-context 4096` start exited silently
   during weight upload (exit -1, no application crash event). Three identical subsequent runs
   succeeded. Keep an eye on it; capture a dump if it recurs.
4. **System monitor is empty on Windows.** The implementation is `/proc`-based and has no Windows
   branch. Serving is unaffected. Porting `cpu_model_name`, memory, CPU load and the NVML loader to
   Windows APIs is follow-up work.
5. **Media is stubbed.** Image/video input and URL acquisition require an FFmpeg/libcurl Windows
   toolchain and `-DNINFER_STUB_MEDIA=OFF`.

## Next steps

1. Produce the independent numerical comparison for Phase 5 L2/L3 (same artifact, same prompts,
   temperature 0, compare token ids; if only a Linux baseline is available, record its ids).
2. Ramp context 8k -> 32k, recording free memory per card; then run the 10-minute soak on a Chinese
   path with concurrency 1.
3. Port `SystemMonitor` to Windows (`GlobalMemoryStatusEx`, `GetSystemTimes`, NVML via
   `LoadLibraryW`).
4. Record the final performance table in `docs/performance.md` only after the soak; the numbers in
   `docs/windows.md` are a first light-load measurement.
5. Regenerate `patches/` from the commit series whenever the port changes.

## Layout

```text
scripts/windows/configure.cmd     configure (portable CUDA, stub media, sm_70)
scripts/windows/build.cmd         build targets
scripts/windows/fetch_cuda_toolkit.py
tools/tp2/win/run_stability.ps1   transport gate + event log
docs/windows.md                   build/run/verify guide
REBUILD.md                        exact reproduction recipe
probe/ (local, untracked)         ad-hoc Phase 0 experiments; not part of the build
```
