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

A second tree, `build-win-media`, builds with `-DNINFER_STUB_MEDIA=OFF` and runs the multimodal
path end to end on Windows: local paths (including non-ASCII names), HTTP(S) URLs, base64 data URIs
and video input.

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
| Media build (`build-win-media`) | links FFmpeg 8.1 (avformat/avcodec/avutil/swscale) + libcurl 8.11.1 Schannel; no `cudart64_*.dll`, `WS2_32.dll` present |
| Image prompt, local UTF-8 path (`图纸.jpg`) | correct floor-plan description, vision 4.41 s, 2144 prompt tokens |
| Image prompt, `http://127.0.0.1:8765/plan.jpg` | fetched over libcurl + Winsock, correct, vision 4.41 s |
| Image prompt, 230 KB `data:image/jpeg;base64` URI | correct, vision 4.54 s |
| Video prompt, 256×256 / 8 fps H.264 (`video_url`) | "Based on the two frames provided", 170 prompt tokens, vision 0.16 s |
| Serve with `--vision`, `image_url` over HTTP | `/health` ok (vision on), text completion `ready`, image completion 200 in 6.4 s with 2,147 prompt tokens |
| Unit test suite (`ctest`, 133 registered) | 121 pass, 12 fail, 1 not built — list and signatures in `docs/windows.md` |
| Text regression in the media build | 40.33 tok/s decode, 190.59 tok/s prefill, `2, 3, 5, 7, 11` |

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
- Media build: `cmake/FindFFMPEG.cmake` (upstream assumed pkg-config or a vcpkg config package),
  `CMAKE_MODULE_PATH` in `CMakeLists.txt`, `scripts/windows/fetch-media-deps.cmd`,
  `scripts/windows/configure-media.cmd`, `NINFER_BUILD_DIR` in `scripts/windows/build.cmd`, and a
  post-build DLL staging step in `apps/CMakeLists.txt`.
- `src/product/media_acquire/acquire.cpp`: `winsock2.h`/`ws2tcpip.h` with a one-shot `WSAStartup`
  and `gai_strerrorA`, `ws2_32` link, UTF-8 `std::filesystem::path` construction (`media_path`),
  and a `native()`-typed parent check for the media-root guard.
- `tests/`: the host suite now builds on Windows. `<unistd.h>`/`<sys/wait.h>` guards and `getpid`
  wrappers, a `_aligned_malloc`-based `aligned_head`/`aligned_release` pair instead of
  `std::aligned_alloc`, `const` instead of `constexpr` where the initializer calls `std::sqrt`, and
  `/utf-8` for every test target (MSVC decodes sources as the ANSI code page otherwise, which both
  broke multi-byte literals and rejected the Chinese prompt fixtures).

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
4. **Media is stubbed in the default tree.** `build-win-media` covers it and the serving API with
   media parts is now driven over HTTP (`_scratch/mediasmoke.ps1`, 5/5). The default `build-win` tree
   still omits FFmpeg/libcurl by design.
5. **12 of 133 unit tests fail on Windows.** Signatures and the per-test list are in
   `docs/windows.md`; none touches the media path. Ten exit `0xC0000409` with no output (the same
   fast-fail signature already recorded for the pre-media `ninfer-compat.exe`), two are numeric or
   capability assertions. `ninfer_qwen3_8_27b_dflash2_real_test` does not build because the portable
   CUDA toolkit ships no `cuda_profiler_api.h`.
6. **Test runs need a writable `TMP`.** Tests calling `std::filesystem::temp_directory_path()` abort
   silently (`0xC0000409`, no output) when it is unavailable, because MSVC's default terminate
   handler calls `abort()`. Point `TMP`/`TEMP` at a real directory before `ctest`.

## Next steps

1. Produce the independent numerical comparison for Phase 5 L2/L3 (same artifact, same prompts,
   temperature 0, compare token ids; if only a Linux baseline is available, record its ids).
   Cover the `vision_disabled` guard next: the positive media path over HTTP is done, the negative
   path (media part sent to a server started without `--vision`) is not.
2. Port `SystemMonitor` to Windows (`GlobalMemoryStatusEx`, `GetSystemTimes`, NVML via
   `LoadLibraryW`).
3. Promote the measured numbers into `docs/performance.md` if the port becomes a supported target.

## Layout

```text
scripts/windows/configure.cmd     configure (portable CUDA, stub media, sm_70)
scripts/windows/configure-media.cmd  configure the media-enabled tree (build-win-media)
scripts/windows/fetch-media-deps.cmd FFmpeg dev tree + libcurl (Schannel) into _toolchain\media
scripts/windows/build.cmd         build targets
scripts/windows/fetch_cuda_toolkit.py
tools/tp2/win/run_stability.ps1   transport gate + event log
tools/tp2/win/make_prompt.py      long-context needle prompt generator
tools/tp2/win/soak.py             long-context soak with VRAM sampling
docs/windows.md                   build/run/verify guide
REBUILD.md                        exact reproduction recipe
probe/ (local, untracked)         ad-hoc Phase 0 experiments; not part of the build
```
