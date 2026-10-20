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

This tree has media stubbed. For image/video input see
[Media: stubbed by default, enabled on demand](#media-stubbed-by-default-enabled-on-demand), which
adds FFmpeg and libcurl DLL dependencies.

For the test suites (including the transport gate):

```powershell
scripts\windows\configure.cmd -DBUILD_TESTING=ON
scripts\windows\build.cmd ninfer_allreduce_test
```

## Media: stubbed by default, enabled on demand

The Windows build defaults to `-DNINFER_STUB_MEDIA=ON`, which removes the FFmpeg and libcurl
dependencies while keeping every target and its link graph. Text, chat, tools and serving work
normally; image/video prompts and HTTP(S) media acquisition throw a clear runtime error instead of
returning partial data:

```text
media decode is unavailable in this build (decode_image): it was configured with
NINFER_STUB_MEDIA=ON, which omits the FFmpeg dependency
```

### Building the multimodal (media-enabled) tree

The media path needs development trees, not just runtimes: headers plus MSVC import libraries. Two
steps, both offline-friendly once the archives are cached under `_toolchain\media\`:

```powershell
scripts\windows\fetch-media-deps.cmd     # FFmpeg 8.1 shared + libcurl 8.11.1 (Schannel)
scripts\windows\configure-media.cmd      # build-win-media, NINFER_STUB_MEDIA=OFF
$env:NINFER_BUILD_DIR = "$PWD\build-win-media"
scripts\windows\build.cmd ninfer ninfer-serve
```

`fetch-media-deps.cmd` honours `NINFER_FFMPEG_URL` and `NINFER_CURL_URL` so a mirror can replace
the upstream hosts. It builds libcurl from source rather than downloading a binary because every
prebuilt curl for Windows ships either a MinGW import library or no headers at all.

Keep the two trees separate: `build-win` stays the stub build, `build-win-media` carries the media
build, and `scripts\windows\build.cmd` selects the tree through `NINFER_BUILD_DIR` (default
`build-win`).

`apps/CMakeLists.txt` stages the FFmpeg and libcurl DLLs next to both executables after linking, so
the media build runs from the tree without touching `PATH`. `ffmpeg-n8.1-...-shared` means shared
FFmpeg libraries: the DLL set must stay next to the binaries when they are copied elsewhere.

### Measured on this port (2×V100, `qwen3_8_27b_nvfp4`, `--tp 2`, `--vision`)

| Input form | Result |
|---|---|
| Local path with a non-ASCII name (`图纸.jpg`) | correct floor-plan description, vision 4.41 s, 2,144 prompt tokens |
| `http://127.0.0.1:8765/plan.jpg` | fetched through libcurl/Winsock, vision 4.41 s |
| `data:image/jpeg;base64,...` (230 KB URI) | correct, vision 4.54 s |
| 256×256 / 8 fps H.264 clip, `video_url` | "Based on the two frames provided", 170 prompt tokens, vision 0.16 s |
| `POST /v1/chat/completions` with `image_url` (`ninfer-serve --vision`) | 200 in 6.4 s, 2,147 prompt tokens, correct floor-plan answer |
| Text-only prompt regression | unchanged: 40.3 tok/s decode, 190.6 tok/s prefill |

Prompt tokens are the quickest sanity check: 57 tokens for a 64×64 clip means the frame produced
almost no merged patches, while 170 tokens for a 256×256 clip means the visual tokens arrived.

Two Windows-specific fixes were required in `src/product/media_acquire/acquire.cpp`:
`winsock2.h`/`ws2tcpip.h` with a one-shot `WSAStartup` and `gai_strerrorA`, and building
`std::filesystem::path` from the UTF-8 JSON value instead of the ANSI code page (otherwise every
non-ASCII media path is reported as "not a regular file").

### Still stubbed

`src/` has no `/proc`-based fallbacks: the Windows system monitor remains empty (open issue in
`HANDOFF.md`), independent of the media build.


## Continuous integration

`.github/workflows/build-windows.yml` is the only build pipeline in this repository — it replaced the
inherited Linux workflow. It calls the same `scripts\windows\*.cmd` a developer calls, so a CI tree
and a local tree come from identical CMake arguments: `sm_70`, `NINFER_STUB_MEDIA=OFF`, portable
CUDA 12.8, FFmpeg 8.1 and libcurl 8.11.1.

Two entry points:

| Trigger | Effect |
|---|---|
| Manual dispatch | Inputs `tag` (base tag) and `prefix`. `prefix: test-` publishes a **prerelease**; an empty prefix publishes a real release. |
| Push of a `v*` tag | Publishes a **real release** under that tag. This is the trap to know about: pushing `v0.1.1-win.3` to publish a tag also starts a build that creates the release for it. |

Selecting a tag manually (`Run workflow`, `prefix: test-`) is the way to exercise the pipeline without
touching the release list. The run publishes `test-<tag>` and marks it prerelease.

Three constraints shaped the job, each discovered the hard way:

- **`windows-2022`, not `windows-latest`.** The newest runner image ships Visual Studio 2026 (MSVC
  14.51). CUDA 12.8's front end rejects it (`crt/host_config.h` allows `_MSC_VER < 1950`) and then
  dies in `cudafe++` with `0xC0000005`. `windows-2022` stays on MSVC 14.4x, inside the accepted range
  and equal to the toolchain used for local builds.
- **`NINFER_BUILD_JOBS=4`.** A hosted runner has 4 cores and 16 GB; each `nvcc` can hold several GB
  while expanding CUTLASS templates, and `-j8` crashed `cicc.exe` on the local tree.
- **The release tag is created through the Releases API** (`softprops/action-gh-release`), not by
  `git push`. `GITHUB_TOKEN` acts as a GitHub App and may not create a ref whose commit touches
  `.github/workflows/*` without the `workflows` scope, which the `permissions:` block cannot grant.

The release ships **one zip archive** (`ninfer-<tag>-windows-x64-cuda12.8-sm70.zip`) plus a
sha256sum-style `.sha256` sidecar. The archive holds both executables and every FFmpeg and libcurl
DLL, because the media build loads its libraries from the executable directory: unzip it into one
directory and run the executables from there. `tar.exe -a -cf` (bsdtar) writes the zip, and a
separate step extracts it again and compares every file against the staged build output, so a
truncated or mis-built archive fails the run instead of the download.

A successful run takes 15-19 minutes on a hosted runner (measured: 15m23s for `v0.1.1-win.5`) and
turns ten files totalling 855,872,132 bytes into a 506,273,857-byte archive. The runner has no GPU,
so CI proves the tree configures, compiles, links and packages — it does **not** prove the binaries
produce correct output. Numerical and media verification stay local: the `ctest` suite, the runs in
[Verification](#verification) below, and the multimodal fixtures under
[`examples/cli`](../examples/cli/README.md).

### Release flow

CI can build any commit on demand, but it cannot verify one, so a release passes through a test
build, a local acceptance run, and only then a promotion or a deletion:

1. Push the code to `master`.
2. `Run workflow` with `tag` set to the next unused `vX.Y.Z-win.N` and `prefix` set to `test-`. The
   run publishes `test-<tag>` as a **prerelease**, so it never becomes Latest and never replaces a
   real release.
3. Download the archive on the V100 host, unzip it into an empty directory, and run the acceptance
   checks in [Verification](#verification); the media profile additionally replays
   [`examples/cli/messages/*.json`](../examples/cli/README.md).
4. Pass: **promote in place** — rename the tag to `<tag>`, clear `prerelease`, drop the `[TEST]`
   title and the body banner, rename both assets, rewrite the `.sha256` sidecar so it names the
   renamed archive, and delete the leftover `test-` tag. The promoted release becomes Latest as the
   newest non-prerelease. Fail: delete the release and its tag, fix the code, and build the next
   number — `win.N` only moves forward.

Promotion is a handful of Releases API calls, and like tag creation it cannot go through `git push`
(see the third constraint above). Renaming a tag with `PATCH /releases/:id` creates the new tag but
leaves the old ref behind, so the stale `test-` tag needs its own `DELETE /git/refs/tags/...`.

Pushing a `v*` tag skips steps 2-4 and publishes a real release straight from CI; use it only for a
build you are willing to publish unverified.

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

Long context is bounded by device memory, not by the artifact's 262,144-position ceiling: the KV
pool and the per-sequence reservation each grow by about 32 KiB per token per card, so the pair
costs about 64 KiB per token per card. `--max-context 32768 --kv-capacity 32768` still leaves
3.81 GiB free per card. Raise the pair one step at a time and read the `gpuN free/total` lines in
the load summary instead of jumping to the ceiling.

```powershell
build-win\apps\ninfer-serve.exe F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer ^
  --host 127.0.0.1 --port 18080 ^
  --max-context 32768 --kv-capacity 32768 --tp 2 --devices 0,1 --max-concurrency 1
```

## Verification

1. **Transport gate.** `tools\tp2\win\run_stability.ps1` runs `ninfer_allreduce_test` several times
   and checks the Windows System event log for NVLink/Xid failures. The test covers both transport
   routes (event/copy and the one-shot kernel), 500 host-drained iterations each, graph capture and
   rank skew, and prints `peer access: enabled (direct P2P)` when peer access qualified.
2. **Reader gate.** `ninfer.exe inspect <artifact>` must print the expected `model_id`/`weights_id`
   and `direct_io_check: ok`.
3. **Graph parity.** The same prompt with and without `--no-cuda-graph` must produce identical token
   ids (`--print-token-ids`). This is the check that a captured cross-device decode graph replays
   the same arithmetic as eager execution.
4. **Serving smoke.** `/health` returns 200 and a chat completion returns 200 with the expected
   content.
5. **Long-context ladder.** `python tools\tp2\win\make_prompt.py --reps 280 --out long.json` builds
   a prompt whose first sentence carries a six-digit code, pads with filler, and asks for the code
   at the end. Run it at `--max-context`/`--kv-capacity` 8192, 16384 and 32768 with reps 280, 763
   and 1556 (about 5.6k, 15.3k and 31.2k prompt tokens), `--no-thinking --greedy --max-new 16`, and
   require the answer to contain the code. This checks that attention still reaches the start of
   the window, not just that the allocation succeeded.
6. **Soak.** `python tools\tp2\win\soak.py --minutes 10` starts `ninfer-serve` on the Chinese-named
   artifact path at 32k context with `--max-concurrency 1`, cycles 8k/16k/24k/32k prompts, and
   samples VRAM at 1 Hz. A pass ends with `fail=0`, `concurrent_ok=True`, `gpu_released=True`, and
   an unchanged VRAM high-water mark between the first and last cycle.

Measured on 2 x V100-SXM2-16GB (TCC, NVLink), CUDA 12.8.61, MSVC 19.38, driver 576.57:

| Quantity | Value |
|---|---|
| Artifact load, 20.16 GiB weights | 73.6 s (H2D 70.1 s); `infer-serve` load plus warmup 81.7 s cold, 25.4 s with the artifact in the OS cache |
| Per-card weight residency | 10.08 GiB (reserved 10.79 GiB, 4.75 GiB free at 4k context) |
| Prefill, 5.6k / 15.3k / 31.2k prompt | 980 / 978 / 944 tok/s |
| Decode, batch 1 | 40.3 tok/s at 4k; 37.9 / 35.7 / 32.6 tok/s at 8k / 16k / 32k |
| Long context 8k / 16k / 32k, per card | KV pool 256 MiB / 512 MiB / 1.00 GiB; free after startup 4.56 / 4.31 / 3.81 GiB; peak 11,574 / 11,830 / 12,342 MiB |
| 10-minute soak, 32k, Chinese path, concurrency 1 | 32/32 requests returned the code; VRAM flat at 12,342 MiB per card |
| All-reduce, 5120 BF16 | one-shot mean 28-32 us, copy mean 43-52 us (host-drained) |

A 25-token prompt reports about 200 tok/s prefill because fixed per-request cost dominates it; the
throughput figure above is the one that scales with context.

### Unit test suite (`ctest` on the media tree)

`BUILD_TESTING=ON` in `build-win-media` registers 133 tests. On 2026-10-20 the suite ran to
completion: **121 pass, 12 fail, 1 not built**. The media-decoder regression
(`ninfer_media_decode_test`, the swscale width-tight RGB24 reproducer from upstream issue #20) and the
vision-attention op test both pass.

Requirements to reproduce the pass rate:

- Put `build-win-media\apps` on `PATH` or copy its DLLs — the test executables import the same
  FFmpeg/libcurl DLLs as the apps.
- Point `TMP`/`TEMP` at a writable directory. Tests that call
  `std::filesystem::temp_directory_path()` throw when it is not; MSVC's default `std::terminate`
  calls `abort()` without printing, so the failure surfaces only as exit code `0xC0000409` with no
  output. 12 of the 24 initial failures were this and pass once `TMP` is fixed.

The 12 remaining failures, with the signature each reports:

| Test | Signature |
|---|---|
| `ninfer_qwen3_6_frontend_test`, `ninfer_concurrent_executor_test` | `0xC0000409`, no output |
| `ninfer_linear_add_split_test`, `ninfer_linear_swiglu_split_test`, `ninfer_attn_input_proj_split_test`, `ninfer_gdn_projections_split_test` | `0xC0000409`, no output |
| `ninfer_attn_input_proj_test`, `ninfer_gdn_input_proj_test`, `ninfer_gdn_input_proj_conv_snapshot_test`, `ninfer_linear_add_nvfp4_test` | `0xC0000409`, no output |
| `ninfer_gdn_headsplit_test` | FP criterion: `shard relative_l2 0.000000 is worse than the tp1 geometry's 0.000000` |
| `ninfer_linear_swiglu_nvfp4_test` | `NVFP4 A4 execution requires an sm_120a GPU`, and `bad allocation` for `T=49`/`T=128` |

The `0xC0000409` group produces no output at all and exits within a second, i.e. the exception is
thrown before anything is flushed; the same fast-fail code and offset already appear in this tree's
WER archive for the pre-media `ninfer-compat.exe`, so the signature predates the media work. The two
numeric failures state capability/size expectations (sm_120a-only kernel, oversized fixture) rather
than decode or serving behaviour. None of the 12 touches the media path.

`ninfer_qwen3_8_27b_dflash2_real_test` is registered but not built here: it includes
`cuda_profiler_api.h`, which the portable CUDA toolkit extracted by
`scripts\windows\fetch_cuda_toolkit.py` does not ship.

## Three-artifact coding benchmark

October 19-20, 2026: the release server binary (`ninfer-serve.exe`, SHA256
`ffd20b9de0aa3e91f4e3dc2595051fe07172284e275bdc3b58df62bf8dbae7b5`, built from commit
`b0c7ffc0`) served three `qwen3.8-27b` artifacts on 2 x Tesla V100-SXM2 16 GB (TP2 over NVLink,
TCC, CUDA 12.8). Every run uses the same five coding prompts and the same profile, with one fresh
server per artifact. This is a same-binary cross-artifact throughput comparison; it is not an
answer-quality or fixed-output comparison. The earlier
[Coder390 Q6_K MTP](performance.md#v100-duo-coder390-q6_k-mtp) section measured the pelican prompt
at a 131,072-token window; these rates use different prompts and are not comparable.

### Profile and artifacts

```text
--tp 2 --devices 0,1 --kv-dtype int8 --max-concurrency 1 --no-prefix-reuse
--max-context 196608 --kv-capacity 196608 --prefill-chunk 2048
--greedy --reasoning-effort medium --spec mtp --draft-tokens 3
```

INT8 KV is what makes the 196,608-token window fit on a 16 GB card; prefix reuse is off, so every
request pays a cold prefill. A and C select the optimized proposal head (`--lm-head-draft`); B uses
the full proposal head.

| Code | Artifact | Weights | Proposal head |
|---|---|---|---|
| A | `qwen3_8_27b_nvfp4.ninfer` | official NVFP4 | optimized |
| B | `Qwen3.8-27B-Coder390-W4A4-MTP-DFlash2-NInferV2-NOT-FOR-V3.ninfer` | Coder390 W4A4 V2 (text-only load) | full |
| C | `Qwen3.8-27B-Coder390-EfficientThink-W4A4-MTP-DFlash2.ninfer` | Coder390 W4A4 V3 (Vision weights loaded but unused) | optimized |

B and C carry the same Coder390 W4A4 weights in different containers; A is a different model.

### Workloads

| id | Content | Output budget |
|---|---|---:|
| s1_geometry | plane quadrilateral / minimum dihedral-angle cosine, full derivation | 6,144 |
| s2_impl | C++20 bounded blocking queue plus GoogleTest | 4,096 |
| s3_debug | root cause and minimal fix for a concurrent-LRU use-after-free | 4,096 |
| s4_refactor | byte-equivalent refactor of a hot `build_report` | 4,096 |
| s5_long_context | ~189,000-character real repository source, cross-file lookup | 2,560 |

Each server executes the five non-streaming Chat Completions requests in order. They share the
process but not a cached prefix. The analyzer reads token counts and phase timings from the
server's `--request-log-jsonl` `request_done` events; HTTP transport and client-side decoding are
excluded. Decode rate is `(completion_tokens - 1) / decode_seconds`.

### Three-artifact results

| Code | Prompt tok | Completion tok | Startup s | Request s | Process s | Decode tok/s | End-to-end tok/s | MTP accept | Truncated / empty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 49,790 | 20,992 | 71.9 | 239.0 | 313.3 | **112.77** | **87.84** | 70.6% | 5 / 2 |
| B | 49,790 | 20,992 | 31.2 | 248.2 | 281.8 | 107.48 | 84.57 | 71.3% | 5 / 1 |
| C | 49,790 | 19,710 | 73.5 | 228.2 | 304.0 | 112.43 | 86.35 | 70.3% | 3 / 2 |
| Total | 149,370 | 61,694 | 176.6 | 715.4 | 899.2 | 110.81 | 86.23 | - | 13 / 5 |

Aggregate prefill for the run is 949.5 / 948.2 / 947.5 tok/s (A/B/C). The 48,056-token s5 prompt
prefills at 958.1 / 956.8 / 955.9 tok/s and takes 50.4 / 50.4 / 50.5 s to first token, which
dominates that scenario. A and C run within 0.3% of each other on decode; B trails by about 5%,
consistent with the full-vocabulary proposal head.

Per-scenario decode rates:

| Scenario | A | B | C |
|---|---:|---:|---:|
| s1_geometry | 112.20 | 108.70 | **114.32** |
| s2_impl | **118.88** | 112.42 | 117.19 |
| s3_debug | **110.01** | 102.57 | 107.15 |
| s4_refactor | **115.39** | 108.00 | 113.41 |
| s5_long_context | 105.75 | 104.48 | **107.46** |

A and C split the five scenarios 3-2, with gaps of 1.4%-2.7%, below the run-to-run resolution of a
single repetition. The only material split is B, which is 4.9%-6.8% behind the best artifact on
the four short prompts.

### DFlash2 on Coder390

C's DFlash2 companion cannot start at 196,608: the runtime reservation needs 5,323,270,656 B and
only 4,386,459,648 B remain after the DFlash2 weights load, 893 MiB short. `--kv-capacity auto`
does not help because the minimum 196,608 reservation already exceeds the free budget. Probing in
64-token steps puts the largest feasible window at **155,648** with `--prefill-chunk 2048`
(5.1 MiB of slack after startup).

The DFlash2 K=7 run therefore changes the speculative block to
`--spec dflash2 --draft-tokens 7 --lm-head-draft` and the window to
`--max-context 155648 --kv-capacity 155648`; everything else matches the main profile. All five
prompts fit in 155,648 tokens, so the smaller pool does not truncate the workload.

| Metric | C MTP3 K=3 @196,608 | C DFlash2 K=7 @155,648 |
|---|---:|---:|
| Completion tok | 19,710 | 19,198 |
| Decode tok/s | **112.43** | 108.71 |
| Request s | **228.2** | 230.2 |
| Acceptance | **70.3%** | 43.9% |
| Process s | **304.0** | 312.4 |

DFlash2 decodes 3.3% slower and its acceptance rate is 26 points lower, so the wider draft window
does not pay for its extra verification cost. Per scenario DFlash2 is +4.6% (s1), -0.2% (s2),
-14.0% (s3), -4.5% (s4) and -4.9% (s5) on decode rate.

A same-context control (C MTP3 at 155,648) separates the backend difference from the window change:

| Metric | MTP3 @155,648 | DFlash2 @155,648 | Delta |
|---|---:|---:|---:|
| Decode tok/s | **112.33** | 108.71 | -3.2% |
| Request s | **228.4** | 230.2 | +0.8% |
| Acceptance | **70.3%** | 43.9% | -26.4 pp |
| Startup slack | 1,101.8 MiB | 5.1 MiB | -1,096.7 MiB |

MTP3 loses only 0.1% when the window drops from 196,608 to 155,648, so the 3.3% gap against
DFlash2 is the backend, not the context ceiling. Per-scenario deltas match the table above:
+4.7%, -0.1%, -13.9%, -4.4% and -4.9%.

### Q6_K on Coder390

`Qwen3.8-27B-Coder390-EfficientThink-Q6_K-MTP.ninfer` (`qwen3.8-27b/gguf-blocks-full`, full
proposal head) also fails at 196,608: it needs 5,304,856,832 B against 4,486,056,960 B available,
781 MiB short. It starts at 155,648 with 165.7 MiB of slack. The comparison below uses the same five
prompts, greedy decoding, `reasoning_effort=medium`, chunk 2,048, INT8 KV and the 155,648 window;
Q6_K runs MTP3 without `--lm-head-draft`, so its proposal head differs from the W4A4 column.

| Metric | W4A4 MTP3 | Q6_K MTP3 | Delta |
|---|---:|---:|---:|
| Completion tok | 19,710 | 20,385 | +3.4% |
| Prefill tok/s | 947.8 | **1,501.0** | **+58.4%** |
| Decode tok/s | **112.33** | 82.30 | **-26.7%** |
| Request s | **228.4** | 281.3 | +23.1% |
| s5 TTFT | 50.5 s | **31.7 s** | -37.2% |
| s5 total | 74.3 s | **62.2 s** | **-16.3%** |
| Acceptance | 70.3% | 68.9% | -1.4 pp |
| Startup slack | 1,101.8 MiB | 165.7 MiB | -936.1 MiB |

Q6_K is a prefill-oriented profile: long prompts prefill about 58% faster, while decode is about
27% slower. On the four short prompts it is 26%-61% slower end to end; only the 48K-token s5
scenario wins (62.2 s vs 74.3 s). A coding workload dominated by long repository context and short
answers favors Q6_K; short prompts with long generations favor W4A4.

### Caveats

- 13 of 15 requests stopped at `max_tokens`, and five responses had empty `content` because the
  whole budget went to reasoning. Completion counts are budget-truncated and not full answer
  lengths; decode rate is unaffected, but totals depend on where each model stopped.
- Each scenario ran once. Decode differences below roughly 3% are not a stable advantage.
- B loaded in 29.8 s against about 71 s for A and C because the artifact was in the OS page cache.
  Startup times here are not a cold-start comparison.
- Response text was corrupted by Windows PowerShell 5.1 decoding UTF-8 as Latin-1 and was repaired
  losslessly. Token and timing data come from the server JSONL and were never affected.
- The largest prompt is 48,056 tokens, 24.4% of the 196,608 window. This campaign does not qualify
  decode with a fully occupied 196K context.

### Reproduction

The local harness (`run_bench.ps1`, `analyze.py`, the five prompt files) and the raw per-run
JSONL/responses stay in the uncommitted `_scratch/bench-coding-3models-20261019/` tree; the tables
above are the published summary. Reproduce a run with the profile block above, the artifact's
explicit path, one resident server per artifact, and greedy non-streaming requests with
`reasoning_effort: medium` and the per-scenario `max_tokens`. Read `request_done` from the request
log for the token counts and `prefill_seconds` / `decode_seconds` used here.

## Known limitations

- Media decode and URL acquisition are unavailable in the default (stubbed) build; the separate
  `build-win-media` tree enables them (see the media section above) and then needs the FFmpeg and
  libcurl DLLs next to the executables.
- 12 of the 133 unit tests fail on Windows; the list and signatures are in
  [Unit test suite](#unit-test-suite-ctest-on-the-media-tree). None of them touches the media path.
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
| `cl : 命令行 error D8050: 无法执行 c1.dll` | `TMP`/`TEMP` points at a directory that does not exist, so `cl` cannot write its debug record. Create the directory before configuring or building (the scripts do). |
| The media build starts but reports `media path is not a regular file` for a correct path | A path was rebuilt from a narrow string, decoding UTF-8 as the ANSI code page. Keep `media_path()` in `acquire.cpp` as the only way a media path becomes a `std::filesystem::path`. |
| `找不到 avformat-62.dll` at startup in the media build | The FFmpeg/`libcurl` DLLs are not next to the executable. Relink (`cmake --build ... --target ninfer`) so the post-build staging runs, or copy `_toolchain\media\ffmpeg-*/bin/*.dll` and `curl\bin\libcurl.dll` alongside. |
