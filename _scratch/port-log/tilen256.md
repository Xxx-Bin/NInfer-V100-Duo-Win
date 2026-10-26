# port/tilen256 — NVFP4 CUTLASS TileN=256

- 日期：2026-10-25
- 基线：`baseline/port-study-34412f4`（HEAD 34412f4f）
- worktree：`.agents/worktrees/port-tilen256`
- 分支：`port/tilen256`（**已删除，未打 keep tag**）
- 快照参照：`_scratch/ref-v100x2/ninfer-V100X2-master/src/ops/linear/nvfp4/nvfp4_cutlass_sm70.cu`

## 判定

**不保留（REJECT）。**

| 判据 | 门槛 | 实测 | 结果 |
|---|---:|---:|---|
| oracle 指纹 | 严格相等 | sha1 `06287348665d3648` / acc 406 / drf 421 / rounds 106 / ar 0.9643705 / al 4.8301887 / pp [105,102,100,99] | **PASS**（两次 rep 全同） |
| prefill（pp60000） | ≥ 955.257 tok/s | 948.458（rep 949.530 / 947.385） | **FAIL**（+0.281%） |
| decode（tg512, MTP4） | ≥ 130.376 tok/s | 129.717（rep 129.696 / 129.737） | **FAIL**（−0.008%） |

两项均低于门槛 → 删除分支，不留 tag。

## 改动摘要（`+40 −12`，单文件）

`src/ops/linear/nvfp4/nvfp4_cutlass_sm70.cu`（diff 存档 `tilen256.diff`）：

1. `ShapeMMAThreadBlock` 与 `EpilogueOp`/`Gemm` 改为模板化：新增
   `template <class ElementOutput, int TileN = 128> using GemmFor`，TileK 固定 32、warp 仍 64x64。
2. `nvfp4_cutlass_sm70_workspace_bytes` 主体移入 anon namespace 改名 `workspace_bytes_impl`。
3. 原 `nvfp4_cutlass_sm70_launch` 主体移入 `launch_impl<ElementOutput, TileN>`。
4. 新增 dispatch：`x.ne[1] >= 2048 && ((w.n==17408 && w.k==5120) || (w.n==5120 && w.k==8704))`
   → `launch_impl<..., 256>`，否则 128。

**刻意未移植**（快照同文件里存在但属其他功能）：
- 快照的 QPN dequant kernel 重写（tuple-wise 遍历，`uint2` 读 + 32B 写，消除每 tuple 双解码）——
  独立优化项，不属 TileN。
- 快照的 `nvfp4_cutlass_sm70_fp32_launch`（FP32 epilogue 出口）——本 fork 无此入口。

**门槛形状确认命中**：prefill chunk 2560 → `x.ne[1]=2560 ≥ 2048`；
`17408×5120`（gate_up）、`5120×8704`（TP2 down）均匹配。

## 关键：A/B 校准（本项的第二个目的）

协议字段核对：chunk 2560 / KV int8 / capture-generation / `-pg 60000,512` / MTP4 + lm-head-draft —— 与 `protocol.env` 全同；
输出 513 IDs、计数 406/421/106 与基线全同。

仍测不到 +1%，于是在**同一 binary、同一 worktree、同一驱动状态**下把 dispatch 门槛临时改成
`x.ne[1] >= 1000000`（256 分支永不命中），重编重测：

| 组 | 代码 | prefill 均值 (sd) | decode 均值 (sd) | sha1 |
|---|---|---:|---:|---|
| BASE | 基线 tag，独立 build | 945.799 (1.573) | 129.727 (0.0086) | 06287348665d3648 |
| CTRL128 | 本 worktree，门槛关 | 943.993 (1.103) | 129.741 (0.0330) | 06287348665d3648 |
| TILE256 | 本 worktree，门槛开 | 948.458 (1.517) | 129.717 (0.0290) | 06287348665d3648 |

结论：
1. **测量链正常** —— CTRL128 复现基线，差 −0.193%（< 1×sd）。
2. **TileN=256 的效果是真的，但只有 +0.473%**（948.458 vs 943.993，同 binary 同批次）；
   相对基线的 +0.281% 里混了 build-to-build 漂移。
3. 增益约为文档值（prefill +1.1466%，`docs/performance-v100.md:101-119`）的 **1/3**。
   文档那轮绝对吞吐是 1297→1312 tok/s，本机是 946→948：prefill 里这两个 GEMM 的占比不同，
   所以同一个相对 GEMM 收益摊到整体 prefill 上更小。
4. decode 三组全在 129.72±0.03 内，与文档「decode 差异不作为稳定收益」一致。

## 附带发现（对后续项有用）

- **缺一条 configure 前提**：worktree 里没有 `.tools/` 也没有 `_toolchain/`（都 gitignore），
  `CMakeLists.txt:34` 的 `FetchContent_MakeAvailable(cutlass)` 会去 git clone v4.4.2，
  在本机失败（`Filename too long` + msys `NtCreateDirectoryObject` 0xC0000022）。
  必须加 `-DFETCHCONTENT_SOURCE_DIR_CUTLASS=<MAIN>/_toolchain/cutlass`。
  已固化到 `_scratch/port-bat/configure-*.bat`。
- 全量 configure+build（ninfer + ninfer_bench，`-j4`）**16m34s**；改单个 `.cu` 的增量重编 + 链接 **≈2.5min**。
- 一次完整测量（加载 + warmup + 2 reps）**4m36s**。
- 测量前的 GPU 为空载（两卡 10 MiB / 50°C）；本轮未逐秒采温度峰值，属记录缺口（基线峰值 71°C）。

## 复现

```bash
# 配置（注意 CUTLASS 覆盖）
cmd //c "_scratch\\port-bat\\configure-port-tilen256.bat"
# 测量
MAIN=/c/Users/lwb6/Documents/git/NInfer-V100-Duo-win
cd $MAIN/.agents/worktrees/port-tilen256
build/bench/ninfer_bench.exe --weights F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer \
  --corpus bench/fixtures/bench_corpus.ids -pg 60000,512 \
  --prefill-chunk 2560 --kv-dtype int8 --mtp-draft-tokens 4 --lm-head-draft \
  --tp 2 --devices 0,1 -r 2 --warmup 1 --capture-generation \
  -o json --output-file $MAIN/_scratch/port-log/tilen256.json
# 判定
python _scratch/port-bat/compare.py
```

原始数据：`tilen256.json`（移植）、`tilen256-control.json`（对照）、`baseline.json`（分母）。
