# prefill 归因战役（2026-10-25 09:2x，分支 `port/prefill-gemm-budget`）

目标：把 prefill 的 63.45 s（60000 tok，chunk 2560，TP2）拆到具体阶段，找出可移植/可改的收益项。

## 0. 起点事实

| 量 | 值 | 来源 |
|---|---|---|
| prefill | 945.623 tok/s = 63.45 s / 60000 tok | `packet-allreduce.json` |
| chunk | 2560 → 23.44 chunk，**2.71 s/chunk** | 同上 |
| 模型 | 64 层 = 48 GDN + 16 full attention，hidden 5120，intermediate 17408 | `config.h` |
| 线性部分 FLOPs | 22.8 GFLOP/token → 2.74 PFLOP / 63.45 s = **43.2 TFLOPS 聚合 = 21.6 TFLOPS/卡 = 17.3% 峰值** | 推导 |
| 通信 | 78 GB/卡，NVLink P2P 已启用（`allreduce.cu:447`），6×25.781 GB/s | `nvidia-smi nvlink -s` |

## 1. 单 op 实测（TP1 形状，cold L2，V100，本分支编出的 op bench）

| op | 形状 | T=2560 耗时 | 有效 TFLOPS |
|---|---|---:|---:|
| `linear_swiglu` (gate_up，当前 cutlass 路由) | N=34816 K=5120 | **11.89 ms** | **76.75** |
| `linear_add` (down，wide-T mma 路由) | N=5120 K=17408 | **5.99 ms** | **76.18** |
| `gdn_input_proj_conv` (nvfp4 a16, snapshot) | rows 10240+6144, K=5120 | **6.48 ms** | — |
| `gated_delta_net` (running, qk16/v48) | T=2560 | **5.33 ms** | — |

**结论 1：prefill 的 GEMM 不是 17% —— 两条路由都跑到 ~76 TFLOPS（61% 峰值）。**
所以"prefill 是 GEMM 效率问题"这个初始假设**被推翻**，17.3% 是混合了低效阶段的全局平均值。

**结论 2：FFN 在 TP2 下每 chunk 只占 ~573 ms（21%）。**
（每层 gate_up 5.95 + down 3.00 ms × 64 层，按 TP2 减半计）

已测可归因项（TP2 折算）：

| 阶段 | 每 chunk | 占 2.71 s |
|---|---:|---:|
| FFN（64 层 gate_up+down） | 573 ms | 21% |
| GDN input proj（48 层，减半） | 156 ms | 6% |
| GDN core（48 层，减半） | 128 ms | 5% |
| **未归因** | **1853 ms** | **68%** |

未归因候选：16 层 full attention（qkv proj + 注意力核 + KV 写入，随上下文增长）、
128 次/ chunk 的 TP2 collective、128 次 rmsnorm、rope、embedding/终层。

## 2. 归因手段（本机无 ncu / nsys / CUPTI）

新增 **lab-only** `src/core/lab_phase_trace.h`：在现有 NVTX 边界（`run_layers_tp2` 的
layer / mixer / post_mixer，以及 chunk 边界）插入 `cudaDeviceSynchronize` 夹持的 host 计时，
按 tag 累加，`atexit` 打印。仅 `NINFER_PHASE_TRACE=1` 时生效，**不合并进产品**。

已知偏差：每个夹持点会消除跨阶段重叠，只用于归因，不用于性能判定。
