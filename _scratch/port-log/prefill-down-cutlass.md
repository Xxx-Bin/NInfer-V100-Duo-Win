# prefill-down-cutlass：plain rank 下投影改走 CUTLASS

**结论：KEEP，已合并 master（merge `f9e5ce82`，tag 见下）。prefill +25.274%，oracle A 类逐位一致，decode 结构性为零。**

## 发现（相位归因，`prefill-trace4.log`）

自写 `lab_phase_trace.h`（host 墙钟 + device 同步 bracket，`NINFER_PHASE_TRACE=1` 启用）对
pp10240 分相位，发现 TP2 down 投影严重 rank 不对称：

| bracket | total_s | mean_ms | 路径 |
|---|---|---|---|
| down.gemm.rank0（fused） | 1.245 | 1.945 | `dispatch_linear_add` → Volta 下 `LinearThenAdd` → `nvfp4_cutlass_sm70` |
| down.gemm.rank1（plain） | 5.156 | 8.057 | `dispatch_linear` → wide-T mma（`mma.sync.m8n8k4`，~28 TFLOPS） |

同形状（T=2560, N=5120, K=8704）差 4 倍。rank0/rank1 权重、FLOPs 完全相同，差异只在 kernel。

## 改动

`src/ops/wrapper/linear_add.cpp` 的 `issue_plain_rank`：NVFP4 + A16Only + T≥33 +
down 投影形状族（N=5120, K∈{3072/6144/8704/17408}）时，直接调
`nvfp4_cutlass_sm70_launch` 写 partial（覆盖输出 == plain 语义，allreduce 契约不变）。
decode（T=1）不进此分支，QPN2 prepacked 路由原样。+15 行，1 文件。

## A/B 与全量判定

- pp10240 A/B（lab env 开关）：prefill 1101.3 → 1441.1 tok/s（**+30.9%**），decode 34.1 不变。
- 标准全量（pp60000+tg512, r2, `measure.sh prefill-down-cutlass packet-allreduce`）：
  - oracle **A 类**：sha1 `06287348665d3648`、acc/drf/rd 406/421/106、al 4.830 —— 逐位一致
  - prefill 945.623 → **1184.617（+25.274%）**，gate PASS，MDD 0.481% 可分辨
  - decode 135.594 → 135.808（+0.158%）—— 低于 MDD 0.433%，结构性为零（路由门 T≥33）
  - judge VERDICT=PENDING（decode 门槛未过），裁决：本改动目标即 prefill，
    decode 无路径可达，按"目标过 + 非目标不回退"判 KEEP。

## 途中修掉的潜在缺陷（lab 记录）

`nvfp4_linear_swiglu_plan.cpp` 的 `LinearW4A4Post` 路线在 Volta 上是死代码但有三处失配：
workspace 查询用 `kNvfp4InternalPolicy`（A16→256B 地板）而 `linear()` 硬编码 `AllowA4`
（需 5MB 瞬态，OOM）；且 AllowA4 在 sm_70 直接拒绝。P0 的 swiglu 路由 A/B 因此得以完成：
**wide-T mma 路线 14.07s vs VoltaCutlass 9.32s —— swiglu 现路由已是正确选择**（负结果，
默认路由保留）。

## 遗留 / 下一步

- prefill MFU 从 ~17% 升到 ~21%；剩余大头是 gate_up 的 VoltaCutlass（每 chunk 全权重
  反量化物化 FP16）与 rank0/1 都走的 down cutlass 本身。下一格：CUTLASS mainloop 融合
  反量化，目标再 ≥1.5×。
- lab worktree `port/prefill-gemm-budget` 保留全部仪表（相位计时、arena OOM 诊断、
  A/B 开关），分支未删。
- `judge.py` 的 PENDING 裁决逻辑可考虑支持"目标指标"参数化（本次人工裁决）。
