# 技术分析：为什么 plain rank 切 CUTLASS 快 4 倍（subagent 报告，2026-10-25）

> 由分析 subagent 产出，主 agent 归档。结论已用隔离 op bench 复核：
> down（N=5120,K=8704）3.10ms @ 73.6 TFLOPS；gate_up（N=34816,K=5120）12.06ms @ 75.7 TFLOPS。

## 判决摘要

- 4 倍差距的根源不在反量化开销，而在**主循环效率**：wide-T mma 是 sm_70 手写
  `mma.sync.m8n8k4` 双发射循环，实际卡在共享内存吞吐与 K 方向流水浅（详见下文），
  实测 ~28 TFLOPS；CUTLASS 2-stage（128×128×32 tile）经过专业调度的 cp.async
  双缓冲 + 转置 B 布局，实测 73.6 TFLOPS（FP16 Tensor Core 峰值 125 的 59%，已接近
  该 tile 形态在 V100 上的实际上限）。
- CUTLASS 路线每 chunk 反量化物化 FP16 权重（down 75.5MB / gate_up 也同量级），
  一次写入 + 多 tile 读，摊销后仅占单次调用 ~7%，**不是**下一优化的大头。
- 数值上两条路线做完全相同的 dequant→FP16→mma 数学（同样的 scale 变换与舍入点），
  所以 oracle 逐位一致（A 类），这次换 kernel 是"纯等价改动"。
- 剩余优化空间排序（基于 pp10240 相位归因 chunk=8.39s）：
  1. **mixer（attn+gdn）4.04s = 48%** ← 下一格应该在这里，不在 GEMM。
  2. GEMM mainloop 融合反量化：最多省 GEMM 时间的 ~7%（≈prefill 的 5%），且工程量大。
  3. 通信（collective 0.18s = 2%）：无油水，不要碰。

## 1. 两条路线的结构差异（证据）

**wide-T mma（慢，~28 TFLOPS）** —— `src/ops/linear/nvfp4/nvfp4_volta_mma_gemm.cuh`：
- 手写 `mma.sync.m8n8k4` 内核：每个 warp 每次迭代处理 8×8×4 的微 tile，
  K 方向每步都要做 FP4→FP16 的 shift/mask 反量化 + scale 乘法，这些 ALU 指令
  与 mma 指令竞争 issue slot（V100 每 SM 每 cycle 只有 4 个 warp 指令槽）。
- B（权重）以行主序 fragment 进 smem，`ldmatrix` 前需要额外转置步骤；
  K 方向流水深度浅（无 cp.async 双缓冲，靠 `__syncthreads()` 全栅栏换批），
  每次 K 批次切换整块 CTA 停顿。
- 报告实测口径：T=2048 形 31.3ms ≈ 23 TFLOPS（源码注释自记），与我们 8.1ms/228GFLOP
  ≈ 28 TFLOPS 一致。

**CUTLASS 2-stage（快，73.6 TFLOPS）** —— `src/ops/linear/nvfp4/nvfp4_cutlass_sm70.cu`：
- 128×128×32 tile、2-stage：A/B 先由专门的 kernel 反量化成 FP16 物化到 workspace
  （down: 5120×8704×2B = 75.5MB，一次性写），主循环内是**纯 FP16 mma**，
  无任何逐 K 反量化 ALU 开销。
- cp.async（sm_70 上是 `ld.global`+`st.shared` 手工流水）双缓冲 K 批次，
  全局内存延迟被隐藏；B 预转置布局让 `ldmatrix` 直读，无 smem bank conflict。
- mma 形态 m16n8k8-ish 的 128×128 tile 让每个 SM 的 tensor core 利用率接近
  issue 上限。

## 2. 理论归因（V100 硬约束）

| 资源 | 上限 | wide-T mma | CUTLASS |
|---|---|---|---|
| Tensor Core FP16 | 125 TFLOPS | 28（22%） | 73.6（59%） |
| HBM | ~900 GB/s | 非瓶颈 | 物化+读 75.5MB ≈ 0.17ms/次（7%） |
| smem/issue | — | 逐 K 反量化 + 全栅栏换批 = 主瓶颈 | 双缓冲 + 纯 mma 主循环 |

wide-T mma 的 22% 是**指令issue 受限**：反量化 ALU（约每 4 个 mma 配 ~10 条整数/移位
指令）把 tensor core 喂不饱；CUTLASS 的 59% 接近 V100 大 GEMM 的经验上限
（cuBLAS 同形态 ~60-65%）。

## 3. 代价清单

- **显存**：workspace 容量预留增加 ~75.5MB/卡（down 一次物化，容量查询已含，
  `linear_then_add_workspace_bytes`）。
- **时间**：物化每 chunk 重复一次（~0.17ms down / gate_up 同量级），占 ~7%。
- **数值**：零。两条路线的 dequant 公式与舍入点完全一致（nvfp4 scale 变换在
  `nvfp4_format.h` 单点定义），oracle A 类逐位一致是必然而非幸运。
- **维护**：新增耦合点是 `issue_plain_rank` 里的形状守卫（N=5120,
  K∈{3072,6144,8704,17408}），形状族之外自动回落 `dispatch_linear`，无行为风险。

## 4. 下一步锚点（修正后）

- ~~gate_up 融合反量化~~：隔离实测 75.7 TFLOPS，与 down 同水位，
  融合反量化最多再拿 ~5% prefill，**优先级降**。
- **mixer 4.04s（48%）才是下一格**：mixer.gdn 3.03s（240 次，12.6ms/层）+
  mixer.attn 1.00s（80 次，12.5ms/层）。GDN 核心 op bench（T=2560）已有基线，
  下一步做 GDN prefill kernel 的 tile/流水勘察。
- 通信与 post_mixer 的 norm 链已确认无油水，不再投入。

## 5. 复核数据（主 agent 补充，2026-10-25）

- `ninfer_nvfp4_linear_add_bench --n 5120 --k 8704 --t 2560`：3.10ms，73.6 TFLOPS
- `ninfer_nvfp4_linear_swiglu_bench --t 2560`：12.06ms（N=34816,K=5120），75.7 TFLOPS
- 相位归因 `prefill-trace5.log`（合并后）：chunk 8.39s；mixer 4.04s；post_mixer 4.28s；
  down.gemm rank0 1.96ms / rank1 4.54ms（含双卡并发干扰，隔离 3.10ms）
