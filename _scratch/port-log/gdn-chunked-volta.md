# GDN chunked Volta 移植：勘察结论与实施计划（2026-10-25 下半场）

> 用户确认只用 27B → 35B 相关项（Gqa35 flash 解锁）仅记档不动工。
> 全部精力收口到 27B 的最后一个深水区：GDN chunked prefill kernel 的 Volta 移植。

## 1. 现状归因（已核实源码）

`gated_delta_net.cpp:254-271`：`NINFER_VOLTA_BUILD` 下 `T_full = 0`，**整个 prefill 宽度**
全部走 `launch_recurrent_inout` → `recurrent_bf16_direct_kernel`（recurrent.cuh:219-269）：

- grid = H_v × (128/16) = 12×8 = **96 blocks**（TP2 本地 12 v-head），每 block 128 线程
- 状态 (128×128 fp32/head) 摊在寄存器：每 warp 持 4×4，**逐 token 串行**
- 每 token 每 warp：transition 4×(4 FMA + warp_sum 5 shfl) + readout 同量
  ≈ 32 FMA + **40 shfl** + 2 exp —— 纯 shfl/延迟瓶颈
- 实测 5.33ms/层，~1.5 TFLOPS（FP32 峰值 10%），流量仅 ~84MB/层（带宽不是瓶颈）

注释自述 trap 原因（gated_delta_net.cpp:254-267）：chunked 路径需要
`ldmatrix`（sm_75+）与 `mma_bf16`(m16n8k16)/`mma_tf32`（sm_80+），`ops/common/mma.cuh`
在 sm_70 下一律 trap。

## 2. 收益上限

- mixer.gdn = **3.02s / 8.39s ≈ 36% prefill 墙钟**（pp10240，mixer-survey.md）
- chunked 公式把每 chunk(64 token) 的递推变成若干 64×128 级 matmul → tensor core 可吃
- FLOP 总量同阶（~8-16 GFLOP/层），但 tensor core 上 <1ms 可完成
- **保守估计 GDN 核心 5.33ms → 0.7~1.2ms，prefill +20~30%**（1184 → 1450~1550 tok/s）
- decode 不受影响（decode 本就走 recurrent 短路径，结构性零变化 → 判据 v3 非目标侧）

## 3. 移植面（chunked/ 共 ~1800 行）

| kernel | 行数 | 依赖 | Volta 对策 |
|---|---|---|---|
| prepare_wy_wu.cuh | 724 | mma_tf32（KᵀK/逆）、mma_bf16、ldmatrix | fp32→fp16 split 2×mma 仿真；bf16→fp16 转换；ldmatrix→手工 LDS |
| state_passing.cuh | 538 | mma_tf32_bits（状态跨 chunk 递推） | 同上 |
| output.cuh | 370 | mma_bf16（QS + AV）、mma_tf32 | 同上 |
| launch/布局 | ~130 | 无 mma | 原样 |

关键约束：
- **Volta mma 只有 fp16 m8n8k4**（无 bf16、无 tf32、无 m16n8k16）→ 全部 fragment 布局重写
- bf16→fp16 有舍入/指数域差异 → **oracle 必变，B 类**。判据用 queue-v2 裁决 #1：
  accepted/drafted/rounds 严格相等 + acceptance_length 相等 + 差异位 ≤2% 且 margin<0.05
- TF32 仿真（fp16 hi/lo split，2×mma）保 KᵀK 求逆的数值稳定性（W 上三角前代不可降精度太多）

## 4. 分阶段实施（每阶段独立可验证）

- **S1**：`ops/common/mma.cuh` 增加.sm_70 分支——`gmma_f16_m8n8k4`（已有 wide-T mma 经验）+
  手工 LDS 替代 ldmatrix + `mma_bf16→fp16 转换垫片` + `mma_tf32→split 仿真`。
  单元测：随机矩阵对照 CPU/cublas。
- **S2**：port prepare_wy_wu（最重）。验收：单层单 head chunked vs recurrent 递推输出
  max rel err < 1e-2（随机输入），端到端 sha1 记录。
- **S3**：port state_passing + output。验收同上 + pp2048 探针。
- **S4**：解开 `T_full=0` 陷阱（加 `NINFER_VOLTA_GDN_CHUNKED` 开关，默认关），
  measure.sh 全量测量，judge.py --target prefill，B 类判据人工核。
- **S5**：达标 → commit + tag `result/gdn-chunked-volta`；不达标 → 记档撤 patch。

每步 `ninja -j2`，测量独占，`nvidia-smi` 前后检查（硬件风险照旧）。

## 5. 风险

1. W 上三角求逆的数值稳定性：tf32 仿真精度需实测；不行就回退 fp32 CUDA-core 求逆
   （该矩阵仅 64×64/块，CUDA core 也能扛，损失部分收益）
2. m8n8k4 fragment 布局繁琐（k=4 需要 2 次拼 8）——wide-T mma 已趟过一遍，可复用其 swizzle
3. B 类 oracle：acceptance rate 若漂移超判据 → 立即停，评估 hi/lo split 精度加强
4. 工程量 3~5 天级，是本战役最贵单项；但不做它，prefill 已无 ≥2% 级别的候选

---

## 6. S4 结果：否决（2026-10-25）

**全量实测（measure.sh gdn-chunked-volta vs prefill-down-cutlass，pp60000 协议）：**

| 指标 | 值 | 门槛 | 判定 |
|---|---|---|---|
| oracle | **A 类逐位一致**（sha1 `06287348665d3648`，acc/drf/rd/al/perpos 全等） | — | 比预期 B 类更好：ema 仿真精度高于硬件 tf32，两条路径输出完全一致 |
| prefill | 897.587 tok/s（**−24.230%**） | ≥1196.463（+1%） | **FAIL** |
| decode | 135.849 tok/s（+0.030%，噪声内） | 结构性零变化 | 符合预期 |

**归因（三层递进）：**

1. 微基准（`_scratch/bench_gdn_chunked.cu`，同 shape T=2560/Hqk=8/Hv=24/NT=40，
   400 reps）：prepare 2.54ms + state 2.90ms + output 1.03ms = **6.47ms/调用**，
   无降频漂移（400 次连续 1.25→2.49ms 稳定）。
2. bench 内嵌 cudaEvent 诊断（LAB-DIAG，rank0）：**dev=7.57ms/调用**（host 仅
   0.03ms）——设备时间与微基准一致量级，排除 host 间隙/同步/workspace 开销。
3. 对比 recurrent 基线 5.33ms/调用：**chunked ema 净慢 ~42%/调用** → 折算
   prefill −24%，与实测 −24.23% 自洽。

**根因**：Volta（sm_70）缺三样东西——bf16 mma（→4×f16 m8n8k4 展开）、tf32 mma
（→3×f16 hi/lo 展开 + 24 shfl）、ldmatrix（→逐 lane LDS.32）。仿真层把
tensor core 的理论吞吐（~130 TFLOPS f16）摊薄到有效 ~30 TFLOPS，再叠加 LDS
仿真与 fragment 重排的 smem 压力，实际反而低于 per-token recurrent kernel 的
"寄存器驻留 + 88GB/s" 工作点。**5.33ms 的 recurrent 并不是低垂果实——它是
Volta 上 GDN 的合理工作点。**

**处置（按 S5 规则）**：不合并 master、不 tag、撤 patch（trap 恢复原样）。
S1-S3 的 ema 层与验证 harness 保留在 lab 分支 `port/gdn-chunked-s1`
（commit d2fb2323 / 3bbb3792 / 9e97b54d / 本节 S4 commit），供未来参考。
所有 kernel 代码正确性已证明（oracle A 类 + 三轮 fp64 对拍）——**败于性能，
非败于正确性**。

**未来若要复活此方向**（按性价比排序）：
1. state_passing 是最大项（2.90ms）：它的 MM1/MM2 是 tf32 路径。若能把
   K/W 预缩放到 fp16 可表示范围直接用 f16 mma（省掉 3× 展开与 shfl），
   state 可能砍半。
2. prepare 次之（2.54ms）：T_inv 求逆的 mma 同为 tf32 路径，同上。
3. ldmatrix 替代改成 LDS.64/LDS.128 宽加载（当前 LDS.32 逐 lane）。
4. 任何一项都不便宜，预期收益上限 = 把 −42%/调用翻正，再挣 +2% prefill
   ——投入产出比低于其他候选方向。
