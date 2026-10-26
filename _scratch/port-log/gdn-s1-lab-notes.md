# GDN chunked Volta 移植 · S1 实验记录（2026-10-25 晚）

Lab worktree 分支 `port/gdn-chunked-s1`（基于 df1f2958）。
harness：`_scratch/probe_m8n8k4*.cu`、`_scratch/validate_volta_qk.cu`（编译脚本同名 .sh，
nvcc 在 `MAIN/_toolchain/cuda-12.8/bin/nvcc.exe`，需 `-std=c++17 -I../src`，TMP 指向本地 tmp）。

## 已确证（真机 V100 实测）

1. **`VOLTA_QK_VALIDATION_PASS`**：仓库 `ops/common/volta_mma.cuh` 的
   volta_load_qp / volta_load_k / volta_mma_qk / volta_d_get_i/j 在随机 32×8·8×8
   QK^T 上与 CPU 参考一致。**这就是 GDN 仿真的 API 基座，直接复用。**
2. m8n8k4 fragment 探针（one-hot 扫描 + 32×32 配对张量，probe_pairs.out）：
   - 配对结构是严格 matmul：slot = Σ_k A[i][k]·B[k][j]，8 行 × 8 列 × k=4，
     A 元素 id = 4*lane+2*reg+half（行=lane/组，k=2*reg+half），B 同构（列=lane/组）。
   - 稀疏探针下单条 PTX m8n8k4 只表现为 lanes 0-7 的两个 4×4×4 块积（quad 局部），
     lanes 8-31 完全不参与、跨 octet 配对为 0 —— 但完整 fragment 下
     volta_mma_qk 两连调用仍产出正确 32×8（见 1）。结论：**不要依赖稀疏探针推断
     全局布局，一切以 API 级验证为准**；疑与 ptxas 对稀疏常量 fragment 的 SASS 编排
     有关，无工程影响，不再追。
   - D fragment：256 个 f32 槽全部是活的累加位（C 直通到全部 256）。

## S1 剩余步骤

> **2026-10-25 深夜更新：S1 仿真层完成并真机验证 PASS**（commit `d2fb2323`）。
> 下面 1/2 已完成，3 未动。

1. ✅ `src/ops/common/gdn_volta_mma.cuh`（lab 新头文件，不动 mma.cuh）：
   - `ema_mma_bf16(...)`：签名与 `ops::mma_bf16` 逐参一致（Ampere m16n8k16 fragment 进出）。
     bf16→fp16 保序转换（|x|<65504 内精确）→ shuffle 重排进 Volta Q/K fragment
     → 2× `volta_mma_qk`（k 对半）→ D 反向重排回 Ampere C 累加。
     **预算：40 shfl + 4 mma / 每次 m16n8k16**（A 16 + B 8 + D 16）。
   - `ema_mma_tf32` / `ema_mma_tf32_bits`：fp32 → fp16 hi/lo 拆分，hh+hl+lh 三次 f16 mma
     （弃 ll）。有效精度 ~21 bit > 硬件 tf32 的 10 bit，对 W 求逆是安全方向。
     **预算：24 shfl + 6 mma**。
   - ⚠️ 精度误差模型（实测标定）：拆分的**绝对**地板 ≈ 3e-8（fp16 次正规量子）× 对侧操作数
     （lo 部分落入次正规区时）；正常量级下仍为 ~21-bit 相对精度。S4 若 B 类判据被卡，
     优先查这里（对策：换尺度感知拆分或该 mma 回退 CUDA-core f32）。
2. ✅ 对拍 harness：`_scratch/validate_gdn_volta_mma.cu`（+同名 build.sh），64 组随机 trial
   + 压力 trial（5e4/1e-4 混合）对照 CPU 参考：**BF16 PASS（worst 0.048×bound）、
   TF32 PASS（worst 0.73×bound）**，真机 V100。
   - 教训：D fragment 寄存器索引手工推导错过一次（j&4 对 j=2t 应为 (t&2)<<1 而非
     (t&1)<<2），被对拍当场逮住 ⇒ **布局一律以对拍为准，勿信手工推导**（与探针结论同向）。
3. ❌ ldmatrix 替代（S2 决策点）：仿真层目前假设 S2 会喂标准 Ampere fragment。
   两条路：(a) 逐 lane LDS 按 Ampere 布局 gather（接口不变，A/B 重排 24 shfl 照付）；
   (b) 给"按 Volta 布局直接从 smem 装载"的替代装载函数（省 A/B 的 24 shfl，只留 D 的 16），
   需动 kernel 装载代码。看完 prepare_wy_wu 的 smem 寻址模式再定。

## Ampere fragment 布局备忘（待真机对拍确认，不可凭记忆写死）

m16n8k16 f16：A 每 thread 8 halves（g=lane>>2, t=lane&3）：
a0=(g,2t),(g,2t+1)；a1=(g+8,2t),(g+8,2t+1)；a2=(g,2t+8),(g,2t+9)；a3=(g+8,2t+8),(g+8,2t+9)。
B 每 thread 4 halves：b0=(2t,g),(2t+1,g)；b1=(2t+8,g),(2t+9,g)。
C 每 thread 4 f32：c0=(g,2t),c1=(g,2t+1),c2=(g+8,2t),c3=(g+8,2t+1)。
m16n8k8 tf32：A a0=(g,t),a1=(g+8,t),a2=(g,t+4),a3=(g+8,t+4)；B b0=(t,g),b1=(t+4,g)。
⇒ 验证法：同 probe 思路，把"仿真函数"当黑盒，对拍 CPU 矩阵乘，布局错了立即暴露。

## S3 完成（2026-10-25，lab commit 9e97b54d）

state_passing + output 全部通过 using-switch 接到 ema 层，三段式 harness
（`_scratch/validate_gdn_pipeline.cu`，kT=128 / 2 chunks）全 PASS：

| 轮 | 输入 | 判定 |
|---|---|---|
| A 端到端 | prepare→state→output | v_new bad=0 (max_abs 7.8e-3)、state 9.2e-4 |
| B clean state | oracle W/U(bf16, d-interleave)/g 直喂 | h_chunk **逐位一致**、state 9.2e-4 |
| C clean output | oracle v_new/h_chunk(bf16) 直喂 | bad=0、max_abs 7.8e-3 |

### 调试战史（三个教训）

1. **NaN 根因是 harness 数据，不是 ema 层**：k 未按 token L2 归一化 → Af 元素
   最大 ±30 → T_inv=(I+Af)^-1 从 t≈25 行起 fp32 发散 → W/U 全 NaN。真实管线里
   l2norm 在 prepare 之前跑（gated_delta_net.cpp:192 normalize_qk），单位向量
   点积 ≤1，Af ≤ β ≤1，完全良态。归一化后 NaN 全消。
2. **NaN 盲区判据**：`e>tol` 对 NaN 为 false、`std::max(worst, NaN)` 不传播 →
   "worst rel 很小 + PASS" 可能是假象（S2 曾中招）。一律 `!isfinite(got) || !(e<=tol)`。
   顺带发现 fp64 oracle 里的 std::max(afmax,|NaN|) 同样吞 NaN，oracle diag 自己也要防。
3. **bf16 输出 vs fp64 oracle 的判据**：先把 ref round 到 bf16 再比；消减产生的小值
   元素（|ref| << row_max）用绝对容差 `3e-3 + 1e-2*row_max`，真正的大值元素
   （>0.25*row_max）加相对 2% 保护。C 轮误差随 t 平滑增长（t=0 ~4e-4 → t=64 ~1e-3），
   无离群突刺 = 64 步 A@v_new 累积舍入的签名，非结构性错误。

### 其他

- h_chunk 设备布局 (chunk*Hv+h)*16384 + d*128 + k，state/output 两核一致。
- Round B 中 h_chunk 逐位一致 = state kernel 的 ldmatrix 快照路径（ema_ldmatrix_x2_t）
  在真实寻址下完全正确——Phase E 的聚焦单测（validate_x2t.cu）结论被端到端证实。
- 陷阱备忘：本环境 shell cwd 每条命令重置回主仓库根——git/构建命令必须
  `cd <worktree> && ...` 单链执行，否则误操作主仓库（S3 期间两次险情，master 未受污染）。

### 下一步：S4

`gated_delta_net.cpp:267` T_full=0 trap → `NINFER_VOLTA_GDN_CHUNKED` 开关（默认关）→
全量 measure.sh → judge.py --target prefill → B 类判据。
