# C3 真实数据复核（2026-10-26 08:42~09:00）· 量化前提被推翻

> 对象：`dd-cards-20261026.md` C3 的裁决条件——"K/W 静态 2 的幂预缩放能否把 T_inv/W 装进 fp16 指数域"。
> 结论：**撤回**。合成统计（`gdn_w_stats.log`，span ≤25~28 bits、subnormal ≤6.8%）不代表真实数据；
> 真实 prefill chunk 上 T_inv/W/Af/expG 的 subnormal 占比 **16.4%~30.7%**，名义 span 到 double 下溢区。

## 证据链

| 环节 | 产物 |
|---|---|
| dump 探针（临时，未进产品） | `src/ops/linear_attention/gated_delta_net/gated_delta_net.cpp` 中 `dump_gdn_operands()`（env `NINFER_GDN_DUMP`），diff 存 `c3-gdn-dump.diff`，**已还原** |
| 一次真实推理 | `build-win/bench/ninfer_bench.exe --weights F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer --corpus bench/fixtures/bench_corpus.ids -pg 60000,512 --prefill-chunk 2560 --kv-dtype int8 --mtp-draft-tokens 4 --lm-head-draft --tp 2 --devices 0,1 -r 1 --warmup 1`，日志 `c3-gdn-dump.log`，exit=0 |
| dump 内容 | 64 个 GDN 调用（每层一次），**全部** `T=2560 H_qk=8 H_v=24 D=128 normalize_qk=1`，`_scratch/gdn-dump/` 1.3 GB |
| host fp64 复算 | `_scratch/gdn_w_stats_real.cu`（数学与 `gdn_w_stats.cu` 逐行同源：Af、T_inv 下三角递推、W、U、v_new、state 递推、同一 `Stats` 报告函数），日志 `c3-gdn-w-stats-real.log`，`GDN_W_STATS_REAL_DONE`，14m41s |

k 的归一化按 `src/ops/kernel/l2norm.cuh` 复刻：fp32 求和 bf16 平方 → `rsqrt(sum+1e-6)` → bf16 回舍入。
（Volta build 下 `T_full=0`，归一化不发生在 `l2norm()`，故必须在分析侧复刻，否则 k 未归一化会让 T_inv 发散——S3 踩过这个坑。）

## 真实 vs 合成（同口径、同代码）

| 操作数 | 真实（64 层合并）max | 真实 span | 真实 subnormal | 合成（8 配置）span | 合成 subnormal |
|---|---:|---:|---:|---:|---:|
| Af | 9.87e-1 | ∞（minNZ 4.9e-324） | **16.39%** | 19.5~23.2 bits | ≤0.21% |
| expG | 1.00 | ∞ | **19.61%** | 0.4~4.1 bits | 0% |
| T_inv | 1.00 | ∞ | **19.77%** | 22.1~30 bits | ≤6.8% |
| W | 9.16e-1 | ∞ | **30.71%** | 21.0~26 bits | ≤0.7% |
| U | 4.33e+1 | 53.8 bits | 5.05% | 20.9~25.4 bits | ≤0.01% |
| v_new | 3.99e+1 | 47.0 bits | 5.66% | 20.1~25.9 bits | ≤0.01% |
| state | **6.12e+1** | 53.2 bits | 12.73% | 23.7~28 bits | ≤0.03% |

真实侧单层极值：state `|S|max` 层间 [0.002, **61.245**]（合成 0.3~2.0，差 30 倍）；
`|Ti|max` 恒为 1.000（对角），与合成一致 —— 说明 T_inv 上界没问题，问题在下界。

## 机制（为什么合成没暴露）

`expG` 的 subnormal 占比 19.6% 是根因：**真实 g 的 64-token 累积和能压到 −10 以下**
（合成配置最狠的 `g_scale=0.075` 也只能到 −4.7）。于是 chunk 内 `exp(G[r]−G[c])` 衰减到 e^-10~e^-300，
Af→T_inv→W 呈 `(1−c)^d` 深衰减，尾部直接落到 double 次正规区。

所以问题不是"预缩放选错了 2 的幂"，而是**同一个静态 scale 要同时覆盖 1.0 和下溢尾部**，
fp16 只有 29 bits 可用域（normal 段）+ 10 bits 次正规段，装不下 700+ bits 的名义动态范围。

## 判决

1. **撤回**：`dd-cards-20261026.md` C3 ③ 与执行状态里"W/T_inv 静态 2 的幂预缩放成立、
   全配置 span ≤25~28 bits < 29 bits"——**该结论仅对合成输入成立，真实数据不成立**。
   相应地，`STATE.md` 横幅里"W/T_inv fp16 静态预缩放指数统计全配置 span ≤25~28 bits"同样撤回。
2. **不是自动 kill**：subnormal 的 16%~31% 全是深衰减尘埃（相对 max 在 1e-19 以下），
   把它们 flush 成 0 对最终输出的影响**未测**。C3 卡里写的 kill 条件是"若需 per-chunk 动态缩放"，
   现在退到那个分支：**必须 per-chunk（或 per-head-per-chunk）缩放**，或者证明 dust-flush 可容忍。
3. **静态缩放这条路不能直接开工程**：state `max|S|` 到 61、U/v_new 到 43/40，
   与合成差一个量级；按合成值定 scale 会在真实数据上溢出。

## 下一步（纯 host，不需要 GPU、不需要重编）

用同一份 dump 做 **fp16 误差传播模拟**，三档对照：

| 档 | 缩放方案 | 判据 |
|---|---|---|
| A | 全局静态 2 的幂（原方案） | 输出 rel err |
| B | per-(head,chunk) 静态 2 的幂（device max-reduce 出标量，无 host sync） | 输出 rel err |
| C | bf16 无损基线（现状 recurrent 路径） | 参照 |

判据：v_new/state/out 相对 bf16 基线的误差是否落在 S3 已实测的量级内（S3: state max_rel 9.2e-4、
out bad=0 / max_abs 7.8e-3）。**A 超标而 B 达标 → C3 立项但方案改为 per-chunk 缩放；
A/B 都超标 → C3 永久封存**（因为那意味着 f16 mma 单次路径精度不够，3× 展开又回到 S4 的老路）。

## 附：本轮环境异常（与结论无关，但影响可复现性）

- 07:54 机器硬挂重启（`LastBootUpTime` 07:54:04，此前 uptime 自 10-25 06:32）；重启前 1 小时 WHEA id=19 共 34 条。
- 冷编期间 `cicc` 两次 ACCESS_VIOLATION（`nvfp4_small_t.cu`、`gqa_attention_prefill.cu`），重试即过。
- **"复用 build-win 增量"证伪**：旧对象（10-25 02:15）+ 新源码混编的二进制，不带 dump 也崩
  （`kernel_attr_once.h:61` → `cudaErrorIllegalAddress`，对照 `c3-nodump-ctrl.log`）。
  必须 `ninja -t clean` + 重配 + 冷编；本轮冷编 08:18 完成、可用。

---

# 附二：fp16 误差传播模拟（2026-10-26 14:06~14:27）

纯 host，用同一份 dump。`_scratch/gdn_fp16_sim.cu` → `c3-fp16-sim.log`。
采样 8/64 层（stride 8）、全 24 头、全 40 chunk，参考解 fp64；state_in 在 chunk 0 取 0。
相对误差按 **chunk 内参考张量的最大幅值** 归一化（S3 口径），不用逐元素（对深衰减尘埃无意义）。

| 档 | 操作数量化 | v_new max_rel | v_new bad_rel | state max_rel | state bad_rel | Ti flush | W flush |
|---|---|---:|---:|---:|---:|---:|---:|
| **C 现状**（Ti fp32、W/U/v_new bf16、state fp32） | bf16 | 1.949e-2 | 100,277 | 7.404e-3 | 17,350 | 0% | 1.89% |
| **A f16 全局静态** | f16, 2^14/2^8 | 1.296e-2 | 88,515 | 7.506e-3 | 17,953 | 3.03% | 9.19% |
| **B f16 per-chunk** | f16, 逐 chunk 2^a | 1.296e-2 | 88,515 | 7.506e-3 | 17,953 | 3.03% | 8.61% |
| **A0 f16 W/state、Ti fp32** | f16 | 1.403e-2 | 87,966 | 7.404e-3 | 17,668 | 0% | 9.19% |

（n: v_new 62,914,560 / state 125,829,120；abs 容差 7.8e-3、rel 容差 9.2e-4 沿用 S3。）

## 三条硬结论

1. **f16 精度不低于现状 bf16**：v_new 相对误差 1.296e-2 vs 1.949e-2（好 33%），
   state 7.506e-3 vs 7.404e-3（差 1.4%，同量级）。⇒ 3% 的 T_inv flush 与 9% 的 W flush
   **没有**造成实质损伤 —— flush 掉的全是深衰减尘埃。
2. **per-chunk 缩放毫无必要**：A 与 B 的误差指标**逐位相同**。原因：`max|T_inv|` 每 chunk 恒为 1.0
   （对角），per-chunk scale 与全局 scale 算出来都是 2^14；W 的 flush 率虽从 9.19% 降到 8.61%，
   但误差一条不变。⇒ C3 卡里"若需 per-chunk 动态缩放"这条兜底**不需要走**。
3. **T_inv 保持 fp32 也不亏**：A0（Ti fp32）与 A 的 state max_rel 都是 7.404e-3/7.506e-3，
   v_new 反而略好。⇒ road#2（T_inv 走 f16）**只有性能意义，没有精度意义**，可后置。

## 因此对上一节的修正

上一节"撤回"仍然成立：**用 span/次正规占比做的论证是错的**（它把深衰减尘埃当成有效动态范围）。
但结论方向要改：**误差传播实测显示 f16 + 全局静态 2 的幂缩放的精度与现状等价**，
C3 的精度前提**不构成否决**。C3 现在只剩性能问题：7.57ms/调用 → 必须 <5.33ms。

## 本模拟的边界（不可外推的部分）

- 未模拟 mma 累加顺序、4×f16/tf32 仿真层的下溢行为；只模拟"操作数 f16 + 累加 fp32"。
- 未覆盖 output 阶段（q @ (...)，即 `out` 张量）——它决定 oracle 是否漂移，需在真机 S4 复测时看。
- state_in 在 chunk 0 取 0；真实 prefill 首 chunk 确实为 0，故影响有限，但多 batch 接续时不同。
- 所有档（含现状 bf16）对 fp64 的相对误差都在 7e-3~2e-2 量级，比 S3 合成的 9.2e-4 大一个量级 ——
  这是**真实数据 + 强衰减**的固有性质，不是量化档位造成的，判据必须继续用 B 类相对口径。

---

# 附三：预登记判据 P1 作废，改判 P2（2026-10-26 17:35）

预登记判据（附二开头沿用 S3）用的是**绝对容差**：v_new max_abs 7.8e-3、state max_rel 9.2e-4。
按字面执行，四档全部超标：

| 档 | v_new max_rel | state max_rel | P1 字面判决 |
|---|---:|---:|---|
| A / B | 1.296e-2 | 7.506e-3 | 超标（封存） |
| **C 现状**（已上线，`final-confirm` oracle A 类通过） | 1.949e-2 | 7.404e-3 | **同样超标** |

⇒ **P1 不可分辨，作废**。`9.2e-4` 是 S3 在合成输入（k 近正交、g 累积只到 −4.7）上标定的；
真实数据 chunk 内累积到 −10 以下，深衰减使 fp64 参考的有效位宽本身收窄，任何有限精度实现都会被否掉。

**P2（取代 P1）**：判据改为「相对现状 bf16 路径的增量误差，不劣于 +10%」。

| 档 | v_new（vs 现状 −33%） | state（vs 现状 +1.4%） | P2 |
|---|---|---|---|
| A f16 全局静态 | 1.296e-2 | 7.506e-3 | ✅ 过 |
| B f16 per-chunk | 同 A | 同 A | ✅ 过，但零收益、机制上不必要 |

**裁决**：预登记的两分支（A 超标 B 达标 / A、B 均超标）**都不命中**。
C3 **立项，方案取 A**（f16 + 全局静态 2 的幂缩放；T_inv 保持 fp32；不走 per-chunk）。
精度不再是风险项，**唯一否决面是性能**：chunked 7.57ms/调用必须 < 5.33ms（recurrent）。
**未验证项**（真机 S4 复测必须覆盖）：`out` 张量（`q @ (…)` 的最终输出）、decode 序列 acceptance 漂移。

---

# 附四：S5-0 mma 闸门实测 + 一次算术修正（2026-10-26 18:08）

## 闸门（`_scratch/bench_gdn_mma.cu` → `s5-0-mma-gate.log`）

V100-SXM2, sm_70, 80 SM, 1530 MHz；单位工作量 = 一个 m16n8k16 tile-MAC（4096 FLOP）。

| 变体 | 时间 | TFLOPS |
|---|---:|---:|
| V0 tf32 仿真（3 products，12 mma + 48 shfl） | 274.80 ms | 3.05 |
| V1 bf16 仿真（1 product，4 mma + 24 shfl） | 75.54 ms | 11.10 |
| V2 单次 f16 + 24 shfl | 53.69 ms | 15.63 |
| V3 单次 f16 无 shfl（上限） | 17.18 ms | 48.82 |

`V0/V2 = 5.12×`（阈值 2.5×）→ **闸门通过**；`V2/V1 = 1.41×`（bf16 块只省 shfl）；
`V3/V1 = 4.40×`。注：V3 = 48.82 TFLOPS 仅为峰值的 39%，寄存器喂养的 m8n8k4 循环上限就在这。

## 顺带纠正一条交接文档的事实

`STATE.md` 原文"bf16→**4×** f16 mma"不成立。读 S1 源码：`ema_mma_bf16` 每个 m16n8k16
只做 **4 条** mma（1 个 product），`ema_mma_tf32` 做 **12 条**（3 products）。真正多算 3 倍的
只有 tf32 路径 —— 而 state_passing 的 MM1/MM2 正是 tf32，所以收益面集中在它身上。

## ⚠️ 算术修正：我 18:08 给的"5.12× → 5.24ms"是错的

那个算法把 5.12× 套在了**整段** 2.90ms 上。正确做法是按工作量折算：

- state_passing 工作量 = 24 头 × 40 chunk × 2 个 matmul × 64×128×128 × 2 = **4.03 GFLOP/调用**
- 实测 2.918ms → 有效 **1.38 TFLOPS**
- 闸门里 tf32 仿真在最佳条件（4 累加器 ILP、寄存器喂养）是 3.05 TFLOPS
  ⇒ state_passing 只跑到闸门条件的 **45%**，其余 55% 是 LDS（逐 lane）、寻址、同步等

按"mma+shfl 占 45%、其余不动"折算：
`2.918 × (0.45/5.12 + 0.55) = 1.86 ms` → 省 **1.06 ms** → chunked 微基准
`6.488 − 1.06 = 5.43 ms`，**仍高于 recurrent 的 5.33 ms（差 0.1ms）**。

⇒ **road#1 单独不够，road#1 + road#3 才翻正**。三项相加的合理区间：
state 省 1.06 + prepare 的 bf16 块省 ~0.29（1.41× × 40% 占比）+ LDS 省 0.3~0.5
→ chunked 4.6~4.85ms → **+5.4~7.6% prefill**（与战前估计一致）。

## 复现与工具

- S4 基线在本机**逐项复现**：prepare 2.532 / state 2.918 / output 1.038 / TOTAL 6.488ms
  （`port/gdn-chunked-s1` @ `de76e0b2`，worktree `.agents/worktrees/s5-prof`，
  `nvcc -arch=sm_70 -O2 -I../src -DNINFER_VOLTA_GDN_CHUNKED`，编译约 10 秒）
- **本机没有 ncu**（`NVIDIA Corporation` 下只有 `Installer2`），所以"mma 占比"是
  由工作量/闸门吞吐反推，不是直接测量。⇒ 下一步应改为**在 harness 里直接实现 road#1
  并测量 state_passing**，用实测值取代 45% 这个反推。

---

# 附五：road#1 在真实内核里落地实测 → C3 封存（2026-10-26 18:17）

## 做法（不碰产品代码）

在 lab worktree `.agents/worktrees/s5-prof`（`port/gdn-chunked-s1` @ `de76e0b2`）上：
1. 新增 `src/ops/common/gdn_road1_f16.cuh`：`ema_mma_f16` / `ema_mma_f16_bits`，
   保留 Ampere→Volta 的 24 条片段重排 shfl，**只把 3 product 降为 1 product**
   （`volta_mma_qk` 1 次而非 3 次），累加器置换复用 `gdn_ema_detail::d_to_c_accum`。
2. `chunked/state_passing.cuh` 的 `mma_tf32_bits` 包装加 `#if defined(NINFER_GDN_ROAD1)` 开关。
3. 用 S4 同一份 harness（`_scratch/bench_gdn_chunked.cu`，T=2560/Hqk=8/Hv=24/NT=40，400 reps）
   分别编基线版与 road#1 版，各跑一次。

编译：`nvcc -arch=sm_70 -O2 -std=c++20 -I../src -DNINFER_VOLTA_GDN_CHUNKED [-DNINFER_GDN_ROAD1]`，
单次约 10 秒。本机未装 ncu。

## 结果

| 阶段 | 基线 | road#1 | Δ |
|---|---:|---:|---:|
| prepare_wy_wu | 2.532 ms | 2.539 ms | 不变（bf16 路径本就是 1 product） |
| **state_passing** | **2.918 ms** | **2.231 ms** | **−0.687 ms（1.31×）** |
| output | 1.038 ms | 1.030 ms | 不变 |
| **TOTAL** | **6.488 ms** | **5.800 ms** | **−0.688 ms（−10.6%）** |
| recurrent 基线 | 5.33 ms | 5.33 ms | 仍需再省 **0.47 ms** |

## 判决：C3 封存

1. 微基准的 5.12× 在真实内核里只兑现 **1.31×**。省掉 2/3 的 mma 只换到 0.687ms，
   ⇒ **mma 总成本 ≈1.03ms，仅占 state_passing 的 35%**；其余 65% 是逐 lane LDS、
   寻址、同步。**这个内核不是 mma-bound**，与 S4 的归因（"LDS 仿真 + fragment 重排的 smem 压力"）
   方向一致，且这次是从收益侧独立证到。
2. 剩余杠杆只有 road#3（LDS 事务，S4 估三阶段合计 0.3~0.5ms）。即使给满，chunked ≈5.3ms，
   **刚好打平、零余量** → 折算 +0~1.5% prefill，压在 +1% 门槛上、与 MDD 0.481% 同量级。
3. 3~5 天工程量 + B 类 oracle（acceptance 漂移）换取一个测不出来的收益，**不成立**。

**闸门本身没错**（V0/V2 = 5.12× 证的是"单次 f16 比 3 乘积便宜 5 倍"），
错在把它当作端到端收益的代理。这条方法论要写进纪律：**微基准的倍率不能线性外推到内核实测，
必须先量目标块在真实内核里的时间占比。**

## 现场

- 基线版 / road#1 版 harness：`s5-prof/_scratch/bench_gdn_chunked{,_road1}.exe`（均已编好）
- 改动仅在 lab worktree 未提交：`src/ops/common/gdn_road1_f16.cuh`（新）、
  `chunked/state_passing.cuh`（+11 行开关）。**主仓 master 未动、`git status` 空。**
- worktree 保留（不删、不 clean），供后续复跑。
