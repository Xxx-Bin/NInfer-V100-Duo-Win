# W8 / 输出头 prepack · 落点勘察
状态：完成

> 只读勘察。fork = `.agents/worktrees/port-packet-allreduce`，
> 快照 = `_scratch/ref-v100x2/ninfer-V100X2-master`。
> 未编译、未跑 benchmark、未调 nvidia-smi；所有数字要么引自代码注释，要么是按
> 已发表的字节数/时间戳算的，算式全部写出。

## 一、判决（值不值得做，一句话）

**"整套 prepack"不值得做——1.31×–1.93× 是「QPN vs 我们早已换掉的 sliced r8_c8」的单 op 比值，
fork 的输出头在 T=3..8 已经跑在 QPN 上，这笔收益已经计入现状，不能重复领取。**
剩下的真实增量只有一处：**scale 平面的 4× DRAM 扇区放大**，它把 W8 QPN 这一族
（输出头 + 五个 MTP 矩阵）的流量从 **1060 MB/卡/round 压到 901 MB（−15.0%）**，
折算端到端 **+0.8%（乐观 +1.9%）**，勉强过 0.5% 门槛。
而这个增量**不需要 prepack**——只要把每 lane 每次读的 scale 块从 8 B 提到 32 B
（1 处改动、不动布局、不动加载器、不动 oracle 路径）就能吃满。

## 二、快照实现（file:line + 片段）

快照里 `prepack` 共三套：W8 `src/ops/linear/w8/w8_prepack_sm70.cu`(311 行)、
FP8 `src/ops/linear/fp8/fp8_prepack_sm70.cu`、NVFP4 `src/ops/linear/nvfp4/nvfp4_prepack_sm70.cu`。
测试 `tests/ops/linear/test_w8_prepacked_sm70.cpp`(151 行)。

### 2.1 加载期做了什么：一次纯全局地址置换

`w8_prepack_sm70.cu:18-35` 就是全部工作（`[row][k]` → `[N/32 tile][K/8 slice][QPN lane][8 code bytes]`）：

```cpp
__global__ void prepack_w8_qpn_kernel(const std::uint8_t* input_codes,
                                      const std::uint16_t* input_scales,
                                      std::uint8_t* output_codes,
                                      std::uint16_t* output_scales, int n, int k) {
    const std::int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= static_cast<std::int64_t>(n) * (k / 8)) return;
    const int lane  = index % 32;
    const int slice = (index / 32) % (k / 8);
    const int tile  = index / ((k / 8) * 32);
    const int row   = tile * 32 + ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) ? 4 : 0);
    reinterpret_cast<uint2*>(output_codes)[index] =
        reinterpret_cast<const uint2*>(input_codes)[row * (k / 8) + slice];   // 8 B 原样拷
    if ((slice & 3) == 0) {
        output_scales[(tile * (k / 32) + slice / 4) * 32 + lane] =
            input_scales[row * (k / 32) + slice / 4];
    }
}
```

- **不重排**：`row` 只是 `lane` 的固定函数，与 `w8_volta_qpn_gemm.cuh:73-76` 的
  `qp=(lane>>2)&3 / r=(lane&3)+((lane&16)?4:0) / col=qp*8+r` 逐位一致。
- **不解码/不缩放**：uint2 原样搬 8 个 code byte，scale 原样搬 fp16。
- **不合并事务**：拷贝粒度 8 B；合并发生在主循环**读取端**。
- scale 平面同时被重排成 `[tile][K/32 group][lane]`（`:31-34`）。
- 落地：`DeviceBuffer scratch` → kernel → `cudaMemcpyAsync` 回原 buffer →
  `cudaStreamSynchronize` → `weight.layout = VoltaQpnPrepacked`（`:250-265`）。
  **没有第二份常驻权重**，是原地改写（`const_cast`）。

官方描述 `docs/maintainer/qwen3.8-27b-artifact.md:162-164`（照抄）：

> SM70 W8 vocabulary output heads also use an in-place load-time permutation. Codes are ordered
> as `[N/32 tile][K/8 slice][QPN lane][8 code bytes]` and the unchanged FP16 scales as
> `[N/32 tile][K/32 group][QPN lane]`. Token embedding retains RowSplit storage.

### 2.2 主循环因此省掉了什么

`w8_qpn_prepacked_kernel`（`w8_prepack_sm70.cu:156-228`）：

```cpp
const std::int64_t scale_base = blockIdx.x * groups * 32 + lane;        // :167  循环外
for (int group = first; group < end; ++group) {
    const half2 scale = __half2half2(__ushort_as_half(scales[scale_base + group * 32])); // :173
    for (int sub = 0; sub < 4; ++sub) {
        const uint2 raw = __ldg(reinterpret_cast<const uint2*>(codes) +
            (blockIdx.x * (k / 8) + group * 4 + sub) * 32 + lane);      // :176-177  32 lane 连续
        ...
        volta_mma_qp_n(c, A[0], A[1], B[0], B[1]);                       // :207
```

省掉的**只有三样**：

1. **每 lane 的行基址**（`col*padded_groups*kCodeB`）→ 变成 `(...)*32 + lane`，`scale_base` 提到循环外。
   这是整数运算，**不是 DRAM**。
2. **32 lane 读 256 B 连续**（`uint2`×32），而不是 32 lane 各读自己一行（行距 5120 B）。
3. **scale 的 32 lane 读 64 B 连续**，而不是每 lane 2 B 却拉一个 32 B 扇区。

对照分析报告第 27 行的三条描述，**有两条与代码不符**，必须在后续引用时纠正：

| 报告说法 | 代码事实 |
|---|---|
| "主循环每 lane 合并 **512 B** 事务" | W8 是 **256 B**（`uint2`×32，`:176-177`）。512 B 是 FP8/NVFP4 的 `uint4`×32（`fp8_volta_qpn_gemm.cuh:134-138`）。W8 一个 group 只有 32 B，物理上做不出 512 B。 |
| "内循环**零** scale 乘法" | 仍每 group 做 4 次 `__hmul2(..., scale)`（`:184`）。scale 乘法一次都没省。 |
| "把 fragment lane↔(row,k) 映射烘进地址" | 属实（见 2.1）。 |

## 三、1.31×–1.93× 的原文与限定条件

### 3.1 出处：`docs/` 里查无此数，源头是 fork 自己的 dispatch 注释

`grep -rn "1\.93\|1\.31" --include="*.md" docs/` 只命中无关表列。
这个数是 `_scratch/ref-v100x2-analysis.md:27` 引用的，但它的真正源头是
**fork 里已经存在的常量注释** `src/ops/linear/w8/w8_dispatch.cpp:247-263`（照抄）：

```
// Band for the quadpair-split-N route (w8_volta_qpn_gemm.cuh), measured on the 27B output head,
// W8 [248320,5120], against whichever route production would otherwise take. Microseconds:
//
//   T:            1     2     3     4     5     6     7     8     9    12    16
//   incumbent  1797  2119  3838  4461  5136  5789  5270  5300  5390  5434  5502
//   qpn        1797  2933  2936  2940  2987  2996  3001  2971  5115  5368  5655
//
// QPN is flat across the band, as the mapping intends -- 2.93-3.00ms at every width -- while the
// incumbent climbs from 2.1 to 5.8ms. Both edges are the incumbent's shape again. T=1 and T=2 are
// the GEMV and its near neighbour, and T=1 in particular is already *at* the machine (752 GB/s of
// this card's ~794 measured ceiling), so there is nothing to take there. From T=9 the route needs
// a second 8-row A tile, which doubles it, while the 32x8 fused route absorbs those rows free.
//
// In band this is 1.31x at T=3 rising to 1.93x at T=6 and 1.78x at T=8, on the op the phase timing
// shows inside the 90%-of-a-round target verification pass.
```

对表验算：T=3 `2936/3838=1.307`、T=6 `2996/5790=1.932`、T=8 `2971/5300=1.784`。
**1.31×–1.93× 就是这张表，一字不差。**

### 3.2 限定条件（逐条，全部有出处）

| 问题 | 答案 |
|---|---|
| 单 GEMM 还是端到端 tok/s？ | **单 op、微秒**（`:249` "Microseconds"）。**不是 tok/s，不是端到端。** |
| 什么 shape | n=**248320**（整张头，**未按 TP2 切**），k=5120，T 见上表。生产是 TP2 → n=124160/卡，**该表一半的工作量**（`bindings.cpp:1323`）。 |
| 带宽受限？ | **是，且基线侧严重欠载**。incumbent 的 GB/s 在同仓给出：T=3=352、T=5=263、T=6=234（`w8_volta_qpn_gemm.cuh:8-10`），只有 ~794 GB/s 上限的 30–44%；QPN 侧 2987 µs / 1.271 GB = **426 GB/s**（54%）。 |
| 有"数值不变"声明？ | **没有**。`docs/performance.md:176` 明说 "QPN changes K-reduction association"；`docs/performance.md:149` 记录 paired NVFP4 QPN "Reverted; slower and **changed output**"；`docs/performance.md:569` 说 W8 "The target vocabulary head **retains its earlier accumulation route**"——快照是**刻意**让 vocab head 保留原结合顺序的。 |

### 3.3 对本战役最要紧的一句

**1.31×–1.93× 的分母（sliced r8_c8 incumbent）在 fork 里已经不存在了。**
`w8_dispatch.cpp:293` 在 T=3..8 直接 `return launch_w8_volta_qpn`。
所以把它当"待移植收益"是重复计入。真正还没被测量的对比是
**QPN + prepack vs QPN + RowSplit**，而快照和 fork **都没有这个数据**。

## 四、fork 现状（输出头走哪条 kernel、差在哪）

### 4.1 输出头的实际发射路径（file:line）

1. `src/targets/qwen3_6/impl/runtime/text_context_impl.h:2062`
   `ops::linear_column_parallel(hidden, {*lm_head_, *lm_head_peer_}, part, execution);`
   → `part[r] = {kShardVocab, columns}`，`kShardVocab = ModelConfig::vocab/2`（`text_context.h:90`）= **124160**。
2. `src/targets/qwen3_6_27b/impl/load/bindings.cpp:1323`
   `output_head = materialized_weight(backing, plan.output_head, 248320 / tp, 5120, device);`
   → **每卡 n=124160、k=5120。权重确实是两卡各一半。**
3. `w8_dispatch.cpp:307` → `select_w8_launch` → `:293`
   `w8_uses_volta_qpn(124160, 5120, 5)` 成立（`:266-269`：`t∈[3,8]` ∧ `n>=4096` ∧ supported）
   → **`launch_w8_volta_qpn`**。
4. `src/ops/linear/w8/w8_volta_qpn_gemm.cu:44` 发射 `w8_volta_qpn_gemm_kernel<1,4>`
   （实现 `w8_volta_qpn_gemm.cuh:57-212`）。

**结论：T=5/6 的输出头在 fork 里已经走 QPN，读的是 RowSplit 布局。**

同一波段还有五个 MTP W8 矩阵（`w8_uses_volta_qpn` 的 n≥4096 门槛全部满足）：

| 矩阵 | 出处 | TP2 每卡 [n,k] | codes |
|---|---|---:|---:|
| `mtp.input_projection` | `bindings.cpp:1352` | [5120, 5120] | 26.2 MB |
| `mtp.attention.packed` | `bindings.cpp:1362` | [7168, 5120] | 36.7 MB |
| `mtp.output` | `bindings.cpp:1377` | [5120, 3072] | 15.7 MB |
| `mtp.post_mixer.gate_up` | `load_mlp` → `bindings.cpp:247`（34816/2） | [17408, 5120] | 89.1 MB |
| `mtp.post_mixer.down` | `load_mlp` → `bindings.cpp:248`（17408/2） | [5120, 8704] | 44.6 MB |

即 **W8 QPN 不是一个 kernel，是一族六个**（合计 848 MB codes/卡），改一处六个都受益。

### 4.2 该 kernel 现在每次加载做多少地址运算 / scale 乘法

`w8_volta_qpn_gemm.cuh:84-88`（每 CTA 一次，循环外）：
```cpp
const std::uint8_t*  crow = codes  + col * padded_groups * kCodeB;   // 行基址，kCodeB=32
const std::uint16_t* srow = (const uint16_t*)(scales + col * padded_groups * 2);
```
主循环 `:104-131`：
- codes：每 4 个 group 组一次 `crow + g*kCodeB`（`:110`）+ 两次 16 B `uint4`（`:111-112`）
  → **每 group 2 次加载 + 1 次整数乘加**。
- scale：每 4 group 一次 `__ldg((const uint2*)(srow + gb))`（`:119-120`，8 B），
  把 16× 放大压到 4×（`:114-117` 自述）。
- scale 乘法：`:153` `__hmul2(__hsub2(...), sc2)` **每 group 4 次**，与快照**相同**。

相对快照 prepack 版，差别只剩两点：codes 是 32 条散落行（非 256 B 连续）；scale 仍是 4× 放大。
而 fork 对第一点**知情且主动选择**——`w8_volta_qpn_gemm.cu:41-43` 原文：

> `// kBlk = 4 groups = 128 B, one cache line per lane per iteration; see the Q4 sibling for why`
> `// that, rather than an offline weight prepack, is what keeps the scattered per-lane row reads`
> `// off DRAM.`

Q4 sibling 的量化依据 `src/ops/linear/q4/q4_volta_qpn_gemm.cuh:120-134`（照抄关键句）：

> `// ... at n=34816 DRAM read measured 2.44x the weight with L2 at 38%, against 1.05x and 64% at`
> `// n=6144 ... Consuming a whole line per lane per iteration means each line is touched once and`
> `// fully used, so residency stops mattering, and it puts 2*kBlk loads in flight to cover their`
> `// latency.`

**即 `kBlk=4` 已把 codes 平面的 DRAM 利用率做到 1.05×。prepack 对 codes 平面只剩
"请求条数 / 寄存器"的余量，没有字节余量。**

### 4.3 fork 有没有别的 prepack？W8 是不是唯一缺的？

`QuantLayout::VoltaQpnPrepacked`（`src/core/tensor.h:102`）在 fork 里已被两套格式使用：

| 格式 | load-time prepack | 加载器调用点 | QPN kernel 分 prepacked 分支？ |
|---|---|---|---|
| FP8 | `fp8_prepack_sm70.cu:51` | `bindings.cpp:1325-1327` | 是，`fp8_volta_qpn_gemm.cuh:134`（`if constexpr (Prepacked)` → 512 B 合并读）+ `:252-287` 派发 |
| NVFP4 | `nvfp4_prepack_sm70.cu:72` | `nvfp4_dispatch.cpp:69` | 是，`nvfp4_volta_qpn_gemm.cuh:456` |
| **W8** | **无** | **无** | **否**（`w8_volta_qpn_gemm.cuh` 全文无 `prepacked`） |

fork 现状 `bindings.cpp:1324-1329`：
```cpp
#ifdef NINFER_VOLTA_BUILD
    if (output_head.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        ::ninfer::ops::detail::fp8_prepack_qpn_sm70(output_head);
    }
#endif
```
快照同处（`ref-v100x2/ninfer-V100X2-master/src/targets/qwen3_6_27b/impl/load/bindings.cpp:1274-1280`）
多两行 `else if (qtype == QType::W8G32_F16S) w8_prepack_qpn_sm70(output_head);`。

→ **确认：W8 是 NVFP4/FP8/W8 三套里唯一没有 prepack 的格式，缺口就在这两行。**

## 五、本机带宽/算术核算（含对我第 4 问算术的核对）

### 5.1 字节数（每卡，TP2，输出头）

| 项 | 算式 | 值 |
|---|---|---:|
| codes | 124160 × 5120 | **635.7 MB** |
| groups/行 | 5120/32 | 160 |
| scale 行距 | 160 × 2 | 320 B |
| scale 名义 | 124160 × 160 × 2 | **39.7 MB** |
| scale 实读（4×） | 见 5.2 | **158.9 MB** |
| **当前合计** | 635.7 + 158.9 | **794.6 MB** |
| **prepack 后合计** | 635.7 + 39.7 | **675.4 MB** |
| 差 | | **−119.2 MB（−15.0%）** |

加上五个 MTP W8 矩阵（每卡 codes 212.3 MB，scale 名义 13.3 MB → 实读 53.1 MB）：

| | 当前 | prepack 后 | 差 |
|---|---:|---:|---:|
| 六个 W8 QPN 矩阵/卡 | **1060.0 MB** | **901.0 MB** | **−159.0 MB（−15.0%）** |
| 其中 codes | 848.0 MB | 848.0 MB | 0（字节不变，只减请求条数） |
| 其中 scales | 212.0 MB | 53.0 MB | −159.0 MB |

### 5.2 scale 的 4× 放大怎么来的

`w8_volta_qpn_gemm.cuh:118-120` 每 lane 读 8 B（`uint2`，4 个 fp16），
32 lane 的行距是 320 B → **32 个互不重叠的 32 B 扇区**，只用掉 8/32 → **4×**。
代码注释自己承认了这一点（`:114-117`："Reading the block's kBlk scales as one vector cuts
that to 4x"）。

**关键观察：把每次读的 scale 块从 8 B 提到 32 B（16 个 group），
`row*320 + gb*2` 恰好 32 B 对齐（320 % 32 == 0，且 gb 为 16 倍数时 `gb*2` 是 32 B 对齐），
32 lane 就变成 32 个**被完全用满**的扇区 → 放大 1×。不需要任何布局重排。**

### 5.3 这个 kernel 是带宽受限还是计算受限（T=5）

- MMA：124160 × 5120 × 5 = 3.18 GMAC = 6.36 GFLOP，对 V100 fp16 TC (~125 TFLOPS) → **0.05 ms**。
- 反量化：每 code ≈ 1.5 个 half2 运算 → 6.36e8 codes × 1.5 / 15.7 Tops/s ≈ **0.06 ms**。
- 实测 2987 µs（整张头）→ 算术占 **< 5%**。
- 指令发射：3880 CTA × 4 warp × ~960 instr ≈ 1.5e7 warp-instr / (80 SM × 4 sched × 1.53 GHz) ≈ **0.03 ms**。

**⇒ 100% 带宽受限。地址算术和 scale 乘法对时间没有贡献，prepack 的"省算术"价值为零。
唯一有价值的是"省 DRAM 字节"。** 这一条可以给分析报告第 9 项的表述定案。

### 5.4 对第 4 问算术的核对（三处要修）

| 你的说法 | 核对结果 |
|---|---|
| "权重 248320×5120 的 W8 = 约 1.27 GB" | ✅ 1.271 GB（整张头）。 |
| "decode 每步读一遍 = 1.27 GB/步" | ⚠️ **两卡合计**才是 1.27 GB；**每卡 0.636 GB codes + 0.040 GB scales**。另外每步还要 allgather：124160×5×2 B×2 = 2.5 MB/步（`text_context_impl.h:2069-2073`），可忽略。 |
| "每次前向是否全读" | ✅ 是。QPN kernel grid = n/32 CTA，每 CTA 读 32 行整行，每行恰好一次（`w8_volta_qpn_gemm.cu:40`）。无跨步复用。 |
| "两卡各半 → 每步 0.7~1.4 ms" | ⚠️ **0.7 ms 是"两卡同时跑满 900 GB/s 名义峰值"的理想值**，实测天花板是 ~794 GB/s（`w8_volta_qpn_gemm.cuh:12`）→ 地板 **0.80 ms**。实测点是 **~1.5 ms/卡**（2987/2，见 3.1 表 T=5）。**正确区间应为 0.8–1.9 ms，实测点 1.5 ms**；0.7 ms 达不到。 |
| 结论 | 输出头 ≈ 1.5 ms；加 MTP W8 ≈ **2.0 ms**，占 round（3.947 s / 106 round = **37.2 ms**；135.6 tok/s 时 ≈35.7 ms）的 **≈5.6%**。 |

### 5.5 换算成时间：为什么"±0.8%"而不是 −15%

- 六个 W8 QPN 矩阵当前 1060 MB，按 532 GB/s 算 = **1.99 ms/round**。
- 只做 scale 合并（效率不变）：901/532 = **1.69 ms** → 省 **0.30 ms** → **+0.80%** decode。
- 若 codes prepack 还能把 532 提到 ~700 GB/s（**未证实**，见 5.6）：901/700 = 1.29 ms → 省 0.70 ms → **+1.9%**。

**但必须同时说清楚一条反向证据：整个 decode 每卡每 round 读 11.23 GB
（`baseline-summary.md`）÷ 37.2 ms = 302 GB/s，只有天花板的 38%。**
round 级别**并未带宽饱和**，是延迟/串行受限。所以砍 1.4% 的**总流量**（159/11230），
不太可能换到 1.4% 的**时间**——上面的 +0.80% 已经是"W8 kernel 局部饱和"这个最有利假设下的上限。

### 5.6 证据不足之处（不编数）

1. fork 没有任何 `QPN + prepack` 的实测（快照也没有）。1.31×–1.93× 无法用来外推。
2. 532 GB/s 是从"注释时间 2987 µs + 推导流量 1.589 GB"算的，不是 profile 读数。
   若实际 scale 只被读 1×（L1/L2 吸收），则流量 1.351 GB → 452 GB/s，
   scale 可省的字节就从 159 MB 掉到 **0**，整个结论直接翻转为"不值得"。
   **这一条必须在动代码前用 ncu 的 `dram__bytes_read` 在 n=124160/T=5 上确认。**
3. codes prepack 能否把 532→700 完全无据（Q4 注释说 1.05× 行利用率已达标，暗示大部分已在）。
4. 3.947 s / 106 round 的 round 时间是整段 decode 的平均值，不是单 round 抖动；
   5.5 的百分比是在这个平均 round 上算的。

## 六、改动清单与风险

### 6.1 方案 A（推荐先做）：只把 scale 块读宽，不动布局

| 文件 | 函数 | 预估行数 |
|---|---|---:|
| `src/ops/linear/w8/w8_volta_qpn_gemm.cuh` | `w8_volta_qpn_gemm_kernel`，scale 预取改为 16 group 一块（32 B/lane），`scv[16]` | ~15 |

- **不动加载器、不动 `Weight`、不动 layout、不动 oracle 路径**。
- **风险：寄存器**。`q4_volta_qpn_gemm.cuh:131-134` 明确说 `__launch_bounds__(128,8)` 的
  64 寄存器预算已被 `cw[2*kBlk]`（32 个）吃满，"双缓冲就 spill，代价 1.6×"。
  再加 `scv[16]`（8 个）大概率 spill。**缓解：`kBlk` 4→2 换出寄存器，或 scale 块走 shared
  （每 CTA 只需 32 行 × 160 scale = 10 KB，完全放得下）——后者更稳。**
- oracle：**逐位不变**（只改 scale 的取数时机，不改取值也不改求和顺序）。

### 6.2 方案 B（不推荐）：完整 prepack

| 文件 | 内容 | 预估行数 |
|---|---|---:|
| `src/ops/linear/w8/w8_prepack_sm70.cu` / `.h` | 从快照搬 `prepack_w8_qpn_kernel` + `w8_prepack_qpn_sm70` | ~200（可精简掉 SIMT/image 分支） |
| `src/targets/qwen3_6_27b/impl/load/bindings.cpp:1324-1329` | 加 `else if (W8G32_F16S)` 一行 | +3 |
| `w8_volta_qpn_gemm.cuh` | 加 `Prepacked` 模板分支（256 B 合并读 + 64 B scale 读） | ~25 |
| `w8_volta_mma_gemm.cuh/.cu` | **prefill 走这条**（t=2560 > 8，`w8_dispatch.cpp:294`），必须同步加分支 | ~25 |
| `w8_rowsplit_gemm_simt.cu`+`w8_small_t.cu` | T=1/2 路径（`w8_dispatch.cpp:27` → `simt_r8_c4`）也要认新布局，或照快照 `w8_qpn_simt_kernel` 做 shared 转置 | ~80 |
| `w8_dispatch.cpp` | prepacked 布局改派 | ~8 |

**风险清单（按严重度）**：

1. **显存可能直接 OOM**。快照 `w8_prepack_sm70.cu:252` 起 675 MB 的 scratch
   （`code_bytes + scale_bytes`）。本机单卡 16 GB，权重已占 11.23 GB
   （`baseline-summary.md`，含 reservation 2.12 GB / workspace 0.755 GB）。
   加载期虽然还没有 KV，但余量很小，**必须改成按 tile 分块 prepack，scratch 压到几十 MB**。
2. **prefill 回归**。输出头 prefill 是 t=2560 的计算受限 GEMM，走 `w8_volta_mma`
   （`w8_dispatch.cpp:294`，`w8_volta_mma_supported` 在 `w8_volta_mma_gemm.cu:15-22`
   对 k=5120 恒真）。换布局要动这条 kernel，而 prefill 门槛是 +1.0%
   （`baseline-summary.md`），风险不对称——为了 decode +0.8% 去碰 prefill 主路径，性价比很低。
3. **oracle**。prepack 是纯置换，lane→row 映射与 group 遍历顺序都不变
   （快照 `w8_prepack_sm70.cu:28` 的 `row` 公式 = fork `w8_volta_qpn_gemm.cuh:73-76`），
   **逐位一致可以做得到**；但快照自己也只在 vocab head 上用它，且 `docs/performance.md:569`
   说 vocab head "retains its earlier accumulation route"。要拿 oracle 指纹
   （`sha1 06287348665d3648 / 406 / 421 / 106`）做硬否决。
4. **原地改写 `weight.qdata`**。快照用 `const_cast`（`:260-263`）。
   fork 的测试套件里有 "immutability checks"（`docs/performance.md:572`），
   原地改写入参会撞上现有不变量测试。
5. **收益区间 0.8%–1.9%，门槛 0.5%，且 5.6 第 2 条一旦证伪就归零。**
   方案 B 的改动量（~340 行 + 三条 kernel 路径 + 加载器）与收益完全不成比例；
   同样 15% 的流量削减，方案 A 只要 15 行。

### 6.3 建议的下一步（不写代码也能做的那一步）

先用 ncu 在**现网形状**（n=124160, k=5120, T=5, 单卡）上取两个数：

```
dram__bytes_read.sum            # 与 635.7 MB（codes）+ 158.9 MB（scale 4×）比
dram__bytes_read.sum (scale 段) # 与名义 39.7 MB 比，判断放大是 1× 还是 4×
```

- 若 scale 确为 4× → 走方案 A（~15 行 + shared 暂存），预期 **+0.8%**，风险极低。
- 若 scale 已是 1×（被 L1/L2 吸收）→ **整个方向到此为止**，
  在 `candidates.md` 里记"prepack 在 V100 上收益有限（scale 放大不成立、codes 已 1.05×）"，
  不再投入。
- 若给 MTP 那五个小矩阵也做（`n=5120..17408`，每卡合计 212 MB），
  方案 A 的 +0.8% 已把它们算在内；不要重复计。
