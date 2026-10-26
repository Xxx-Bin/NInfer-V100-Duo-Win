# 第 9 项 · GDN controls 融合（split-10 精确保序）· 落点勘察
状态：完成

> 只读勘察，未编译、未运行、未 git 操作。
> fork = `C:/Users/lwb6/Documents/git/NInfer-V100-Duo-win`
> 快照 = `_scratch/ref-v100x2/ninfer-V100X2-master`

## 一、判决

**值得做，而且原判「B 类 → 默认跳过」是错的。按快照原样搬，本项是 A 类（逐位一致），oracle 不变。**

三条硬结论：

1. **`shard_exact` 与 fork 现有 split-10 两核路径的浮点归约树完全同构**（下面第三节逐条对到
   线程/lane/累加次序）。它不是「换一种求和顺序」，而是「把同一棵求和树从两个 kernel 塞进
   一个 kernel」。所以输出 `g`/`beta`（FP32）**逐位相同**，不是末位 ulp 级别的差异。
   依据：元素→线程映射逐项相同（`kernels.cu:92-117` vs 快照 `:207-224`）、warp 蝶形相同
   （同一个 `warp_reduce_sum`）、10 路上升序累加相同（`kernels.cu:149-154` vs 快照 `:231-232`）。
2. **上游自己的文档也说是精确的**：`docs/performance.md:558` 该阶梯行输出列写着
   `Exact 513 IDs and counters to producer publish`；`docs/v100-decode-survey.md:102-103`
   明确「norm-image 与后续两步 GDN 融合的全部 513 IDs 和接受计数则与合格的 244.655 阶段相同」，
   并说改浮点关联的是 `Split-D`，不是这一步。
3. **+1.76% 里的「融合」那半截 fork 已经有了**（`gdn_projected_conv.cu`，见第六节），
   所以预期收益只剩 split-10 单 launch 那部分。上游在 TP4 线上把两件事分开测过：
   split-10 单 launch = +1.78%，projection/conv/record 融合 = +2.88%
   （`docs/performance.md:79-84`）。fork 是 TP2，那半截已具备，**只剩 ≤1.76% 的上限，
   按 launch 数折算约 +0.6~0.9%**。

成本：**只碰 1 个文件、纯新增 ~61 行、不外溢 runtime、无 API 变更**（第五节）。

## 二、shard_exact 实现（文件:行号 + 片段）

定义：快照 `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu:197-236`

```cpp
// :197  A head/token CTA retains the split-10 warp dot products and their final
// :198  ascending FP32 sum, so the one-launch route preserves the represented controls.
// :199  template <int Heads>
// :200  __global__ void bf16_gdn_gating_proj_shard_exact_kernel(
// :201      const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
// :202      const __nv_bfloat16* __restrict__ b_weight, const float* __restrict__ A_log,
// :203      const float* __restrict__ dt_bias, float* __restrict__ g, float* __restrict__ beta) {
// :204      const int token = static_cast<int>(blockIdx.y), head = static_cast<int>(blockIdx.x);
// :205      const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, split = tid >> 5;
// :206      float a = 0.0F, b = 0.0F;
// :207  #pragma unroll
// :208      for (int vec = lane; vec < 64; vec += 32) {
// :209          const int k = split * 512 + vec * 8;
// :210          const uint4 xv = load_vec<uint4>(x + token * kK + k);
// :211          const uint4 av = load_vec<uint4>(a_weight + head * kK + k);
// :212          const uint4 bv = load_vec<uint4>(b_weight + head * kK + k);
// :213          ...  // xp/ap/bp 取 4 个 32-bit 字
// :217          for (int pair = 0; pair < 4; ++pair) {
// :218              const auto xf = bf16x2_bits_to_float2(xp[pair]);
// :219              const auto af = bf16x2_bits_to_float2(ap[pair]);
// :220              const auto bf = bf16x2_bits_to_float2(bp[pair]);
// :221              a = fmaf(af.x, xf.x, a); a = fmaf(af.y, xf.y, a);
// :222              b = fmaf(bf.x, xf.x, b); b = fmaf(bf.y, xf.y, b);
// :223          }
// :224      }
// :225      a = warp_reduce_sum(a); b = warp_reduce_sum(b);
// :226      __shared__ float partial[2][10];
// :227      if (lane == 0) { partial[0][split] = a; partial[1][split] = b; }
// :228      __syncthreads();
// :229      if (tid == 0) {
// :230          a = 0.0F; b = 0.0F;
// :231  #pragma unroll
// :232          for (int s = 0; s < 10; ++s) { a += partial[0][s]; b += partial[1][s]; }
// :233          g[token * Heads + head] = -expf(A_log[head]) * softplus(a + dt_bias[head]);
// :234          beta[token * Heads + head] = sigmoid(b);
// :235      }
// :236  }
```

挂载点：快照 `kernels.cu:524-556` 的 `bf16_gdn_gating_dispatch_shard`，Volta 分支在最前

```cpp
// :529  #ifdef NINFER_VOLTA_BUILD
// :530      if ((a_weight.n == kShardN || a_weight.n == 12) &&
// :531          x.ne[1] >= 2 && x.ne[1] <= kSmallTMax) {
// :535              bf16_gdn_gating_proj_shard_exact_kernel<Heads>
// :536                  <<<dim3(Heads, x.ne[1]), 320, 0, stream>>>(...);
// :546          return;
// :547      }
// :548  #endif
```

- 网格：`dim3(Heads, T)`，**每 block 320 线程 = 10 warp**（`kSmallTSplits * 32`）。
  `warp id == split`，`blockIdx.x == head`，`blockIdx.y == token`。
- 生效窗口：`T ∈ [2, 8]`（`kSmallTMax = 8`，`kernels.cu:26`）。`T == 1` 仍走 gemv，`T > 8` 仍走两核。

**fork 侧现状**：全仓 `grep shard_exact` **未找到**（`src/` 下 0 命中）。fork 的
`bf16_gdn_gating_dispatch_shard`（`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu:466-478`）
只有两条路：`T == 1 → gemv_shard`，`T >= 2 → split10_shard`（两核：partial + reduce）。

## 三、与 fork 现有 controls 核的逐条差异

对照对象：fork 的同一 op 的两核路径
`bf16_gdn_gating_proj_small_t_partial_kernel`（`kernels.cu:52-130`，shard 实例化在 `:447-453`，
`N = kShardN = 24`）+ `bf16_gdn_gating_proj_small_t_reduce_kernel`（`kernels.cu:133-160`，
shard 实例化在 `:458-462`）。

### 3.1 结构性差异（6 条，全部不影响浮点值）

| # | 维度 | fork 两核 | shard_exact | 是否改数值 |
|---|---|---|---|---|
| D1 | launch 数 | 2（partial + reduce，`kernels.cu:392` 与 `:402`） | 1（`快照:536`） | 否 |
| D2 | block 数 / 线程 | `partial_grid = (48/4, 10, ceil(T/8))` = (12,10,·)，128 线程（`:389-391`）；`reduce = ceil(24T/128)` = **T=6 时仅 2 个 block**（`:399-402`） | `(24, T)`，320 线程 | 否 |
| D3 | warp 分工 | 1 warp = 1 个 logical_row（0..47，含 a 段与 b 段），循环 10 个 split 靠 `blockIdx.y` | 1 warp = (1 head, 1 split, 1 token)，a 与 b 在同一 warp 两个累加器 | 否 |
| D4 | token 维 | 一个 warp 持 `acc[8]`，处理 8 token（`:88-117`） | 一个 block 一个 token，`acc` 标量 | 否 |
| D5 | x 的搬运 | 先整块搬进 `__shared__ bf16 x_sh[8][512]`（`:61,71-79`），再按 uint4 读 smem | 直接从 global 读 `x + token*kK + k`（`快照:210`），无 smem staging | 否（bf16 位拷贝，无转换） |
| D6 | 跨 split 归约 | 走 global workspace：`partial[(split*t+token)*48 + row]`（`:125-126`），独立 reduce kernel 读回 | `__shared__ float partial[2][10]`（`快照:226`），tid==0 就地累加 | 否 |

### 3.2 浮点求和顺序 / 分组：**逐项相同**（这是本项的关键）

| # | 维度 | fork 两核 | shard_exact | 判定 |
|---|---|---|---|---|
| F1 | 元素→split 归属 | `k0 = split * KSlice`，`KSlice = 512`（`:27,69`） | `k = split * 512 + vec * 8`（`快照:209`） | 相同 |
| F2 | split 内向量→lane | `for (vec = lane; vec < kVecsPerCol(=64); vec += 32)`（`:59,92`）→ 每 lane 拿 `vec = lane` 与 `lane+32` | 同一表达式 `vec = lane; vec < 64; vec += 32`（`快照:208`） | 相同 |
| F3 | 向量内 8 个元素的 fmaf 次序 | `fmaf(wf0.x,xf0.x) → (wf0.y) → (wf1.x) → (wf1.y) → (wf2.x) → (wf2.y) → (wf3.x) → (wf3.y)`，累加进同一 `acc[tt]`（`:107-114`） | `pair = 0..3` 升序，每 pair 先 `.x` 后 `.y`，`fmaf(af.x,xf.x,a)`（`快照:217-223`）—— `xp[1..4]` 就是 `xv.x/y/z/w`，故次序完全一致 | 相同 |
| F4 | 操作数先后 | `fmaf(w, x, acc)`（权重在前） | `fmaf(af, xf, a)`（权重在前） | 相同 |
| F5 | warp 内归约 | `warp_reduce_sum(acc[tt])`（`:122`），`warp.cuh:21-28` 的 `__shfl_down_sync` 蝶形，32 lane 满掩码 | `warp_reduce_sum(a)`（`快照:225`）**同一个函数** | 相同 |
| F6 | 10 路跨 split 累加次序 | `for (split = 0..9) acc_a += partial[base+row]`（`:149-154`，初值 0.0f） | `for (s = 0..9) a += partial[0][s]`（`快照:231-232`，初值 0.0f） | 相同（都是上升序、线性链） |
| F7 | epilogue | `softplus(acc_a + dt_bias[row])`、`-expf(A_log[row]) * sp`、`sigmoid(acc_b)`（`:157-159`） | 同式同序（`快照:233-234`） | 相同 |
| F8 | a/b 是否互相干扰 | 分属不同 warp / 不同 `logical_row` | 同一 warp 两个独立累加器，无交叉 | 相同（各自独立链） |

**结论：不存在任何一处求和顺序或分组差异。** fork 的 partial 核里一个 warp 只覆盖 1 个 split
切片（`k0 = blockIdx.y * 512`），所以「split」这个分组维度在两核版里也存在，只是由 grid 表达；
shard_exact 把它挪进了 block 内的 warp 维。元素到 lane、元素在每 lane 内的累加次序、warp 蝶形、
10 路上升序线性累加，四层全部一一对应。

唯一「看起来像」改序的地方是 D5/D6（smem staging 与 workspace 中转），但它们只搬位不搬值：
x 是 bf16 位拷贝，partial 是 FP32 精确写出再精确读回，不引入任何一次额外舍入。

## 四、数值影响评估（能否保住 B 类三段式）

**不需要 B 类三段式——它是 A 类。**

- 输出张量 `g` / `beta` 是 **FP32**（`src/ops/wrapper/gdn_gating_proj.cpp:250-253`
  `require_sequence_tensor(g, DType::FP32, kShardHeads, ...)`），不是 BF16。第三节 F1–F8 已证明
  两条路径产生**逐位相同**的 FP32 值，所以不存在「末位 ulp 抖动」这一档，更不可能动 argmax。
- 因此 `accepted / drafted / rounds`、`acceptance_length / per-position`、513 IDs 都应
  **严格相等**（这是 A 类判据，比 B 类三段式更强）。
- 上游背书：`docs/performance.md:558` 该行输出列 `Exact 513 IDs and counters to producer publish`；
  `docs/performance.md:80-81`：「single-launch small-T GDN controls that retain the split-10 arithmetic
  order … passes the independent FP64 control oracle」；`docs/v100-decode-survey.md:102-103`
  把「改浮点关联」归给 Split-D，并明确 norm-image 及其后两步 GDN 融合的 513 IDs / 接受计数不变。

**但这三个前提必须守住，否则立刻退化成 B 类：**

1. 必须保留 `512` 这一 K 切片宽度（`kSmallTKSlice`，`kernels.cu:27`）。改成 20 片 / 256 片
   → F1 的划分变了 → **B 类**。
2. 必须保留 `vec = lane; vec < 64; vec += 32` 的两趟遍历（F2）。改成「每 lane 连续 4 个 uint4」
   之类 → lane 归属变了 → **B 类**。
3. 必须保留 `for s = 0..9` 的**上升序线性链**（F6）。改成两两配对树 / `#pragma unroll` 后让编译器
   重排（务必保持 `#pragma unroll` 顺序累加，不要引入中间 float 数组的成对求和）→ **B 类**。

只要照抄快照的 `:207-235` 这 29 行不「优化」，就 A 类。

## 五、规模（文件 / 行数 / 是否外溢）

**只 1 个文件、纯新增、无 API 变更、不外溢。**

| 动作 | 文件 | 行号 | 行数 |
|---|---|---|---|
| 新增 kernel（照抄） | `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu` | 插在 `:195` 之后（原 `bf16_gdn_gating_proj_35_simt_kernel` 之前，即 fork 的 `:197` 位置） | +41 |
| 新增 dispatch 分支 | 同上，`bf16_gdn_gating_dispatch_shard` 头部，即 `:470` 之后 | | +20 |
| 头文件 | `bf16_gdn_gating_proj_kernels.h` | — | **0**（无新公开符号） |
| wrapper / runtime | — | — | **0** |

**为什么不需要动 runtime（与快照的差别）：**

- 快照的 shard 路径是通用的 `Heads ∈ {12, 24}`（`快照:520-521` 按 `a.n` 分派），
  因此快照改了三个外部可见签名：`require_shard_shape` 允许 `w.n == 12`（`快照:313`）、
  `bf16_gdn_gating_shard_workspace_bytes(tokens, heads)`（`快照:558`，`heads` 默认 24）、
  `dispatch_shard` 里 `a_weight.n == 12` 分支。对应 wrapper 侧
  `bf16_gdn_gating_shard_workspace_bytes(tokens, a_weight.n)`（快照
  `src/ops/wrapper/gdn_gating_proj.cpp:305` 一带）。
- **fork 只有 TP2**：`kShardHeads = 24`（`src/ops/wrapper/gdn_gating_proj.cpp:241`），
  `bf16_gdn_gating_shard_workspace_bytes(std::int32_t tokens)` 单参
  （`kernels.cu:480`、`wrapper:297`、`wrapper:323`），`require_shard_shape` 只认 `kShardN`
  （`kernels.cu:271-276`）。
- 所以 **12-head 那套泛化可以整段丢掉**：只保留 `Heads = kShardN = 24` 一个特化
  （甚至直接写死 24，不模板化），`workspace_bytes` 签名不动，wrapper 一行不改。
- 这同时消掉了快照里 `bf16_gdn_gating_dispatch_shard` 的 `#ifdef` 分支对 `require_shard_shape`
  的重复调用（`快照:532-533`）——fork 的 `dispatch_shard` 已经在校验（`wrapper:296-310` 上层已过
  `validate_column_rank_semantics`），可省。

**净工作量：`kernels.cu` +61 行，2 处插入，0 删除。**

补充：`NINFER_VOLTA_BUILD=1` 在 `CMakeLists.txt:25` 就已定义，所以照抄快照的 `#ifdef
NINFER_VOLTA_BUILD` 包裹会正常生效；这个核是纯 SIMT，不加宏也能编过，加宏只是保留上游的
「Volta 专属调优」语义。

**副作用（可接受，但要写进报告）**：`T ∈ [2,8]` 时 wrapper 仍会按
`bf16_gdn_gating_shard_workspace_bytes(tokens)` 分配 workspace（`wrapper:296-310`）再被新核忽略。
每次每 rank 多分配 `10 * T * 48 * 4 B`（T=6 时 ≈ 11.5 KB），arena 分配，零拷贝，无 GPU 代价。
不要为省这点去改签名，否则就是自找外溢。

**L2 流量重算（怕「读放大」吃掉收益的担心可以排除）**：新核每个 (head, token, split) 各自
读权重与 x，权重读放大约 T 倍、x 读放大约 24 倍。但一个 rank 的 controls 权重只有
`2*24*5120*2 B ≈ 491 KB`、x 只有 `T*10 KB`（T=6 时 60 KB），V100 有 6 MB L2，全部常驻；
放大量级落在 L2 带宽（≈2 TB/s）上折合不到 2 µs/层，而省下的是一次 launch（见下）。

## 六、+1.76% 的拆解（三件事里 fork 已有几件）

### 6.1 上游到底声明了什么

- 阶梯行：`docs/performance.md:558`
  `| Exact split-10 GDN controls + projection/conv/record fusion | 186.422 | Exact 513 IDs and counters to producer publish |`
  上一行 `183.201`（`performance.md:557`）→ **+1.76%**。这条是 **TP2 packet-push 血统**
  （`performance.md:552-565` 整表）。
- 两件事在 **TP4 血统里是分开测的**，给了可拆的绝对口径（`docs/performance.md:79-84`）：
  - 「single-launch small-T GDN controls that retain the split-10 arithmetic order」
    → 246.431 → 250.824，**+1.78%**，并「passes the independent FP64 control oracle」；
  - 「TP4 GDN projection/record/conv fusion」→ 250.824 → 258.037，**+2.88%**。
  - 中文侧同义：`docs/v100-decode-survey.md:97-100`「随后将 T2..8 的 GDN controls 合为单次
    launch，保留 split-10 的点积和合并顺序，相对 norm-image 阶段再提升约 1.78% …… TP4 GDN
    projection/record/conv 随后融合为直接输出 …… 相对 250.824 阶段再提升 2.88%」。
  注意 `T2..8` 与新核的生效窗口一致，与 `kSmallTMax = 8` 一致。

### 6.2 三件事 fork 各有几件

| # | 子项 | fork 是否有 | 证据 |
|---|---|---|---|
| A | **split-10 controls 单 launch（shard_exact）** | **没有** | 全仓 `shard_exact` 0 命中；`kernels.cu:466-478` 的 `dispatch_shard` 只有 gemv / 两核 split10 |
| B | **projection/conv/record 融合** | **有** | `src/ops/gdn_input_proj/gdn_projected_conv.cu`（`gdn_projected_conv_snapshot_launch` :150、`gdn_projected_conv_record_launch` :163），声明在 `gdn_projected_conv.h:9,15`；接线在 `src/ops/wrapper/gdn_input_proj.cpp:456,472,645,1120`，tp2 走 `Variant::gdn_input_projection_record`（`src/targets/qwen3_6_27b/impl/variant.cpp:794`）。与快照逐行比对只差「TP4 的 2560 通道几何」被删掉（`diff gdn_projected_conv.cu`：`Channels == 2560` 分支与 `projected.ne[0]==2560` 入口），TP2 用的 10240 / 5120 几何**两边一致** |
| C | **norm + controls 融合** | **部分有，但不在 tp2 路径上** | `src/ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu`（`gdn_norm_gating_27_simt` :14，`bf16_gdn_norm_gating_proj_27_launch` :94）写死 `D=5120, H=48`，**只有 tp1 全 48 头形态**；`variant.cpp:746-749` 注释明说「There is no split form of the fused kernel」→ tp2 的 norm 仍是独立 launch |

**所以 `queue-v2.md:48` 那句「fork 已有 `bf16_gdn_norm_gating_proj_27.cu`（每头同时算 control
点积与完整 norm）→ 融合部分已存在」需要修正**：那份文件对应的是 norm 融合，且只服务 tp1；
阶梯里的「projection/conv/record fusion」其实是 `gdn_projected_conv.cu`——**它确实已有**。
结论方向不变（三件里已具备两件），但「哪两件」要说准：是 **B（conv/record）+ 部分 C（norm，仅 tp1）**，
缺的是 **A（split-10）**。

### 6.3 折算

- 上限：整条阶梯步 = +1.76%（TP2 血统实测）。fork 已含 B，所以纯 split-10 的增量**严格小于** +1.76%。
- 直接估法（launch 计数）：27B 有 **48 个 GDN 层**（`src/targets/qwen3_6_27b/impl/config.h:67`
  `static_assert(TextConfig::gdn_layers() == 48)`），每层每步省 **1 次** launch ⇒ **48 launch/步**。
  Windows/WDDM 下 3~5 µs/launch ⇒ 0.14~0.24 ms/步。186 tok/s、接受长度约 4~6 tok/步 ⇒ 步长约
  22~28 ms ⇒ **0.5%~1.0%，中值 ≈0.7~0.8%**。与队列里给的「折算后约 +0.72~0.79%」一致。
- 生效覆盖：`kMaximumMtpDraftTokens = 5`（`config.h:89`）⇒ verify 的 `T ≤ 6`，全部落在
  `[2,8]` 内 ⇒ **MTP 每一步都吃满**，不存在「只在长上下文才命中」的折扣。
- 额外未计入的小账：省掉 reduce kernel（T=6 时只有 2 个 block、`kernels.cu:399-402`，纯
  launch 延迟）与一轮 workspace 写读 `10*T*48*4 B ≈ 11.5 KB/层`。

**预期：+0.5%~0.9%，保守取 +0.6%。**

### 6.4 顺带发现（不在本项范围，别顺手做）

tp2 路径的 norm 仍是独立 launch（`variant.cpp:746-749` 注释确认）。把 norm 融进 shard 的 controls
（把 `bf16_gdn_norm_gating_27_simt` 的 `H=48` 泛化到 `H=24` 并做 split-10 切片）能再省 48 launch/步，
但 `gdn_norm_gating_27_simt` 的累加结构是「每线程跨 `Threads*8` 步长串行 + 跨 warp 归约」
（`bf16_gdn_norm_gating_proj_27.cu:24-57`），与 split-10 的 `32 lane × 2 趟 × 8 元素`完全不同
⇒ 一碰就是 **B 类**。本项不要碰它。

## 七、实施清单（供后续执行者，本次未落地）

1. `kernels.cu`：`:195` 之后插入 `shard_exact` 核。**建议直接写死 `kShardN`（24），不模板化**，
   把快照的 `template <int Heads>` / `Heads` 全替换成 `kShardN`。
2. `kernels.cu:470`（`dispatch_shard` 开头）插入
   `if (x.ne[1] >= 2 && x.ne[1] <= kSmallTMax) { … <<<dim3(kShardN, x.ne[1]), 320, 0, stream>>> … return; }`。
   外层 `#ifdef NINFER_VOLTA_BUILD` 可保留（`CMakeLists.txt:25` 已定义）。
3. 断言保住第四节三个前提：`kSmallTKSlice == 512`、`vec < 64 && vec += 32`、`for s in 0..9` 线性上升序。
4. 验收按 **A 类**：513 IDs + 全套 MTP 计数**严格相等**（不是 ≤2%）；若出现任何一位不同，
   说明第 3 条被破坏了，回退而不是改判据。
5. 基准：只跑 `packet-allreduce` 同口径（当前 baseline）对比；注意 `_scratch/port-log/incident-20261025-freeze.md`
   的冻结事故，重测前先确认机器空闲。
