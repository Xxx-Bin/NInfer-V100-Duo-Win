# 第 7 项 · projection epilogue 发布 peer packet · 落点勘察
状态：完成（只读勘察，未改任何源码）

基线说明：fork 侧的"现状"指 `port/packet-allreduce`（提交 `df1f2958`），其工作树在
`.agents/worktrees/port-packet-allreduce`（当前 HEAD 与该分支同点）。勘察期间该工作树被其他
进程改写/回退过一次（见第六节"工作树churn"），以下所有 fork 行号均按**当前磁盘内容**核对。
快照指 `_scratch/ref-v100x2/ninfer-V100X2-master`。

## 一、判决

**有 TP2 落点，不建议出队。** 但要挂在第 6 项（融合 RMSNorm）之后做，不能单独成一个可测量的项。

三条依据：

1. 快照里确实有 **TP2 版本**的"epilogue 直接发布"：
   `src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu:44-73` 的 `Nvfp4PacketOutput`
   是**单对端**（`peer`、`epochs`、`epoch_pairs`）策略，启动器
   `nvfp4_linear_add_qpn_publish()` 在 `:157-177`。这条不是 TP4 专用，
   "TP4-only" 的判断只适用于 `Nvfp4Packet4Output`（`:74-113`）。
2. 但它的**消费端只有一个**：`allreduce_published_sum_rmsnorm_to`
   （`src/ops/common/allreduce.cu:950-954`）。快照里没有"只发布不 norm"的消费者
   （`include/ninfer/ops/allreduce.h` 全文只有 `:113` 一处 `published` 字样，
   `src/ops/launcher/allreduce.h:35-40` 只声明两个 `published_*_rmsnorm`）。→ 本项与第 6 项是**同一条路径上的两半**。
3. fork 侧成本比快照低很多：epilogue 要写的那块邮箱（1024 B/epoch 域、80 域、行宽 5120=10 域）
   与 fork 现有 packet collective 的邮箱**逐字节同构**，且 fork 的两个 row-parallel 站点
   （5120×3072 / 5120×8704）正好落在 QPN 路径上。所以"发布"这一半是 ~110 行新代码，
   不涉及新 kernel 几何。

预期收益的定位：上游阶梯 +1.55% 是**顺序测量、非独立归因**（`performance.md:550-565`），
且发生在两个 norm 站点都已融合之后；折算 +0.64~0.70% 是整条链的值。本项单独贡献的只是
"省掉 collective 里那一次本地 50 KiB 读"，是本队列里最薄的一项。**排在第 6 项之后**。

## 二、快照实现（文件:行号 + 片段）

核心：QPN GEMM 的输出策略在 `store()` 里**除了写本地输出，还把一个 BF16 pair 打包成 u32
直接 `st.volatile.global` 写到对端的当前 epoch 邮箱**。

TP2 策略，`src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu:44-73`：

```cpp
template <bool AddResidual>
struct Nvfp4PacketOutput {
    const __nv_bfloat16* residual;
    __nv_bfloat16* output;
    unsigned* peer;
    const unsigned* epochs;
    int epoch_pairs;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        const int index = token * 5120 + row;
        auto represented = __float2bfloat16_rn(value);
        if constexpr (AddResidual) {
            represented = __float2bfloat16_rn(
                __bfloat162float(represented) + __bfloat162float(residual[index]));
        }
        output[index] = represented;
        unsigned bits = *reinterpret_cast<const unsigned short*>(&represented);
        if (bits == 0xffffU) { bits = 0x7fc0U; }
        const unsigned high = __shfl_down_sync(0xffffffffU, bits, 1);
        bits |= high << 16;
        // Every QPN output warp owns one contiguous 32-column row, so pairs
        // share the same live warp, packet domain, and aligned 32-bit address.
        if ((row & 1) == 0) {
            const int pair = index / 2;
            auto* address = peer + (epochs[pair / 256] & 1U) * epoch_pairs + pair;
            asm volatile("st.volatile.global.u32 [%0],%1;"
                         : : "l"(address), "r"(bits) : "memory");
        }
    }
};
```

启动器 `:157-177`（TP2，单对端，`AddResidual` 由 `residual.data` 是否为空决定）：

```cpp
void nvfp4_linear_add_qpn_publish(const Tensor& x, const Weight& weight,
    const Tensor& residual, const Tensor& output, WorkspaceArena& workspace,
    void* local_storage, void* peer_storage, std::size_t epoch_bytes, cudaStream_t stream) {
    auto launch = [&]<bool AddResidual>() {
        const Nvfp4PacketOutput<AddResidual> epilogue{
            static_cast<const __nv_bfloat16*>(residual.data),
            static_cast<__nv_bfloat16*>(output.data), static_cast<unsigned*>(peer_storage),
            reinterpret_cast<const unsigned*>(static_cast<const std::uint8_t*>(local_storage) +
                                                2 * epoch_bytes),
            static_cast<int>(epoch_bytes / sizeof(__nv_bfloat162))};
        ... launch_nvfp4_volta_qpn_with_precast_output / _with_output(x, weight, epilogue, ...);
    };
    if (residual.data) { launch.template operator()<true>(); }
    else { launch.template operator()<false>(); }
}
```

TP4 版本是另一个结构（`Nvfp4Packet4Output`，`:74-113`，`st.volatile.global.v4.b32` 一次发布
8 个 BF16，`offset = epoch * 4 * channel_pairs + Rank * channel_pairs + pair`），
启动器 `nvfp4_linear_add_qpn_publish4()` 在 `:127-155`。**两者不共用代码**。

策略头注释把意图写死了，`src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.h:48-52`：

```
// Same represented projection/add boundaries, additionally publishing each pair
// into the peer's current packet epoch before the norm consumer is issued.
void nvfp4_linear_add_qpn_publish(...);
```

## 三、与 packet collective 的关系

**就是 `allreduce_published_sum_rmsnorm_to` 那条路**，发布端换成 GEMM epilogue。

调用链（快照）：

| 层 | 位置 | 关键点 |
|---|---|---|
| wrapper | `src/ops/wrapper/linear_add.cpp:507-523` | `publish` 分支：先 `nvfp4_linear_add_qpn_publish`，再 `allreduce_published_sum_rmsnorm_to` |
| 公共入口 | `src/ops/common/allreduce.cu:950-954` | `sum_rmsnorm_to(..., published=true)` |
| 几何闸门 | `src/ops/common/allreduce.cu:923-933` | `packet_storage(0) && ne[0]==5120 && ne[1]<=8 && ne[2/3]==1 && 16B 对齐` |
| 启动器 | `src/ops/launcher/allreduce.cu:126-144` | `published` 只是模板实参分派 |
| 内核 | `src/ops/kernel/allreduce.cuh:330-350` | `if constexpr (!Published)` 才做 publish |

"发布"只关掉内核里的推送段，消费端仍然轮询自己邮箱（`allreduce.cuh:343-351`、`:356-371`）。
所以两边的**邮箱协议完全一致**：

- epoch 域粒度 1024 B（消费端 `epochs[i / 256]`，i 是 `__nv_bfloat162` 索引 → 256×4 B；
  发布端 `epochs[pair / 256]`，pair 是 u32 索引 → 256×4 B，**同一个域划分**）
- 一行 5120 BF16 = 2560 pair = 10 个域（`allreduce.cuh:328-329` 注释 "Each complete D5120 row
  owns ten domains, so fused/unfused and different-width calls may interleave"）
- 两个 epoch 交替，`epoch_bytes = 81920`，epoch 计数数组紧跟在 `2 × 81920` 之后
  （`launcher/allreduce.cu:130`；快照 `include/ninfer/ops/allreduce.h:194-195` 给出
  `kPacketMaxBytes = 81920`、`kPacketBlocks = 81920/(16*64) = 80`）

**与 fork 现有 packet collective 的兼容性：逐字节相同。** fork
`src/ops/common/allreduce.cu:347-353`（`kPacketMaxBytes=81920`、`kPacketBlocks=80`、
`kPacketEpochPacks=5120`、`kPacketBytes = 2*81920 + 80*4`）与 `:637-648` 的分配
（`packets[rank] = base`，`packet_epochs[rank] = base + 2*kPacketMaxBytes`，
`memset(packet_base, 0xff, 2*81920)`）和快照的布局一致；fork 内核的域粒度也是
`epochs[blockIdx.x]`、每 block 64 pack × 16 B = 1024 B（`:375-414`）。
→ **不需要改邮箱分配、不需要改 epoch 协议**。

## 四、几何条件（含是否 TP4-only）

快照 TP2 发布的完整闸门，`src/ops/wrapper/linear_add.cpp:480-506`：

```cpp
bool out_of_place_qpn = ec.device_count == 2 && policy == LinearPolicy::A16Only &&
                        x[0].ne[1] <= 8;
for (int rank = 0; rank < ec.device_count; ++rank) {
    out_of_place_qpn &= ec.dev[rank]->sm() == 70 && w[rank].qtype == QType::NVFP4 &&
                        w[rank].layout == QuantLayout::VoltaQpnPrepacked &&
                        w[rank].n == 5120 && (w[rank].k == 3072 || w[rank].k == 8704);
}
...
bool publish = norm != nullptr && norm->weight.size() == 2 &&
               norm->output.size() == 2 && events.packet_storage(0) != nullptr;
```

逐条：

| 条件 | 值 | 说明 |
|---|---|---|
| 设备数 | **2** | 这条分支写死 TP2；4 卡是上面另一个 `published_tp4` 块（`:443-479`） |
| policy | `A16Only` | 走 A16/QPN，不走 W4A4 |
| token 数 | `1..8` | 单 chunk 上限；`:68` 的 `epochs[pair/256]` 索引 8 token 正好 80 域 = `kPacketBlocks` |
| n | `5120` | `:53` `token * 5120 + row` 行宽硬编码，非 5120 一律不可发布 |
| k | `3072` 或 `8704` | 即 row-parallel 的两个 shard |
| 权重 | `NVFP4` + `VoltaQpnPrepacked` | 只有 QPN 内核有这个 warp-uniform 32 列行 |
| SM | `70` | |
| 对齐 | staging/residual/norm.weight/norm.output 均 16 B；v4 版本要 8 列对齐 |
| 消费者 | `ReplicatedRmsNorm` 双卡 + `packet_storage(0)` | **必须有 norm** |

**结论：不是 TP4-only。** 快照存在活的 TP2 变体，且被真实调用：
`src/targets/qwen3_6/impl/runtime/text_context_impl.h:1797-1800` 在 TP2 下把 `norm`
传进 `attention_output_projection`，一路到 `variant.cpp:787-789` → `linear_add.cpp:436`。

fork 侧的对应几何**全部满足**：

| 条件 | fork 现状 | 位置 |
|---|---|---|
| 两个 row-parallel shard 存在 | `Nvfp4GemvGeometry<5120,3072>` / `<5120,8704>` | `src/ops/linear/nvfp4/nvfp4_config.h:146-147` |
| 走 QPN 内核 | `k==8704` 被 `tp2_down` 命中；`nvfp4_volta_qpn_supported` n%128、k%chunk、t≤32 | `src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh:459`、`nvfp4_volta_qpn_gemm.cu:17-29` |
| decode 走 QPN 而非 GEMV | `dispatch → launch_nvfp4_volta_qpn` | `src/ops/linear/nvfp4/nvfp4_dispatch.cpp:95-97` |
| kColsPerCta = 32（warp 独占一行） | 常量 32，行=tokens、列=输出列，warp 内 32 lane 连续列 | `nvfp4_volta_qpn_gemm.cuh:63`、`:396-406` |
| T=5（MTP4 verify） | 实测窗口就是 T=5、50 KiB | `_scratch/port-log/packet-allreduce.md`（"5120(BF16) × T=5 ≈ 50 KiB"） |
| 邮箱 | 已有 `packets[2]` / `packet_epochs[2]`，公开访问器 `events.direct()->packets[rank]` | `include/ninfer/ops/allreduce.h:151-159` |

一个**必须注意的差异**：fork 的 rank0 走的是
`Nvfp4LinearAddRoute::LinearThenAdd`（`src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp:41-49`），
即"QPN 写投影到 workspace → 再单独 `residual_add`"。而快照的发布是**在 epilogue 里做加法**
（`AddResidual`）。所以本项在 fork 上顺带要把 rank0 从 `LinearThenAdd` 换成
"QPN epilogue 带 AddResidual"，否则发布出去的还是没加 residual 的投影。

## 五、fork 侧改动清单（文件 / 函数 / 预估行数）

前提：**第 6 项先落地**（`linear_add_row_parallel` 拿到 `ReplicatedRmsNorm*`，
并有 TP2 版 `allreduce_published_sum_rmsnorm`）。当前 fork 交付物里这两样都**不存在**
（`include/ninfer/ops/linear_add.h:139-146` 两个重载都没有 norm 形参；
`src/ops/common/allreduce.cu` 里没有 `ReplicatedRmsNorm`、没有 `allreduce_sum_to`、
没有 rmsnorm 融合）。

| # | 文件 | 函数 / 位置 | 改动 | 预估行数 |
|---|---|---|---:|---:|
| 1 | `src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu` | 新增 `Nvfp4PacketOutput<AddResidual>` + `nvfp4_linear_add_qpn_publish()` | 照快照 `:44-73`、`:157-177` 抄，改成 fork 的 `launch_nvfp4_volta_qpn_with_output`（fork 没有 `precast` 变体，见下） | +55 |
| 2 | `src/ops/linear_add/nvfp4/nvfp4_linear_add_plan.h` | 声明 `nvfp4_linear_add_qpn_publish` | `#ifdef NINFER_VOLTA_BUILD` 内 | +4 |
| 3 | `src/ops/common/allreduce.cu` | TP2 融合内核加 `bool Published` 模板实参 | `if constexpr (!Published)` 包住 publish 段（快照 `kernel/allreduce.cuh:343`）；再导出一个 `detail::allreduce_published_sum_rmsnorm()` | +20 |
| 4 | `include/ninfer/ops/allreduce_norm.h` | 声明 published 出口 | 与第 6 项同一个头 | +4 |
| 5 | `src/ops/wrapper/linear_add.cpp` | `linear_add_row_parallel`（当前 `:400-426`） | 在现有 `target/scratch` 组装之前插入 `publish_qpn` 闸门 + `for_each_rank` 发布 + 融合消费 + `return` | +45 |
| 6 | — | 调用点 | **不需要**（第 6 项已经把 norm 传进来了） | 0 |
| | | | **合计** | **≈ +128** |

真正属于本项（不含第 6 项）的约 **+105 行**，且大多是照抄。

两点必须按 fork 现状修正，不能照抄快照：

1. fork 的 QPN 启动器只有 `launch_nvfp4_volta_qpn_with_output`
   （`src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh:485-492`），**没有**
   `nvfp4_volta_qpn_precast_supported` / `..._with_precast_output`。快照那两个是 FP16-image
   变体，本项用不到，直接用 `with_output` 即可。
2. 行宽 5120 与 `epoch_pairs` 的偏移在 fork 里仍要手算：epoch 计数数组 =
   `packets[rank] + 2*kPacketMaxBytes`（`src/ops/common/allreduce.cu:637-640`）。
   这个 `kPacketMaxBytes` 目前在 `.cu` 的匿名 namespace 里（`:348`），wrapper 看不到，
   需要提一个常量或访问器（≤5 行）。

## 六、风险

1. **依赖倒挂（最大风险）**：快照的发布分支只在 `publish && norm` 时成立
   （`linear_add.cpp:497-506`）。fork 若没有 norm 融合，本项只能自建一个
   "published 版 in-place `allreduce_sum`" 消费者（约 +10 行），收益仅等于"省一次本地读"，
   很可能淹没在 sd≈0.09 tok/s 的噪声里（`_scratch/port-log/packet-allreduce.md` 的复测
   sd 0.0923 / 135.6 tok/s → 0.5% ≈ 0.68 tok/s）。**没有第 6 项就不该单独测量本项。**
2. **数值边界**：发布值必须是"已加过 residual 的 BF16"。fork 现在 rank0 是
   `linear → residual_add`（两次舍入：`bf16(proj)` 再 `bf16(bf16(proj)+bf16(res))`），
   换成 epilogue 内联加法后要保证同样的两次舍入（快照 `:54-58` 正是这个顺序），
   否则 oracle 逐位一致会被打破。
3. **索引无边界检查**：`epochs[pair / 256]`（快照 `:68`）在 T>8 或 row>5120 时会越界读 epoch、
   越界写邮箱。fork 侧闸门必须显式写死 `n==5120 && 1<=T<=8`，且不能只靠
   `packet_admits` 的 `count % 8` 判断（`src/ops/common/allreduce.cu:416-423`）。
   快照在退化路径上是用 `require(!published, ...)` 抛异常（`allreduce.cu:934`），
   fork 应沿用 fail-fast，不要静默退化。
4. **跨卡可见性**：发布写对端显存只靠 `st.volatile.global` + 对端轮询，
   没有 device fence。和现有 packet collective 同风险，但这次写发生在**GEMM 内核**里，
   而不是专门的 collective 内核里 —— 若该 GEMM 之后有别的东西复用同一个 mailbox
   域（例如同层另一个 collective），epoch 会错位。**一对一绑定：一次发布只能被一次消费。**
5. **自旋上限 15e9 cycle 会 `__trap()`**（`src/ops/common/allreduce.cu:355`、`:403`）。
   发布端是 GEMM，若某层条件不满足走了非发布路径而消费端以为已发布，就是硬 trap 而不是
   报错返回 —— 闸门判断必须在 wrapper 里一次算好、两端共用同一个布尔值。
6. **工作树 churn**：勘察期间 `.agents/worktrees/port-packet-allreduce` 被其他进程改写
   （曾短暂出现一个带 `kNormWidth` / `allreduce_packet_rmsnorm_kernel` 的融合实现和
   `include/ninfer/ops/allreduce_norm.h`，随后被回退；同一 worktree 当前还有
   `src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu` 处于 modified）。
   → 本项开工前必须先把基线固定到一个 tag，否则"第 6 项已落地"的前提可能随时不成立。
