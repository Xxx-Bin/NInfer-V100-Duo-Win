# 第 6 项 · 融合 RMSNorm 进 collective（+0.69%）· 落点勘察

状态：**静态勘察完成，未动代码**。基线 = `result/packet-allreduce`（待复测确认后打 tag）。
撰写时间 2026-10-25 06:10。

## 一、结论先说

**能做，但不是"20 行移植"，是"加一条 API + 改 signature"。**
**而且必须一次做两个点，只做一个过不了门槛。**

重校准后（详见 `candidates.md`）：上游阶梯把 RMSNorm 融合拆成两步，
`+0.69%` 和 `+1.23%`，机制相同、调用点不同。按本机折算系数 ×0.41~0.45：

| 做法 | 上游值 | 折算后 | 门槛 +0.5% |
|---|---:|---:|---|
| 只做 o_proj 那点 | +0.69% | +0.28~0.31% | ❌ |
| 只做 down_proj 那点 | +1.23% | +0.50~0.55% | ⚠️ 压线 |
| **两点都做** | **+1.92%** | **+0.79~0.86%** | ✅ 唯一稳过 |

所以本项的正确形态是 **"RMSNorm 融合 ×2 站点"**，不是一个点。

## 二、快照里有两个入口，只有一个对 TP2 有效

| 入口 | 位置（快照） | 生效条件 | 对 TP2 |
|---|---|---|---|
| `allreduce_sum_rmsnorm` | `allreduce.cu:768 sum_rmsnorm()` | `published && device_count == 4 && ne[0] == 5120 && ne[1] ≤ 8` | ❌ **TP4 专用** |
| `allreduce_sum_rmsnorm_to` | `allreduce.cu:896 sum_rmsnorm_to()` | `packet_storage(0) && ne[0] == 5120 && ne[1] ≤ 8 && ne[2/3] == 1` | ✅ **就是按 2 卡写的** |

关键点：`sum_rmsnorm()` 的快路**写死了 4 卡**（`allreduce_packet_sum4_rmsnorm_launch`、`std::array<void*,4>`）。
若照它移植，TP2 上条件恒假，会退到 `allreduce_sum` + 独立 `rmsnorm` —— **收益精确为 0**，
就是把 fork 现有行为原样重写一遍。

`sum_rmsnorm_to()` 才是 TP2 版本：`norm.weight.size() == 2`、`rank < 2`、
`packet_storage(rank)` / `packet_storage(1-rank)`，条件 `D == 5120 && T ≤ 8`，
**正好命中 decode**（hidden 5120，MTP4 时 T=5）。

→ 第 6 项应当移植的是 **`allreduce_sum_rmsnorm_to` 这条路**，不是文档字面写的那个。

## 三、fork 缺什么

| 快照有 | fork 现状 |
|---|---|
| `struct ReplicatedRmsNorm { weight, output, eps, unit_offset }` | **无**（`include/ninfer/ops/allreduce.h`） |
| `validate_sum_to()` / `allreduce_sum_to()` | **无**（fork 只有 in-place 的 `allreduce_sum`） |
| `detail::allreduce_packet_rmsnorm_launch()` | 无（快照 `src/ops/launcher/allreduce.cu:126`） |
| `allreduce_sum_rmsnorm_to()` | 无 |
| `PeerEvents::packet_storage(i)` + `kPacketMaxBytes` | packet 存储已有（`packets[2]` / `packet_epochs[2]`，见 `df1f2958`），但**在匿名 namespace 里**，需提升为可访问的常量/访问器 |

## 四、真正的难点不在 allreduce，在调用方

**归约的收敛点只有一个**：`ops::linear_add_row_parallel`（`include/ninfer/ops/linear_add.h:139/146`
两个重载 → `src/ops/wrapper/linear_add.cpp` 末尾 `allreduce_sum(target, staging, ec, events)`）。

`variant.h:133-139` 的注释把话说全了：

> the two row-parallel leaves (`attention_output_projection`, `gdn_output_projection`,
> and `post_mixer`'s trailing `down`) carry the block's single all-reduce inside
> `ops::linear_add_row_parallel`, which also folds the residual in exactly once, on rank 0,
> before the reduce.

也就是**三个 row-parallel 叶子共用一个归约入口**：

| 叶子 | 归约后消费方 | 融合点 |
|---|---|---|
| `attention_output_projection`（o_proj） | `mlp_tail_tp2` 入口的 `post_attn_norm` | **点 A** |
| `gdn_output_projection` | 同上（GDN 层） | 点 A（同机制） |
| `post_mixer` 尾部 `down` | 下一层 `attn_mix` 入口的 `input_norm` | **点 B** |

所以只需**改一处签名**就能覆盖全部站点：

```cpp
// include/ninfer/ops/linear_add.h:139 / :146 两个重载
struct NormEpilogue {                    // 或者直接用 allreduce.h 的 ReplicatedRmsNorm
    std::array<Tensor, 2> weight;
    std::array<Tensor, 2> output;
    float eps;
    bool unit_offset;
};
linear_add_row_parallel(..., const PeerEvents& events, const NormEpilogue* norm = nullptr);
```

`norm == nullptr` → 走原 `allreduce_sum`，行为逐位不变（回归安全）。
`norm != nullptr` 且几何合格（`D==5120 && T<=8 && 16B 对齐`）→ 走 `allreduce_sum_rmsnorm_to`。

再往上，`Variant::attention_output_projection` / `gdn_output_projection` / `post_mixer`
（`src/targets/qwen3_6_27b/impl/variant.h`）的 TP2 形态要各加一个透传参数，
`run_layers_tp2`（`text_context_impl.h`）负责把「下游那层的 norm 权重」递进来。

**点 B 的跨层依赖是本项唯一的结构性麻烦**：`mlp_tail_tp2` 结束时不知道下一层的 `input_norm`，
必须由 `run_layers_tp2` 在循环里按 `layer+1` 取权重传下去；最后一层要传 `nullptr`
（下一层是 final norm，走别的路径）。

## 五、改动清单（预估）

| 文件 | 内容 | 行数 |
|---|---|---:|
| `include/ninfer/ops/allreduce.h` | `ReplicatedRmsNorm` + `packet_storage()` 访问器 + `kPacketMaxBytes` | +20 |
| `src/ops/common/allreduce.cu` | `validate_sum_to` + `allreduce_sum_to` + `sum_rmsnorm_to`（TP2 版）+ 常量外提 | +110 |
| `src/ops/kernel/allreduce_rmsnorm.cuh`（新增） | packet + norm 融合内核（照抄快照 `launcher/allreduce.cu:126` 的实现） | +90 |
| `include/ninfer/ops/linear_add.h` + `src/ops/wrapper/linear_add.cpp` | `NormEpilogue*` 可选参数，`nullptr` 时行为不变 | +25 |
| `src/targets/qwen3_6_27b/impl/variant.h` + 实现 | 三个叶子透传参数 | +18 |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | 点 A / 点 B 传 norm，删掉两处独立 `rmsnorm` | +20 / −8 |
| **合计** | | **≈ +303** |

比 packet（+136）大一倍。**编译一整轮 + 测量一轮 ≈ 1.5 小时**。

## 六、内核细节（`kernel/allreduce.cuh:331-386`，56 行）

```cuda
__global__ void allreduce_packet_rmsnorm_5120_kernel(
    const __nv_bfloat162* source, __nv_bfloat162* sum_output,
    const __nv_bfloat162* weight, __nv_bfloat162* norm_output,
    unsigned* local, unsigned* peer, unsigned* epochs, int epoch_pairs, float eps)
```

单核一趟做完 5 件事：**发布 peer → 自旋等 peer → FP32 求和 → 行内 RMS → 归一化**。
`grid = T`、`block = 512`、每线程 5 个 `bfloat162`（2560 pairs = 5120 个 BF16）。

三个必须照抄的设计点：

1. **`sum_output` 和 `norm_output` 都写**。原始和仍然落地 → 归约结果逐位不变，
   oracle 风险只来自 norm 那一步，不会因为"省掉中间结果"而扩散。
2. **空槽标记是 per-`u32` 的 `0xffff`**，粒度是 `bfloat162`（4 字节）；而我在 `df1f2958` 实现的是
   per-`Bf16x8Pack`（16 字节）。**标记粒度不同，但存储规模一致**（更正先前误算）：
   - epoch 下标是 `epochs[i / 256]`，`i` 是 **pair**（4 字节）下标，最大 `8×2560-1 = 20479`
     → `20479/256 = 79`，即 **80 个计数器**
   - 我现有实现 `kPacketBlocks = 81920/(16×64) = 80` 个计数器 —— **数量相同**
   - 缓冲区同为 `2 × 80 KiB`，`epoch_pairs = 81920/4 = 20480`
   → **可以共用同一块 packet 存储与同一组 epoch 计数器**，只需换标记语义。
     而且两者的 **epoch 索引映射完全一致**：现有内核每 block 覆盖
     `kPacketThreads(64) × 8 × 2B = 1 KiB`，融合内核每 256 pairs × 4B = **同样 1 KiB**
     → `epochs[blockIdx.x]` 与 `epochs[i/256]` 指向同一个计数器，状态可以互相继承。
     但两种协议的"就绪"判定不同，同一轮里不能混用，必须按几何二选一（见风险 4）。
3. **epoch 翻在位**：`if (tid < 10) epochs[row*10 + tid] ^= 1U;`（每行 10 个块），
   行与行之间互不干扰 → 天然支持 `T` 行并发。

依赖 `ops::block_reduce_sum<Block>` 与 `rmsnorm_epilogue<RmsEpilogue>`（后者需一并搬）。

## 七、风险与判据

1. **数值**：融合后归一化在 FP32 累加后、单次 BF16 舍入前做，理论上应逐位等于
   "先 BF16 归约再 norm" —— 但**快照注释明说**这是"preserves the BF16 rounding"的精细活。
   如果顺序有偏差，oracle 会从 A 类掉到 B 类（可解释差异）。
   **这是本项最大风险**，也是它值得先做静态对拍的原因。
2. **门槛**：相对 packet 基线（复测 135.594），`DECODE_MIN = 136.272`（+0.5%）。两点全做折算后 +0.79~0.86%，
   余量约 0.3 个百分点 —— 比只做一点时（0.19）宽松，但仍不算厚。
   → 若实测落在 +0.3%~+0.5%，按现判据仍是不保留，不要"测到就算成果"。
3. **折算系数的适用性**：×0.41~0.45 是从 TileN 与 packet 两个样本推的。
   若本项实测显著偏离（比如只有 +0.2%），说明这两个样本不代表"消除 launch 类"改动，
   **应停下来重估系数**，而不是继续烧后面的项。
4. **两种 packet 协议共存**：现有 packet（per-pack 标记，≤80 KiB 任意形状）
   与新融合内核（per-u32 标记，`D==5120 && T<=8`）共用同一块存储。
   必须满足：同一时刻只有一个内核在跑（同一条流上顺序发射即可），
   且切换前 epoch 状态干净。验收方式就是第二轮"`norm=nullptr` 零变化"那一跑 ——
   若那次出现任何漂移，说明协议互相污染。

## 八、建议的执行顺序（等 packet 复测通过后）

1. 建 `port/rmsnorm-fusion`，从 `result/packet-allreduce` 起。
2. **第一轮只搬 API + 内核，`norm` 参数全程传 `nullptr`** → 编译 + 全量测量。
   期望 oracle 逐位一致、性能**零变化**（±0.2%）。
   这一步是回归护栏：证明新加的路径没有被意外走到。
3. **第二轮才接线**：`linear_add_row_parallel` 加参数 → 三个叶子透传 → 点 A（o_proj，简单）。
   → 编译 + 测量。若点 A 单独就有 +0.3~0.5%，说明机制有效，再做点 B。
4. **第三轮加点 B**（跨层 handoff，`run_layers_tp2` 传下一层 `input_norm`）→ 编译 + 测量。
5. 若 oracle 掉到 B 类，停止并记录，不硬推。

分三轮而不是一把梭，因为点 B 的跨层依赖是本项唯一容易写错的地方，
混在一起失败时无法区分是内核问题还是接线问题。

## 九、顺带发现的可用资产

- 快照 `launcher/allreduce.cu:92` 的 `allreduce_packet_sum4_rmsnorm_launch` 是 TP4 专用，**不要搬**。
- 快照 `kPacketMaxBytes` 是 `PeerEvents` 的公有常量；fork 侧我实现 packet 时放在了匿名 namespace，
  第 6 项需要外提，改动很小但**会让 packet 那 136 行 diff 变**——所以第 6 项必须在 packet 之后单独成分支。
