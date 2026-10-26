# 队列 v2：后续 6 项的落点勘察与测试规划（2026-10-25）

> 本文只做**规划**，不改任何源码。勘察结论全部基于静态 diff（fork vs `_scratch/ref-v100x2/ninfer-V100X2-master`），
> 目的是把"6 项 × 30~45 分钟编译测量"的盲目投入，压缩成"先杀掉不该做的，再跑该跑的"。

## 一、先说三个会改变结论的前提

### 前提 A：文档里的 +x% 都是**阶梯逐步值**，不是独立归因

`performance.md:550-565` 的 retained ladder 是**顺序累加测量**：每一步的百分比是"相对上一步"。
所以 `+0.69%`（RMSNorm）只有在 packet 已经存在时才成立；单独把这个改动搬到当前 fork 上，
可能一点收益都没有，甚至变慢。**任何一项的收益都必须在本机重新实测，不能引用文档数字当预期。**

### 前提 B：TP2 上 argmax 合并本身在噪声内（分析报告修正 3）

`performance-v100.md:140-148`（TP2、matched）：完整 logits 81.797 → exact shard-winner 82.396 →
+parallel shard argmax **82.434（噪声内）**。收益来自 **small-message sum** 那一步（+3.37%），不是 argmax。

⇒ 队列里的"fused shard-winner collective (+1.37%)"在 TP2 上的**真实预期接近 0**
（那个 +1.37% 是 packet 已在位之后的一级）。必须按"噪声内"的预期去做，别指望它过门槛。

### 前提 C：oracle 判据会直接杀掉"改求和顺序"的项

本战役的硬否决是 **513 IDs 的 sha1 必须等于 `06287348665d3648`**。据此把 6 项分两类：

| 类 | 含义 | 能否过 oracle |
|---|---|---|
| **A 类·纯传输/调度** | 只改数据怎么搬、kernel 怎么发，算术与舍入边界完全不变 | 能，且必须逐位不变 |
| **B 类·改数值路径** | 改变求和顺序、拆分量、舍入次数 | **大概率变** → 按现判据直接判无效 |

- A 类：packet collective、RMSNorm 融进 collective（若沿用同一 FP32 累加 + 单次 BF16 舍入）、
  projection epilogue 发布、tiled argmax（精确 value/index，tie 策略不变）。
- **B 类：GDN `shard_exact`（split-10 精确保序）** —— 它改的就是归约/分组顺序。

⇒ **需要你裁决一件事**：B 类项是否允许"oracle 变化但可解释"（例如要求 accepted/drafted/rounds
不变、513 IDs 仅个别位置不同、且与参考实现对齐）？如果不允许，**GDN controls 融合这项要直接移出队列**，
不要浪费一轮 45 分钟。默认我按"不允许"处理，即跳过。

## 二、逐项落点勘察（已完成 5/6）

| # | 项 | 文档 delta | 快照锚点 | fork 现状 | delta 规模 | 依赖 | 结论 |
|---|---|---:|---|---|---|---|---|
| ① | 融合 RMSNorm 进 collective | +0.69% | `allreduce.cu:768-833`（`sum_rmsnorm` / `ReplicatedRmsNorm` / `allreduce_packet_sum4_rmsnorm_launch`）、`allreduce.h:113,253,272` | **无**：`allreduce.h` 只有 `allreduce_sum`；全仓仅 3 个调用点（`linear.cpp:316`、`linear_add.cpp:425,470`） | API +3 函数、kernel 加 norm 尾巴、3 处调用点重接线，约 150–200 行 | **packet** | 可做，但必须排在 packet 之后 |
| ② | projection epilogue 直接发布 peer packet | +1.55% | `launcher/allreduce.cu:65-98` 的 `Published` 模板位；`allreduce_packet_sum4*` 直接写下游输入 | **无** | 要改 CUTLASS/MMA 的 epilogue 写出口 + 协议，**数百行** | **packet** | 最贵的一项，收益中等 → 排在最后 |
| ③ | fused shard-winner collective | +1.37% | `kernel/allreduce.cuh:35-62`（winner + id_map remap 并入 collective） | **无**（fork 无 argmax 版 collective） | 在 packet 内核里加 argmax 尾巴，约 80–120 行 | packet + argmax | 预期接近 0（前提 B）→ **降级为可选** |
| ④ | five-query packed KV vector16 | +1.99% | 仅存在于 `gqa_attention_decode_i8_qpn_verify5_volta.cuh`（fork 无此文件） | fork 的 T=5 走 `key_split` 核，每线程 8 codes = 8 字节连续（stride 2048 B）→ 只能 `int2` | 改成 `int4` 需动 **KV cache 布局 / 量化写入**，不是 20 行 | — | **维持跳过**（见第三节复判） |
| ⑤ | MTP shortlist tiled argmax / 跨卡 argmax 合并 | +1.83% | `kernel/argmax.cuh:201`（`argmax_with_value_kernel`）、`wrapper/argmax.cpp:71-83`、`common/allreduce.cu:1126`（`allreduce_argmax`，59 行）、`kernel/allreduce.cuh:28-62`、`runtime/text_context_impl.h:2170-2201` | fork 有 `argmax.cuh:130` / `argmax.cu:83`，**缺 with_value、缺 `allreduce_argmax`、且仍在传完整 logits**（`text_context_impl.h:2664` `argmax(whole[0])`） | **约 350–400 行、跨 4 个文件**（不是 110 行） | 无 | **仍是最优选**，但预期只有 +0.7%（见下） |
| ⑥ | GDN controls 融合（split-10 精确保序） | +1.76% | `bf16_gdn_gating_proj_kernels.cu:734` 行的 `bf16_gdn_gating_proj_shard_exact_kernel` | fork 有 `split10` / `split10_shard`，**且已有 `bf16_gdn_norm_gating_proj_27.cu`（每头同时算 control 点积与完整 norm）** → 融合部分已存在 | 只剩 `shard_exact` 一个核，约 80–120 行 | 无 | **B 类**：改序 → oracle 必变 → 默认跳过 |

关键发现：**④⑤⑥ 三项在 fork 里都不是"干净缺失"**。⑤ 是唯一有完整参考链路的增量；
⑥ 的大部分融合 fork 已经有了（`bf16_gdn_norm_gating_proj_27.cu` 就是 controls+norm 融合），
剩下的 `shard_exact` 是数值路径改动；④ 是布局问题不是移植问题。

### ⑤ 的真实机制与预期（重要）

文档的 `+1.83%` 是 **"MTP shortlist tiled argmax **+** GDN activation staging"** 两件事的合计，
不能整块归给 argmax。它的实际内容是一整条链路：

1. 每 rank 只对自己那片词表做 `argmax_with_value` → 产出 `[1,T]` 的 (FP32 value, I32 index)
2. `allreduce_argmax` 只交换 `[2,T]` 的 value/index（不再传 `[1,152k]` 的完整 BF16 logits）
3. 按 `(value 大者优先, index 小者优先)` 合并，可选 id_map remap

而 `performance-v100.md:140-148`（TP2、matched）实测这一步是
完整 logits 81.797 → exact shard-winner **82.396（+0.73%）** → 再上并行 shard argmax **82.434（噪声内）**。

⇒ **本项在本机预期 +0.5%~0.8% decode**，正好压在门槛线上，属于"值得赌一轮、但不保证过"。
好处是它完全是 A 类（tie 策略与精确比较都保留），oracle 可以逐位校验，跑砸了也能干净回滚。

## 三、④ five-query vector16 的复判（你要求重新评估）

之前判"无落点"，重新核对后结论**不变，但理由要说清楚**：

- 快照的 `vector16` 加载出现在 `gqa_attention_decode_i8_qpn_verify5_volta.cuh`，该文件在 fork 中不存在。
- fork 的 T=5（MTP4 验证）路由是 `gqa_attention_decode_i8_key_split_volta.cuh`，其 staging 是
  **每线程 8 个 code、跨 key 的 stride 为 2048 B**，8 个 code 在内存里连续 → 天然只够一个 `int2`（8 B）。
- 想凑满 `int4`（16 B 连续）必须让 16 个 code 连续，也就是改 **KV cache 的 pitch / 量化 kernel 的写出布局**。
  这是布局级改动，会连带影响 prefill 的 KV 写入与 attention 的所有读取路径，风险远高于"~20 行"。

⇒ 建议：**不进本轮队列**。若要做，单独立一项"KV cache 布局换 vector16"，预期要 2–3 轮编译测量，
且只在 decode 生效（门槛 decode ≥130.376）。收益 1.99% 是阶梯值，本机能否复现未知。

## 四、依赖图与分支法则（需要定规则）

```
baseline ──┬── ⑤ tiled argmax            （独立）
           ├── ⑥ GDN shard_exact         （独立，但 B 类）
           └── packet(进行中) ──┬── ① RMSNorm 进 collective
                                ├── ② epilogue 发布 peer packet
                                └── ③ shard-winner collective（还要 argmax）
```

**分支规则冲突**：战役规定"每项从 `baseline/port-study-34412f4` 建分支、绝不合并回基线"。
但 ①②③ 依赖 packet，从 baseline 起分支就必须把 packet 的几百行一起搬进去，否则编不过。

三条可选路线：

| 路线 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **R1（推荐）** | packet 通过后打 `keep/packet-allreduce`；①②③ 从该 tag 起分支 | 分支干净、增量小 | ① 的对照组是"packet 版"而不是原始基线，等于承认 packet 已进基线 |
| R2 | ①②③ 的 worktree 里手工叠加 packet 改动（不走 git 关系） | 不碰 tag | 每项都要重搬一次，易漂移 |
| R3 | 每项都从 baseline 起，包 packet + 目标改动 | 严格合规 | 测出来的是"packet + 目标"的合成收益，无法归因 |

R1 的前提是把 `keep/*` 当作"新基线"，这需要在规则上松一格。**这一点请你确认**，否则 ①②③ 无法开始。

## 五、建议的执行顺序与成本

按"独立优先、便宜优先、B 类垫底"排：

| 序 | 项 | 类 | 预估单轮成本 | 理由 |
|---|---|---|---:|---|
| 1 | **⑤ 跨卡 argmax 合并（tiled argmax + allreduce_argmax）** | A | 50–60 min | 不依赖任何东西；oracle 可逐位验证；预期 +0.7% decode，压线 |
| 2 | packet（进行中，队列第 4 项） | A | 已在跑 | 后面 ①②③ 的门槛 |
| 3 | ① RMSNorm 进 collective | A | 40–50 min | 依赖 packet；+0.69% 刚过 decode +0.5% 门槛，容错小 |
| 4 | ② epilogue 发布 peer packet | A | 60–90 min | 最贵；+1.55% |
| 5 | ③ shard-winner collective | A | 40 min | 预期接近 0，只作"拼图验证" |
| — | ⑥ GDN shard_exact | B | 40 min | 默认跳过，等 oracle 裁决 |
| — | ④ vector16 | — | — | 维持跳过，见第三节 |

单轮 = configure（1 min）+ `-j2` 全量编译（25–40 min）+ 测量（≈5 min）。
总计约 **3–4.5 小时**，且必须串行（一次只能一个 GPU 任务）。硬件若再抖，按 `BUILD-LOCAL.md` 的
三形态判别处理。

## 六、每项的测量与判定（与现协议一致，只差基准）

- 协议参数不动：`-pg 60000,512 --prefill-chunk 2560 --kv-dtype int8 --mtp-draft-tokens 4 --lm-head-draft --tp 2 -r 2 --warmup 1`
- 门槛：prefill ≥ **955.257**、decode ≥ **130.376**（相对原始基线）；对依赖 packet 的 ①②③，
  门槛改为**相对 packet 版实测值** +1.0% / +0.5%，并在报告里同时列出两档数字，避免归因混淆
- oracle：A 类必须逐位一致（`06287348665d3648`）；B 类见第一节裁决
- 一次测量不算数：≥2 次 cold request 都过门槛

## 七、待确认（已裁决，见第八节）

1. ~~B 类（GDN shard_exact）是否放行~~ → **放行，改可解释判据**
2. ~~分支路线~~ → **R1：keep tag 当新基线**
3. ~~vector16~~ → **只做探针实验**

## 八、裁决结果与判据修订（2026-10-25，执行依据）

### 8.1 B 类：oracle 改为"工作负载等价"判据

原判据（513 IDs sha1 严格相等）对改求和顺序的项不可用。改为**三段式**，全部通过才算有效：

| 级别 | 判据 | 不通过的后果 |
|---|---|---|
| 硬否决 1 | **accepted=406 / drafted=421 / rounds=106** 必须严格相等 | 判无效——说明每轮验证的工作量变了，tok/s 不可比 |
| 硬否决 2 | `acceptance_length` 与 `per-position=[105,102,100,99]` 必须相等 | 同上 |
| 可解释性 | 513 个输出 ID 与基线逐位比对，**差异位占比 ≤ 2%**，且每个差异位的 top1–top2 logit margin < 0.05 | 超出则判无效（说明是逻辑变化，不是边界重排） |

同时必须记录：差异位的具体位置、个数、以及该步 logits margin。报告里同时给"基线 sha1"和"本项 sha1"。

> 理由：`decode_output_tok_s = 512 / decode_seconds`，只要**接受/草稿/轮次完全一致**，
> 两边做的 GPU 工作量就一致，tok/s 可比；此时个别 token 因求和顺序改变而翻转，属于数值边界问题，
> 不是性能作弊。反过来，只要 accepted 变了，工作量就变了，任何 tok/s 对比都失去意义。

### 8.2 R1：`keep/*` 作为新基线

- packet 通过 → `git tag keep/packet-allreduce port/packet-allreduce`，**该 tag 立即成为后续 ⑥⑦⑧ 的基线**
- 后续项 `git worktree add ... -b port/<feat> keep/packet-allreduce`
- 这三项的门槛改为**相对 packet 版实测值** +1.0% prefill / +0.5% decode，
  且报告里必须**同时列出**两档数字：`vs packet 版` 与 `vs 原始基线`
- 合并回 `baseline/port-study-34412f4` 的禁令不变；`keep/*` 只是"分支起点"，不是"合并目标"

### 8.3 vector16：探针实验（不承诺收益）

范围严格限定，**不改 KV cache 布局**，只回答一个问题：key_split 核能否在不改布局的前提下凑出 128-bit 加载。

1. 在 `gqa_attention_decode_i8_key_split_volta.cuh` 的 staging 循环里，尝试把相邻 key 的
   8 字节 code 段拼成 16 字节、用 `int4` 一次取回（需要 `__ldg` 探针 + 手工偏移）
2. **先看 NVCC 是否真的生成 `LDG.E.128`**（`cuobjdump -sass` 检查），没生成就直接放弃，不测
3. 只有生成 128-bit 时才跑一轮测量；oracle 必须逐位一致（纯加载宽度变化，属 A 类）
4. 越界风险：跨页的 `int4` 会读越界，必须保留边界分支；探针若无法保证边界安全 → 放弃

判据：decode ≥ 130.376 且 prefill ≥ 955.257（对 packet 版基线则是 +0.5%/+1.0%）。
若 SASS 没产出 128-bit 加载，本项记为"已评估·技术上不成立"，不占测量轮次。
