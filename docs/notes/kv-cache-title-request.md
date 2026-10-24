# 旁路请求导致前缀缓存失效：机制、定性、社区对照与优化方案

> **状态：已归档 · 被决策记录取代（不再据此做判断）**
> 归档：2026-10-24 · 归属：superseded · 角色：note
> 原文适用条件：Windows 11 · 2×V100-SXM2-16GB · sm_70 · CUDA 12.8 · TP2 · `--max-concurrency 1`
> 现行替代：[prefix-reuse-decision.md](prefix-reuse-decision.md)（方案已选定并实现）
> 保留理由：方案枚举与判据推导的原始过程；其中「提并发能缓解」「N≥k+1 通用」已被撤回，见替代文档 §5

> 场景：`--max-concurrency 1`（即 llama.cpp 语义的 `PARALLEL=1`）。
> 发起一轮对话时，客户端同时发出一个「改会话标题」请求，导致该会话下一轮的前缀缓存全部不命中。
>
> 本文所有代码引用均来自当前工作区（`docs/notes/` 为本笔记目录，不参与构建）。
> 目标模型 Qwen3.8-27B 的代码在 `src/targets/qwen3_6_27b/`（架构常量与权重绑定）
> 与 `src/targets/qwen3_6/`（共享 runtime / frontend），由 artifact 的 `model_id` 分派，
> 对应关系见附录 A 首节。
> §6.1 的数字是本机实测（记录见 `kv-cache-title-request-p0.md`），其余性能数字引自仓库既有记录，
> 出处随文标注。

---

## 0. 先纠正一个前提：这不是「驱逐」

「因为被驱逐所以缓存不命中」这个描述在**效果上**成立，在**机制上**不成立。

`--max-concurrency 1` 时，准入层的驱逐函数 `evict_retained_lane` **一次都不会被调用**。
`find_admission_lane` 分两趟扫描，第一趟只要有任何 lane 可原地准入就立即返回，第二趟（唯一会设置
`evict_retained = true` 的地方）根本不执行：

```cpp
// src/runtime/engine/concurrent_executor.h:779-790（第一趟）
for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
    if (slots_[lane] != nullptr) { continue; }
    ensure_lane_plan(request, lane);
    const Plan& plan          = *request->lane_plans[lane];
    const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
    if (instance_.program->can_admit_lane(lane, plan) &&
        (!selected || reuse > selected_reuse)) {
        selected       = LaneChoice{.lane = lane};
        selected_reuse = reuse;
    }
}
if (selected) { return selected; }
```

`:790` 这一行 `return` 就是分界：第二趟（唯一会设置 `evict_retained = true` 的地方，`:840`）不再执行。

C=1 时唯一那条 lane 在上一轮结束时就已空闲（`slots_[0] == nullptr`），且标题请求很短、配额充裕，
第一趟必然成功。**没有任何东西「驱逐」了缓存。**

真正的销毁发生在别处：请求自己的复用计划被判成 `FullReset`，而 `FullReset` 的第一件事就是释放本 lane 的
KV 分配。是**请求在 lane 内自毁**，不是调度器抢走。

这个区分不是措辞洁癖——它直接决定了修复方向。因为「驱逐」会让人去调驱逐策略或单纯加并发，
而 C=1 下这两条路什么都不做（连代码都进不去）。

但要说准：**C≥2 时准入层确实有一个可用杠杆，且不用改所有权模型** —— 是 tie-break（§6.3 方案 4），
不是驱逐策略。C=1 时它退化掉，才只剩脱钩一条路。

---

## 1. 现象

| 时点 | 期望 | 实际（实测值，§6.1） |
|---|---|---|
| turn 1 正常结束 | KV 保留，`retained = true` | 成立（`program_impl.h:1141`） |
| 标题请求插入 | 只用自己的少量 token，会话缓存不动 | 42 token 的请求把会话缓存销毁（见 §2） |
| turn 2 | `reuse ≈ 上一轮全长` | 对照组 961/1030 命中；插入标题后 `0/1025`、`full_reset` |

后面所有量化都基于这个模型的规模（`src/targets/qwen3_6_27b/impl/config.h:10-31`）：

| 项 | 值 |
|---|---|
| 总层数 | **64** |
| full attention 层 | **16**（每 4 层一个，`hybrid_topology.h:7-11`） |
| GDN（linear attention）层 | **48**（`config.h:66-67` 的 `static_assert` 固定了 16 / 48） |
| hidden / intermediate | 5120 / 17408 |
| GQA | query 24 / kv 4，head_dim 256 |
| GDN | key 16×128，value 48×128，conv kernel 4，`convolution_dim = 2·2048 + 6144 = 10240` |
| vocab（输出行） | 248,320 |

**64 和 48 不要混**：KV 容量按 16 个 full attention 层算，状态槽容量按 48 个 GDN 层算。

代价可以量化：缓存丢失后就等于一次 cold prefill，可直接用冷启动实测折算。
`docs/performance.md:312-318`（180,224 capacity、4,096 chunk、TP2、INT8 KV、optimized MTP3、CUDA Graph）：

| 请求 | token 数 | TTFT | prefill 吞吐 |
|---|---:|---:|---:|
| Codex | 11,718 | 10.40 s | 1,127 tok/s |
| Claude Agent | 16,368 | 14.71 s | 1,113 tok/s |

按 ~1,110 tok/s 折算，丢掉 3 万 token 上下文 ≈ **多花 27 秒**；丢掉 1.6 万 ≈ 14.7 秒。

---

## 2. 源码路径：它是怎么发生的

### 2.1 第一轮的收尾：缓存是「保留」的

自然结束时走 `terminal` 分支，只归还**行**（block table row），不归还**页**：

```cpp
// src/targets/qwen3_6/impl/runtime/program_impl.h:1137-1142
trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
if (terminal[row]) {
    release_sequence_growth_entitlement(sequence);
    unbind_sequence_kv(sequence);      // 还 row
    sequence.retained = true;          // 页仍由 sequence.kv 持有
    request.lifecycle = Lifecycle::Complete;
}
```

此时 lane 空闲，但 `sequence.kv`（`PagedKVAllocation`）仍然活着，`page_ids_` 仍在，`prefix_identity`
与 `rewrite_checkpoint` 仍在。这是下一轮能命中的全部物质基础。

**时序细节**：C=1 时标题请求不能抢占，只能排队。`outstanding_` 的硬上限是
`max_concurrency + max_pending_requests`（默认 `1 + 16 = 17`，`include/ninfer/types.h:114-116`），
标题请求先进 `pending_`，等 turn 1 结束、`slots_[0]` 释放后才被准入。
所以它不是「插进」对话中间，而是「紧接着」对话发生——这也解释了为什么受害的恰好是 turn 2，
而不是被中断的 turn 1。

### 2.2 标题请求进入规划阶段

复用判定只有两个候选基座，且都要求**从 token 0 起全等**：

```cpp
// src/targets/qwen3_6/impl/runtime/request_plan_impl.h:194-211
if (base.allow_prefix_reuse && prompt.identity.reusable && sequence.retained) {
    if (sequence.execution_frontier != 0 && dflash_append_ready &&
        prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                       sequence.execution_frontier)) {
        plan->reuse      = ReusePath::AppendAtFrontier;
        plan->reuse_base = sequence.execution_frontier;
    } else if (sequence.rewrite_checkpoint.valid && sequence.rewrite_checkpoint.frontier != 0 &&
               sequence.rewrite_checkpoint.frontier <= prompt.token_ids.size() &&
               prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                              sequence.rewrite_checkpoint.frontier)) {
        plan->reuse      = restore_path(sequence.rewrite_checkpoint.kind);
        plan->reuse_base = sequence.rewrite_checkpoint.frontier;
    }
}
```

`prefix_matches` 是逐 token 的 `std::equal`，没有最长公共前缀（LCP）回退。两个候选都不成立时，
`plan->reuse` 保持默认值 `FullReset`（`program.h:77`）。

多数客户端会自带指令模板，例如「请为以下对话生成标题：{首条消息截断}」。这种 prompt 的 token 0
就与 `ledger[0]` 不同，两个候选同时失败，`reuse` 停在默认值 `FullReset`：

```cpp
// src/targets/qwen3_6/impl/runtime/program_impl.h:705-712
if (request_plan.reuse == ReusePath::FullReset) {
    sequence.kv.reset();
    ordered_reset(sequence);
    sequence.ledger.clear();
    sequence.text_kv_valid = 0;
    sequence.mtp_kv_valid  = 0;
    reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement,
                        request_plan.backend_kv_page_entitlement);
}
```

`:706` 的 `sequence.kv.reset()` 就是销毁点：它析构 `PagedKVAllocation`，页立即回池。

`PagedKVAllocation` 的析构即释放，链路是 `~PagedKVAllocation()` → `release()` →
`unbind_row()` + `pool_->return_pages(page_ids_)`（`src/core/paged_kv_cache.cpp:287`、`:413-421`）。

**这是唯一一处「驱逐」字面成立的地方**，但执行者是请求自己的 plan，不是准入策略。

**即使标题 prompt 恰好以整段会话为前缀，结局也不变**——入口会命中 `AppendAtFrontier`，
`sequence.kv` 不被析构、页不回池，但 checkpoint 依然在 `:808-810` 被清空，`ledger` /
`prefix_identity` 依然在 `:819-820` 被换成标题请求的。所以 turn 2 仍然是 `FullReset`（§2.4）。

唯一的实际差别是**页在什么时刻回池**：入口 `FullReset` 就当场回池，入口命中则要拖到 turn 2 才回池。
这个差别不产生任何可观测行为差异，只决定「§6.3 的方案 2（`PreserveExisting`）有没有可能生效」——
因为那个方案要求页还活着。**所以这是同一个机制、同一条销毁链，不是两种问题。**

### 2.3 两个销毁点，支点那次是无条件的

规划阶段决定 checkpoint 动作，五态枚举（`program.h:44-50`）：

```cpp
enum class RewriteCheckpointAction : std::uint8_t {
    Drop, KeepExisting, ReclassifyExisting, CaptureNew, DeferCapture,
};
```

对应的判定（`request_plan_impl.h:235-255`）：

```cpp
const std::optional<RewriteCheckpointSpec>& desired = base.rewrite_checkpoint;
const bool existing_checkpoint_matches =
    desired && plan->reuse != ReusePath::FullReset && sequence.rewrite_checkpoint.valid &&
    sequence.rewrite_checkpoint.frontier == desired->frontier &&        // ← 必须 frontier 完全相等
    prefix_matches(prompt, sequence.ledger, sequence.prefix_identity, desired->frontier);
if (!desired) {
    plan->rewrite_checkpoint_action = RewriteCheckpointAction::Drop;
} else if (existing_checkpoint_matches) {
    plan->rewrite_checkpoint_action = sequence.rewrite_checkpoint.kind == desired->kind
                                          ? RewriteCheckpointAction::KeepExisting
                                          : RewriteCheckpointAction::ReclassifyExisting;
} else if (desired->frontier > plan->reuse_base) {
    plan->rewrite_checkpoint_action  = RewriteCheckpointAction::CaptureNew;
    plan->rewrite_checkpoint_capture = desired;
} else {
    plan->rewrite_checkpoint_action = RewriteCheckpointAction::DeferCapture;
}
```

执行阶段只有一处写：

```cpp
// program_impl.h:808-810
if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::Drop ||
    request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::CaptureNew) {
    sequence.rewrite_checkpoint = {};
}
```

`existing_checkpoint_matches` 的第一个条件就是 `plan->reuse != ReusePath::FullReset`（`:237`）。
这一行决定了**只要入口 `FullReset`，checkpoint 必然被判成 `Drop` 或 `CaptureNew`，必然被清空**，
不存在「页没了但支点还在」的中间态。于是销毁只有两处，而且第二处对所有形状都成立：

| 销毁点 | 触发条件 | 代码 | 观测证据 |
|---|---|---|---|
| **1. 页被释放** | 入口 `FullReset`（标题 prompt 不以会话开头） | `:706` `sequence.kv.reset()` | 标题请求自身 `full_reset` + 0（§6.1 典型形状组） |
| **2. 支点被清空** | `Drop` 或 `CaptureNew` —— 两者写在同一个条件里 | `:808-810` | 标题请求命中 962（页还在），下一轮仍 `full_reset` + 0（§6.1 postfix 组） |

**两处可以被分别观测到**：A 组证明第 1 处，B 组绕过第 1 处之后第 2 处照样生效。

第 2 条里最容易被忽略的是 **`Drop` 的语义不是「不动」，是「清空」**：标题请求如果走非 chat 模板、
不声明 checkpoint，就会落到 `Drop`（`:241-242`），和 `CaptureNew` 一起在 `:810` 被清掉。
如果它声明了自己的 checkpoint，边界必然 `> reuse_base`（`:247`），走 `CaptureNew`，同样清掉。
**两条路出口相同**，所以「让标题请求别声明 checkpoint」并不能保住会话的支点。

同时，身份指纹被整体替换：

```cpp
// program_impl.h:819-820
sequence.ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
sequence.prefix_identity.assign(prompt);
```

### 2.4 第二轮为什么必然 `FullReset`

turn 2 的 prompt 是「原会话 + 新消息」。两个候选：

- `execution_frontier` 是标题请求的长度。turn 2 的 prompt 在标题指令处就分叉，`prefix_matches` 失败。
- `rewrite_checkpoint` 已被 §2.3 清成 `{"valid": false}`，`sequence.rewrite_checkpoint.valid` 为 false，
  分支不进入。

→ `reuse` 停在默认 `FullReset` → 才能解释 `reused_prompt_tokens = 0`。

### 2.5 关键事实：页和行本来就是两种资源

`release()` 只有两件事：

```cpp
// src/core/paged_kv_cache.cpp:402-421
void PagedKVAllocation::unbind_row() noexcept {
    if (bound_row_ < 0) { return; }
    pool_->release_row(bound_row_);      // 行号只是一个 int
    bound_row_ = -1;
}
void PagedKVAllocation::release() noexcept {
    if (!valid()) { return; }
    unbind_row();
    pool_->return_pages(page_ids_);      // 物理页在 host 侧 vector 里
    pool_->replace_entitlement(page_entitlement_, 0);
    page_ids_.clear();
    ...
}
```

`bound_row_` 是整数下标，`page_ids_` 是 host 侧数组，**两者没有耦合**。页之所以跟着 lane 一起死，唯一原因是
`PagedKVAllocation` 挂在 `sequence` 上，而 `sequence` 挂在 lane 上，RAII 一析构就全还。

维护文档对此有明确表述：`docs/maintainer/paged-kv-cache.md:790`（§10.3）——「一个 retained bundle 同时只能被
一个 active request claim。所谓 pin 是 retained entry 持有各 pool allocations，使相应 IDs 不属于各自 free sets；
**不需要 page refcount 或特殊 allocator role**。」

---

## 3. 属于什么问题

四层，由浅到深：

| 层次 | 问题 | 证据 |
|---|---|---|
| **判定粒度** | 复用只接受「从 token 0 起全等」，无 LCP、无块级匹配；只有两个候选基座 | `request_plan_impl.h:198-210` |
| **快照容量** | 每 lane 仅 1 个 rewrite checkpoint（`2×C` 个状态槽），且任何请求都能改写它 | `linear_state_slots.h:13-36`；`:808-810` |
| **所有权** | 缓存以 lane 为唯一所有者；lane 是调度资源，天生被抢占 | `concurrent_executor.h:1291` `slots_{}` |
| **概念模型** | 系统里没有「会话」这个实体，lane 被迫兼任「调度单位」与「身份单位」 | 全仓库无 `session_id` / `conversation_id` |

**不是**下列任一：不是 LRU 淘汰策略问题、不是内存不足、不是并发竞争、不是 page size 问题。
C=1 时准入层的驱逐路径完全不可达（§0）。

维护文档已把这几层写成契约，不是遗漏：

- 判定粒度 ← `paged-kv-cache.md:921`：「只有更短 token prefix match，但该位置没有 checkpoint → cache miss」
- 快照容量 ← `paged-kv-cache.md:748-749`（§10.1）：retained sequence 只含「current resume frontier」+「一份
  有效时的 typed rewrite checkpoint」
- 所有权 ← `paged-kv-cache.md:790-803`（§10.3）：「一个 retained bundle 同时只能被一个 active request claim」

---

## 4. 其他框架有类似问题吗

没有一家是靠「改调度」解决的。差距在**缓存的所有权与回收方式**，不在调度器。

| 引擎 | KV 归属 | 旁路请求的影响 | 关键机制 |
|---|---|---|---|
| **vLLM** | 16-token 块，链式哈希（hash 含父块 hash + 额外键，如 LoRA / 多模态），全局块池 | 只对共享前缀块 refcount +1，另分配自己的块；两段都不搬动 | 块 refcount + LRU；请求结束反向入队，尾块先淘汰；`enable_prefix_caching` |
| **SGLang** | radix 树节点，每节点一组槽 | 分裂节点共享父路径；MAMBA checkpoint 留在原 endpoint，复用者取 COW 私有副本 | 节点 refcount + leaf-first LRU；hybrid 模型拆 FULL / SWA / MAMBA 三组件 |
| **TensorRT-LLM** | block + radix 搜索树 | 碰不到高优先级块 | `KvCacheConfig.enable_block_reuse`；**prioritized LRU**（priority 0–100，默认 35，低优先级先淘汰）；`KvCacheRetentionConfig` / `TokenRangeRetentionConfig(token_start, token_end, priority, duration_ms)`；`host_cache_size` 二级 offload |
| **llama.cpp** | 每 slot 一份 KV | 抢占 slot 时旧状态可进 RAM checkpoint 池 | `--cache-prompt`（默认开）按 slot 复用前缀；`-sps/--slot-prompt-similarity`（默认 0.10）选 slot；**`-ctxcp/--ctx-checkpoints` 默认 32 个，在 user message 边界创建**，`-cms/--checkpoint-min-step` 默认 8192 限制最小间距；`-cram/--cache-ram` 默认 8192 MiB；`--slot-save-path` + `/slots/{id}?action=save\|restore` 落盘 |
| **NInfer** | 每 lane 一份，**独占** | 会话缓存被销毁（§2） | 无 refcount、无优先级、无快照池、仅 1 个 checkpoint |

### 4.1 值得借鉴的三条（① 只作对照，不构成方案）

**① llama.cpp 的 32 个 context checkpoint（值得参考，但不构成本文的方案）**
它默认给每个 slot 保留 32 个 checkpoint，**在 user message 边界创建**，并用 `-cms` 避免间距过近的重复快照。
NInfer 现在只有 `2×C` 个槽（current + 唯一 checkpoint），全系统一次回退机会。
它的实测代价：单个 checkpoint 62–214 MiB（随位置增长），32 个约 4.7 GiB，超过默认 8 GiB 缓存的一半。

**但多槽救不了本文场景**：槽再多，只要页在旁路请求入口就被 `sequence.kv.reset()` 回收，
快照就指向已释放的页。所以它是「页能活下来之后」才谈得上的优化，不是本问题的解。

**② TensorRT-LLM 的 retention priority**
它的驱逐是「全局最低优先级先淘汰」，不是按 lane 号。NInfer 决定「谁被毁」的地方有两处，都不看优先级：

- 第一趟准入的 tie-break：`(!selected || reuse > selected_reuse)`（`concurrent_executor.h:784-788`）。
  `reuse` 相同时**严格大于不成立**，而 lane 从 0 递增，于是**下标最小的可准入 lane 赢**。
  C≥2 时旁路请求就是靠这个落回 lane 0 的（详见 §6.3 方案 4）。
- 第二趟的兜底驱逐：`evict_retained_lane` 从 `retained_lane = 0` 逐个往上清
  （`concurrent_executor.h:840-850`），只在第一趟整体不可准入时才走。

给请求加 `retention_priority`（会话高、旁路低）能替掉这两处的顺序。**本文场景（C=1）用不上**：
单 lane 时 tie-break 没有第二个候选，兜底路径也不可达。它的价值在 C≥2。

**③ SGLang 的 checkpoint 只读语义**
「MAMBA checkpoint 留在它原本的 prefix endpoint，复用它的请求通过 COW 获得私有副本，然后继续更新自己的状态」
——分支写自己的 state，绝不回头改支点。这正是 §6 方案 2（`PreserveExisting`）的外部依据。

### 4.2 一个必须澄清的社区方案

open-webui #6677《Postfix title/tag query instead of prefix to avoid trashing KV cache》提出把标题请求改成会话的
**续写**：

```
User: Hello.   AI: Hi.
User: Generate a title for the above conversation.   AI: Just Greetings
→ 丢弃最后两条消息，恢复正常对话
```

它能让标题请求自身**命中全前缀**（省掉那一次 prefill），但**救不了下一轮**：该请求仍然会声明自己的
`TurnClosure` 边界（在「请起标题」之后的 assistant header），仍然 `> reuse_base`，仍然 `CaptureNew`，
仍然覆盖会话支点。所以 postfix 必须与方案 2 组合才有效，单独使用只是把损失从「两次」降到「一次」。

### 4.3 一个反面证据（支持「无 LCP 是物理约束」）

llama.cpp 提供 `--cache-reuse N` 做非前缀块的 KV shifting，但这在 hybrid/recurrent 模型上失效，服务端会打印：

```
forcing full prompt re-processing due to lack of cache data
(likely due to SWA or hybrid/recurrent memory)
```

Qwen3.6 的 GDN 就是 hybrid/recurrent。这与 NInfer 的取舍一致：**recurrent state 是整条序列累积出来的量，
KV 页只承载 attention 那一半；没有对应位置的快照，页就算全在也无法继续**。
维护文档 `paged-kv-cache.md:775-776` 也写了同一句话：「最后一种情况不是 page-size limitation。缺少的是对应
Linear Attention/backend continuation state，而不是 KV page。」

---

## 5. 这是为了性能取舍后故意设计的吗

**是「设计」，不是 bug；但它的优化目标不是性能，是代码可经济性地证明正确。**

要把「会话身份与执行车道绑成一个东西」拆开看，三层性质完全不同：

| 耦合 | 是否强制 | 运行时收益 |
|---|---|---|
| 请求 ↔ lane（`slots_[i]`） | 是，调度本意 | — |
| KV 页 ↔ lane（`sequence.kv` 挂在 lane 上） | **否**，只是 RAII 放置位置 | **0** |
| 线性状态槽 ↔ lane（`i` 与 `C+i`） | 否，索引算术 | 约 73 MiB/槽 的内存代价 |

### 5.1 页的耦合没有任何运行时收益

- 页不会因为「绑定」而少占——它们本来就在池内、本来就算在这个 lane 的 entitlement 里。
- 行只是一个 `int`，`unbind_row()` 一行代码。
- 释放反而要做一次 `page_ids_` 遍历 + 重建三万个 token 的前缀。单用户场景下是**净负**。

所以这个耦合**不省时间、不省内存**。它省的是证明成本：一个所有者、一处释放，就免于证明
「悬垂引用」「双重释放」「attach 时 frontier 与 state 是否一致」这三类不变量。

### 5.2 但有一件事确实是承重的

必须区分两个决定：

| 决定 | 是否承重 | 理由 |
|---|---|---|
| checkpoint 不持有第二份 KV 拷贝 | **是** | 128k / INT8 / TP2 已约 2 GiB/卡；再来一份加 10.08 GiB 权重，16 GiB 卡装不下。文档 §10.1 明写「checkpoint 不是第二份 KV allocation」 |
| retained 序列必须绑在 lane 上 | **否** | 见 §5.1 |

**这两件事被一起实现了，但只有一件不可退让。** 把它们当成同一件事，是得出「只能这样」的原因。

### 5.3 真正的动因：系统里没有「会话」

`session_id`、`conversation_id`、`sequence_id` 在全仓库一个都不存在；
`slots_` 就是 `std::array<std::shared_ptr<Request>, kMaximumConcurrency>`（`concurrent_executor.h:1291`）。

lane 因此兼任两个角色：**调度单位**（该由谁执行）与**身份单位**（这段 KV 属于谁）。角色冲突时必然牺牲一方，
而被牺牲的是持久性——因为调度资源天生要被抢占。

一句话定性：**这不是「为性能牺牲了缓存」，而是把「缓存持久性」寄生在了「调度资源」上。**
它的代价在单用户交互场景全额兑现，收益在任何场景都接近零。

---

## 6. 优化方案

修复手段分岔成两条互斥分支（§6.1），取决于标题请求的 prompt 以什么形状进入引擎。
下面只保留被这一分歧支撑的方案：不改代码的（§6.2）与改代码的（§6.3）。

### 6.1 判据：一次观测就能分辨两条修复路线

分歧点只有一个：标题请求进入规划阶段时，页是否还活着。字段现成
（`GenerationResult.prefix_reuse_path`，`concurrent_executor.h:473`），
配置为 `qwen3.8-27b` / nvfp4 / TP2 / `--max-concurrency 1` / `prefix_reuse = true` 时实测：

| 组 | 请求 | prompt | `cached` | `prefix_reuse_path` |
|---|---|---:|---:|---|
| **A** 对照：无旁路 | turn 2 | 1030 | **961** | `restore_turn_checkpoint` |
| **B** 典型客户端标题（42 token） | title | 42 | 0 | **`full_reset`** |
| | turn 2 | 1025 | **0** | `full_reset` |
| **C** postfix 标题（1033 token） | title | 1033 | **962** | **`restore_turn_checkpoint`** |
| | turn 2 | 1031 | **0** | `full_reset` |

四条读数：

1. **对照组 turn 2 走 `restore_turn_checkpoint`，不是 `append_at_frontier`。**
   正常多轮靠支点恢复，所以 §2.3 的「清空 checkpoint」不是次要风险，**它就是主销毁路径**。
2. **B 组：标题请求自己是 `full_reset` + 0。** 按 §2.2，`sequence.kv.reset()` 在入口就执行了，
   页当场回池，支点指向的页不存在。方案 2 无从生效。
3. **C 组：标题请求 `restore_turn_checkpoint` + 962，页当时还活着；紧接着一个只短 2 token 的
   prompt（1031 vs 1033）却 `full_reset` + 0。** 页存活 ≠ 支点存活，两个销毁点被分别钉住。
4. 把同一组实验搬到 `--max-concurrency 2`，B/C 组的 turn 2 仍是 0。**lane 数不是变量。**

⇒ 修复路线由此确定：

| 标题请求形状 | 观测 | 可用方案 |
|---|---|---|
| 典型客户端（token 0 就分叉，自带指令模板） | `full_reset` + 0 | **方案 3**（C=1）／**方案 4**（C≥2）。页已回池，方案 2 无效 |
| postfix（以整段会话为前缀） | `restore_turn_checkpoint` + 962 | **方案 1 + 方案 2**（~30 行）。页还在，只缺「不清支点」 |

两个形状互斥，走哪条只看一件事：**标题请求的 prompt 由谁构造、能不能改成 postfix。**

**残余判据**（排障时要顺手排除的两种情形）：

| 观测到 | 结论 |
|---|---|
| turn 2 本来就是 `RestoreTurnCheckpoint` 且命中 ≈ 全量 | 缓存没丢，问题不在引擎，查 serve 层 |
| 所有轮次都是 0，含没发标题请求的对话 | 复用被整体关闭，查启动参数 |

后者对应：`--no-prefix-reuse` 会关掉整条复用逻辑
（`src/serve/serve_options.cpp:316-317`），此时任何轮次都命中不了。它默认是开的
（`include/ninfer/types.h:201`，`allow_prefix_reuse = true`），只在评测脚本里被硬编码
（`tools/bench/run_peak_sweep.py:271`、`run_serve_concurrency.py:340`、`run_serve_corpus.py:708`），
目的是拿 cold-start 可比数据——所以 `bench-runs/` 下所有 `launch.json` 都带它，**不代表服务配置**。
自查一次自己的启动命令即可。观测字段：`GenerationResult.prefix_reuse_path`
（`concurrent_executor.h:473`）与 `reused_prompt_tokens`（`:474`），serve 层还把它们写进了
每请求日志（`src/serve/request_log.cpp:599`）。

### 6.2 不改代码的路径

只有这两条，都只降低损失，**都不能保住主会话的缓存**。

**方案 1：postfix（应用层）**

把标题请求改成会话续写（§4.2），即让它的 prompt 以整段会话为前缀、末尾追加「请起标题」。

- 收益：标题请求自身命中全前缀。**实测**（§6.1 postfix 组）：1033 token 只用算 71 个，
  路径 `restore_turn_checkpoint`。所谓「多算一段 prefill」在实际调用里不存在。
- 局限：**救不了下一轮**。同一实测里紧接着的 turn2 仍是 `full_reset` + 0。支点仍在 `:810` 被清。
- 所以它单独没有独立价值，**唯一意义是把形状从典型搬到 postfix**，从而让方案 2 变得可用。

**方案 0：不用大模型生成标题**

直接取首条消息前 N 字符做截断式标题，彻底绕开旁路请求。引擎一行不动，代价是标题质量。
如果目标只是「产品体验不被打断」而不在乎标题质量，这条最省事。

### 6.3 改代码的路径

按改动量从小到大：方案 4（C≥2，~10–50 行）→ 方案 2（~30 行）→ 方案 3（200–400 行）。
**适用域不同**：方案 4 只在 C≥2 成立，方案 2 / 3 是本场景（C=1）的两条路，
且方案 2 必须先由方案 1 把标题请求搬到 postfix 形状（§6.1 分支表）。

#### 方案 4：把「谁被销毁」从 lane 下标改成显式代价（C≥2，约 10–50 行）

先撤回一个旧结论。早前把「提高 `--max-concurrency`」判为无效，理由是「这两条路径与 lane 数无关」——
**这个理由不成立**。C>1 单独无效，是因为 tie-break，不是因为 lane 数无关。
实验上也是这个结果：同一组实验搬到 `--max-concurrency 2` 重跑，两种形状的 turn 2 仍是 `0`
（§6.1 第 4 条）。所以「提并发」是下面两步的前提，不是方案本身。

C≥2、两 lane 都空闲时，标题请求在两个 lane 上的 `reusable_prompt_tokens` **都是 0**：
`ensure_lane_plan` 按 lane 独立算 plan，lane 0 上它与 retained 会话首 token 不同，lane 1 上没有 retained。
同时 `can_admit_lane` 并不因为「该 lane 有 retained」而拒绝，它只查 lifecycle 与页配额，
把 retained 的 entitlement 当作可替换的 `old`（`program_impl.h:529-551`）。于是第一趟两 lane 全中：

```cpp
// concurrent_executor.h:784-788
if (instance_.program->can_admit_lane(lane, plan) &&
    (!selected || reuse > selected_reuse)) {
```

`reuse` 相等时**严格大于不成立**，lane 从 0 递增 ⇒ **lane 0 先被选中并锁定**，lane 1 永远轮不到。
所以 C=2 时旁路请求照样落在会话所在的 lane 0 上，照样把它毁了。**杀死缓存的是「下标最小」这个 tie-break，不是 lane 数不够。**

改法两步，缺一不可：

1. **第一趟 tie-break**：键从 `reuse` 换成 `(reuse, destruction_cost)` 的字典序，
   `destruction_cost = sequences[lane].retained ? kv entitlement : 0`，即**优先选没有 retained bundle 的 lane**。
   `has_retained_lane` 已有现成接口（`program_impl.h:1162`）。
2. **第二趟兜底顺序**：容量吃紧时会走 `admit_planned_request` 的驱逐循环，它从 `retained_lane = 0`
   往上毁（`concurrent_executor.h:840-850`），同样先毁会话。要一起改成按 `retention_priority` 排序，
   否则第 1 步的收益在容量紧张时被吃掉。这才是原来设想的 `retention_priority` 真正该落的地方。

效果：标题请求落 lane 1，只毁 lane 1；turn 2 回 lane 0 全前缀命中，收益上界同方案 3（省一次全量 prefill）。

生效条件与代价：
- 仅 C≥2。C=1 只有一个 lane，无第二个候选，这条路不存在。
- 需要两个 lane 的 KV entitlement 同时放得下（标题请求很短，通常成立；但 256k 配置下两 lane
  各要一份配额，**必须实测 `can_admit_lane` 在目标配置下确实为 true**，不能假设）。
- 多一份状态槽 ≈ 73 MiB/卡（算式见 §7）。
- 改动集中在一个选择键 + 一个排序键，不新增不变量（不需要证明脱钩/重放，页从不离开 lane）。

#### 方案 2：`PreserveExisting`（约 30 行，只在页未释放时有效）

现状五态枚举里缺的正是「不捕获也不清除」这一态。改四处：

1. `program.h:44` 加 `PreserveExisting`。
2. `request_plan_impl.h:247-249` 当请求带旁路标记时，把 `CaptureNew` 降级为 `PreserveExisting`
   （`Drop` 分支 `:241-242` 同样要降级）。
   **注意不能降级为 `Drop`**——`:808` 对 `Drop` 同样执行清空。
3. `program_impl.h:808-810` 对 `PreserveExisting` 不做任何写操作，`sequence.rewrite_checkpoint` 保持原值，
   checkpoint 状态槽不被覆盖。
4. 标记来源：`serve_options` 加请求级开关，走 `ResolvedExecutionOptions` 透传（`allow_prefix_reuse` 已有现成通道）。

**生效前提有两个，缺一不可**，两者被分别观测到：

- 标题 prompt 以会话前缀开头，否则 `prefix_matches` 会合法地失败（这是正确行为，不是 bug）
  —— 典型形状不满足，postfix 形状满足；
- 入口没有走 `FullReset`，否则 `sequence.kv.reset()` 已经把页还回池，支点指向的页不存在了
  —— 典型形状 `full_reset`，postfix 形状 `restore_turn_checkpoint` 命中 962。

所以本方案**只在 postfix 形状下有效**，且该形状必须先由方案 1 造出来（§6.1 分支表）。
连续多个旁路请求也安全——只要都不捕获，支点一直有效。

#### 方案 3：retained 序列脱钩（C=1 下唯一独立有效的方案，200–400 行）

**不需要 refcount**（文档 §10.3 也明确说 pin 不需要 refcount）。需要的是 single-owner 的
**detached registry**：让 `PagedKVAllocation` + 状态槽 + 指纹三样脱离 lane 存活。

- 存活物：`page_ids_`（host 数组）、一个状态槽（约 73 MiB）、`prefix_identity`、`rewrite_checkpoint`。
- `FullReset` 时若当前 lane 有 retained 序列，把它移入 registry 而不是 `sequence.kv.reset()`。
- 准入时按指纹匹配 registry；命中则把行 `bind_row` 到新 lane，状态槽换过去，跳过一次重算。
- `bind_row` / `unbind_row` 已存在且无副作用依赖，接得上。
- 治理：registry 容量上限 + 淘汰键用 `reusable_prompt_tokens`（数据在 `plan.summary()` 里现成）。

页一个不多占（本来已分配、已 retained）；新增内存只有 `N × 73 MiB` 状态槽，算式见 §7。

### 6.4 明确不要做

| 方案 | 为什么不做 |
|---|---|
| LCP / 块级部分前缀匹配 | 对 GDN 无效。缺少的是 continuation state 不是页（§4.3） |
| 延后或去抖标题请求 | 时间点无关。只要它进同一 lane 并释放/覆盖，损失是固定的，推迟只把 `FullReset` 从 turn 2 挪到 turn 3 |
| 只把 `--max-concurrency` 提到 2、不动选择键 | 无效。旁路请求的 `reuse` 在两个 lane 上都是 0，tie-break 严格大于 + 下标优先 ⇒ 仍落 lane 0（§6.3 方案 4）。**提并发是方案 4 的前提，不是方案本身。** |
| 把标题 prompt 拼成整段会话前缀「多复用一点」 | 这样它必然 `CaptureNew`（自己的边界 > `reuse_base`），照样拆支点，还多算一段 prefill |
| 让标题请求不声明 checkpoint（走 `Drop`） | `Drop` 语义是清空（`:808`），与会 `CaptureNew` 同样销毁支点 |

---

## 7. 改造难度与性能影响

| 方案 | 改动点 | 代码量 | 需新证明的不变量 | 内存 | 性能影响 | 风险 |
|---|---|---|---|---|---|---|
| **1. postfix（应用层）** | 客户端 prompt 构造 | 0（引擎） | — | 0 | 标题请求只算 71/1033 token；**对主会话无保护** | 依赖客户端可控；单独用无价值 |
| **2. `PreserveExisting`** | 枚举 + 决策 + 收尾 + 选项透传 | ~30 行 | 无（复用现有五态逻辑，仅加「不写」分支） | 0 | ~0（无复制、无搬页） | **仅 postfix 形状有效**；典型形状入口即 `full_reset`，页已回池，本方案无效 |
| **3. retained 脱钩（registry）** | 分配器归属 + attach 协议 + 淘汰键 | 中，约 200–400 行 | **有**：detach 后状态槽不被 lane 复用、页不被两次释放、attach 时 frontier 与 state 一致 | +N × 73 MiB | ~0 到 −1%（多一次 `bind_row`，微秒级） | 结构性改动，需单测覆盖脱钩/回收/复用三条路径 |
| **4. lane 选择键（仅 C≥2）** | 第一趟 tie-break 加 `destruction_cost`；兜底驱逐改按 `retention_priority` 排序 | 小，约 10–50 行 | 无（页从不离开 lane，不需证明脱钩/重放） | +1 份状态槽 ≈ 73 MiB | 标题请求改落空闲 lane，主会话零损失 | **C=1 无效**；只提并发已被证伪；需确认两 lane 配额 |

补充三点量化（方案 3 与方案 4 共用同一个收益上界——两者都是「保住一次全量 prefill」）：

- **收益上界** ≈ 一次全量 prefill 的时间。按 1,110 tok/s，3 万 token 约 **27 秒**；1.6 万约 **14.7 秒**。
  小规模直测与之吻合：1k token 会话命中时 turn 2 从 2.07 s 降到 1.53 s，约 **0.5 s / 1k token**。
- **状态槽成本** 是 `N × 73 MiB`（每槽，TP2 每卡）。算式（维度 `config.h:21-26`，TP 分片与精度
  `layouts_impl.h:135-141`，形状 `linear_attention_state.cpp:95-99`）：

  ```
  recurrent（FP32，shape {key_head_dim 128, value_head_dim 128, value_heads 24, slot}）
    = 48 层 × 24 value heads（48 / tp=2） × 128 × 128 × 4 B = 75,497,472 B ≈ 72.0 MiB

  conv（BF16，shape {conv_channels 5120, conv_width 3, slot}）
    = 48 层 × 5120 通道/卡 × 3 步 × 2 B = 1,474,560 B ≈ 1.4 MiB

  合计 ≈ 73.4 MiB / 槽
  ```

  conv 只占 1.4 MiB 而不是 2.8，因为 `conv_dtype = DType::BF16`（`layouts_impl.h:141`）；
  recurrent 固定 FP32（`linear_attention_state.cpp:140`）。hidden 未计入，量级更小。
  现在 2 槽 147 MiB；方案 3 / 方案 4 各再加 1 槽。
- **与 KV 的比值**：按你的实际配置（64 层 / 16 个 full attention 层 / kv 4→2 每卡 /
  head_dim 256 / INT8 / TP2）计算 KV：

  KV per token per card = 16 × 2 (K,V) × 2 heads × 256 × 1 B + 256 组 scale × 4 B ≈ **17 KiB**

  | max-context | KV / 卡 | 2 个脱钩槽（147 MiB）占比 |
  |---|---:|---:|
  | 32k | 544 MiB | 27% |
  | 128k | 2.13 GiB | 6.8% |
  | **256k（你的 `--max-context 262144`）** | **4.25 GiB** | **3.4%** |

  也就是说**上下文越长，这个方案越划算**——在你的 256k 配置下，两个槽只占 KV 的 3.4%，
  基本等同于免费。这也与「长会话才值得保缓存」一致。

---

## 8. 怎么选

分岔判据是「标题请求的 prompt 由谁构造」，不是改动量。改动量只在选定分支后用来排先后。

1. **先确认控制权在谁手里**：

   | 情况 | 走法 | 说明 |
   |---|---|---|
   | 客户端 / 服务端可改标题 prompt 构造 | **方案 1 + 方案 2**（~30 行） | 把形状从典型搬到 postfix；不动所有权模型 |
   | 标题 prompt 由前端框架或第三方固定 | **方案 3**（C=1）／**方案 4**（C≥2） | 页在入口就回池，只能动所有权或动 lane 选择 |

2. **走方案 1+2 时**：先只改 prompt 构造，确认标题请求的 `prefix_reuse_path` 从 `full_reset`
   变成 `restore_turn_checkpoint` 且有命中——这一步不改引擎就能验证 postfix 形状成立
   （§6.1 C 组就是该形状）。确认之后才需要那 30 行枚举与分支。
3. **走方案 3 或方案 4 时，先算一笔账**：

   | 前提 | 选 | 理由 |
   |---|---|---|
   | 可以接受 C=2 | **方案 4**（~10–50 行） | 改动最小、不新增不变量；前提是确认两 lane 配额放得下 |
   | 必须 C=1 | **方案 3**（200–400 行） | 单 lane 没有第二个候选，只能把页从 lane 上摘下来 |

   先确认 C=2 下 `can_admit_lane` 在 lane 1 上确实成立，再决定要不要动所有权模型。
   省下来的不是几十行代码，是「脱钩 / 回收 / attach 一致」三条不变量的证明成本。
   **光把 C 提到 2 不成立**（§6.1 第 4 条），必须先改选择键。
4. **走方案 3 时**：先做最小版本（registry 容量 1–2、单会话），验证 attach 路能命中，再考虑扩容。
   256k 配置下 1 个槽只占 KV 的 1.7%。
5. 只想让产品体验不被打断、不在乎引擎改造：**方案 0**（截断式标题）直接绕开。

---

## 附录 A：关键代码位置索引

### A.1 模型与代码目录的对应

目录名是**代码世系**，不是模型名。Qwen3.8-27B 没有独立目录，它与 Qwen3.6-27B 共用同一组 C++ 类型，
在注册表里按 artifact 的 `model_id` 分派：

| 目录 | 作用 | 与 Qwen3.8-27B |
|---|---|---|
| `src/targets/qwen3_6/` | 家族共享层：`impl/runtime`（KV 复用 / checkpoint / 调度对接）、`impl/frontend`（chat template）、`impl/state` | 与权重无关，3.6 / 3.8 共用 |
| `src/targets/qwen3_6_27b/` | 27B 架构：`impl/config.h` 维度常量、`impl/load` 权重绑定、`impl/package.cpp` 采样默认值 | **就是这里** |
| `src/targets/qwen3_6_35b_a3b/` | 35B-A3B 变体 | 不适用 |

**`src/targets/qwen3_6/impl/runtime/*.h` 对 Qwen3.8-27B 是同一份源码，不是「相似」或「对应」。**
27B 目录里根本没有 `program_impl.h` / `request_plan_impl.h` 的副本；它只提供一个 `Variant`
（`impl/config.h` 的常量 + `impl/variant.cpp` 的算子叶子），然后用宏把共享源码**就地展开**：

```cpp
// src/targets/qwen3_6_27b/impl/variant.cpp:19-21
#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"
```

`instantiate.h` 逐个 include 共享实现（`program_impl.h`、`request_plan_impl.h`、`layouts_impl.h` …），
而共享实现的文件头是 `namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {`
（`program_impl.h:26`）。也就是说：

- 同一份源码被**编译两次**（27B 一次、35B-A3B 一次，见 `qwen3_6_35b_a3b/impl/variant.cpp:15`），
  只是落进不同命名空间，行号完全一致；
- 差异只来自 `Variant` 的编译期常量——`kNativeContext`、层数、head 数、`maximum_mtp_draft_tokens`
  这类**数值**，不改变任何判定分支的结构。
- 更直接的证据：`program_impl.h` 与 `request_plan_impl.h` 里 `Variant::` 出现 **0 次**、
  `#if` / `#ifdef` 出现 **0 处**。复用判定链上不存在任何按模型分叉的代码，
  连编译期常量都没用到——它是纯粹的逻辑。

所以正文引用的 `program_impl.h:706`、`:808-810`、`request_plan_impl.h:237-249` 等行号，
对 Qwen3.8-27B 直接成立；凡是真的按模型取值的地方（64 层、16 个 full attention 层、48 个 GDN 层、
kv 4、head_dim 256），正文都另外引用了 `qwen3_6_27b/impl/config.h`。

```cpp
// src/targets/registry.cpp:268-275
if (identity.model_id == Qwen3_6_27B::model_id) {
    return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
        options, execution, reader, load_start, Qwen3_6_27B::target_key);
}
if (identity.model_id == Qwen3_6_27B::qwen3_8_model_id) {
    return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
        options, execution, reader, load_start, Qwen3_6_27B::qwen3_8_target_key);
}
```

两个 `model_id` 落到**同一组 C++ 类型**，外部可见的差异只有 target key：

```cpp
// src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h:87-89
static constexpr std::string_view target_key         = "qwen3_6_27b";
static constexpr std::string_view qwen3_8_model_id   = "qwen3.8-27b";
static constexpr std::string_view qwen3_8_target_key = "qwen3_8_27b";
```

采样默认值见 `impl/package.cpp:61-76`（`kQwen3_8Defaults`，`:80` 按 `model_id` 选中）；
权重 profile 见 `:86-107`（`resolve_weights`，按 `weights_id` 分流为 `groupwise-int` / `nvfp4` /
`gguf-*`）。

### A.2 正文引用索引

| 位置 | 内容 |
|---|---|
| `src/targets/qwen3_6_27b/impl/variant.cpp:19-21` | 用 27B 的 `Variant` + `qwen3_6_27b_runtime` 命名空间展开共享 runtime |
| `src/targets/qwen3_6/impl/runtime/instantiate.h:7-27` | 逐个 include 共享实现（`program_impl.h`、`request_plan_impl.h` …） |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:26` | 共享实现落在 `detail::NINFER_QWEN36_RUNTIME_NS` 命名空间 |
| `src/targets/registry.cpp:268-275` | 按 `model_id` 分派：`qwen3.8-27b` → `qwen3_8_27b`，同一组 C++ 类型 |
| `src/targets/qwen3_6_27b/export/ninfer/targets/qwen3_6_27b/package.h:87-89` | `target_key` / `qwen3_8_model_id` / `qwen3_8_target_key` |
| `src/targets/qwen3_6_27b/impl/config.h:10-31` | 维度常量：64 层、kv 4、head_dim 256、GDN 48×128 |
| `src/targets/qwen3_6_27b/impl/config.h:66-67` | `static_assert`：full attention = 16、GDN = 48 |
| `src/targets/qwen3_6/impl/runtime/layouts_impl.h:135-141` | 状态池 spec：按 `tp` 分片，`conv_dtype = BF16` |
| `src/core/linear_attention_state.cpp:95-99` | conv / recurrent 的 shape 与精度（recurrent 固定 FP32） |
| `src/targets/qwen3_6_27b/impl/package.cpp:61-76` | `kQwen3_8Defaults`（采样默认值） |
| `src/targets/qwen3_6_27b/impl/package.cpp:86-107` | `resolve_weights`：按 `weights_id` 分发权重 profile |
| `tools/bench/run_peak_sweep.py:271` 等三处 | bench 脚本硬编码 `--no-prefix-reuse`（cold-start 可比） |
| `src/targets/qwen3_6/export/ninfer/targets/qwen3_6/hybrid_topology.h:7-19` | 每 4 层 1 个 full attention；`gdn_layers = total − full` |
| `src/serve/serve_options.cpp:316-317` | `--no-prefix-reuse` 关闭复用 |
| `include/ninfer/types.h:201` | `allow_prefix_reuse` 默认 `true` |
| `src/runtime/engine/concurrent_executor.h:779-807` | `find_admission_lane` 两趟扫描；`:790` 第一趟命中即返回 |
| `src/runtime/engine/concurrent_executor.h:784-788` | tie-break：`(!selected \|\| reuse > selected_reuse)`，严格大于 + lane 递增 ⇒ 下标最小者赢 |
| `src/runtime/engine/concurrent_executor.h:840-851` | retained 驱逐兜底循环（从 lane 0 顺序清），仅第二趟可达 |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:529-551` | `can_admit_lane`：只查 lifecycle 与页配额，不因「lane 有 retained」拒绝 |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:1162` | `has_retained_lane`（方案 4 的 tie-break 键可直接复用） |
| `src/runtime/engine/concurrent_executor.h:1291` | `slots_{}`：lane 与请求一一对应 |
| `src/runtime/engine/concurrent_executor.h:473-474` | `prefix_reuse_path` / `reused_prompt_tokens`（观测字段赋值） |
| `include/ninfer/types.h:437-438` | 上述两字段的声明与默认值（`FullReset`） |
| `src/targets/qwen3_6/impl/runtime/request_plan_impl.h:194-211` | 复用路径决策（只有两个候选基座） |
| `src/targets/qwen3_6/impl/runtime/request_plan_impl.h:235-255` | checkpoint 动作决策（五态 → 四分支） |
| `src/targets/qwen3_6/impl/runtime/program.h:44-50` | `RewriteCheckpointAction` 五态枚举 |
| `src/targets/qwen3_6/impl/runtime/program.h:77` | `reuse` 默认值 = `FullReset` |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:703` | 进入 prefill 即 `retained = false` |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:705-712` | `FullReset`：`sequence.kv.reset()`（页回池） |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:736-786` | checkpoint 恢复分支：trim + `copy_slot` |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:808-810` | `Drop` / `CaptureNew` 清空 checkpoint |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:819-820` | ledger / identity 被整体替换 |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:1137-1142` | 自然结束：`unbind` + `retained = true` |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:1179-1195` | `clear_lane`（取消/故障共用的硬复位） |
| `src/targets/qwen3_6/impl/runtime/linear_state_slots.h:13-36` | `2×C` 状态槽布局 |
| `src/core/paged_kv_cache.cpp:402-421` | `unbind_row` / `release`（行与页分离） |
| `docs/maintainer/paged-kv-cache.md:759` | 「checkpoint 不是第二份 KV allocation」 |
| `docs/maintainer/paged-kv-cache.md:775-776` | 「缺少的是 continuation state，而不是 KV page」 |
| `docs/maintainer/paged-kv-cache.md:790-803` | §10.3 独占 bundle；pin 不需要 refcount |
| `docs/maintainer/paged-kv-cache.md:921` | 边界条件：更短前缀匹配但无 checkpoint → miss |
| `docs/maintainer/paged-kv-cache.md:924` | 边界条件：取消 → boundary release bundle |
| `docs/performance.md:312-318` | 冷启动 prefill 实测（1,110 tok/s 量级） |

## 附录 B：术语对照

| 本文 / llama.cpp | NInfer | 说明 |
|---|---|---|
| `PARALLEL` / `-np` | `--max-concurrency` | 默认 1，取值 [1,8] |
| target key `qwen3_8_27b` | 目录 `src/targets/qwen3_6_27b/` | 同一份实现，按 `model_id` 分派 |
| slot | lane | `slots_` 数组下标；也是状态槽下标 |
| prompt cache | retained sequence | `sequence.retained` |
| context checkpoint | rewrite checkpoint | `RewriteCheckpointSpec{kind, frontier}` |
| `cache_n`（timings） | `reused_prompt_tokens` | 判定是否命中的直接观测值 |
| `--cache-ram` | 无对应 | NInfer 的 retained 只在显存内 |
