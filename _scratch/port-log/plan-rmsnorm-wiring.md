# 第 6 项 · RMSNorm 融合接线方案（静态设计，未落代码）

状态：**设计完成，未动任何源码、未编译、未运行**
撰写时间 2026-10-25
阅读基线：lab worktree `C:/Users/lwb6/Documents/git/NInfer-V100-Duo-win/.agents/worktrees/port-packet-allreduce`
（分支 `port/gdn-split10`）；已提交的内核部分读自 `git show 8bb59c64:<路径>`。

---

## 一、结论先说

1. **路线 (a)（把 stage scope 提到层级）单独用不了**，原因是它对**站点 A 也无效**：
   站点 A 的归一化目标 `h` 是 `mlp_tail_tp2` **内部**才分配的
   （`text_context_impl.h:1865-1867`），而写入发生在 `attn_mix_tp2` 末尾的归约
   （`text_context_impl.h:1678-1680`）—— 时间顺序上 `h` 那一刻还不存在。
   scope 提到层级只能让「早分配的东西晚释放」，不能让「晚分配的东西早存在」。
2. **必须走路线 (b)：在层内（实际是 `run_layers_tp2` 函数级）显式分配，再把地址传给两个 stage。**
   这不只是「选一条」，而是 (a) 的必要补充：真正需要的是**一次显式分配**，(a) 连提 scope 都不需要。
3. **只要 1 个 buffer 就够**（每卡每 call），因为站点 A 的目标和站点 B 的目标可以在同一层内串行复用同一块
   `[hidden, T] BF16`（证明见 §2.3）。
4. **峰值 workspace 增量 = `10240 × T` 字节/卡**（T=1..8 → 10~80 KiB）。若按「不共用」的保守写法是
   `20480 × T`。这个量级**不会**突破 `layouts_impl.h` 的既有容量——因为容量是**从同一份 recipe 重新算出来的**
   （`layouts_impl.h:331-337` 的 `post_mixer_stage` 等），不是硬编码常量；但**必须同步改 recipe**，
   否则运行时 `DeviceArena::alloc_bytes` 在 `arena.cu:208` 抛 `std::bad_alloc`。
5. **prefill 建议按 `T <= 8` 门控关掉整条融合**（`T=2560` 永远走不进融合几何）。
   这样 prefill 的指令序列与今天**逐字相同**，oracle 的 prefill 部分天然不可变 —— 这是本项最重要的护栏。
6. 站点 A 的权重是 `FullLayerW::post_attn_norm` / `GdnLayerW::post_attn_norm`；
   站点 B 的权重是**下一层**的 `input_norm`，必须由 `run_layers_tp2` 按 `layer+1` 取，**最后一层传 nullptr**。
7. **两个最容易写错、且都不会崩的地方**（详见 §5）：
   - **第一层不能当成「已归一化」**：`norm_hidden` 在 layer 0 之前没人写过。
     所以「输入已归一化」必须是独立条件 `fuse_norm && layer > 0`，不能复用「归约要融合」那个条件。
   - **叶子接受 `norm` 就必须兑现它**。`gdn_output_projection` 的 GGML_K/GGUF 分支
     （`variant.cpp:731-735`）必须一起接（`ggml_k_gdn_output_norm`），
     否则 `mlp_tail_tp2` 会读到未初始化的 arena 残值 → 静默错误输出。
8. **不需要改 `linear_add.h`**（19 个 include 者）。新增的融合入口放在
   `include/ninfer/ops/linear_add_norm.h`，只有 `linear_add.cpp` 和两个 `variant.h` 侧会 include。

---

## 二、scope 重构方案

### 2.1 现状（带行号的证据）

`src/targets/qwen3_6/impl/runtime/text_context_impl.h`

```
1875: void TextContext::run_layers_tp2(...)
1879:     for (int layer = 0; layer < kCfg.n_layers; ++layer) {
1880:         if (ModelConfig::is_full(layer)) {
1885:                 nvtx::ScopedRange mixer_range(...);
1886:                 auto scope_0 = work_.scope();
1887:                 auto scope_1 = tp_->work->scope();
1888:                 attn_mix_tp2(a, b, x, fidx, ph, staging);        // 归约发生处（站点 A）
...
1896:             {
1898:                 nvtx::ScopedRange post_mixer_range(...);
1899:                 auto scope_0 = work_.scope();
1900:                 auto scope_1 = tp_->work->scope();
1901:                 mlp_tail_tp2(a.post_attn_norm, b.post_attn_norm, a.mlp, b.mlp, x, ph, staging);
1902:             }                                                    // ← 出块即回收
```

GDN 分支完全对称：`1908-1917`（`gdn_mix_tp2`）+ `1919-1925`（`mlp_tail_tp2`）。

`mlp_tail_tp2` 的 `h` 是 arena 临时分配：

```
1857: void TextContext::mlp_tail_tp2(const Tensor* post_norm_0, const Tensor* post_norm_1, ...)
1864:     std::array<Tensor, 2> h;
1865:     for (std::size_t r = 0; r < 2; ++r) {
1866:         h[r] = workspace_recipe::post_mixer_hidden<TextConfig>(*ws[r], T);
1867:     }
1868:     for_each_rank(execution, [&](int rank) { ops::rmsnorm(x[r], *norm[r], ..., h[r], ...); });
1872:     Variant::post_mixer(h, {...}, x, staging, ph, ws, execution, *tp_->events);
```

`workspace_recipe::post_mixer_hidden` = `matrix(allocator, BF16, Config::hidden, tokens)`
（`workspace_recipe.h:145-148`），`hidden = 5120`（`qwen3_6_27b/impl/config.h:12`）。

### 2.2 两条路线的量化对比

**(a) 把 `scope_0/scope_1` 从两个 `{}` 提到「层」级**（即 `1879` 的循环体里只开一次）
→ 层内所有 stage 的临时张量互相叠加。单层的名义张量足迹（tp=2，`kTensorParallelWidth=2`）：

| 来源 | 行数 × T（BF16 行数） | 出处 |
|---|---:|---|
| `text_attention_projection`：hidden 5120 + query 3072 + gate 3072 + key 512 + value 512 | 12 288 | `workspace_recipe.h:61-71`（`/tp`），`config.h:41-47` |
| `text_attention_results`：qn 3072 + kn 512 + a 3072 | 6 656 | `workspace_recipe.h:79-87` |
| `gdn_control`：hidden 5120（+ g/beta FP32） | ≈5 120 | `workspace_recipe.h:95-102` |
| `gdn_projection`：3072+1024+1024+3072 | 8 192 | `workspace_recipe.h:111-119` |
| `gdn_recurrent_output` + `gdn_normalized_output` | 6 144 | `workspace_recipe.h:135-143` |
| `post_mixer_hidden` | 5 120 | `workspace_recipe.h:145-148` |

一层内 full 与 GDN 互斥，所以实际叠加的是 `max(attn, gdn) + post_mixer` ≈
`(19456 + 5120) × T × 2B = 49 152 × T 字节`。
T=8 → **+384 KiB/卡**。

**(b) 显式分配 1 个 `[5120, T]` BF16 并传给两个 stage**
→ `5120 × T × 2B = 10 240 × T 字节`。T=8 → **+80 KiB/卡**。

**(a) 比 (b) 贵 4.8 倍，而且 (a) 单独用根本不解决问题**（§一.1）。故取 **(b)**。

> 补一条 (a) 的额外代价：`layouts_impl.h` 的模拟把 `attention_stage`/`gdn_stage`/`post_mixer_stage`
> **三段全部串行调用**（`target_body`，`331-337`），而运行时一层只跑 `attn` 或 `gdn` 之一。
> 一旦把 scope 提到层级，模拟会按「三段叠加」规划，比真实需求再多算一整个 attention stage
> —— 也就是 (a) 会让**规划值**比**实际值**高出约 `24 576 × T` 字节，纯浪费。

### 2.3 为什么 1 个 buffer 够：同层串行复用证明

设 `B[r]` 为每卡一块 `[5120, T]` BF16。层 L 的动作序列（同一个 `rank` 的 stream，顺序执行）：

```
attn_mix(L):
  A1  attention_projection(h = B)          // 读 B（B 里是层 L 的 input_norm 结果）  ← 站点 B 的消费者
  A2  o_proj 归约: x += o_proj(a) 同时 post_attn_norm → 写 B             ← 站点 A 的生产者
mlp_tail(L):
  A3  post_mixer: swiglu 读 B → activation                                // 读 B（A2 的产物）
  A4  down 归约: x += down(activation) 同时 下一层 input_norm → 写 B      ← 站点 B 的生产者
attn_mix(L+1):
  A5  attention_projection(h = B)          // 读 B（A4 的产物）
```

每个「写」都排在它要覆盖的那次「读」之后，且都在该 rank 自己的 stream 上（`ws[r]`/`stream_for(rank)`
一一对应，`1685-1690`、`1860-1862`）。跨 rank 不构成依赖：融合内核**只读自己的 inbox、
只把自己的 `buffer[r]` 发布到对端 inbox**，从不直接读对端 `buffer`（`allreduce.cu` 融合内核
`store_volatile_u32(peer + …)` / `load_volatile(slot)`，见 §五）。

GDN 分支同理：`h` 的读者 `gdn_control_projection`（`1717`）与
`gdn_input_projection{,_snapshot,_record}`（`1759` / `1802` / `1836`）**全部早于**
`gdn_output_projection`（`1852-1854`），所以 A2 时刻覆盖 `h` 安全。
`attn_mix` 里 `h` 的读者只有 `1627` 的 `attention_projection`，早于 `1678` 的归约。

**序列的起点例外**：`A5` 只在 `layer >= 1` 成立。第一层没有前驱，`B` 里是 arena 残值，
所以第一层的 `attention_projection` 必须仍然吃自己算出来的 `projection.hidden`
（即 `input_prenormalized == false`）。这一点在签名上被一个独立的 bool 承载（§2.6）。

**若评审不接受这种同址复用**，退化为 2 块（`B_attn` / `B_next`），增量变 `20 480 × T`，其余设计不变。

### 2.4 峰值增量表（每卡每 call，BF16）

公式：

```
Δbytes = buffers × 5120 × T × 2 = buffers × 10 240 × T
```

| T | 1 buffer（推荐） | 2 buffers（保守） |
|---:|---:|---:|
| 1 | 10 240 B（10 KiB） | 20 480 B（20 KiB） |
| 2 | 20 480 B（20 KiB） | 40 960 B（40 KiB） |
| 3 | 30 720 B（30 KiB） | 61 440 B（60 KiB） |
| 4 | 40 960 B（40 KiB） | 81 920 B（80 KiB） |
| 5（MTP4 decode） | 51 200 B（50 KiB） | 102 400 B（100 KiB） |
| 6 | 61 440 B（60 KiB） | 122 880 B（120 KiB） |
| 7 | 71 680 B（70 KiB） | 143 360 B（140 KiB） |
| 8 | 81 920 B（80 KiB） | 163 840 B（160 KiB） |
| 2560（prefill，**门控后为 0**） | 0 | 0 |

（不门控的话 prefill 是 `10 240 × 2560 = 26 214 400 B ≈ 25.0 MiB` —— 纯浪费，见 §2.5。）

### 2.5 会不会突破 `layouts_impl.h` 已算好的容量

**不会，但前提是同一提交里改 recipe，否则运行时就崩。** 证据链：

1. 容量不是常量，是**回放同一份 recipe 得到的峰值**：
   `layouts_impl.h:331-337`
   ```cpp
   const auto post_mixer_stage = [&](WorkspaceLayoutBuilder& layout, int first, int last, ...) {
       auto stage = layout.scope();
       (void)workspace_recipe::post_mixer_hidden<TextConfig>(layout, last);
       scratch(layout, Variant::post_mixer_workspace_capacity_bytes(...));
   };
   ```
   `target_body`（`331`）依次调用 `attention_stage`（`285` 起）、`gdn_stage`（`297` 起）、
   `post_mixer_stage`；`target_body` 的调用点有 6 处（`442/452/464/501/594/619`），
   每处对应一个 builder（prefill=chunk、ordinary=逐 batch、mtp、dflash…），
   最终 `out.capacity = max({...})`（`633-635`）。
2. 运行时 arena 的容量就是这个值：`program_impl.h:232-233` / `260-261`
   `workspace_storage(plan.workspace.capacity)`，`work(DeviceSpan{workspace_storage.base(), ...})`。
3. 溢出是硬失败，不是静默越界：`src/core/arena.cu:207-208`
   ```cpp
   const std::size_t end = aligned_offset + bytes;
   if (end > cap_) { throw std::bad_alloc(); }
   ```
   → **只改 runtime 不改 recipe = 启动后第一次 decode 就 `std::bad_alloc`。**
4. 反向的「容量够不够」担心不成立：recipe 与 runtime 同构，**recipe 改了容量自动 +Δ**。
   真正的次级影响是 `device_reservation_bytes` 也 +Δ（`layouts_impl.h:855-861`：
   `persistent.bytes + workspace.capacity + request_transient + graph_allowance`），
   即 KV cache 预算少 10~80 KiB —— 相对 16 GiB 完全可忽略，但它是**启动期会参与显存校验的数字**，
   所以验收时要盯着启动日志里的 reservation 是否仍然通过。

### 2.6 可直接粘的代码骨架

**`text_context_impl.h:1875` 起，`run_layers_tp2` 开头插入**（在 `const bool prefill = ...` 之后、
`if (sink != nullptr)` 之前）：

```cpp
    const int T = x[0].ne[1];
    const std::array<WorkspaceArena*, 2> ws = workspaces();
    // Fused-norm scratch. One [hidden, T] BF16 per rank, alive for the whole call so the row-parallel
    // reduce can write the normalized row before the stage that consumes it has allocated anything.
    // Skipped entirely outside the fused packet geometry (T > 8), where the same code takes the
    // unfused route it always took -- that keeps prefill's instruction sequence byte-for-byte
    // identical to the pre-fusion build.
    const bool fuse_norm = T <= 8;   // mirrors src/ops/common/allreduce.cu's kNormMaxColumns
    std::array<Tensor, 2> norm_hidden{};
    if (fuse_norm) {
        for (std::size_t r = 0; r < 2; ++r) {
            norm_hidden[r] = workspace_recipe::post_mixer_hidden<TextConfig>(*ws[r], T);
        }
    }
    // Layer L's post-mixer reduce writes the *next* layer's input-norm result, so the weight has to
    // be fetched here: the MLP leaf has no idea which block follows it. The final layer has no
    // successor -- its epilogue norm is `final_norm_`, run by the caller after this loop returns
    // (2150 / 2377 / 2763), so it passes nullptr and keeps the standalone rmsnorm.
    const auto layer_input_norm = [&](int layer) -> std::array<const Tensor*, 2> {
        if (layer >= kCfg.n_layers) { return {nullptr, nullptr}; }
        if (ModelConfig::is_full(layer)) {
            const auto i = static_cast<std::size_t>(ModelConfig::full_idx(layer));
            return {full_.at(i).input_norm, full_peer_.at(i).input_norm};
        }
        const auto i = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
        return {gdn_.at(i).input_norm, gdn_peer_.at(i).input_norm};
    };
```

**full 分支（`1885-1894` 的两处调用改成）**：

```cpp
                attn_mix_tp2(a, b, x, static_cast<int>(fidx), ph, staging, norm_hidden,
                             /*input_prenormalized=*/fuse_norm && layer > 0);
...
                mlp_tail_tp2(a.post_attn_norm, b.post_attn_norm, a.mlp, b.mlp, x, ph, staging,
                             norm_hidden, layer_input_norm(layer + 1));
```

**GDN 分支（`1916` / `1924`）** 同样两处替换，只是第一个调用是 `gdn_mix_tp2`。

> **`layer > 0` 这个条件不能省**：`norm_hidden` 在第一层之前**没有人写过**。
> 若第一层也按「已归一化」处理，`attention_projection` 会把 arena 里的残值当 hidden 用
> —— 静默数值灾难，且 oracle 只会表现为 B 类偏移，极难定位。
> 所以「输入是否已归一化」（`input_prenormalized`，= `fuse_norm && layer > 0`）与
> 「归约是否融合」（`norm_hidden` 非空，= `fuse_norm`）**必须拆成两个独立条件**。

**`mlp_tail_tp2`（`1857-1873`）整段替换**：

```cpp
void TextContext::mlp_tail_tp2(const Tensor* post_norm_0, const Tensor* post_norm_1, const MlpW& m0,
                               const MlpW& m1, std::array<Tensor, 2>& x, Phase ph,
                               const std::array<Tensor, 2>& staging,
                               const std::array<Tensor, 2>& norm_hidden,
                               const std::array<const Tensor*, 2>& next_input_norm) {
    const ExecutionContext& execution       = ec();
    const int T                             = x[0].ne[1];
    const std::array<WorkspaceArena*, 2> ws = workspaces();
    const std::array<const Tensor*, 2> norm = {post_norm_0, post_norm_1};
    const bool fused = norm_hidden[0].data != nullptr;
    std::array<Tensor, 2> h;
    if (fused) {
        // attention_output_projection's reduce already wrote post_attn_norm into this buffer.
        h = norm_hidden;
    } else {
        for (std::size_t r = 0; r < 2; ++r) {
            h[r] = workspace_recipe::post_mixer_hidden<TextConfig>(*ws[r], T);
        }
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::rmsnorm(x[r], *norm[r], kCfg.rms_eps, true, h[r], stream_for(rank));
        });
    }
    // The next layer's input norm rides along on this block's trailing down reduce. `nullptr` on the
    // last layer (final_norm_ is the caller's), and on the unfused route.
    ops::ReplicatedRmsNorm epilogue{};
    const ops::ReplicatedRmsNorm* epilogue_ptr = nullptr;
    if (fused && next_input_norm[0] != nullptr && next_input_norm[1] != nullptr) {
        epilogue = ops::ReplicatedRmsNorm{
            {*next_input_norm[0], *next_input_norm[1]}, norm_hidden, kCfg.rms_eps, true};
        epilogue_ptr = &epilogue;
    }
    Variant::post_mixer(h, {m0.payload, m1.payload}, x, staging, ph, ws, execution, *tp_->events,
                        epilogue_ptr);
}
```

> 注意 `ReplicatedRmsNorm` 的成员顺序是 `{weight, output, eps, unit_offset}`
> （`allreduce_norm.h`，`8bb59c64`），所以聚合初始化写成
> `{ {w0, w1}, norm_hidden, kCfg.rms_eps, true }`。`unit_offset=true` 与
> `1868-1870` / `1625` 现有 `ops::rmsnorm(..., /*unit_offset=*/true, ...)` 一致。

**`attn_mix_tp2`（`1590`）签名与内部三处改动**：

```cpp
void TextContext::attn_mix_tp2(const FullLayerW& w0, const FullLayerW& w1, std::array<Tensor, 2>& x,
                               int fidx, Phase ph, const std::array<Tensor, 2>& staging,
                               const std::array<Tensor, 2>& norm_hidden, bool input_prenormalized) {
    ...
    const bool fuse_output = norm_hidden[0].data != nullptr;   // 站点 A
    ...
    for (std::size_t r = 0; r < 2; ++r) {
        const auto projection = workspace_recipe::text_attention_projection<TextConfig>(...);
        h[r]         = input_prenormalized ? norm_hidden[r] : projection.hidden;   // ← 改 1613
        ...
    }
    if (!input_prenormalized) {                                          // ← 包住 1623-1626
        for_each_rank(execution, [&](int rank) {
            const auto r = static_cast<std::size_t>(rank);
            ops::rmsnorm(x[r], *w[r]->input_norm, kCfg.rms_eps, true, h[r], stream_for(rank));
        });
    }
    ...
    // 1678-1680
    ops::ReplicatedRmsNorm epilogue{};
    const ops::ReplicatedRmsNorm* epilogue_ptr = nullptr;
    if (fuse_output) {
        epilogue = ops::ReplicatedRmsNorm{{*w0.post_attn_norm, *w1.post_attn_norm}, norm_hidden,
                                          kCfg.rms_eps, true};
        epilogue_ptr = &epilogue;
    }
    Variant::attention_output_projection({a[0].view({kShardQSize, T}), a[1].view({kShardQSize, T})},
                                         {*w0.o_proj, *w1.o_proj}, x, staging, ph, ws, execution,
                                         *tp_->events, epilogue_ptr);
```

> 两个标志互相独立：`fuse_output` 控制「归约是否顺便归一化」，`input_prenormalized` 控制
> 「本 stage 入口是否还要自己归一化」。第一层是 `(true, false)`，第 2..n-1 层是 `(true, true)`，
> 非融合路径（`T > 8`）是 `(false, false)`。

`gdn_mix_tp2`（`1683`）改法完全同构，只需注意 `h[r] = control.hidden`（`1700`）、
输入 norm 在 `1713-1716`、归约在 `1852-1854`，以及 **GGML_K/GGUF 分支**（见 §五.1）。

**`text_context.h:398-406` 三个声明同步改**（当前形态见 `398-406`），
新增参数都放末尾、给默认值 `{}` / `= nullptr` 可以让其余调用点零改动：

```cpp
    void attn_mix_tp2(const FullLayerW& w0, const FullLayerW& w1, std::array<Tensor, 2>& x,
                      int index, Phase phase, const std::array<Tensor, 2>& staging,
                      const std::array<Tensor, 2>& norm_hidden, bool input_prenormalized);
    void gdn_mix_tp2(const GdnLayerW& w0, const GdnLayerW& w1, std::array<Tensor, 2>& x, int index,
                     Phase phase, const std::array<Tensor, 2>& staging,
                     const std::array<Tensor, 2>& norm_hidden, bool input_prenormalized);
    void mlp_tail_tp2(const Tensor* post_norm_0, const Tensor* post_norm_1, const MlpW& m0,
                      const MlpW& m1, std::array<Tensor, 2>& x, Phase phase,
                      const std::array<Tensor, 2>& staging,
                      const std::array<Tensor, 2>& norm_hidden,
                      const std::array<const Tensor*, 2>& next_input_norm);
```

（这三个函数只在 `text_context_impl.h` 内部被调用 —— `1893/1901/1916/1924` 四处，
grep 确认无其他调用者，所以**不需要**默认值，直接改签名即可。）

**`layouts_impl.h` 的镜像改动（必须同提交）**：`target_body`（`331-337`）里、三个 stage lambda
调用之前插入与 runtime 同一条门控：

```cpp
    const auto target_body = [&](WorkspaceLayoutBuilder& layout, std::int32_t first,
                                 std::int32_t last, ...) {
        // Fused-norm scratch: mirrors run_layers_tp2's `fuse_norm` buffer, alive for the whole call.
        if (last <= 8) { matrix(layout, DType::BF16, TextConfig::hidden, last); }
        attention_stage(...);
        gdn_stage(...);
        post_mixer_stage(...);
    };
```

放在最前面（而不是某个 stage 的 scope 内），才和 runtime 的「函数级、跨层」寿命一致。
`last` 在 6 个调用点分别是 `chunk(2560)` / `batch` / `aggregate`，与 runtime 的 `T` 精确对应。

**顺带**：`mlp_tail_tp2` 融合后不再调用 `post_mixer_hidden`，但 `layouts_impl.h:327` 仍然规划它。
保留即可（模拟值 ≥ 实际值，安全）；要省这 10 KiB/T 就得再给 recipe 加个 bool 参数，
**不建议**在本项里做——它会把「容量对不上」的排查面扩大一倍。

**按 §6.1 分三步落地时，上面的骨架只需三处开关**：
- S0：`const bool fuse_norm = false;`
- S1：`const bool fuse_norm = T <= 8;`，且调用点写 `/*input_prenormalized=*/false`、
  `layer_input_norm(...)` 恒返回 `{nullptr, nullptr}`（即 `mlp_tail_tp2` 收到空 epilogue）。
- S2：全部打开。
三处都是单行/单表达式改动，`layouts_impl.h` 的镜像只在 S1 起需要（S0 门控恒假 → 不规划）。

---

## 三、逐文件改动清单

约定：`path : 函数 : 位置` → 「需要的代码形状」。行号取自 lab worktree 当前 HEAD（`port/gdn-split10`）。

### 3.1 新增 `include/ninfer/ops/linear_add_norm.h`

**为什么另开一个头而不是改 `linear_add.h`**：`linear_add.h` 被 **19 个** TU include
（`grep -rl "ninfer/ops/linear_add.h"` → 19 个文件，含 `src/targets/**`、`src/ops/**`），
改它会触发近全量重编译；过去两个提交（packet、gdn-split10）都刻意避开了它。
新头只被 `linear_add.cpp` 和两个 `variant.h` 的 `.cpp` 侧 include，重编译面可控。

```cpp
#pragma once

// ninfer::ops - the row-parallel residual GEMM with an RMSNorm epilogue folded into its all-reduce.
//
// A separate header from linear_add.h on purpose: that one is included by 19 translation units,
// and the fused form is only reachable from the three tp2 row-parallel leaves.

#include "ninfer/ops/allreduce_norm.h"
#include "ninfer/ops/linear_add.h"

#include <array>

namespace ninfer::ops {

// Same contract as linear_add_row_parallel(), with `norm` describing the RMSNorm that consumes the
// reduced row. `norm.output[r]` receives the normalized row; the all-reduced sum still lands in
// `residual[r]` exactly as linear_add_row_parallel leaves it.
//
// `norm` is REQUIRED here (no nullptr form): callers that do not want the epilogue keep calling
// linear_add_row_parallel(). Inside, allreduce_sum_rmsnorm() degrades to allreduce_sum() + rmsnorm()
// when the geometry does not admit the fused packet schedule, so the arithmetic is the same either
// way -- but the degradation still *writes* norm.output, so the caller must always supply it.
void linear_add_row_parallel_norm(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                                  const std::array<Tensor, 2>& residual,
                                  const std::array<Tensor, 2>& staging, LinearPolicy policy,
                                  const std::array<WorkspaceArena*, 2>& workspace,
                                  const ExecutionContext& ec, const PeerEvents& events,
                                  const ReplicatedRmsNorm& norm);

// GGML_K / GGUF variant of the GDN output projection (linear_add.h:86-95's tp2 overload), with the
// same epilogue. Declared here rather than in linear_add.h for the reason above.
void ggml_k_gdn_output_norm(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                            const std::array<Tensor, 2>& residual,
                            const std::array<Tensor, 2>& staging, LinearPolicy policy,
                            const std::array<WorkspaceArena*, 2>& workspace,
                            const ExecutionContext& ec, const PeerEvents& events,
                            const ReplicatedRmsNorm& norm);

} // namespace ninfer::ops
```

### 3.2 `src/ops/wrapper/linear_add.cpp`

| 位置 | 改动 |
|---|---|
| `:1` 之后 | 新增 `#include "ninfer/ops/linear_add_norm.h"` |
| `:434` 之后（`linear_add_row_parallel` 的 A16 重载之后、`namespace {` 之前） | 新增 `linear_add_row_parallel_norm` 定义 |
| `:471` 之后（`ggml_k_gdn_output` 的 tp2 重载之后、`} // namespace ninfer::ops` 之前） | 新增 `ggml_k_gdn_output_norm` 定义 |

两处都能直接复用本 TU 匿名 namespace 里的 `validate_add_split_pair`（`:329`）、
`validate_add_split_residency`（`:345`）、`issue_fused_rank`（`:366`）、
`issue_plain_rank`（`:392`）、`block_gdn_output`（`:437`）—— **一行都不用复制**：

```cpp
void linear_add_row_parallel_norm(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                                  const std::array<Tensor, 2>& residual,
                                  const std::array<Tensor, 2>& staging, LinearPolicy policy,
                                  const std::array<WorkspaceArena*, 2>& workspace,
                                  const ExecutionContext& ec, const PeerEvents& events,
                                  const ReplicatedRmsNorm& norm) {
    validate_policy(policy);
    validate_add_split_pair(x, w, ec);
    validate_add_split_residency(x, w, residual, ec);

    std::array<Tensor, 2> target{residual[0], residual[1]};
    std::array<Tensor, 2> scratch{staging[0], staging[1]};
    detail::for_each_rank(ec, [&](int rank) {
        const auto slot      = static_cast<std::size_t>(rank);
        const cudaStream_t s = ec.dev[slot]->stream;
        if (rank == 0) {
            issue_fused_rank(x[slot], w[slot], target[slot], scratch[slot], policy,
                             workspace[slot], s);
        } else {
            issue_plain_rank(x[slot], w[slot], target[slot], policy, workspace[slot], s);
        }
    });
    allreduce_sum_rmsnorm(target, staging, norm, ec, events);   // ← 唯一的差别在这一行
}
```

```cpp
void ggml_k_gdn_output_norm(const std::array<Tensor, 2>& x, const std::array<Weight, 2>& w,
                            const std::array<Tensor, 2>& residual,
                            const std::array<Tensor, 2>& staging, LinearPolicy policy,
                            const std::array<WorkspaceArena*, 2>& workspace,
                            const ExecutionContext& ec, const PeerEvents& events,
                            const ReplicatedRmsNorm& norm) {
    validate_add_split_pair(x, w, ec);
    validate_add_split_residency(x, w, residual, ec);
    detail::for_each_rank(ec, [&](int rank) {
        Tensor target = residual[rank];
        block_gdn_output(x[rank], w[rank], target, rank == 0, policy, workspace[rank],
                         ec.dev[rank]->stream);
    });
    allreduce_sum_rmsnorm(residual, staging, norm, ec, events);   // ← 原 :470 是 allreduce_sum
}
```

> **注意**：`allreduce_sum_rmsnorm` 的 `buffer` 参数必须和 `residual` **同址**，
> 且 `norm.output` 不得与 `buffer` 重叠（`allreduce_norm.h` 的注释要求）。
> `norm_admits` 只校验对齐/shape，**不校验重叠**（见 §五.4）。

### 3.3 `src/targets/qwen3_6_27b/impl/variant.h`

| 行 | 位置 | 改动 |
|---|---|---|
| `:7` | include 区 | `ninfer/ops/allreduce.h` → 追加 `#include "ninfer/ops/allreduce_norm.h"`（或直接替换，它包含前者） |
| `:150-156` | tp2 `attention_output_projection` | 末尾加 `, const ops::ReplicatedRmsNorm* norm = nullptr` |
| `:181-187` | tp2 `gdn_output_projection` | 同上 |
| `:227-232` | tp2 `post_mixer` | 同上 |

默认值 `= nullptr` 让 **MTP 那三个调用点（`variant.cpp` 的 `mtp_post_mixer` / MTP o_proj）零改动**，
且语义就是「不融合」。这是本项最重要的一条零风险设计：新增参数不改变任何既有调用者的行为。

### 3.4 `src/targets/qwen3_6_27b/impl/variant.cpp`

| 行 | 函数 | 改动形状 |
|---|---|---|
| `:658-665` | `attention_output_projection` | 签名加 `const ops::ReplicatedRmsNorm* norm`；函数体改成 `norm == nullptr ? ops::linear_add_row_parallel(...) : ops::linear_add_row_parallel_norm(...)` |
| `:725-735` | `gdn_output_projection` | 同上，但**两个分支都要处理**（GGML_K/GGUF → `ggml_k_gdn_output_norm`，否则 `linear_add_row_parallel_norm`） |
| `:770-789` | `post_mixer` | 尾部 `linear_add_row_parallel`（`:787-788`）按 `norm` 二选一 |

```cpp
void Variant::attention_output_projection(const std::array<Tensor, 2>& attention,
                                          const std::array<Weight, 2>& weight,
                                          const std::array<Tensor, 2>& residual,
                                          const std::array<Tensor, 2>& staging, qwen3_6::TextPhase,
                                          const std::array<WorkspaceArena*, 2>& workspace,
                                          const ExecutionContext& ec, const ops::PeerEvents& ev,
                                          const ops::ReplicatedRmsNorm* norm) {
    if (norm == nullptr) {
        ops::linear_add_row_parallel(attention, weight, residual, staging, text_policy(weight[0]),
                                     workspace, ec, ev);
        return;
    }
    ops::linear_add_row_parallel_norm(attention, weight, residual, staging, text_policy(weight[0]),
                                      workspace, ec, ev, *norm);
}
```

```cpp
void Variant::gdn_output_projection(const std::array<Tensor, 2>& hidden,
                                    const std::array<Weight, 2>& weight,
                                    const std::array<Tensor, 2>& residual,
                                    const std::array<Tensor, 2>& staging, qwen3_6::TextPhase,
                                    const std::array<WorkspaceArena*, 2>& workspace,
                                    const ExecutionContext& ec, const ops::PeerEvents& ev,
                                    const ops::ReplicatedRmsNorm* norm) {
    const bool ggml = weight[0].qtype == QType::GGML_K || weight[0].qtype == QType::GGUF;
    if (ggml) {
        if (norm == nullptr) {
            ops::ggml_k_gdn_output(hidden, weight, residual, staging, text_policy(weight[0]),
                                   workspace, ec, ev);
        } else {
            ops::ggml_k_gdn_output_norm(hidden, weight, residual, staging, text_policy(weight[0]),
                                        workspace, ec, ev, *norm);
        }
        return;
    }
    if (norm == nullptr) {
        ops::linear_add_row_parallel(hidden, weight, residual, staging, text_policy(weight[0]),
                                     workspace, ec, ev);
        return;
    }
    ops::linear_add_row_parallel_norm(hidden, weight, residual, staging, text_policy(weight[0]),
                                      workspace, ec, ev, *norm);
}
```

`post_mixer`（`:770`）只改尾部两行：

```cpp
    if (norm == nullptr) {
        ops::linear_add_row_parallel(activation, pair_of(w[0]->down, w[1]->down), residual, staging,
                                     text_policy(w[0]->down), workspace, ec, ev);
    } else {
        ops::linear_add_row_parallel_norm(activation, pair_of(w[0]->down, w[1]->down), residual,
                                          staging, text_policy(w[0]->down), workspace, ec, ev,
                                          *norm);
    }
```

### 3.5 `src/targets/qwen3_6_35b_a3b/impl/variant.h` + `.cpp`（接口同步，行为不变）

35B-A3B **没有 TP2 路径**：`variant.h:47-56` 的注释写明「every one throws」，
`variant.cpp:344-431` 的每个 tp2 叶子都是 `reject_tensor_parallel(...)`。
但它和 27b 共用一个 `targets/qwen3_6/impl/runtime`，所以**签名必须逐字一致**，
否则 `text_context_impl.h` 在同一份源码下无法为两个 Package 各编一份。

| 文件 | 行 | 改动 |
|---|---|---|
| `variant.h` | `:8` | 加 `#include "ninfer/ops/allreduce_norm.h"` |
| `variant.h` | `:60-66` / `:82-88` / `:95-100` | 三个 tp2 叶子末尾加 `, const ops::ReplicatedRmsNorm* norm = nullptr` |
| `variant.cpp` | `:344-351` / `:370-376` / `:385-391` | 形参列表末尾加 `const ops::ReplicatedRmsNorm*`（**不带形参名**，与既有风格一致），函数体保持 `reject_tensor_parallel(...)` |

> 35b 的 `PostMixerWeights` 是 `SparseMoePayload`（`variant.h:29`），MoE 的 down 归约也不在
> `post_mixer` 里 —— 所以即使将来 35b 有了 TP2，这三个叶子的语义也不能照抄 27b。
> 本次只是**接口占位**，不要顺手实现。

### 3.6 `src/targets/qwen3_6/impl/runtime/text_context.h`

`:398-406` 三个声明（`attn_mix_tp2` / `gdn_mix_tp2` / `mlp_tail_tp2`）按 §2.6 的形态改。
`run_layers_tp2`（`:405-406`）**签名不变**（`sink` 默认参数保留）。

### 3.7 `src/targets/qwen3_6/impl/runtime/text_context_impl.h`

| 位置 | 改动 |
|---|---|
| `:1590` `attn_mix_tp2` 签名 | 末尾加 `, const std::array<Tensor, 2>& norm_hidden, bool input_prenormalized` |
| `:1613` | `h[r] = projection.hidden;` → `h[r] = input_prenormalized ? norm_hidden[r] : projection.hidden;` |
| `:1623-1626` | 外面包 `if (!input_prenormalized) { ... }` |
| `:1678-1680` | 传 `epilogue_ptr` 给 `Variant::attention_output_projection` |
| `:1683` `gdn_mix_tp2` 签名 | 同上 |
| `:1700` | `h[r] = control.hidden;` → `h[r] = input_prenormalized ? norm_hidden[r] : control.hidden;` |
| `:1713-1716` | 包 `if (!input_prenormalized)` |
| `:1852-1854` | 传 `epilogue_ptr` 给 `Variant::gdn_output_projection` |
| `:1857-1873` `mlp_tail_tp2` | 整段按 §2.6 替换 |
| `:1875-1878` `run_layers_tp2` 头部 | 插入 `fuse_norm` / `norm_hidden` / `layer_input_norm` |
| `:1893` / `:1901` | full 分支两次调用加实参（`input_prenormalized = fuse_norm && layer > 0`） |
| `:1916` / `:1924` | GDN 分支两次调用加实参 |

### 3.8 `src/targets/qwen3_6/impl/runtime/layouts_impl.h`

| 位置 | 改动 |
|---|---|
| `:331` `target_body` 开头 | 插 `if (last <= 8) { matrix(layout, DType::BF16, TextConfig::hidden, last); }` |

只有这一行。**漏掉它 = 运行时 `std::bad_alloc`**（§2.5 证据 3）。

### 3.9 改动量估计

| 文件 | 净增行 |
|---|---:|
| `include/ninfer/ops/linear_add_norm.h`（新） | +45 |
| `src/ops/wrapper/linear_add.cpp` | +40 / −2 |
| `qwen3_6_27b/impl/variant.h` | +3 |
| `qwen3_6_27b/impl/variant.cpp` | +34 / −8 |
| `qwen3_6_35b_a3b/impl/variant.h` | +2 |
| `qwen3_6_35b_a3b/impl/variant.cpp` | +3 |
| `text_context.h` | +6 / −3 |
| `text_context_impl.h` | +48 / −14 |
| `layouts_impl.h` | +1 |
| **合计** | **≈ +180** |

（比 `plan-rmsnorm-fusion.md` 的 +303 少，因为内核与 API 已在 `8bb59c64` 落地。）

---

## 四、两个站点的 norm 权重来源（确切表达式）

### 4.1 站点 A：`o_proj` 归约 → `post_attn_norm`

| 层类型 | 权重成员 | 确切表达式 |
|---|---|---|
| full | `FullLayerW::post_attn_norm`（`text_context.h:105`） | `{*w0.post_attn_norm, *w1.post_attn_norm}` |
| GDN | `GdnLayerW::post_attn_norm`（`text_context.h:115`） | `{*w0.post_attn_norm, *w1.post_attn_norm}` |

绑定处：`text_context_impl.h:294`（`out.post_attn_norm = &source.post_attention_norm;`，rank 0 本层）
与 `:304`（GDN 同理）；peer 侧在 `:310` 与 `:322`。
调用现场：`attn_mix_tp2` 的 `w0`/`w1` 就是 `full_.at(fidx)` / `full_peer_.at(fidx)`
（`run_layers_tp2:1881-1884`），`gdn_mix_tp2` 的是 `gdn_.at(gidx)` / `gdn_peer_.at(gidx)`（`:1904-1907`）。
**所以站点 A 的权重完全在本函数作用域内，不需要跨层传递。**

`eps`：`kCfg.rms_eps`（`= TextConfig::rms_epsilon = 1.0e-6F`，`config.h:36`）。
`unit_offset`：**`true`**（现有调用 `1625` / `1715` / `1870` 都是 `/*unit_offset=*/true`）。
目标缓冲：`norm_hidden`（= `h`），shape `[5120, T]`、dtype BF16，由
`workspace_recipe::post_mixer_hidden` 分配（`workspace_recipe.h:145-148`）。

### 4.2 站点 B：`post_mixer` 尾部 down 归约 → **下一层** `input_norm`

- 权重的**所属层是 `layer + 1`**，不是当前层。
- `layer + 1` 可能是 full 也可能是 GDN，所以必须按 `layer+1` 重新分派索引：

```cpp
    // text_context_impl.h, run_layers_tp2 内
    const auto layer_input_norm = [&](int layer) -> std::array<const Tensor*, 2> {
        if (layer >= kCfg.n_layers) { return {nullptr, nullptr}; }       // 最后一层 → 不融合
        if (ModelConfig::is_full(layer)) {
            const auto i = static_cast<std::size_t>(ModelConfig::full_idx(layer));
            return {full_.at(i).input_norm, full_peer_.at(i).input_norm};
        }
        const auto i = static_cast<std::size_t>(ModelConfig::gdn_idx(layer));
        return {gdn_.at(i).input_norm, gdn_peer_.at(i).input_norm};
    };
    ...
    mlp_tail_tp2(..., layer_input_norm(layer + 1));
```

`ModelConfig::is_full` / `full_idx` / `gdn_idx` 是 `constexpr`（`text_context.h:59`、`:67`、`:71`），
`full_` / `gdn_` 是成员容器，既有用法 `.at(fidx)`（`:1882`）。

- 目标缓冲：**同一个 `norm_hidden`**（§2.3 的复用证明），shape `[5120, T]` BF16。
- **最后一层（`layer == n_layers - 1`）→ 传 `{nullptr, nullptr}` → 不融合**，保留调用方在循环外做的
  独立 `ops::rmsnorm`：
  - prefill `:2150-2151`（写 `xf[r]`，`xf` 在 `:2144-2149` 分配，位于 `run_layers_tp2` **之后**）
  - ordinary decode `:2377-2378`
  - MTP/dflash verify `:2763-2764`
  最终 norm 权重是 `final_norm_` / `final_norm_peer_`（`:278` / `:286`，类型 `const Tensor*`，`text_context.h:567` / `:578`）。
- **不建议**把最后一层也接上：目标 `xf[r]` 要么是 `prefill_hidden_`（外部缓冲），要么是
  `run_layers_tp2` 返回后才知道的 arena 地址，需要把分配前移 + 把 `final_norm_` 递进
  `run_layers_tp2`。收益是 1/64 的站点 B（64 层里最后一层），**收益/风险比极差**。
  MTP 的 `mtp_post_mixer`（`variant.cpp:886`）同理 —— 它的尾部 reduce 之后跟的是
  `mtp.final_norm`，本次**不接**。

### 4.3 两个站点的权重不能被合并的原因

站点 A 用 `post_attn_norm`（本层），站点 B 用 `input_norm`（下一层）。**没有任何一层里二者是同一个
Tensor 对象**（绑定源分别是 `source.post_attention_norm` 与 `source.input_norm`，
`text_context_impl.h:294` vs `:291`），所以同一个 `ReplicatedRmsNorm` 只能服务一个站点，
`attn_mix_tp2` / `mlp_tail_tp2` 各自构造一个。

---

## 五、风险清单

按「会不会静默算错」排序，不是按「会不会崩」。

### 5.1 `gdn_output_projection` 的 GGML_K/GGUF 分支（**最高风险，必须一起改**）

`src/targets/qwen3_6_27b/impl/variant.cpp:731-735`
```cpp
    if (weight[0].qtype == QType::GGML_K || weight[0].qtype == QType::GGUF) {
        ops::ggml_k_gdn_output(hidden, weight, residual, staging, text_policy(weight[0]),
                               workspace, ec, ev);
        return;
    }
```
`ops::ggml_k_gdn_output` 的 tp2 重载在 `src/ops/wrapper/linear_add.cpp:457-471`，
结尾是裸的 `allreduce_sum(residual, staging, ec, events)`（`:470`），**不认识 norm**。

**必须融合，不能「这条路上不融合就好」。** 原因：
`mlp_tail_tp2` 在 `fused == true` 时**跳过自己的 rmsnorm**（§2.6），它假定
`norm_hidden` 已经被 o_proj 归约写好。若 GGML_K 分支不写，`norm_hidden` 里是从未初始化过的
arena 残值 → `post_mixer` 的 gate_up 吃进一堆垃圾 → **静默错误输出**，而且 oracle 只会表现为
B 类偏移，排查成本极高。

所以：**「叶子接受 `norm != nullptr` 就必须亲自兑现它，否则必须抛错」** 是本项的不变式。
三个叶子的每个内部分支都要覆盖：
- `attention_output_projection`：单一分支 → 已覆盖。
- `gdn_output_projection`：**两个分支**（GGML_K/GGUF、其余）→ 都要覆盖（§3.4 的代码）。
- `post_mixer`：单一分支（`down` 的 `linear_add_row_parallel`）→ 已覆盖。

**「同一层前后两次 collective 走不同协议会不会 epoch 错乱」——不会。** 分三层论证：

1. **同一次 collective 内两卡不会走不同协议。**
   `validate_add_split_pair` 硬校验 `w[0].qtype == w[1].qtype`（`linear_add.cpp:336-338`）；
   `norm_admits` 与 `packet_admits` 都是**同时遍历两卡**再返回（`allreduce.cu` 的
   `for (int rank = 0; rank < 2; ++rank)`，`norm_admits` 在 512-541、`packet_admits` 在 404-412），
   是**全局谓词**而不是每卡独立判断。所以不存在「rank0 用 packet 内核自旋、rank1 走 events」的
   分裂执行 —— 那会变成 15e9 周期后 `__trap()`（`kPacketSpinCycles`，`allreduce.cu:365`），
   是 GPU fault 不是挂起。**这条不变式必须保住：将来不要往 `norm_admits` 里加任何 rank-local 条件。**
2. **两个协议不共享任何状态。** packet 路由用 `direct->packets[r]` + `direct->packet_epochs[r]`；
   events 路由用 `inputs_ready[r]` / `pull_done[r]`（`allreduce.h:147-155`、
   `allreduce_sum` 的三相位注释 `allreduce.cu:11-24`）。互不读写。
3. **两个 packet 变体共享 epoch 计数器，且它们的翻转集合逐位相同。**
   这是我原本最担心的点，结论是**兼容**：
   - 普通 packet 内核：`epochs[blockIdx.x] = epoch ^ 1U`（`allreduce.cu:399`），
     `blockIdx.x` 覆盖 `packs/64` 个块，`packs = numel/8`。对 `[5120,T]`：`packs = 640T`
     → 翻转 `epochs[0 .. 10T-1]`。
   - 融合内核：`epochs[row * kNormEpochPerRow + tid] ^= 1U`（`tid < kNormEpochPerRow`），
     `kNormEpochPerRow = 2560/256 = 10`，`row < T` → 翻转 `epochs[0 .. 10T-1]`。
   - **集合完全相同**，且 `kNormEpochPerRow * kNormMaxColumns = 80 == kPacketBlocks`
     由 `static_assert` 钉死（`allreduce.cu:446-447`）。
   - 空槽标记的**字节表示也相同**：普通内核写 8 个 `0xffff`（`allreduce.cu:379-386`），
     融合内核 retire 时写 `store_volatile_u32(slot, 0xffffffffU)`（4 字节两个 `0xffff`，
     `allreduce.cu:487`）—— 逐字节一致。
   - 结论：**同一轮里 A 站点走融合、B 站点走普通 packet，epoch 奇偶保持一致**。
     MTP 层（`mtp_post_mixer`，`variant.cpp:886`）不接线也不会污染。
   
   ⚠️ 但反过来**不成立**：`[5120,5]` 的融合调用翻转 `0..49`，`[5120,1]` 的普通调用只翻转 `0..9`。
   只要**两卡执行完全相同的调用序列**（现在就是这样：TP2 的两个 rank 走同一份 host 代码），
   每个计数器的奇偶就一致。**任何「按 rank 分叉走不同路径」的改动都会破坏它**，
   且症状是随机错值而不是崩溃。

### 5.2 prefill `T = 2560` 的退化路径（**确认存在，且不会崩**）

`src/ops/common/allreduce.cu:816-829`（`8bb59c64`）：
```cpp
    if (direct == nullptr || !norm_admits(buffer, norm) || !packet_admits(buffer, buffer[0].numel())) {
        allreduce_sum(buffer, staging, ec, events);
        const CurrentDeviceGuard guard;
        for (int rank = 0; rank < 2; ++rank) {
            CurrentDeviceGuard::set(ec.dev[rank]->device);
            Tensor normalized = norm.output[rank];
            rmsnorm(buffer[rank], norm.weight[rank], norm.eps, norm.unit_offset, normalized,
                    ec.dev[rank]->stream);
        }
        return;
    }
```
- **退化条件存在**：`norm_admits` 要求 `buffer[0].ne[1] >= 1 && <= 8`
  （`allreduce.cu:436`、`515-517`），`T=2560` 直接不满足 → 退化。
- **退化路径的算术与今天完全相同**：`allreduce_sum` + `rmsnorm`，与
  `mlp_tail_tp2:1870` 现在做的事一字不差（同 eps、同 `unit_offset=true`、同 row-major BF16）。
  → 即使不门控 prefill，**oracle 也应当逐位不变**。这是很强的安全边际。
- **缺少 `norm.output` 校验会不会崩？** 分两问：
  - 融合内核那条路：`norm_admits` 只检查 `output.ne[d] == buffer.ne[d]`（`:538-540`）和
    「三个地址按位或之后 16 字节对齐」（`:528-531`）。一个 **shape 正确但 `data == nullptr`**
    的 `output` 能通过这两条（nullptr 或上非空地址仍可能对齐），然后内核往 nullptr 写
    → **segfault**。这是 `8bb59c64` 里一个**真实缺口**，建议补一行：
    `if (norm.output[r].data == nullptr) { return false; }`（放在 `:528` 之前）。
  - 退化那条路：`rmsnorm_impl` 会校验
    `require_same_shape(x, out, "out")`（`src/ops/wrapper/rmsnorm.cpp:63`）和
    `out.data == nullptr → throw`（`:79-82`），所以空 `output` 会得到
    `std::invalid_argument("rmsnorm: x/out shapes must match")` —— **受控抛错，不是崩溃**。
  - 结论：**退化路径本身是安全的；缺口在融合路径的 `data == nullptr` 上**，与本项接线无关
    （接线保证永远传真 buffer），但既然要动这个文件，顺手补掉。
- **本方案的建议仍然是门控**（`fuse_norm = T <= 8`）：不门控要给 prefill 多规划
  `10 240 × 2560 = 25.0 MiB/卡`，换来的收益是 **0**（该几何永远走不进融合内核）。
  门控后 prefill 的**指令序列与今天逐字相同**，oracle 的 prefill 段天然不可变。

### 5.3 单缓冲复用依赖 stream 顺序（**可接受，但必须写进注释**）

§2.3 的证明成立的前提是：**同一 rank 的 `swiglu` 与 `down` 归约在同一个 stream 上，且按该顺序发射**。
现状满足（`Variant::post_mixer` 内 `linear_swiglu_column_parallel` 在
`variant.cpp:784-786`，`linear_add_row_parallel` 在 `:787-788`，同一个 `ec.dev[rank]->stream`）。
风险在于**将来有人把 post_mixer 的两半拆到两条 stream**，那时 `B` 会被提前覆盖。
→ 在 `run_layers_tp2` 的 `norm_hidden` 声明处写一句「alive across the layer; the aliasing between
the post_attn_norm destination and the next layer's input_norm destination is only safe because the
two writes are stream-ordered behind their reads」。
**若评审不愿承担这个隐式约束**，改用 2 块 buffer（+10 KiB/T），其余不变。

### 5.4 `norm.output` 与 `buffer` 的重叠没有被检查

`allreduce_norm.h` 的注释要求「`output[r]` … must not overlap `buffer[r]`」，
但 `norm_admits` 里**没有 overlap 检查**（`allreduce.cu:512-541` 只查 dtype/contiguous/对齐/shape）。
本方案不会触发（`B` 与 `x` 是两个独立 `alloc`），但这是同一类「静默错误」的入口，
建议在 `linear_add_row_parallel_norm` 里加一句 `require(!overlaps(...))`（`overlaps` 已存在于
`linear_add.cpp:62-66`，只需提到外层或复制一份）。

### 5.5 第一层不能当成「已归一化」（**已在本方案内解决，但极易写错**）

见 §2.3 结尾与 §2.6：`input_prenormalized = fuse_norm && layer > 0`。
若漏掉 `layer > 0`，第一层的 `attention_projection` 会读到 arena 残值。
症状：oracle 全错但程序不崩、loss 不 NaN（BF16 残值大概率是有限数）。
**这一条是本项最容易犯、也最难从日志看出来的错误。**

### 5.6 数值：融合内核的求和顺序必须等于 `rmsnorm_generic_kernel`

`8bb59c64` 的提交说明（`git show --stat 8bb59c64`）写明：融合内核**不复用** packet 的求和顺序，
而是先把和写回 `buffer`，再按 `rmsnorm_generic_kernel` 的顺序（256 线程、stride-256 标量累加、
共享内存二叉树、`rsqrtf(sum/d + eps)`）从那一行**重算**行内平方和
（`allreduce.cu:441-448`、`482-511`）。这是有意的、也是唯一的正确做法。
残余风险：
- `unit_offset` 的 `w+1` 在 FP32 里做（`allreduce.cu:507-509`），与 `rmsnorm` 的 tp1 内核
  是否**逐位**一致 —— 静态无法确认（我没读 `rmsnorm_generic_kernel` 的 `unit_offset` 实现）。
- 因此**只能靠 oracle 判定**，且必须分两步（先站点 A、再站点 A+B，见 §6），
  否则失败时分不清是内核数值问题还是接线问题。

### 5.7 `std::array` 是 **rank 序**，不是「本卡/对端」序

`ReplicatedRmsNorm::weight = {w0, w1}` 必须严格是 `{rank0 的绑定, rank1 的绑定}`：
- 站点 A：`w0` / `w1` 来自 `full_.at(fidx)` / `full_peer_.at(fidx)`
  （`run_layers_tp2:1881-1884`，peer 绑定在 `text_context_impl.h:283-325`）。
- 站点 B：`layer_input_norm` 返回 `{full_.at(i).input_norm, full_peer_.at(i).input_norm}`。
写反了会让融合内核在 rank1 的 GPU 上解引用 rank0 的指针 —— UVA 下**不会立刻崩**，
只会读到另一张卡的显存（如果可访问）或触发 illegal address。
`packet_admits`/`norm_admits` 的对齐检查也拦不住。
→ 建议在 `Variant::attention_output_projection` 等处的注释里点明这一点（`post_mixer` 的
`{m0.payload, m1.payload}` 已经是这个约定，可以引用它）。

### 5.8 未接线的两个 collective（可接受，但要记录）

- MTP 层：`mtp_post_mixer`（`variant.cpp:886-931`）与 MTP 的 o_proj
  （`text_context_impl.h:2560-2562`）**不接线**。后者走的甚至是 `ops::linear_row_parallel` +
  独立的 `residual_add`（`:2563-2575`），`linear_add.h` 的 `linear_add_row_parallel` 它根本不用
  —— 所以本项的「只改一处签名覆盖全部站点」对 MTP **不成立**，这是既有的路径分裂，不是新引入的。
  MTP 只 1 层，收益损失极小，保持不动。
- 最后一层：站点 B 关闭（§4.2）。损失 1/64 的站点 B。

### 5.9 workspace 规划漏改 = `std::bad_alloc`

见 §2.5 证据 3（`src/core/arena.cu:207-208`）。**`layouts_impl.h` 的那一行是本项唯一
「漏了就必崩」的改动**，值得单独 commit。

### 5.10 性能可能不达标（与 `plan-rmsnorm-fusion.md` §七.2 同）

折算后预期 +0.79~0.86%（两点全做），门槛 `DECODE_MIN = 136.272`（相对 135.594 的 +0.5%）。
本项**新增**的一个不确定因素：单缓冲复用把两次归约的 norm 写合并到同一地址，
理论上不改变发射次数，但会让 L2 上的写-读局部性变好/变差，方向未知。不影响判据，只是提醒。

---

## 六、验收方式

### 6.1 三步走（每步都必须过 oracle 才进下一步）

| 步 | 内容 | 期望 |
|---|---|---|
| **S0** | 只加 API + 三个叶子透传参数，`run_layers_tp2` 里 `fuse_norm = false` 硬编码 | oracle **逐位一致**；性能 ±0.2%；`workspace.capacity` **不变** |
| **S1** | 打开站点 A（`norm_hidden` 非空 → o_proj 归约带 epilogue；`input_prenormalized` 恒 `false`；`mlp_tail_tp2` 走 `fused` 分支但 `next_input_norm` 恒 `nullptr`） | oracle **逐位一致**；`capacity` = 基线 + `10 240 × T` |
| **S2** | 打开站点 B（`layer_input_norm(layer+1)`） | oracle **逐位一致**；性能测量，判据 +0.5% |

S0 是回归护栏：证明新参数没有被任何既有调用者接住。
S1 单独跑的意义：把「内核数值」与「跨层 handoff」两类失败分开。

### 6.2 判据（来自 `packet-allreduce.md:34-37` 与 `baseline-summary.md:47`）

- `token_ids(513)` 的 `sha1` **必须** = `06287348665d3648`
- `acc / drf / rd` = **406 / 421 / 106**
- 三条同时满足才算 A 类通过。任何一条不符 → 停止，不硬推（`queue-v2.md:24` 的硬否决）。

### 6.3 不做全量测量的前置确认（省掉一轮编译+测量）

1. **容量自证**：S1/S2 之后，启动日志/统计接口里的
   `ArenaMemorySummary{workspace.capacity(), work.used(), work.peak_used()}`
   （`program_impl.h:3691-3694`）必须显示 `capacity` 恰好增加 `10 240 × T`
   （decode 用 T = batch 或 1+sig）。**不等**就说明 `layouts_impl.h` 的门控写错了
   —— 这一步能在跑 benchmark 之前抓到最常见的错误。
2. **接线自证（负对照）**：把 `epilogue.eps` 临时改成 `1.0F`（或把 `unit_offset` 改成 `false`），
   重跑 oracle。**必须变成非 `06287348665d3648`**。
   为什么必须做：**一个「从未执行的融合」与「执行且正确」的融合，在 oracle 上表现完全相同。**
   不做负对照，"oracle 逐位一致" 可能只证明代码没跑到。这是本项唯一能证明「路径真的生效」的
   廉价手段，成本 = 一次编译 + 一次 oracle（不做性能测量）。
3. **prefill 不变性自证**：门控后 prefill 不进融合，可用一次 prefill-only 的
   hidden/logits 对拍确认。若连 prefill 都变了，说明 `fuse_norm` 的门控条件写错
   （例如用了 `T` 之外的量）。
4. **前缀一致性**：先只跑 `--limit 32` 之类的短前缀，确认前 32 个 token 逐位一致，
   再跑 513。短跑能把「第 1 层」的错误（§5.5）与「深层累加」的错误分开。

### 6.4 需要一起提交/记录的东西

- `layouts_impl.h` 的那一行**单独一个 commit**（它是「漏了就崩」的改动的唯一标记）。
- 在 `plan-rmsnorm-fusion.md` 末尾追加一行指向本文件，避免两份方案的 scope 方案冲突
  （旧文 §七.4 与 §八.3-4 仍可作为执行顺序参考，本文替换其「改动清单」与「难点」两节）。
