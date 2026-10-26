# 快照保留优化清点（对照 fork）

状态：**完成**（第一节 22 条，第二~五节已补齐；代理在中断处被回收，第二~五节由主代理综合）

> 任务：从快照文档中抽出所有被描述为 "retained / 保留 / 采用" 的改动，逐条在 fork 中验证是否已有。
> 只读。不改源码、不编译、不跑 benchmark。
>
> 路径简写：
> - `SNAP` = `_scratch/ref-v100x2/ninfer-V100X2-master`
> - `FORK` = `.agents/worktrees/port-packet-allreduce`
>
> 重要前提（已核实）：**fork 的基线不是快照**。fork 来自一个更新的上游 master
> （有 `dflash2` / `candidate_selector` / `context_kv_materialize` / `dynamic_grouped_conv` /
> `kv_cache` / `linear_topk` / `rmsnorm_rope` / `softmax_attention` 等目录，快照没有；
> 快照的 `dflash_conv.cu` / `dflash_selector.cu` 在 fork 里被 `candidate_selector` +
> `dynamic_grouped_conv` 取代）。所以"fork 没有"既可能是"没移植"，也可能是"上游重构掉了"。
> 判定按**机制**是否存在，不按文件名。
>
> 验证手段：`diff -rq --strip-trailing-cr FORK/src SNAP/src`
> → 204 个文件内容不同，40 个仅快照有，114 个仅 fork 有。绝大多数文件在两棵树里逐字节相同，
> 因此下面的判定以实际 diff 为准，不是靠名字猜。

---

## 一、总表

按"文档声称的量级从大到小"排序。
量级口径列区分 **倍率（×）** 与 **阶梯百分比（Δ%，顺序叠加、非独立归因）**。

### A. 默认路径（TP2，`--volta-long-attention --volta-fast-decode` 之外的常开项）

| # | 改动名（原文措辞） | 文档出处 | 文档声称的量（原文照抄） | 量级口径 | TP4 专属 | 快照实现位置 | fork 是否已有 | 判定 |
|---|---|---|---|---|---|---|---|---|
| A1 | "W8 vocabulary head has an exact load-time code/scale permutation, without requantization or a second resident copy" | `docs/performance.md:937` | 无量（定性："exact…without requantization or a second resident copy"） | 无语义量化 | 否（TP2/TP4 都有，248320/124160 头） | `SNAP/src/ops/linear/w8/w8_prepack_sm70.cu:239`（`w8_prepack_qpn_sm70`）、`.../w8_prepack_sm70.cu:268`（`launch_w8_prepacked_sm70`）、调用点 `SNAP/src/targets/qwen3_6_27b/impl/load/bindings.cpp:1278` | **否**：fork 的 `src/ops/linear/w8/` 下完全没有 `w8_prepack_sm70.*`（`diff` 显示仅快照有）；`bindings.cpp` 只 prepack FP8/NVFP4（`FORK/.../bindings.cpp:1326`），W8 分支不存在 | **缺失** |
| A2 | 输出头 248320 行实际走哪条 kernel（回答提问，非独立优化） | `docs/performance.md:569`（"The target vocabulary head retains its earlier accumulation route"） | 无量 | — | 否 | 快照：prepack 后 `layout==VoltaQpnPrepacked` → `launch_w8_prepacked_sm70`；TP2 每 rank n=124160，`mtp = (n<=17408)` 为假，故 t==1 → `w8_qpn_simt_kernel<1>`、t==2 → `<2>`、t∈3..8 → `w8_qpn_prepacked_kernel<4>`、t>8 → `launch_w8_volta_mma`（`w8_prepack_sm70.cu:268-308`） | **否**：fork 的 n=124160 走 `select_w8_tp2_shard_launch` → t≤4 `launch_w8_simt_r8_c4`、t≤16 `launch_w8_simt_r8_c8`，否则 `launch_w8_mma_r64_c128`（`FORK/src/ops/linear/w8/w8_dispatch.cpp:14-34`） | **缺失**（配置口径） |
| A3 | "GDN output activation staging"（原文另述："GDN output N5120/K3072/T1..8 converts BF16 activation to the same FP16 MMA operands once instead of inside every consumer CTA"） | `docs/performance.md:559`（阶梯 178.213 行）、`docs/performance.md:601-602` | 无独立量（含在 178.213 组合里） | 阶梯 Δ%（非独立归因） | 否（TP2 GDN 输出 n=5120,k=3072） | `SNAP/src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cu:24-32`（`gdn_output = n==5120 && k==3072 && tokens 1..8`）、`...:43`（`nvfp4_volta_qpn_cast_activation`）、`...:50`（`launch_nvfp4_volta_qpn_precast`）；分发 `SNAP/src/ops/linear/nvfp4/nvfp4_dispatch.cpp:72`、`SNAP/src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu:120` | **否**：fork 的 `nvfp4_launch.h` 无 `launch_nvfp4_volta_qpn_precast` / `nvfp4_volta_qpn_cast_activation` / `nvfp4_volta_qpn_precast_supported`（`diff` 显示这 7 行仅快照有）；`nvfp4_dispatch.cpp` 既无 precast 也无 CUTLASS 分支 | **缺失** |
| A4 | "precast selected narrow NVFP4 activations"（TP4 文中同一机制的 TP4 侧） | `docs/performance.md:935` | 无量 | — | 部分（TP4 另有 `volta_tp4.cu` 调用点） | `SNAP/src/ops/attn_input_proj/volta_tp4.cu:87`、`SNAP/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_sm70.cu:96,153-155` | **否**（同 A3） | **缺失** |
| A5 | "GDN gated-norm BF16-rounded FP16 image" / "GDN gated-norm output image" | `docs/tp4.md:279-284`、`docs/tp4.md:283`（表内 +0.94%/+0.67%） | **+0.94% / +0.67%**（long/fast 与 flags-off 两档，顺序测量） | 阶梯 Δ% | **是**（`tp==4` 才返回 FP16） | `SNAP/src/targets/qwen3_6_27b/impl/variant.cpp:898`（`gdn_output_input_dtype`：`tp==4 && tokens 1..8 && NVFP4 && VoltaQpnPrepacked && n==5120 && k==1536 ? FP16 : BF16`）、`SNAP/.../workspace_recipe.h:141-144`（`gdn_normalized_output(..., DType dtype = BF16)`） | **否**：fork 的 `workspace_recipe.h:141-143` 硬编码 BF16，无 `dtype` 参数；`gdn_output_input_dtype` 在 fork 里不存在 | **缺失** |
| A6 | "post_mixer_input_dtype"（同一机制的 MLP 侧，文档未单列） | 未见文档表述（代码侧） | 无量 | — | **是**（`tp==4`，n==8704,k==5120） | `SNAP/src/targets/qwen3_6_27b/impl/variant.cpp:903-907` | **否** | **缺失**（且文档未声称，不列入待评估） |

### B. QUASAR v3 TP2 MTP4 阶梯（`docs/performance.md:550-566`，逐行顺序叠加）

阶梯起点是"parent FP32 epilogue"之前的实现；每行的 tok/s 是**新增该行之后**的两请求 wall 值。
文档明确写："these are sequential measurements, not isolated attribution"（:575），
"about +0.84% versus the preceding 196.452 pair; these are sequential measurements, not isolated attribution"（:570-572）。

| # | 改动名（原文措辞） | 文档出处 | 文档声称的量（原文照抄） | 量级口径 | TP4 专属 | 快照实现位置 | fork 是否已有 | 判定 |
|---|---|---|---|---|---|---|---|---|
| B1 | "Parent FP32 epilogue + shape-specific streaming" | `docs/performance.md:552` | 空阶梯（157.312 为起点，无 Δ） | 基准行 | 否 | `SNAP/src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh:8-9`（`Streaming` / `Fp32Scale` 模板参数）、`:50-54`（`Streaming ? __ldcs(...) : __ldg(...)`）、`:38`（`epilogue_scale = Fp32Scale ? 256*inv_divisor : 1`）、`:103,123-126`；权重标志 `w.volta_fp32_scale`（`:172` 等 6 处） | **否**：fork 的 `nvfp4_volta_qpn_gemm.cuh` 无 `Streaming` / `Fp32Scale` / `ColumnOutput` / `PairedProjection` 模板参数，scale 在 K 归约前乘（`sc2 = __hmul2(decode_scale(scales[i]), divisor2)`，fork 全文只有这一种）；无 `w.volta_fp32_scale` | **缺失** |
| B2 | "TP2 two-epoch BF16 packet push" | `docs/performance.md:553` | 173.803（接 A1 的 157.312 → 名义 +10.5%） | 阶梯 Δ%（名义） | 否 | `SNAP/src/ops/kernel/allreduce.cuh`（全套 packet 内核）、`SNAP/src/ops/launcher/allreduce.{cu,h}` | **是（自研等效）**：fork 有独立实现 `FORK/src/ops/common/allreduce.cu:335-425`（`kPacketThreads=64`、`kPacketMaxBytes=81920`、`kPacketEpochPacks`、`allreduce_packet_sum_kernel`、`ld.volatile.global.v4.b32` 16B 收发、`epochs[blockIdx.x] & 1` 双 epoch）；与本战役已实测的"packet 两 epoch 邮箱 decode +4.5%，有效"是同一机制 | **已有**（且已实测有效） |
| B3 | "Mixer output packet sum + RMSNorm" | `docs/performance.md:554` | 175.006（名义 +0.69%） | 阶梯 Δ% | 否 | `SNAP/src/ops/kernel/allreduce.cuh`（fused norm）、`SNAP/src/targets/qwen3_6/impl/runtime/text_context_impl.h:2059,2081,2103`（`next_input_norm` 管线） | **部分已有**：fork 有融合内核 `FORK/src/ops/common/allreduce.cu:428-445`（"Fused packet all-reduce + RMSNorm (row width 5120, at most eight columns)"、`kNormEpochPairs=20480`）与 site A 接线 `FORK/.../text_context_impl.h:1948,1880-1884,1919-1923`，且 `kFuseNormByCollective = true`（`text_context.h:103`）；但**本战役已实测该融合为 −2.75%（负）** | **部分已有** ／ 标注"**预期负**"（把宽归约塞进按列切分 CTA → 已实测 −2.75%） |
| B4 | "MTP shortlist tiled argmax" | `docs/performance.md:555` | 178.213（与 B5 合并一格，名义 +1.83%） | 阶梯 Δ% | 否 | `SNAP/src/ops/launcher/argmax.cu:60-87`（`argmax_with_value_launch`，`proposal_shard = physical_rows==131072/2 && valid_rows==131072/2 && t<=64`）、`SNAP/src/ops/kernel/argmax.cuh:98,138-147` | **否**：fork 的 `argmax.h` 只导出 `argmax()`（`FORK/include/ninfer/ops/argmax.h:21`），`argmax.cu` 全文 83 行，无 `argmax_with_value_launch` / `merge_argmax_shards_launch` / `argmax_with_value_kernel` / `merge_argmax_shards_kernel`；fork 的 `tiled_block_for` 仍保留 `kShortlistRows=131072` 分支（`FORK/src/ops/launcher/argmax.cu:19,29-31`），但那是通用 tiled atomic，不是 MTP 短的 value+index 融合路径 | **缺失** |
| B5 | "GDN output activation staging" | `docs/performance.md:555` | 同上（178.213 两机制合并） | 阶梯 Δ% | 否 | 见 A3 | **否**（见 A3） | **缺失** |
| B6 | "MLP down sum + next-layer input RMSNorm handoff" | `docs/performance.md:556` | **180.410 / 180.566**（名义 +3.1%） | 阶梯 Δ% | 否 | `SNAP/src/targets/qwen3_6/impl/runtime/text_context_impl.h:2024-2060`（`handoff_norm = x[0].ne[1] <= 8`、`next_input_norm`、`input = handoff_norm && layer>0 ? &prepared_input : nullptr`）、`SNAP/.../text_context.h:431,435`（`normalized_input` 形参） | **否（设计已就位但未接线）**：fork 的 `mlp_tail_tp2` 有 `next_input_norm` 形参与 `ReplicatedRmsNorm` epilogue（`FORK/.../text_context_impl.h:1889,1919-1923`），但**两个调用点都传 `{nullptr, nullptr}`**（`:1983, :2009`），且 `kNextInputNormalized = false`（`text_context.h:107`）。即"下一层 input norm 被跳过"这条**没有**生效，每层仍跑独立 rmsnorm | **缺失**（host 侧 launch 已存在，wire 缺失） |
| B7 | "Projection epilogue publishes peer packet" | `docs/performance.md:557` | 183.201（名义 +1.55%） | 阶梯 Δ% | 否（TP4 另有 `Nvfp4Packet4Output`） | `SNAP/src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu:29-67`（`Nvfp4PacketOutput`，QPN epilogue 里 `st.volatile.global.u32` 直写 peer 双 epoch 槽）、`:68-108`（`Nvfp4Packet4Output`，4-rank 8 路 fork） | **否**：fork 的 `nvfp4_linear_add_decode.cu` 无 `Nvfp4PacketOutput` / `Nvfp4Packet4Output`（`diff` 显示整块仅快照有） | **缺失** |
| B8 | "Exact split-10 GDN controls + projection/conv/record fusion" | `docs/performance.md:558` | 186.422（名义 +1.76%） | 阶梯 Δ% | 否（TP4 侧见 D 组） | `SNAP/src/ops/gdn_input_proj/gdn_projected_conv.cu:84`（`Channels==10240||5120`）、`SNAP/src/ops/gdn_input_proj/*/..._gdn_input_plan.cpp`（单 kernel 控制+conv+record） | **部分已有**：fork 的 `gdn_projected_conv.cu:84` 同样有 `Channels==10240||5120`（TP2 的 projection/conv/record 融合**已有**）；`FORK/.../wrapper/linear_add.cpp:532` 注释表明 linear_add_norm epilogue 也已有；但"split-10 单 launch 控制投影"本战役实测 **+0.37%，低于门槛** | **部分已有** ／ 标注"**降低 host launch 数、不改 kernel 内并行度 → 预期中性偏好**"，但已实测 +0.37% 低于门槛 |
| B9 | "Five-query packed KV vector16 loads" | `docs/performance.md:559` | 190.124（名义 +1.99%） | 阶梯 Δ% | 否（long profile TP2 五 query） | `SNAP/src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh`（724 行；fork 同文件 509 行，差 371 行）、`SNAP/src/ops/kernel/gqa_attention_decode_i8_qpn_verify5_volta.cuh`、`..._qpn_verify_volta.cuh`、`..._qpn_volta.cuh` | **否**：fork 的 `gqa_attention_decode_i8_tc_volta.cuh` 与快照有 371 行实质差异（`diff --strip-trailing-cr`），且 fork **完全没有** 三个 `qpn_*_volta.cuh` 文件（仅快照有）；fork 用的是 `gqa_attention_decode_i8_key_split_volta.cuh`（仅 fork 有） | **缺失** |
| B10 | "Output projection streaming" | `docs/performance.md:560` | 190.919（名义 +0.42%） | 阶梯 Δ% | 否 | 同 B1 的 `Streaming = __ldcs` 参数（`SNAP/.../nvfp4_volta_qpn_gemm.cuh:8,50-52`） | **否**（见 B1） | **缺失** |
| B11 | "W8 byte-permute decode" | `docs/performance.md:560` | 同上 | 阶梯 Δ% | 否 | `SNAP/src/ops/linear/w8/w8_volta_qpn_gemm.cuh:150-153`：`const uint32_t bits = __byte_perm(src, 0x64646464U, (j&1) ? 0x4342U : 0x4140U);` | **否**：fork 同位置是移位/掩码版 —— `shift=(j&1)*16; bits = ((src>>shift)&0xff) \| (((src>>shift)&0xff00)<<8) \| 0x64006400`（`FORK/src/ops/linear/w8/w8_volta_qpn_gemm.cuh:149-153`）。算术等价，fork 是 `__byte_perm` 之前的写法 | **缺失**（等价改写，纯 instruction-count 收益） |
| B12 | "Single-query packed KV vector16 loads" | `docs/performance.md:561` | 191.397（名义 +0.25%） | 阶梯 Δ% | 否 | `SNAP/src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh`（同一改动覆盖 1/5/6-query）、`SNAP/src/ops/launcher/gqa_attention_decode.cu:461` | **否**（同 B9） | **缺失** |
| B13 | "FP32 proposal scores through top-1" | `docs/performance.md:562` | **193.294 / 193.221**（名义 +0.99%） | 阶梯 Δ% | 否 | `SNAP/src/ops/proposal_argmax/proposal_argmax.cpp:20-35`（`device_sm==70 && Q4G64_F16S && (n==32768\|\|65536) && k==5120 && x.ne[1]==1` → `proposal_q4_sm70_launch`，FP32 scores 直出）、`SNAP/src/ops/proposal_argmax/q4_sm70.cu`、`SNAP/include/ninfer/ops/proposal_argmax.h` | **否**：fork 无 `proposal_argmax` op、无 `proposal_q4_sm70`。fork 的提案路径是 `TextContext::proposal_argmax`（`FORK/.../text_context_impl.h:641-658`）：先 `ops::linear(hidden, *proposal_head_, proposal_logits /*BF16 [n,T]*/)` 物化 BF16 logits，再 `ops::argmax`；TP2 版（`:2694-2757`）还额外 `allgather_rows` 拼成全量 BF16 `whole[2]` 再 argmax。**即 fork 停在"先舍到 BF16 再 argmax"的旧路径** | **缺失**（且是明确可对照的旧几何） |
| B14 | "Fused shard-winner collective" | `docs/performance.md:563` | **195.941 / 196.209**（名义 +1.37%） | 阶梯 Δ% | 否 | `SNAP/src/ops/common/allreduce.cu:1178`（`merge_argmax_shards`）、`SNAP/src/ops/wrapper/argmax.cpp:86-99`、`SNAP/src/ops/kernel/argmax.cuh:138`（`merge_argmax_shards_kernel`，含 `indices[t*ranks+rank] + rank*first_shard_rows`） | **否**：`merge_argmax_shards` 与其 kernel 在 fork 里都不存在（见 B4） | **缺失** |
| B15 | "MTP tail fusion"（"Map shortlist winner inside the TP2 argmax collective, after tie-breaking"） | `docs/performance.md:147`（决策列："Retained; removes launches, speed gain unproven"）、`docs/performance.md:153-160`、`docs/performance.md:621-625` | 197.062 / 197.318；原文："the roughly **0.10%** mean difference does **not** establish an end-to-end gain" | 阶梯 Δ%（但文档自己否定其增量） | 否 | `SNAP/src/ops/kernel/argmax.cuh`（tie-break 后 remap）、`SNAP/src/ops/common/allreduce.cu:1178` 邻域、`SNAP/.../text_context_impl.h:330`（"the remap runs where the argmax runs"） | **否**（依赖 B14 的 shard-winner collective，fork 无） | **缺失** ／ 标注"**减少 host 侧 launch 次数但不改 kernel 内部并行度 → 预期中性偏好**"（文档自己也说 +0.10% 未成立） |
| B16 | "MTP hidden/FC buffer reuse"（"MTP AR calls now write directly over their dead hidden input, removing separate next-hidden storage and copy-back nodes"） | `docs/performance.md:628-631` | 无独立量（含在 196.471/196.433 一格） | 阶梯 Δ% | 否 | `SNAP/src/targets/qwen3_6/impl/runtime/text_context_impl.h:1469-1470`（`mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ..., io_.mtp->ar_hidden /* in==out */, ...)`）、`:2464-2465`（TP2 同） | **否**：fork 用**两块独立缓冲 + 显式回拷** —— `v.ar_hidden` / `v.next_hidden` 分离，每步 `cudaMemcpyAsync(v.ar_hidden.data, v.next_hidden.data, ..., DeviceToDevice)`（`FORK/.../mtp_impl.h:322-346` TP1 路径；`:381-412` TP2 路径）。多出一条 D2D copy node + 一个 [5120,batch] BF16 缓冲 | **缺失** ／ 标注"**减少 host/D2D 拷贝但不改 kernel 并行度 → 预期中性偏好**" |
| B17 | "MTP stem sum/norm fusion" | `docs/performance.md:566`（TP2 阶梯）、`docs/tp4.md:299`（TP4 侧同机制**被撤回**） | TP2：**198.105 / 198.097**（与 B18 合并一格，名义 +0.84%）；TP4：282.938/283.156 vs 283.242，"**no benefit is established, so it is withdrawn**" | 阶梯 Δ% | 否（TP2/TP4 都有尝试） | **未找到**（TP2 侧）。快照 `mtp_pack.cu` 与 fork 只差 TP4 `attn_in.ne[0]==3584` 分支（`SNAP/src/ops/launcher/mtp_pack.cu:50-53`）；未见独立的 stem sum+norm 融合 kernel | **未找到**（fork 与快照在这一步的差异无法从静态 diff 定位；见第五节） | **不确定（缺什么：需要 profile 名 / 具体 kernel 名；文档只给了阶梯格，没有给实现入口）** |
| B18 | "packed W8 MTP"（"prepackages five QUASAR MTP W8 matrices once at load time, preserving every code/scale and keeping one resident copy. Width one uses FP32 split GEMV; widths 3..8 use shape-specific QPN partitions (N5120:16, N7168/N17408:8)"） | `docs/performance.md:568-575` | 198.105 / 198.097（+0.84% vs 196.452 pair，顺序测量） | 阶梯 Δ% | 否 | `SNAP/src/ops/linear/w8/w8_prepack_sm70.cu:268-308`（`launch_w8_prepacked_sm70`：t==1&&mtp → `w8_qpn_mtp_gemv_kernel<16>`；t∈3..8&&n==5120 → `<16>`；否则 `<8>`）、`SNAP/.../w8_rowsplit_gemm_decode.cu:57,137-151`（`launch_volta_gemv` / `launch_w8_volta_gemv`）、`SNAP/.../w8_dispatch.cpp:283-286,322-326`、调用点 `SNAP/.../load/bindings.cpp:1322-1323,1346-1348`（5 个 MTP 矩阵） | **否**：fork 既无 `w8_prepack_sm70.*`，也无 `launch_w8_volta_gemv`（`w8_launch.h` 缺该声明）、无 `w8_rowsplit_sm70_gemv_kernel`、`w8_dispatch.cpp` 无 `t==1` 的 gemv 分流、无 `n==7168/17408/62080` 分片登记（fork 只有 `n==7168/17408` 出现在 `col==5120` 的 column_shard 里，且走通用 small_t/mma） | **缺失** |

### C. 六项 decode 试验（2026-10-22，`docs/performance.md:136-171`）

| # | 改动名 | 文档出处 | 文档声称的量 | 量级口径 | TP4 专属 | 快照实现位置 | fork 是否已有 | 判定 |
|---|---|---|---|---|---|---|---|---|
| C1 | "MTP tail fusion"（同 B15） | `docs/performance.md:147` | 197.062 / 197.318（≈+0.10%，原文否认定性） | 阶梯 Δ% | 否 | 见 B15 | 见 B15 | **缺失** |
| C2 | "GDN verifier value tile"（"Eight value rows per CTA instead of sixteen, record path only"） | `docs/performance.md:150,162-165` | 初测 197.308；复测 **197.614 / 197.698**；2048-token 配对 **181.580 → 181.880（+0.16%）** | 阶梯 Δ%（小样本顺序） | 否 | 见 `SNAP/src/ops/linear_attention/gated_delta_net/recurrent.cuh`（fork 与快照有差异） | **不确定**：fork 的 `gated_delta_net/recurrent.cuh` / `recurrent.cu` 与快照不同，但未定位到"Dv2/W4、8 vs 16 value rows"的常量切换。需进一步查（见第五节） | **不确定（缺什么：需确认 fork 的 record 路径 value-tile 高度）** |
| — | 未采用项（仅记录以排除误判）：INT8 attention load overlap 188.754 / TP2 packet 向量化 193.073 / QPN split-4 191.874 / verified history lookup 187.220 | `docs/performance.md:145-151`、`docs/v100-decode-survey.md:65` | 均**未采用**（slower 或 changed output） | — | — | — | 不需要查 | （非保留项，不入表） |

### D. TP4 阶梯（`docs/tp4.md:176-300`）

| # | 改动名 | 文档出处 | 文档声称的量 | 量级口径 | TP4 专属 | 快照实现位置 | fork 是否已有 | 判定 |
|---|---|---|---|---|---|---|---|---|
| D1 | "1Cat-inspired large allreduce"（reduce-scatter/allgather for large payloads） | `docs/tp4.md:229-232` | 相对上一阶段 decode **22.46%**（long/fast）/ **16.86%**（flags off）；prefill ≈**34.2%** | 阶梯 Δ%（顺序，非独立） | **是**（TP4 大消息才走该分支；narrow 仍走 packet） | `SNAP/src/ops/kernel/allreduce.cuh`、`SNAP/src/ops/launcher/allreduce.cu` | **否**：fork 的 `src/ops/common/allreduce.cu` 只有 TP2 双 epoch packet（≤80 KiB）+ fused norm，无 reduce-scatter/allgather 大消息分支（fork 的 `kPacketMaxBytes=81920`，超过直接 `packet_admits()` 返回 false） | **缺失**（TP4 专属） |
| D2 | "1Cat / fishlikeX Split-D prefill" | `docs/tp4.md:233-246` | prefill 1740 → 2065 tok/s（表内） | 阶梯 Δ% | **是**（TP4 D256 attention 专用） | `SNAP/src/ops/launcher/gqa_attention_volta_splitd.cu/.h` | **否**（仅快照有） | **缺失**（TP4 专属） |
| D3 | "Split-D prefill N64"（N32→N64 tile，57856 B opt-in smem） | `docs/tp4.md:243-246` | prefill **+3.46% / +3.45%**；decode 0.81%（但改 prefill 关联，非独立 decode 收益） | 阶梯 Δ% | **是** | 同上 `gqa_attention_volta_splitd.cu` | **否** | **缺失**（TP4 专属） |
| D4 | "Projection publication, 64-pack epochs / 16-byte stores" | `docs/tp4.md:288-292` | 231.433 → 238.454 | 阶梯 Δ% | **是**（4-rank） | `SNAP/src/ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu:68-108`（`Nvfp4Packet4Output`） | **否**（见 B7） | **缺失**（TP4 专属） |
| D5 | "Published sum/norm, 640 threads" | `docs/tp4.md:288-292` | 238.454 → 241.457 | 阶梯 Δ% | **是** | `SNAP/src/ops/kernel/allreduce.cuh`（640-thread sum/norm 调度） | **否**：fork 的 fused norm 是 row width 5120、≤8 columns、`kPacketThreads=64` 的 TP2 布局（`FORK/src/ops/common/allreduce.cu:428-445`） | **缺失**（TP4 专属） |
| D6 | "Verify5 K/V prefetch + FP32 proposal N32768/rank" | `docs/tp4.md:290`、`docs/tp4.md:249-251` | 241.457 → 244.655 | 阶梯 Δ% | **是**（TP4 每 rank N32768） | `SNAP/src/ops/proposal_argmax/q4_sm70.cu`（N32768/rank 路由）、`SNAP/src/ops/kernel/gqa_attention_decode_i8_qpn_verify5_volta.cuh` | **否**（见 B9、B13） | **缺失**（TP4 专属） |
| D7 | "Exact BF16→FP16 norm output image" | `docs/tp4.md:252-255` | 244.655 → 246.431（**约 +0.71%**"vs that same-session control"） | 阶梯 Δ% | **是** | `SNAP/src/targets/qwen3_6_27b/impl/variant.cpp:898`（FP16 dtype 选择）、`SNAP/.../workspace_recipe.h:141` | **否**（见 A5） | **缺失**（TP4 专属） |
| D8 | "Single-launch small-T GDN controls"（"a single 320-thread launch handles T2..8 for 12/24 value heads, preserving the ten split dot products, warp reduction and split-combine order"） | `docs/tp4.md:256-260,270-275`、`docs/tp4.md:270`（**+1.78%**） | 246.431 → 250.824；**+1.78%** | 阶梯 Δ% | 部分（TP4 320-thread / 12-24 heads；TP2 侧见 B8） | `SNAP/src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.{cu,h}`（fork 同文件有差异） | **不确定→部分**：fork 的 `gdn_gating_proj` 走 `Variant::gdn_norm_control_projection`（`FORK/.../text_context_impl.h:959`），`gdn_gating_proj_kernels.cu` 与快照有差异，但不是 320-thread/24-head 的 TP4 形状。TP2 侧的"split-10 单 launch"本战役实测 +0.37%（低于门槛） | **部分已有** ／ 标注"**降低 launch 数、不改 kernel 并行度 → 预期中性偏好**" |
| D9 | "TP4 GDN projection/record/conv fusion" | `docs/tp4.md:261-268,276-279` | 250.824 → 258.037（**+2.88%**；普通档 **+1.99%**） | 阶梯 Δ% | **是** | `SNAP/src/ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh:37-44`（`Nvfp4GdnInputSections<Nvfp4GdnInputTp4ColumnGeometry>{kKeyRows=512,kValueRows=1536}`）、`SNAP/src/ops/gdn_input_proj/gdn_projected_conv.cu:143-148`（`Channels==2560` 分支） | **否**：fork 的 `nvfp4_gdn_input_output.cuh` 无 TP4 sections 特化；`gdn_projected_conv.cu:84` 无 `Channels==2560`、无 2560/512/512/1536 的 launch | **缺失**（TP4 专属） |
| D10 | "Ordered attention reducer"（final five max/sum tree levels use warp shuffles with the original paired order） | `docs/tp4.md:265-268` | N64 → +**1.87% / +0.80%** | 阶梯 Δ% | **是** | `SNAP/src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh`（371 行差异区） | **否**（见 B9） | **缺失**（TP4 专属） |
| D11 | "SwiGLU output image"（FP32→BF16→FP16 typed image，去掉 standalone cast） | `docs/tp4.md:269-271` | +**0.94% / 0.67%**（原文把 GDN norm image 与 SwiGLU image 分列，此处为 SwiGLU 侧 1.87/0.80 之后的下一档） | 阶梯 Δ% | **是** | `SNAP/src/ops/linear_swiglu/swiglu_fp32.{cu,h}`（仅快照有） | **否**（仅快照有 `swiglu_fp32.*`） | **缺失**（TP4 专属） |
| D12 | "Last-CTA GDN record/norm fusion"（192-CTA Dv2/W4，per-head completion counter） | `docs/tp4.md:279-287` | +**0.55% / 0.30%** | 阶梯 Δ% | **是** | `SNAP/src/ops/linear_attention/gated_delta_net/recurrent.cuh`（fork 同文件有差异） | **不确定**：fork 的 `recurrent.cuh`/`recurrent.cu` 与快照不同，但未定位到 per-head completion counter。需进一步查 | **不确定（缺什么：需确认 fork 的 GDN recurrence 是否有 last-CTA norm fusion）** |
| D13 | "W8 verifier input image"（SM70 N62080/K5120, T3..8，FP16 [K,T] operands 一次物化） | `docs/tp4.md:289-296` | +**0.65% / 0.45%** | 阶梯 Δ% | **是**（n=62080） | `SNAP/src/ops/linear/w8/w8_prepack_sm70.cu:277-286`（`w8_prepacked_workspace_bytes(62080,5120,t,t)` + `w8_input_image_kernel`）、`SNAP/src/ops/linear/w8/w8_dispatch.cpp:308-315` | **否**（fork 无 w8 prepack / 无 62080 登记） | **缺失**（TP4 专属） |
| D14 | "Direct-peer TP4 argmax"（rank-zero kernel 读四份 score/index 分片） | `docs/tp4.md:297-302` | +**4.05% / 2.78%** | 阶梯 Δ% | **是** | `SNAP/src/ops/kernel/argmax.cuh`（`merge_argmax_shards_kernel`）、`SNAP/src/ops/common/allreduce.cu:1178` | **否**（见 B14） | **缺失**（TP4 专属） |
| D15 | "Direct-peer metadata broadcast"（≤4 KiB I32 单 origin publication） | `docs/tp4.md:303-309` | +**0.56% / 0.38%** | 阶梯 Δ% | **是** | `SNAP/src/ops/common/allreduce.cu`（metadata broadcast 分支） | **不确定**：未在 fork 定位到同机制；需进一步查（第五节） | **不确定（缺什么：需确认 fork 的 4-rank metadata 广播路径）** |
| D16 | "MTP stem sum/norm fusion"（TP4 侧，**被撤回**） | `docs/tp4.md:299` | 282.938/283.156 vs 283.242 → **无收益，已撤回** | 阶梯 Δ% | **是** | 未找到 | 不需要 | （非保留项） |

---

## 二、fork 明显缺失且值得评估的（≤8 条，按本机折算后预期排序）

折算规则：**阶梯 Δ% × 0.45**（本机三个实测样本：TileN 0.411 / packet 0.447 / GDN 约 0.5）。
**倍率与"非阶梯值"不适用该系数**，单列。

| 排名 | 项 | fork 缺失证据 | 上游声称 | 折算后本机预期 | 成本 | 排序理由 |
|---|---|---|---|---|---|---|
| 1 | **B1 Parent FP32 epilogue + shape-specific streaming** | fork 的 `nvfp4_volta_qpn_gemm.cuh` 无 `Streaming` / `Fp32Scale` / `ColumnOutput` / `PairedProjection` 模板参数；scale 在 K 归约前乘（全文只有 `sc2 = __hmul2(decode_scale(...), divisor2)` 一种）；无 `w.volta_fp32_scale` | **无 Δ**——它是阶梯的**起点**（157.312），14 步全部叠在它之上 | **未知，可能高于任何单步 Δ** | 中（改 QPN gemm 模板 + 权重标志 + 6 处实例化） | **唯一"非阶梯值"的 decode 结构项**。上游是先把这一层铺好才开始记 14 步，说明它是"地基"而不是"一步" |
| 2 | **D1 大消息 allreduce（reduce-scatter / allgather）** | fork 只有 `kPacketMaxBytes=81920`；超过 `packet_admits()` 直接返回 false，退回 one-shot | prefill **≈+34.2%**（TP4 顺序测量）；decode +22.46% | prefill 若可移植，量级远超所有 decode 候选 | 高（新 collective + 协议） | 文档标 TP4 专属，**但我们的 prefill 每 chunk 26 MB 也走 one-shot**，机制可能对 TP2 成立。**prefill 是整块无人碰过的地**（我们只试过 TileN） |
| 3 | **A3/B5 GDN output activation staging** | fork 的 `nvfp4_launch.h` 无 `launch_nvfp4_volta_qpn_precast` / `nvfp4_volta_qpn_cast_activation` / `..._precast_supported` | 含在 +1.83% 格里，**无独立量** | 0.2~0.4% | 中低 | 机制是"BF16→FP16 转一次而不是每个消费 CTA 各转一次"——**cast 是精确的，数值应逐位不变**，属 A 类 |
| 4 | **W8 scale 平面 4× 扇区宽读**（非文档项） | 代码分析：`w8_volta_qpn_gemm.cuh:118-120` 每 lane 读 8 B、行距 320 B → 32 个 32 B 扇区只用 8/32 | 文档无；代码注释自认 `cuts that to 4x` | **≤ +0.80%（上限）** | 极低（约 15 行） | 六个 W8 矩阵流量 −15%；但 round 级带宽只用到 38%（302/794 GB/s），**延迟受限，砍流量未必换来时间**。寄存器预算已满，直接加 `scv[16]` 可能 spill |
| 5 | **B13 FP32 proposal scores through top-1** | fork 无 `proposal_argmax` op、无 `proposal_q4_sm70`；fork 先物化 BF16 logits 再 argmax，TP2 版还 `allgather_rows` 拼全量 | +0.99% / +0.97% | ~0.45% | 中高（新 op + dispatch） | 压线；但 fork 的旧路径确实更重（BF16 落地 + 全量 gather），是"减少一次大传输"而非微优化 |
| 6 | **B16 MTP hidden/FC buffer reuse** | fork 用两块独立缓冲 + 每步 `cudaMemcpyAsync(D2D)`（`mtp_impl.h:322-346` / `:381-412`） | 无独立量 | 0.1~0.2% | 极低（改 io 绑定，删一条 copy） | 纯删节点；风险低但收益也低，**适合和其他改动打包** |
| 7 | **B11 W8 byte-permute** | fork 用移位/掩码替代 `__byte_perm`（`w8_volta_qpn_gemm.cuh:149-153`） | 含在 +0.42% 格里 | <0.1% | 极低（1 行） | 纯指令数；**只值一次编译，不值一次测量** |
| 8 | **B18 packed W8 MTP** | fork 无 `w8_prepack_sm70.*`、无 `launch_w8_volta_gemv`、无 `t==1` gemv 分流 | +0.84% | ~0.38% | 高（prepack 基础设施） | 依赖完整 prepack；且 MTP 五矩阵（n=5120/7168/17408）与输出头形态不同，**不能复用输出头已吃到的 QPN 收益** |

**结论：前 4 名之外全部低于或接近本机噪声地板（±0.2~0.33%），不值得单独占一轮编译+测量。**
第 1、2 名的价值在于它们**不是阶梯增量**，不能用 ×0.45 估。

## 三、fork 已有、不必再看的

- **B2 TP2 two-epoch packet push** —— fork 自研等效（`allreduce.cu:335-425`），本战役实测 **decode +4.5%**
- **A1 W8 输出头 prepack** —— 那 1.31×–1.93× 是"QPN vs 旧 sliced incumbent"，而 `w8_dispatch.cpp:293` 生产路径**已 `return launch_w8_volta_qpn`**。→ **收益已吃掉，不能重复领**
- **A2 输出头配置口径** —— fork 用自己的 `select_w8_tp2_shard_launch`（`w8_dispatch.cpp:14-34`），是"实现不同"不是"能力缺失"
- **B8 projection/conv/record 融合** —— fork 的 `gdn_projected_conv.cu:84` 同样覆盖 `Channels==10240||5120`；split-10 单 launch 本战役已试（+0.37%，低于门槛）
- **smem +8 padding** —— fork `gqa_attention_decode_i8_tc_volta.cuh:123` 已有
- **B3 Mixer output packet sum + RMSNorm** —— fork 已实现（`allreduce.cu:428-445` + site A 接线），但**实测 −2.75%，负**

## 四、TP4 专属、TP2 无对象的

D2（Split-D prefill）、D3（Split-D N64）、D4（64-pack epochs 4-rank）、D5（640-thread sum/norm）、
D6（verify5 prefetch + FP32 proposal N32768/rank）、D7（BF16→FP16 norm image）、D9（TP4 GDN
projection/record/conv，含 `Channels==2560`）、D10（ordered attention reducer）、D11（SwiGLU
output image）、D13（W8 verifier image n=62080）、D14（direct-peer TP4 argmax +4.05%）、
D15（≤4 KiB metadata broadcast）。

D8（single-launch small-T GDN controls）**部分**适用 —— TP2 侧已试 +0.37%。

## 五、不确定、需要进一步查证的

| 项 | 缺什么信息 |
|---|---|
| **B17 MTP stem sum/norm fusion** | 文档只给阶梯格（198.105），**没有实现入口**；`mtp_pack.cu` 只差 TP4 分支。且 TP4 侧同机制已被**撤回**（`tp4.md:299`："no benefit is established, so it is withdrawn"）→ 建议不再追 |
| **C2 GDN verifier value tile** | 需确认 fork 的 record 路径 value-tile 高度（8 vs 16 行）；文档只给复测 +0.16% |
| **D12 last-CTA GDN record/norm fusion** | 需确认 fork 的 `recurrent.cuh` 有没有 per-head completion counter；文档 +0.55%/0.30%（TP4） |

**工具限制（影响所有"需要 profile 才能定论"的项）**：本机 `_toolchain/cuda-12.8/bin/` **无 ncu**，
无法取 `dram__bytes_read` 一类的硬件计数器。凡依赖 profile 才能判定的候选（如 W8 scale 是否真 4×），
只能靠"实现 + 端到端测量"来验，成本更高。
