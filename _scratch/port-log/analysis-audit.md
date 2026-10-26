# `ref-v100x2-analysis.md` 全文审计

审计对象：`_scratch/ref-v100x2-analysis.md`（181 行）  
依据：快照 `_scratch/ref-v100x2/ninfer-V100X2-master` 与 fork 源码逐条对账  
时间：2026-10-25 06:45

## 一、结论摘要

**整体可信，但有 2 处实质性错误、1 处路径不精确，且错误正好落在"决定做什么"的那张表上。**

| 类别                                      | 结论                                                       |
| --------------------------------------- | -------------------------------------------------------- |
| 引用位置（8 条重点）                             | **7 条准确**，1 条路径少了子目录（`ggml_k/`）                          |
| 阶梯算术（7 个 delta）                         | **全部正确**，与 `performance.md:550-565` 原文逐行吻合               |
| 缺口表（5 条）                                | **3 条成立，1 条成立但理由错（⑤），1 条正确（③）**                          |
| 最容易误导决策的一处                              | 缺口④ 把 TP4 的 +4.05% 外推成"TP2 同理" —— **TP2 受控实测只有 +0.046%** |
| 核心判断（"差距在 collective 形态，不在 kernel 微优化"） | **成立**，且已被本战役实测验证（packet +4.5~4.7%）                      |

## 二、引用核对表

| 报告位置                                                 | 声称引用                               | 实际内容                                                                                                                                                                            | 判定                   |
| ---------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| §二 #1 `program_impl.h:113-226`                       | profile 家族 + GraphExecUpdate       | `:113` 即 `select_graph_profile(DecodeGraphFamily&, batch, frontier, greedy_target)`                                                                                             | ✅ 成立                 |
| §二 #4 `gqa_attention_decode_i8_tc_volta.cuh:370-396` | KV 寄存器预取                           | `:371-372` 注释"Keep the next tile's packed bytes in registers while the current QK/softmax/PV runs"，`:374-396` 为 `prefetch_kv` lambda + `int2 next_k[PrefetchKv ? KvChunks : 1]` | ✅ 成立（精确）             |
| §二 #6 `kernel/allreduce.cuh:86-272`                  | 两 epoch packet 邮箱                  | 该区间即 `allreduce_packet_*` 全部实现，含 `0xffff` 空槽与 epoch 翻转                                                                                                                          | ✅ 成立（本战役已照此移植成功）     |
| §二 #7 `allreduce.cu:1126-1246`                       | 跨卡 argmax 合并                       | `:1126` `void allreduce_argmax(...)`，`:1161/:1163` 分派 tp2/tp4 launch                                                                                                            | ✅ 成立                 |
| §二 #8 `ggml_k_cutlass_sm70.cu:196-212`               | 融合 epilogue                        | 文件实际在 `src/ops/linear/**ggml_k**/ggml_k_cutlass_sm70.cu`                                                                                                                        | ⚠️ **路径缺子目录**，内容未逐行核 |
| §二 #8 `w8_rowsplit_gemm_mma.cuh:403-458`             | residual 进 epilogue                | `:403` `else if constexpr (Epilogue == W8Epilogue::Residual)`                                                                                                                   | ✅ 成立                 |
| §二 #11 `...:141-150`                                 | smem +8 padding，ncu 724M conflicts | `:141-150` 注释原文含 "ncu measured 724M bank conflicts against 891M wavefronts"，`:148` `constexpr int SmemPad = 8`                                                                  | ✅ 成立（精确）             |
| §四 `performance.md:550-565`                          | retained ladder                    | 表头即 "Retained combination, added in order"，共 14 行                                                                                                                               | ✅ 成立                 |
| §六 修正3 `performance-v100.md:140-148`                 | TP2 matched campaign               | 四行表 + "The isolated argmax step is within end-to-end noise"                                                                                                                     | ✅ 成立                 |

## 三、数字来源分级与算术核验

抽查 7 个 delta，全部在 `performance.md:550-565` 原文对上，算术正确：

| delta   |              原文两数 | 计算      | 判定 |
| ------- | ----------------: | ------- | -- |
| +10.48% | 157.312 → 173.803 | 10.483% | ✅  |
| +0.69%  | 173.803 → 175.006 | 0.692%  | ✅  |
| +1.83%  | 175.006 → 178.213 | 1.833%  | ✅  |
| +1.23%  | 178.213 → 180.410 | 1.233%  | ✅  |
| +1.55%  | 180.410 → 183.201 | 1.547%  | ✅  |
| +1.76%  | 183.201 → 186.422 | 1.758%  | ✅  |
| +1.37%  | 193.294 → 195.941 | 1.369%  | ✅  |

**来源性质**：全部为**顺序阶梯**（每步叠加前一步，非独立归因）。报告 §四 已明确标注这一点，  
并在 §六修正 4 里撤回旧结论，这部分**没有注水**。

补充两条原文限定条件，报告未引：

- `performance.md:547`："The user stopped that campaign after the final W8 trial. **200 tok/s was not reached**."
- `performance.md:147`："MTP tail fusion ... Retained; **removes launches, speed gain unproven**."

## 四、必须修正的条目

### 错误 1（实质性）：缺口⑤"≤80KiB 单 kernel peer-read"**fork 本来就有**

- 报告原文（§三 缺口⑤）："launcher 无 allreduce kernel；combine 未合入单核"，  
  并据此在 §四 末行建议移植"≤80KiB peer-read 单核 sum，+3.37%，成本低"。
- 实际：fork 的 `src/ops/common/allreduce.cu:281` 已有  
  `direct_allreduce_kernel`（`kDirectThreads = 256`，CTA 切片 `kDirectSlicePacks`），  
  **单个 kernel 完成 peer-read + 本地累加**，`:527-535` 就是它的 launch 点。
- 影响：§四 末行"fork 现状=无"是错的；这条本可以省掉一次立项评估。
- 修正建议：改为"fork 已有单核 one-shot peer-read；缺的是**两 epoch 邮箱协议**（本地贡献留寄存器、  
  免 event 边、单次 launch 覆盖多轮），这才是 packet 的增量"。  
  —— 已被本战役实测证实：packet 相对基线 decode **+4.5~4.7%**，说明增量真实存在，  
  但**不是"从 0 到 1"**，而是"从 one-shot 到 mailbox"。

### 错误 2（最具误导性）：TP4 的 +4.05% 被外推成"TP2 同理"

- 报告原文（§三 缺口④）："greedy 跨卡 argmax 合并 … 预期收益锚点：**TP4 +4.05% decode，TP2 同理**"。
- +4.05% 确实有出处，但**只限四卡**：`performance.md:106-107`  
  "Direct-peer **four-rank** argmax advances this to **281.662 / 191.373** (+4.05% / +2.78%)"，  
  同文 `tp4.md:294` 亦称 "Direct-peer TP4 argmax … adds 4.05% / 2.78%"。
- TP2 的受控实测在 `performance-v100.md:140-148`：shard-winner +0.73%，  
  再加 parallel shard argmax 只 +0.046%（原文 "within end-to-end noise"）。
- 影响：**"TP2 同理"这四个字把一个噪声级改动推到了优先级前列**，本战役第 5 项因此立过项。
- 修正建议：删除"TP2 同理"，改写为"TP4 +4.05%；**TP2 受控实测 +0.046%，等于无收益**，  
  不应立项"。

### 错误 3（轻微）：路径不精确

`§二 #8` 的 `ggml_k_cutlass_sm70.cu` 实际在 `src/ops/linear/ggml_k/` 下。不影响结论。

### 内部不一致（非错误，但需注意）

报告 §六修正 3 已经写出"TP2 上 argmax 合并本身在噪声内"，但 §三 缺口④ 与 §四 优先级表  
没有同步改。**同一份文档里两处相反**。以修正 3 为准。

## 五、经核对成立、不必再怀疑的部分

1. **核心判决**："差距在 collective 的数据流形态，不在 kernel 微优化"。  
   → 已被实测验证：packet（形态改动）拿到 decode +4.5~4.7%；  
   TileN（kernel 微优化）只有 prefill +0.473%。
2. **阶梯表本身**：数字与算术全部准确，且报告已正确标注"顺序测量、非独立归因"。
3. **缺口①②③④ 的存在性**（不含其收益估计）：
   - ① `nvfp4_cutlass_sm70.cu` 固定 `GemmShape<128,128,32>` ✅
   - ② fork `gqa_attention_decode_i8_tc_volta.cuh` **无任何 prefetch/PrefetchKv/next_k**  
     （509 行 vs 快照 724 行）✅
   - ③ `src/ops/linear/w8/` 无 `*prepack*` 文件 ✅
   - ④ fork 无 `allreduce_argmax` ✅
4. **§二 #13 PDL 在 sm_70 是 no-op**：与"不要移植"清单一致，无需复核。
5. **§五 提醒**（实验开关、阈值表必须本机重测）：合理且重要，保留。

## 六、对当前队列的影响

| 队列项                 | 审计后动作                                            |
| ------------------- | ------------------------------------------------ |
| 第 5 项 argmax        | **保持出队**。原立项依据就是被审计出的"TP2 同理"外推                  |
| 第 8 项 shard-winner  | **保持出队**。TP2 受控值 +0.73%，折算后 +0.30~0.33%，过不了 0.5% |
| 第 6 项 RMSNorm       | **不受影响**，两项 delta 有独立出处（阶梯 +0.69% / +1.23%）      |
| 第 7 项 epilogue 发布   | **不受影响**（+1.55% 算术已核）                            |
| "≤80KiB 单核 sum" 那一行 | **删除**。fork 已有，不是缺口                              |
