# 候选清单重校准（2026-10-25 06:20）

**背景**：原队列的 6 个百分比不是独立实测，是从 `ref-v100x2` 的**顺序阶梯**抄来的
（`docs/performance.md:550-565`，每步叠在前一步之上）。用本次战役的两次真实移植反推折算系数后，
清单顺序需要改。

## 一、折算系数的两个实测样本

| 项 | 上游文档值 | 本机实测 | 比值 |
|---|---:|---:|---:|
| TileN=256（prefill） | +1.15% | **+0.473%** | **×0.411** |
| packet collective（decode） | +10.48% | **+4.685%** | **×0.447** |

两个完全不同的机制、不同的指标，比值却高度一致（0.41 / 0.45）。
**采用 ×0.41~0.45 作为折算区间**，凡不在这个范围的结果先怀疑测量。

## 二、上游阶梯原文（`docs/performance.md:550-565`，顺序测量，非独立归因）

| 加入顺序 | Wall tok/s | 相对上一步 |
|---|---:|---:|
| Parent FP32 epilogue + streaming | 157.312 | — |
| TP2 two-epoch BF16 packet push | 173.803 | +10.48% |
| **Mixer output packet sum + RMSNorm** | 175.006 | **+0.69%** |
| MTP shortlist tiled argmax + GDN activation staging | 178.213 | +1.83% |
| **MLP down sum + next-layer RMSNorm handoff** | 180.410 | **+1.23%** |
| **Projection epilogue publishes peer packet** | 183.201 | **+1.55%** |
| Exact split-10 GDN controls + fusion | 186.422 | +1.76% |
| Five-query packed KV vector16 | 190.124 | +1.99% |
| Fused shard-winner collective | 195.941 | +1.37% |

注意：阶梯底数是 157 tok/s，我们是 129.7 tok/s，**不是同一配置**。比值迁移只是近似，但方向可信。

## 三、另有一份更权威的 TP2 matched campaign（`performance-v100.md:140-148`）

同一语料、同 8 个请求、同 513 IDs 的**受控对比**，比阶梯可信得多：

| 实现 | decode tok/s | 相对上一行 |
|---|---:|---:|
| Control: complete target logits | 81.797 | — |
| Exact shard-winner verification | 82.396 | **+0.73%** |
| + parallel shard argmax | 82.434 | **+0.046%（噪声）** |
| + direct-peer small-message sum | **85.212** | **+3.37%** |

**结论一：`parallel shard argmax` 的收益是 +0.046%，上游文档自己写明"within end-to-end noise"。**
**结论二：这份 campaign 里 +3.37% 的那一步，正是我们已经移植的 packet collective**
（我们测到 +4.685%，比它还好，方向一致 —— 这是 packet 结论的独立交叉验证）。

上游另一处更直白（`performance-v100.md:271-274`）：
> "All 513/2049 output IDs and every acceptance counter match the prior argmax-only route.
> That route measured 99.32/95.12 decode tok/s respectively; the 0.6–0.8% single-run difference
> **does not establish a stable selector speedup**. No full-logit versus argmax-only controlled A/B
> has been rerun for these windows, so older reports do not establish an isolated argmax speedup."

## 四、重校准后的排序

| 排名 | 项 | 上游值 | 折算后 | 门槛 | 结论 |
|---|---|---:|---:|---:|---|
| **1** | **RMSNorm 融合（两个点一起做）** | +0.69% +1.23% = **+1.92%** | **+0.79~0.86%** | +0.5% | ✅ **值得做**，唯一稳过门槛的 |
| 2 | projection epilogue 发布 peer packet | +1.55% | +0.64~0.70% | +0.5% | ⚠️ 过线但**最贵**（改 CUTLASS epilogue + 协议） |
| 3 | Exact split-10 GDN controls | +1.76% | +0.72~0.79% | +0.5% | ⚠️ 过线，但 B 类（改求和顺序）+ 复杂度高 |
| 4 | five-query packed KV vector16 | +1.99% | +0.82~0.90% | +0.5% | ⚠️ 过线，但**要改 KV cache 布局**，非移植 |
| — | fused shard-winner collective | +1.37% | 受控实测 +0.73% → 折算 +0.30~0.33% | +0.5% | ❌ **出队** |
| — | MTP shortlist tiled argmax | +1.83%（含 GDN staging） | 受控实测 **+0.046%** | +0.5% | ❌ **出队**（400 行换噪声） |
| — | KV vec16 / 寄存器预取（原队列 2/3） | — | — | — | ❌ 已否决，目标核不存在 |

## 五、关键洞察：第 6 项不是一个点，是两个点

阶梯把 RMSNorm 融合拆成两步：
- `Mixer output packet sum + RMSNorm`（+0.69%）→ o_proj 归约 + **post_attn_norm**
- `MLP down sum + next-layer RMSNorm handoff`（+1.23%）→ down_proj 归约 + **下一层 input_norm**

机制完全相同（同一条 `allreduce_sum_rmsnorm_to`），只是两个调用点。
**单独做任一个都过不了门槛（0.28~0.31% / 0.50~0.55%），两个一起做才到 +0.79~0.86%。**

fork 侧两个落点已核实存在：
- 点 A：`text_context_impl.h:1868` —— `mlp_tail_tp2` 入口的 `ops::rmsnorm(x, *post_norm, ..., h)`
  （x 由 `attn_mix_tp2` 的 o_proj row-parallel 归约产出）
- 点 B：`text_context_impl.h:898` —— `attn_mix` 入口的 `ops::rmsnorm(x, *w.input_norm, ..., h)`
  （x 由上一层 `mlp_tail_tp2` 的 down row-parallel 归约产出）

## 六、前置原件已在 fork（无需新增依赖）

| 原件 | 位置 | 状态 |
|---|---|---|
| `block_reduce_sum<Block>` | `src/ops/common/warp.cuh:41` | ✅ 已有 |
| `RmsEpilogue` / `rmsnorm_epilogue` | `src/ops/kernel/rmsnorm.cuh:14-24` | ✅ 已有 |
| packet 存储 | `PeerEvents::packets[2]`（`df1f2958`） | ✅ 已有，但协议粒度要换 |
| `hidden == 5120` | `qwen3_6_27b/impl/config.h:12` | ✅ 与硬编码内核吻合 |

## 七、建议执行顺序

1. **第 6 项（两点全做）** —— 唯一稳过门槛的，先做。
2. 若第 6 项实际 ≥ +0.8%，说明折算系数偏保守，再评估第 2 名（epilogue 发布）。
3. 若第 6 项落在 +0.3~0.5%，**停下来重估折算系数**，不要顺着往下烧编译时间。
4. 第 5/8 项（argmax 系列）**不再开工**，理由见第三节的受控数据。
