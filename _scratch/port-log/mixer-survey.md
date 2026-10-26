# mixer 勘察：无快赢，确认 GEMM/attention 均已就位（2026-10-25 下半场）

**结论：mixer（attn+gdn，chunk 的 48%）拆解后没有"选错路由"级别的快赢。
prefill attention 已在走 VoltaFlash；剩余优化都是 kernel 级深水区。**

## 拆解（pp10240，trace5/6，chunk 8.39s）

| 块 | 耗时 | 构成（隔离实测） | 评估 |
|---|---|---|---|
| mixer.gdn | 3.02s | GDN 核心 5.33ms/层（40 chunk 递推，流量 448MiB @ 88GB/s）+ input_proj_conv 6.48ms + conv/norm 少量 | 两者相加吻合。GDN 核心是 chunked recurrence，带宽利用率 10% 但受限于递推结构 |
| mixer.attn | 0.99s | QKVZ proj GEMM + **VoltaFlash** + o_proj GEMM + rope/rmsnorm/sigmoid_mul | **attention 核心已是最优路线**（见下） |
| post_mixer | 4.28s | gate_up 5.8ms + down 2.4-4.5ms + norms，GEMM 均 ~74 TFLOPS（60% MFU） | 已到 V100 cuBLAS 水位 |

## 关键修正（撤回我之前的判断）

- **"attention 核心 11ms @ 2.4 TFLOPS"是错的**。路由追踪（lab printf）证明
  prefill 宽 T 一直走 `GqaAttentionRoute::VoltaFlash`（qh=24 = 27B 的 24q/4kv，
  32 次/5chunk = 16 层×2 卡）。27B 是 24q/4kv（ratio 6），不是 32q/4kv；
  Gqa35Geometry（16q/2kv）只在 35B 模型用。mixer.attn 的 12.4ms 主要是
  两个 GEMM + rope/norm 链，不是 attention core。
- 实验过程：解锁 q_heads==16 的 flash 路由 → pp2048 探针输出与基线**逐位一致**
  （sha1 `51f95c8258e8ce05`）→ pp10240 性能零变化 → 路由追踪发现 16q 根本没被调用。
  附带收获：**Gqa35 的 4×8 tiling 在真实负载下没复现注释里的 shared-memory 损坏**
  （gqa_attention.cpp:296-303 的禁用理由可能已过时，但无收益，不追）。
- proj.swiglu 5.8ms 是双 rank 时序重叠假象（隔离 12.06ms @ 75.7 TFLOPS，健康）。

## 顺带发现（不影响当前路线）

- `causal_conv1d_silu_bench` 的参数校验疑似 bug：`--tokens >1024`（或组合
  `--channels/--slots`）会拒绝并打印 usage，默认 1024 才能跑。conv 本体 ~250µs 量级，
  无性能影响，但 bench 覆盖不了真实 T=2560。
- GDN input_proj_conv bench 的 `--form both --sweep 2560:2560` 会先报一个
  "invalid B/T domain"再正常出数（snapshot form 的域校验问题）。

## 剩余优化空间排序（更新）

1. **GDN 核心 5.33ms/层**（mixer.gdn 的一半）：chunked recurrence，40 chunk×64，
   88GB/s。深水区：recurrence 结构决定难以 FlashAttention 化，需重读
   `gated_delta_net` kernel 找 tile/流水机会。工程量大，收益上限 ~1s/chunk 的 1/3。
2. **GEMM 融合反量化**：省 GEMM 时间 ~7%（≈prefill 5%），工程量大。
3. 通信、norm、conv、attention core：确认无油水，不碰。

⇒ prefill 的"路由级"收益已挖尽（本轮 +25.3% 是最后一口大的）。
下一步若继续 prefill，应先做 GDN kernel 源码精读再决定是否投入。
