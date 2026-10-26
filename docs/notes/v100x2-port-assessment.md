# tuxKOH/ninfer-V100X2 移植评估

> **⚠️ 状态：`superseded` — 本文的决策已执行完毕，仅作评估链留存，不要按本文开工。**
> 本文的 Phase 0/1/2 计划已全部执行，结论与终态指标见
> [v100-port-campaign.md](v100-port-campaign.md)（2026-10-26 战役终局记录）。
> 简短结论：V100X2 的七段接力**不能**快速移植（卡间互联假设不同，本机是 PCIe 3.0 x4
> 而非 NVLink）；最终只采纳了 packet collective 与 prefill-down-cutlass 两项，累积
> prefill +25.511% / decode +4.671%。本文「关键未知：本地 NVFP4 的真实基线」已由
> `34412f4f` → `945.799 tok/s` 回答。
>
> 记录：2026-10-24 · 角色：assessment · 状态：~~待决策~~ **已终结（superseded）**
> 参考源码：`_scratch/ref-v100x2/ninfer-V100X2-master`（master @ 02fec273，2026-10-23，本地 gitignore）
> 对照：本仓库 `perf/v100x2-port-study` 分支，base = `c96744938293e1bbe77124f0d7ba5dd1c3045dd2`

## 问题

`tuxKOH/ninfer-V100X2` 在 2 × V100-SXM2 16 GB 上测到 85K 输入 198.105 tok/s，
本仓库同硬件、同上下文是 53.41 tok/s（Q4_K_M）。差 3.7 倍。要判断：为什么快、能不能搬过来、值不值。

## 谱系：两条平行分支，不是前后版本

```
Neroued/ninfer (RTX 5090, sm_120a)
├── geoffwatts/ninfer-v100           单卡 V100
├── Neroued RTX 3060 TP2 fork        sm_86
├── plus1998/NInfer-V100-Duo  ──►   本仓库 Xxx-Bin/NInfer-V100-Duo-Win（+51 commits）
│                                    tuxKOH/ninfer-V100X2（自己的 100+ commits）
```

`c9674493`（本仓库 base）在 V100X2 的历史里**不存在**（GitHub API 返回 422）。
V100X2 的提交里 1381/1592 在本仓库对象库中可见（共同的 Neroued 上游），但**V100 专属工作两边各写各的**。

| | 本仓库 | V100X2 |
|---|---|---|
| 定位 | Windows 原生移植 | V100 性能深挖 |
| 卡间互联 | NVLink 直连 | PCIe P2P（SXM2 转接板，无 NVLink） |
| 自建提交 | +51（119 文件，+9080/−1674） | +100 余（含 40 个新 src 文件） |
| 推荐模型 | 官方 NVFP4 / Q4_K_M / Coder390 | QUASAR NVFP4 v3 |
| 85K decode | 53.41（Q4_K_M，MTP3） | 198.105（QUASAR，MTP4） |

## 一、V100X2 为什么快：七段接力，每段都有 A/B

全部是同一 artifact（QUASAR NVFP4 v3）、85K 占用、180K 容量、TP2 + MTP4 + INT8 group-64 KV + CUDA Graph、
greedy、513 token 窗口、committed wall decode。来源 `docs/v100-decode-survey.md`。

| # | 机制 | wall tok/s | 增量 | 来源 |
|---|---|---:|---:|---|
| 0 | 起点（普通 collective） | 113.306 | — | — |
| 1 | 固定 rank 特化 allreduce + 保留本地贡献 | 145.556 | **+28.5%** | 1Cat `custom_all_reduce.cuh` |
| 2 | QPN4/QPN2：`__ldcs` 流式读、code/scale 捆绑、M8 paired gate/up | 157.312 | **+8.1%** | 1Cat `nvfp4_qpn{4,2}_sm70.cu` |
| 3 | BF16 TP2 双 epoch packet push | 173.803 | **+10.5%** | 1Cat（源头 sglang-V100） |
| 4 | packet sum/norm + 跨层 norm 交接 | 180.410 | +3.8% | 自研 |
| 5 | 控制投影单 kernel（保 split-10 顺序）+ QPN 列 epilogue → conv/record 融合 | 186.422 | +3.3% | Augur |
| 6 | MTP argmax/remap 尾部融合 | 197.062 | **+5.7%** | 自研 |
| 7 | FP32 proposal top-1 + GDN value tile | 198.105 | +0.5% | Augur / 1Cat |

**总计 +74.8%（113.306 → 198.105）。**

结论：没有单一魔法。最大三项是 **allreduce 特化（+28.5%）**、**PCIe packet push（+10.5%）**、
**QPN 取数优化（+8.1%）**，合计贡献约 47 个点；其余四段各 3–6 个点。
注意力侧的长上下文优化（split-KV reducer、共享 QK score tile、QPN verify5 预取）被记为
"已落实"但未单独列 A/B 增量——它们和 MTP4 的验证形状耦合。

## 二、能不能快速移植：不能

`src/` 下两仓库共有文件 **626 个，逐字节相同的只有 4 个**。差异不是增量而是两套实现：

| 文件 | 本仓库 | V100X2 |
|---|---:|---:|
| `src/ops/common/allreduce.cu` | 31,575 | **63,683** |
| `src/targets/qwen3_6/impl/runtime/program_impl.h` | 207,479 | 195,100 |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | 172,567 | 190,763 |
| `src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh` | 29,884 | 40,509 |
| `src/ops/linear/ggml_k/ggml_k.cu` | 33,300 | 23,454 |

- V100X2 独有 src 文件 40 个（`gqa_attention_decode_i8_qpn_*`、`allreduce.cuh`、`ggml_k_cutlass_sm70`、
  `proposal_argmax/`、`fp8_block`、`w8_prepack_sm70`、`swa_volta`、`splitd` 等）
- 本仓库独有 src 文件 126 个

⇒ `git merge` / `cherry-pick` 均不可行。任何移植都是**手工重写一层**，不是"拿过来编译"。

## 三、收益评估

### 可拿到的收益

| 项 | 上限 | 前提 |
|---|---|---|
| allreduce 特化 | +28.5% | 本机是 NVLink，V100X2 有 `active_nvlink_mesh()` 专门分支，适用 |
| QPN `__ldcs` + 捆绑取数 | +8.1% | 需要先切到 QUASAR NVFP4 路线 |
| packet push | +10.5% | 面向 PCIe；本机 NVLink 下**大概率不适用** |
| 控制投影/argmax 尾部融合 | +9% | 与 GDN/MTP 实现强耦合，需重写 |
| long attention | 未单独测 | 85K 多请求并发时才明显 |

### 三个方案

| 方案 | 做法 | 工作量 | 风险 | 上限 |
|---|---|---|---|---|
| **A. 给 V100X2 加 Windows 层** | 把本仓库 51 个提交里的 Windows 部分重放到 V100X2 | 中（119 文件里的平台层，机械性强） | 中（V100X2 文件已大改，需逐个适配） | **198 tok/s 全量** |
| **B. 往本仓库挑单点** | 只搬 allreduce 特化 + argmax 融合 | 中高（接口已变） | 高（改坏现有 53.4 基线） | 约 +30–40% |
| **C. 只换模型** | 本地已有 v3 magic，试跑 QUASAR artifact | **低（先验证）** | 低 | 未知，可能已到 100+ |

### 关键未知：本地 NVFP4 的真实基线

本仓库 HANDOFF 记的是 "NVFP4 ~40 tok/s (batch 1)"，但同一份代码的上游记录是
V100-PCIe 短上下文 219 tok/s。**40 这个数很可能缺 MTP / 缺 Graph / 是短 prompt 的固定开销**。
在没测出本地 NVFP4 85K 真实数字之前，方案 A/B/C 的收益都是猜的。

⇒ **Phase 0 必须先做这件事。**

## 四、计划

### Phase 0 — 基线体检（阻塞在 GPU 占用）
两张卡现在被本项目自己的 `ninfer-serve.exe`（PID 24468）占满 15.9/16.4 GiB。
需要停服务才能测。

要测的四个点，同 artifact、同 prompt、greedy、513 token 窗口：

| 配置 | 目的 |
|---|---|
| NVFP4，MTP0 | 无投机基线 |
| NVFP4，MTP3 + `--lm-head-draft` | 与文档 219 对照 |
| NVFP4，MTP5 | 本地 bench 上限 |
| NVFP4，chunk 4096 vs 1024 | 确认 prefill 档位 |

判据：若 MTP3 能到 150+，说明本地代码没问题，差距全在 QUASAR 模型 + 长上下文 kernel ⇒ 走 C 或 A；
若只有 60–80，说明 decode 路径本身落后 ⇒ 走 A。

### Phase 1 — 模型可行性（不占 GPU 显存）
`ninfer.exe inspect` 打开 QUASAR NVFP4 v3 artifact，验证 identity 绑定。
本仓库 `reader.cpp:1217` 已有 v3 magic 分支，但绑定是否覆盖 QUASAR 的 mixed FP8/NVFP4 需要实测。
障碍：23.7 GB 下载，本机到 GitHub/HF 实测约 30 KB/s（7.3 MB 的源码 zip 用了 294 s）。

### Phase 2 — 按 Phase 0/1 结论二选一
- 选 A：在 `perf/v100x2-windows` 分支上重放 Windows 层，先过 `configure → build → inspect`，
  再过 `--no-cuda-graph` 一致性门，最后跑 85K 对照。
- 选 B：在 `perf/v100x2-port-study` 上逐个单点，每点独立 A/B，回归门用现有 121/133 测试。

### 已排除
- **WSL2 直接跑 V100X2**：`wsl -d Ubuntu` 内 `nvidia-smi` 返回 `Failed to initialize NVML: N/A`，
  `/dev/dxg` 存在但 TCC 模式的 V100 无法直通。零移植路线不通。

## 五、待决策

1. 是否现在停 PID 24468 跑 Phase 0 基线？停多久（4 个配置 × 加载 75 s + 推理）。
2. 若 Phase 0 结论指向方案 A，是否接受"在 V100X2 之上重建 Windows 树"（等于把本仓库的
   Windows 工程化成果迁走，而不是并入）。
3. QUASAR artifact 的下载预算（23.7 GB，按当前带宽约 9 天，需要换镜像或代理）。
