# V100-Duo 移植战役 · 收官报告（2026-10-26）

> 本文是战役总结。断点恢复细节仍以 `STATE.md` 为准；单项详情见各 `<feat>.md`。

## 最终确认（final-confirm，master `f9e5ce82` 全量复测）

| 指标 | 原始基线（`34412f4f`） | 收官实测（2 rep 均值） | 累积提升 | 门槛 | 判定 |
|---|---|---|---|---|---|
| prefill | 945.799 tok/s | **1187.077** tok/s（1188.778 / 1185.376） | **+25.511%** | +1.0% | **PASS** |
| decode | 129.7266 tok/s | **135.7862** tok/s（135.7612 / 135.8111） | **+4.671%** | +0.5% | **PASS** |
| oracle | sha1 `06287348665d3648` | **同 sha1，A 类逐位一致**（acc/drf/rd 406/421/106、al=4.8301887 两 rep 全等） | — | — | ✅ |

两项 delta 均远超 MDD(95%, 2reps)（prefill 0.481% / decode 0.433%），可分辨。
tag：`result/final-confirm` = `f9e5ce82`（已核对 ≠ `34412f4f`）。

## 战役时间线与判决定案

| # | 项 | 结果 | 证据 |
|---|---|---|---|
| 0 | 原始基线实测 | prefill 945.799 / decode 129.727 | `baseline.json` |
| 1 | NVFP4 CUTLASS TileN=256 | 实测 +0.473% < +1.0% 门槛，**不保留**（A/B 同 binary 对照） | `tilen256.md` |
| 2/3 | five-query packed KV vector16 / KV 寄存器预取 | **否决（源码核证）**：目标核不存在、机制已在 key_split | `STATE.md` 第 2/3 节 |
| 4 | 两 epoch packet 邮箱 collective | **合并** `df1f2958`，decode **+4.523%**（复测 135.594），oracle 逐位一致 | `merge-packet.md` |
| 5 | 跨卡 argmax 合并 | **出队**：受控实测 +0.046%（噪声内），400+ 行不值 | `candidates.md` §3 |
| 6 | 融合 RMSNorm 进 collective | **否决**：oracle A 类但 decode −2.748%（归约几何退化） | `rmsnorm-wiring.md` |
| 7 | projection epilogue 发布 peer packet | 未立项（折算 +0.64~0.70% 过线但最贵，收益余量薄于 MDD） | `queue-v2.md` |
| 8 | fused shard-winner collective | **出队**：受控实测 +0.73% → 折算 ~+0.3%，不过门槛 | `candidates.md` |
| 9 | GDN controls 融合（split-10） | decode +0.369% < 门槛 +0.5% 且 < MDD 0.433%（**本来测不出**），**不保留** | `gdn-split10.md` |
| 10 | prefill 相位归因 + prefill-down-cutlass | **合并** `f9e5ce82`：plain rank 宽 T 改走 cutlass，prefill **+25.274%**，oracle 逐位一致 | `prefill-down-cutlass.md` |
| 11 | GDN chunked prefill Volta 移植（S1-S4） | 正确性 A 类逐位一致，性能 **−24.230%**，**否决**（ema 仿真层吃光 tensor core 优势） | `gdn-chunked-volta.md` §6 |

方法学沉淀：判据 v3（目标项卡门槛、非目标项卡 MDD）、噪声地板常数化进 judge.py、
阶梯值折算系数 ×0.41~0.45（由 TileN/packet 两样本反推）、lab worktree 常驻工作流（`LAB-WORKFLOW.md`）。

## 主分支终态

- `master` = `f9e5ce82`，包含全部采纳改动（packet + prefill-down-cutlass），工作区干净。
- **无需再合并任何东西**：GDN chunked 已否决，封存在 lab 分支 `port/gdn-chunked-s1`
  （S1-S4 四个提交：`2d3641ca`/`d2fb2323`/`3bbb3792`/`9e97b54d`/`c01799dd`，
  含 ema 仿真层、全部 fp64 对拍 harness 与微基准——败于性能，非败于正确性）。
- worktree `.agents/worktrees/port-packet-allreduce` 保留（已 detach 到 `f9e5ce82`）。

## 剩余方向盘点（全部有结论，见 2026-10-25 23:04 会话分析）

- **vector16 SASS 探针**：唯一低成本项（只看 `LDG.E.128` 是否生成，不占测量轮次）；
  但 8 维/lane 布局障碍是硬的，大概率"技术上不成立"。
- **#7 epilogue publish**：唯一纸面过线存量项，折算后余量 ~0.15-0.2 点（贴 MDD），
  立项前需用 decode 直测数据重校折算系数。
- **GDN chunked 复活**：state_passing fp16 预缩放等三条路，需先把 −42%/调用翻正，
  ROI 劣于其他方向（详见 `gdn-chunked-volta.md` §6）。
- prefill GEMM（60% MFU 已到 cuBLAS 水位）、attention（已是 VoltaFlash）、通信/norm/conv：无油水，不碰。

## 硬件教训（留给下一场战役）

`-j4` 满载编译两次无 dump 硬挂（WHEA Bus/Interconnect 风暴，见 `incident-20261025-freeze.md`），
两卡只协商到 PCIe 3.0 x4 → 永远 `-j2`；测量前后查 `nvidia-smi`。
