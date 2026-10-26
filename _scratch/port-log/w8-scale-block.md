# W8 scale 32 字节超块读取 · **失败，已回退**

分支 `port/w8-scale-block`（已删除，无提交），基线 `result/packet-allreduce`。
改动：`src/ops/linear/w8/w8_volta_qpn_gemm.cuh` +25 行，试图把 scale 的 DRAM 扇区放大从 4× 降到 1×。

## 动机（数字来自代码注释的实测，不是推算）

`w8_volta_qpn_gemm.cuh:113-117` 原文：

> "a 2-byte load per lane fetches a whole 32B sector for it -- a 16x amplification that **measured as
> 1.89 GB of DRAM read** against a 1.31 GB weight. Reading the block's kBlk scales as one vector
> **cuts that to 4x**."

| | 标称 | 实测 DRAM |
|---|---:|---:|
| 权重 codes | 1.31 GB | 1.31 GB |
| scales（4× 放大后） | 0.145 GB | **0.58 GB** |
| 合计 | 1.35 GB | **1.89 GB** |

`1.89 GB / 2987 µs = 633 GB/s` = 天花板（794）的 **80%** → 该 kernel 是**带宽受限**的，
所以砍掉 0.435 GB 的无用流量理论上应该换来时间。预期 round 级 +0.4~1.0%。

## 改法

`kBlk` 同时管代码宽度（`cw[2*kBlk]`）和 scale 宽度（`scv[kBlk]`），直接改成 16 会让
`cw[32]` 占 128 个寄存器 → 必 spill。所以只动 scale 侧：

- 保留 `kBlk = 4` 的代码读取与累加顺序（否则改变 FP32 求和顺序 → B 类）
- 外层按 16 group 对齐（`sb = gb & ~15`）一次性读 32 B 到 `sc16[16]`（8 个寄存器）
- 内层仍按 4 group 取代码，scale 从 `sc16[(gb+e) & 15]` 取
- 越界（`sb + 16 > gend`）回退原 uint2 路径

**warp→group 的划分 `g0 = warp*gq` 完全不动** —— 它决定累加顺序，改了就是 B 类。

## 结果：oracle FAIL

| rep | prefill | decode | sha1 | acc/drf/rd | alen |
|---|---:|---:|---|---|---:|
| 0 | 946.086 | **74.440** | 06287348665d3648 | **319/766/193** | 2.653 |
| 1 | 945.817 | **73.721** | 06287348665d3648 | 319/766/193 | 2.653 |

对照 `packet-allreduce`：`406/421/106`、alen 4.830、decode 135.594。

**签名很有信息量**：

- **sha1 完全一致** → 目标头的 logits 没变，最终 513 个 token 也没变
- **接受计数崩了**（4.83 → 2.65），轮次从 106 涨到 193 → **草稿质量下降**
- 结论：**这个 QPN W8 核同时服务 MTP 的草稿矩阵**，我把它们的 scale 读错了；
  目标头走的是别的路径（或不同 shape），所以没受影响

decode 掉到 74 tok/s 完全是接受率崩溃的次生结果，不是 kernel 变慢。

## 为什么没能定位

按代码逐条推演，四处检查都成立：

| 检查项 | 结论 |
|---|---|
| `sb` 取整与索引 `(gb+e)&15` | 正确（warp1 的 g0=40 → sb=32，索引 8..11 = group 40..43） |
| 32 B 对齐 | 成立：`padded_groups = padded_shape[1]/kGroupK`，用到的 k 全是 512 的倍数 → 行距是 32 的倍数 |
| 尾部回退 | 成立：`blk < kBlk` 时走原 uint2 / 标量路径 |
| 累加顺序 | 未变（`gb` 仍按 4 递增） |

**但实测就是错的。** 说明真实原因在静态推演覆盖不到的地方（`padded_shape` 的实际取值、
不同矩阵的行距、`good` 语义、或 16 位索引在某种 shape 下的越界）。

## 决策：回退，不开调试战役

理由：
1. 收益上限只有 +0.4~1.0%（round 级），而且是"带宽受限"假设下的上限
2. 定位需要 kernel 级单测（逐 group 比对 scale 值），成本超过收益
3. 本项本来就是我自己加的，不在用户给的优先级里

已 `git checkout -- .` 回到 `df1f2958`，重建 `BUILD_OK`，分支删除。lab worktree 干净。

## 如果将来要重做，前置条件

1. **先写一个 kernel 级对拍**：给定一个小的 W8 矩阵（如 n=256,k=512），用 CPU 参考实现逐 group
   核对 scale 索引，确认 `padded_groups`、`kScaleBytesPerGroup`、行距的真实取值。
2. **先确认哪些 shape 真的走这个核**：用一行日志或 `--print-routing` 把 (n,k,T) 打出来，
   而不是靠 dispatch 代码推。
3. 一次只改一个 shape（先 n=5120 的 MTP 矩阵），确认 oracle 不变再推广。

## 仍然成立的结论

- 该 kernel 在 T=5 时 DRAM 读 1.89 GB，达 633 GB/s（80% 天花板）——**它确实是带宽受限的**，
  scale 放大占 23% 流量。这个事实不因本次失败而改变。
- 但"带宽受限"不等于"改对了就有收益"：本次失败说明**这条路径对正确性极其敏感**，
  而收益上限并不高。
