# port/gdn-split10 · GDN controls 单 launch（split-10 精确保序）

提交 `a039c2c7`，分支 `port/gdn-split10`，基线 `result/packet-allreduce`（`df1f2958`）。

## 改动摘要

TP2 的 GDN controls 原本是两个 kernel：partial 核把每个 (head, token) 的 10 个 split 点积写进
global workspace，reduce 核再读回来做 epilogue。改成**一个 CTA 管一个 (head, token)**，
10 个 warp 各管一个 512 元素的 K 切片，partial 走 shared memory 而不是 global。

**没有任何数值被重新结合**，四层都对得上：

| 层 | 两者一致 |
|---|---|
| 元素→lane | `vec = lane; vec < 64; vec += 32` |
| 向量内 8 个 fmaf 次序 | `wf0.x → wf0.y → wf1.x → … → wf3.y`，权重在前 |
| warp 归约 | 同一个 `warp_reduce_sum` |
| 跨 split | 10 路上升序线性链，初值 0.0f |

`x` 是 BF16 位拷贝，两个中间量都是 FP32 精确写出/读回，不引入额外舍入。
输出 `g`/`beta` 是 FP32，因此本项是 **A 类**，不需要三段式判据。

规模：**1 个文件 +104 行（含注释），0 删除，wrapper / runtime 一行未改。**

生效窗口 `T ∈ [2, 8]`（`kMaximumMtpDraftTokens = 5` ⇒ verify 的 T ≤ 6，每步都命中）。
`T == 1` 仍走 gemv，`T > 8` 仍走两核。

## 测量

参照 = `packet-allreduce`（复测值 135.594 / 945.623）。

### 第 1 轮（干净：无并行任务，GPU 空载 10 MiB）

| rep | prefill tok/s | decode tok/s | sha1 | acc/drf/rd |
|---|---:|---:|---|---|
| 0 | 947.233 | **136.318** | 06287348665d3648 | 406/421/106 |
| 1 | 946.761 | **135.871** | 06287348665d3648 | 406/421/106 |
| 均值 | 946.997 | **136.095** | A 类逐位一致 | 逐项相同 |

相对 packet：prefill **+0.145%**、decode **+0.369%**。

门槛（tier1，相对 packet）：prefill ≥ 955.079、decode ≥ 136.272。
**decode 差 0.13 个百分点未达；且 rep0 过、rep1 不过。**

### 第 2 轮：**作废**（测量窗口被污染）

| rep | prefill | decode |
|---|---:|---:|
| 0 | 944.087 | **129.948** |
| 1 | 942.170 | 136.024 |

`129.948` 几乎等于原始基线 129.727 —— 不是变慢，是 packet 的收益整轮消失。

**归因（已排除 GPU 侧）**：两 rep 总时长几乎相同（67.490 / 67.452 s），
但 rep0 的 decode 慢 0.176 s、prefill 快 0.133 s ⇒ 是 host 侧干扰，不是 GPU 降频。
当时我正在派一个全仓 Grep 的准备代理。decode 每步有 128+ 次跨卡 collective，
对 launch 延迟高度敏感，host CPU 争用会直接抬高它。

**结论：r2 不计入。这是操作失误（违反「测量独占」），不是被测代码的问题。**

## 判定：待裁决（PENDING），且暴露出协议噪声地板问题

- oracle **A 类**，逐位一致 → 不触发硬否决
- decode +0.369% < 门槛 +0.5%
- 但更严重的是：**本协议（2 reps）的噪声地板就在 ±0.2~0.33% 量级**
  （本项 r1 两 rep 相差 0.33%；packet 两次测量相差 0.23~0.33%）
  ⇒ **2 reps 的协议在原理上无法裁决 0.5% 以下的差异**，而本项的目标值恰好落在这个区间

因此本项的"未过门槛"不是"证明无效"，而是"这个协议分辨不出来"。
要真正定论，需要换成重复次数更多的测量（例如 `-r 6`），或做 kernel 级微基准。
**建议在统一裁决时按"证据不足"处理，而不是按"性能不足"处理。**

## 资产

- 报告：`_scratch/port-log/gdn-split10-r1.json`（有效）、`gdn-split10-r2.json`（作废）
- 日志：`gdn-split10-r1.log` / `gdn-split10-r2.log`
- 分支：`port/gdn-split10` @ `a039c2c7`
- 勘察：`plan-gdn-split10.md`（含四层求和顺序的逐条对账，以及 +1.76% 的三件事拆解）

## 顺带修正（勘察产物，与本次实现无关）

- `queue-v2.md` 说"fork 已有 `bf16_gdn_norm_gating_proj_27.cu`，融合部分已存在" **不准确**：
  那是 norm+controls 融合，**只服务 tp1**（`variant.cpp:746-749` 注释明说 tp2 没有 split 形式）。
  真正已具备的是 `gdn_projected_conv.cu`（projection/conv/record 融合）。
  三件事里缺的确实只有 split-10，结论方向不变。
