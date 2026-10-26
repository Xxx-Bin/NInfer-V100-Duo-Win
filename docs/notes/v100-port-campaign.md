# V100 Duo 移植与优化战役 · 终局记录

> 记录：2026-10-26 · 角色：record · 状态：**已终结，无挂起项**
>
> **本文件是该战役的唯一版本控制载体。** 过程日志、原始测量输出、尽调卡与规划文档
> 在 `_scratch/port-log/`（本地，`.gitignore` 隔离，**不受版本控制**）。任何与本文件
> 冲突的本地记录以本文件为准；本文件不再引用本地文件作为判决依据，所有关键数值
> 均已内联。
>
> 取代 `docs/notes/v100x2-port-assessment.md`（2026-10-24 的前置评估，其 Phase 0/1/2
> 计划已在此执行完毕，结论见下文）。
>
> 当时硬件：Windows 11 · 2 × Tesla V100-SXM2 16 GB · `sm_70` · CUDA 12.8 · TP2 ·
> 卡间 PCIe 3.0 x4（**非 NVLink**）· 30.9 GB RAM。

---

## 一、终态指标

| 指标 | 战役起点 `34412f4f` | 终态 `77a493a5` | 累积 | 门槛 | 判定 |
|---|---|---|---|---|---|
| prefill | 945.799 tok/s | **1187.077** tok/s | **+25.511%** | +1.0% | PASS |
| decode | 129.7266 tok/s | **135.7862** tok/s | **+4.671%** | +0.5% | PASS |
| oracle | sha1 `06287348665d3648` | 同 sha1，**A 类逐位一致** | — | — | PASS |

终态 2-rep 实测：prefill 1188.778 / 1185.376；decode 135.7612 / 135.8111。
两项 delta 均远超 MDD(95%, 2 reps)（prefill 0.481% / decode 0.433%），可分辨。
tag `result/final-confirm` → `f9e5ce82`；其后 docs 提交 `e7a53442` 与
`docs(port-log)` 若干，代码面终态 `77a493a5`。

**只采纳了两项改动**：两 epoch packet 邮箱 collective（decode +4.523%）、
prefill-down-cutlass（prefill +25.274%）。其余候选全部否决或出队。

### 测量口径与复现

上表数字来自 `ninfer_bench` 的合成语料基准，**不是**自然结束请求的速率，
与 README 的 MTP3/4/5 表不可比（那张表是 74-token prompt、峰值 5s 区间与全程均值）。
引用上表时必须带完整口径：

```bash
ninfer_bench --weights /path/to/qwen3_8_27b_nvfp4.ninfer \
  --corpus bench/fixtures/bench_corpus.ids \
  -pg 60000,512 --prefill-chunk 2560 \
  --kv-dtype int8 --mtp-draft-tokens 4 --lm-head-draft \
  --tp 2 --devices 0,1 -r 2 --warmup 1 --capture-generation
```

| 项 | 值 |
|---|---|
| artifact | `qwen3_8_27b_nvfp4.ninfer`（23,719,715,844 B） |
| workload | `bench/fixtures/bench_corpus.ids`，60,000 token prefill + 512 decode |
| prefill chunk | 2560 |
| KV | INT8 |
| MTP | `--mtp-draft-tokens 4 --lm-head-draft`（MTP4 + 优化草稿头） |
| 环境 | Windows 11 · 2 × Tesla V100-SXM2 16GB · TP2 devices 0,1 · CUDA runtime 12.8 / driver 12.9 |
| 统计 | `-r 2 --warmup 1`，2 rep 均值；MDD 见第四节 |
| oracle | 双方 sha1 `06287348665d3648`，A 类逐位一致 |

`ninfer_bench` 会把这条命令行原样写进报告 JSON 的 `command` 字段，所以任何一份历史报告
都能自证口径。原始报告（`baseline.json`、`final-confirm.json`）**已不在版本控制内**，
上面这条命令是唯一可重建口径的载体。

---

## 二、判决定案（17 项，全部有结论，勿再复议）

### 第一阶段

| # | 项 | 判决 | 依据 |
|---|---|---|---|
| 0 | 原始基线实测 | 记录 | prefill 945.799 / decode 129.727 |
| 1 | NVFP4 CUTLASS TileN=256 | 不保留 | 同 binary A/B 实测 +0.473% < +1.0% 门槛 |
| 2 | five-query packed KV vector16 | 否决 | 源码核证：目标核不存在，机制已在 key_split |
| 3 | KV 寄存器预取 | 否决 | 同上，机制重复 |
| 4 | 两 epoch packet 邮箱 collective | **合并** `df1f2958` | decode **+4.523%**（复测 135.594），oracle 逐位一致 |
| 5 | 跨卡 argmax 合并 | 出队 | 受控实测 +0.046%（噪声内），400+ 行不值 |
| 6 | 融合 RMSNorm 进 collective | 否决 | oracle A 类但 decode **−2.748%**（归约几何退化，T=5~6） |
| 7 | projection epilogue 发布 peer packet | 维持否决 | 折算 +0.64~0.70% 过线但余量薄于 MDD；且与 #6 同内核同几何 |
| 8 | fused shard-winner collective | 出队 | 受控实测 +0.73% → 折算 ~+0.3%，不过门槛 |
| 9 | GDN controls 融合（split-10） | 不保留 | decode +0.369% < 门槛 +0.5% 且 < MDD 0.433%（本来测不出） |
| 10 | prefill 相位归因 + prefill-down-cutlass | **合并** `f9e5ce82` | prefill **+25.274%**，oracle 逐位一致 |
| 11 | GDN chunked prefill Volta 移植（S1-S4） | 否决 | oracle A 类但 prefill **−24.230%**；**败于性能，非正确性** |

### 第二阶段

| 项 | 判决 | 依据 |
|---|---|---|
| C1 prefill 大消息 collective | 出队 | 占比 ≤1.4%（kill 线 5%）+ TP2 流量守恒（RS+AG 总流量 = 2M = one-shot） |
| C2 GEMM 融合反量化 | 出队 | 折算 0.15~0.25% prefill |
| C3 GDN 复活三路 | **封存（内核实测）** | road#1 内核只兑现 1.31×；chunked 5.800 vs recurrent 5.33 ms |
| C4 `Streaming`/`Fp32Scale` | 出队 | 改 oracle 类 / 折算 +0.19% < MDD |
| C5 vector16 SASS 探针 | 永久结案 | `cuobjdump -sass` 0 条 `LDG.E.128`，int4 被拆 2×64 位 |
| #7 epilogue publish | 维持否决 | prefill +0.4~0.7% < 1% 红线 |

---

## 三、C3 封存：三条证据链与两条撤回

C3 是本战役唯一被**实测推翻**的立项，也是唯一留下方法学教训的一项。

**立项时的纸面预期**：GDN chunked 复活三条路，state_passing 走 fp16 预缩放，
微基准显示 mma 路径有 5.12× 加速，折算后预期待可观。

### 证据 1 — 真实数据复核，撤回「静态预缩放成立」

合成统计曾得出「chunk 内 span ≤ 25~28 bits，fp16 静态预缩放成立」。
**该结论撤回。** 用真实权重与真实输入 dump 复核后：

- subnormal 占比 **16.4~30.7%**（合成数据远低）
- state max **61.2**
- 根因：真实 g 的 chunk 内累积和到 **−10 以下**，动态范围远超合成假设

即：合成统计的分布假设与真实数据不符，静态预缩放在真实数据上不成立。

### 证据 2 — 判据 P1 作废，改 P2

原判据 P1 规定绝对容差 **9.2e-4**。**该判据作废**——因为**现状 bf16 路径在同一数据下
也超标 8~21 倍**：判据比现状实现还严，等于用不可能达到的标准否决一项改动。

改为 P2（相对现状 **≤ +10%**）后，f16 全局静态缩放通过，per-chunk 缩放无必要。

### 证据 3 — 内核落地实测，1.31× 而非 5.12×

road#1（3 product → 1 product）接入真实 `state_passing` 内核后：

| 场景 | 加速 |
|---|---|
| 微基准（孤立 mma 循环） | 5.12× |
| **真实内核 `state_passing`** | **1.31×** |

反推 mma 只占 `state_passing` 的 **35%** → 该内核**不是 mma-bound** → 剩余 road#3
最好只能打平。叠加 chunked 整体 **5.800 ms vs recurrent 5.33 ms**（仍慢 8.8%），
判定无复用可能。

### 由此确立的纪律

> **微基准倍率不得线性外推到内核实测。先量目标块在真实内核里的时间占比，
> 再用 Amdahl 折算。** 5.12× 的孤立加速在占比 35% 的块上只能兑现 ~1.3×。

同族教训（第一阶段 #11）：GDN chunked 移植花了 S1–S4 四个阶段才在测量时发现方向性
错误——**先纸面尽调筛掉不合适项，再动手**。

### 分支现状

C3 判定无复用可能，全部 lab 分支已删除，`s5-prof` worktree 已移除，`git branch` 只剩
`master`。`port/gdn-chunked-s1`（S1–S4：`2d3641ca`/`d2fb2323`/`3bbb3792`/`9e97b54d`/`c01799dd`，
含 ema 仿真层、fp64 对拍 harness、微基准）与 `port/gdn-road1-f16`（road#1）**代码已不可取得**，
仅结论与数字保留在本文件。

---

## 四、判据与噪声地板（复用时直接照抄）

| 角色 | 判据 |
|---|---|
| 目标指标 | `delta ≥ 该项门槛`（decode +0.5% / prefill +1.0%） |
| 非目标指标 | `delta ≥ −MDD`（落在 MDD 内视为无变化） |

**噪声地板**（2-rep，95%）：MDD decode **0.433%** / prefill **0.481%**。
噪声几乎全部来自运行之间 → **加运行次数才有效**，加 rep 内采样无效。
低于 MDD 的差异本协议测不出来，立项即浪费。

**B 类判据**（若 oracle 变类）：accepted / drafted / rounds 严格相等 +
acceptance_length 相等 + 差异位 ≤2% 且每差异位 margin <0.05。

**立项红线**：折算预期 < 0.5%（decode）/ < 1.0%（prefill）不立项。
理论值先乘折算系数 **×0.41~0.45**（由 TileN 0.411 与 packet 0.447 两实测样本反推）。
低于 MDD 的项「做了也测不出」，只存档。

---

## 五、必须遵守的纪律（都是踩过的坑）

1. **`-j2` 是编译安全上限。** `-j4` 两次触发 WHEA Bus/Interconnect 风暴并无 dump 硬挂。
   两卡只协商到 **PCIe 3.0 x4**，这是硬约束。测量前后查 `nvidia-smi`。
2. **测量独占。** 测量窗口内不派代理、不编译、不做全仓搜索。
   曾因起只读代理导致 decode rep0 掉 −4.5%。
3. **shell cwd 每条命令重置回仓库根**：git/构建一律 `cd <worktree> && ...` 单链或 `git -C`。
   本战役三次险情（两次误操作落主仓库、一次 commit 落空）。
4. **先提交再切分支**；不删 lab worktree、不 `git clean -xdf`（会连热 build 一起删）。
5. **中断三形态要分清**：硬件硬挂 / WHEA 预警 / 后台任务回收。回收态直接重跑 ninja 续，
   别当硬挂处理。
6. **子代理会被回收**，关键工作主代理自己做；文档先落盘骨架、逐节追加。
7. **微基准 ≠ 端到端**（见第三节）。

---

## 六、工具位置（本地，不受版本控制）

| 工具 | 路径 | 用途 |
|---|---|---|
| 构建 | `_scratch/port-bat/build-port.sh <feat> build` | `JOBS=2` 固定；只有改 `CMakeLists.txt` 才传 `configure` |
| 测量 | `_scratch/port-bat/measure.sh <feat> <ref>` | 约 4 分钟；自动前置检查 GPU 空载 → bench → judge |
| 重判 | `python _scratch/port-bat/judge.py <feat> --ref <ref> --target decode\|prefill` | 噪声地板已固化进常数 |
| SASS | `_toolchain/cuda-12.8/bin/cuobjdump -sass <obj> \| grep LDG.E.128` | 指令生成核查 |
| lab worktree | `.agents/worktrees/port-packet-allreduce` | **目录名不要改**（Ninja 烧了绝对路径）；已 detach 到 `f9e5ce82` |
| 参考源码 | `_scratch/ref-v100x2/ninfer-V100X2-master`、`_ref/` | **只读，不要改任何东西** |

主分支代码验证用 `git checkout --detach f9e5ce82`（master 被主仓库占用，
worktree 里 checkout master 会报错）。

---

## 七、空间已排空：不需要再挖的方向

- **prefill 路由级收益已挖尽**：GEMM 两侧 ~74 TFLOPS（60% MFU，已到 cuBLAS 水位）、
  attention 已是 VoltaFlash、通信 / norm / conv 无油水。
- **候选池已清空**：C1–C5 与 #7 全部有终局判决，无一存活。
- **GDN chunked**：代码已不可取得，结论为「败于性能」。复活需先把 −42%/调用翻正，
  ROI 劣于其他方向。

### 唯一还站得住的方向

> **recurrent GDN 自身是否带宽受限。** 一个约 1 小时的 roofline 判断即可定性。
> 这是 C3 三段证据链之外唯一没有被否决的问题是：既然 mma 只占 35%，
> 剩下 65% 在做什么、是否被显存带宽钉死。

若重启优化，从这里开始，不要重开 C3 的三条路。

---

## 八、本地证据仓（不受版本控制）

以下内容在 `_scratch/port-log/`，`.gitignore` 隔离，**可能随时丢失，且不属于版本控制**。
本文件不把它们作为判决依据——判决与数值已全部内联于上文。它们只是复核用的原始材料：

| 类别 | 内容 |
|---|---|
| 测量数值快照 | `baseline.json`、`confirm-a/b.json`、`noisefloor-r1~r3.json`、`final-confirm.json`、`prefill-trace1~6.json`、`gdn-split10*.json`、`tilen256*.json`、`packet-allreduce*.json` 等 35 份 |
| 测量 stdout | 上述 JSON 对应的 `.log`，以及 `s5-0-mma-gate.log`、`flash16-probe.log` |
| 代码差异 | `c3-gdn-dump.diff`、`c5-vec16-probe.diff`、`tilen256.diff` |
| 硬件证据 | `whea-raw.txt`（WHEA Bus/Interconnect 报错原文） |
| 过程文档 | `HANDOFF.md`、`STATE.md`、`campaign-close.md`、`c3-real-check.md`、`gdn-chunked-volta.md`、各 `plan-*.md` 等 34 份 |

**取舍说明**：这些文件曾有 131 份被强行纳入 git（含 6.36 MB 构建日志，其中一份 4.6 MB
是 MSVC `/showIncludes` 的乱码刷屏，零信息量且 zlib 压不动）。最终改为「一个自包含
记录 + 本地证据仓」：判决与数值进版本控制，原始输出留本地。代价是原始输出失去历史，
换取的是记录可读、可 diff、不会被 6 MB 噪声淹没。
