# 交接文案（2026-10-26 00:45 · 第二阶段：收官后再挖掘）

> **■ 本战役已于 2026-10-26 18:24 终结，不要再开工。**
> 六项判决全部落地：C1/C2/C4/#7 出队、C5 结案、**C3 封存**（road#1 内核实测 1.31×，
> mma 只占 state_passing 35%，chunked 5.800 vs recurrent 5.33ms —— 见 `c3-real-check.md`）。
> 终态 master `77a493a5` + docs `e7a53442`，prefill **1187.077（+25.5%）** / decode
> **135.786（+4.7%）**，oracle A 类，tag `result/final-confirm`。**无挂起项、无待办。**
> 若将来要重启优化，唯一还站得住的方向是"recurrent GDN 自身是否带宽受限"（1 小时的 roofline
> 判断即可定性），其余候选池已清空。下文为第二阶段开工时的原始交接，留档。

> 复制本文件即可接手。目标、状态、判据、工具、教训、下一步都在这里，不需要重新做源码分析。
> 上一阶段（至 GDN chunked 否决 + final-confirm 收官）的完整记录见 `campaign-close.md`。

---

## 〇、本阶段工作流（用户指令，硬性顺序）

1. **以合并后版本为基线**：一切对照都相对 master `f9e5ce82`（tag `result/final-confirm`）。
2. **尽调前置**：候选先从**理论推算 + 参考项目源码**两个维度评估，**排除不合适的**。
   尽调不写产品代码、不编译、不测量。每项产出一张尽调卡（预期/机制/风险/kill 条件）。
3. **合适选项才进快速验证**：快速验证 = 探针/微基准/单轮 measure，过了再走全量测量
   → 判定 → 写档 → commit → tag 的既定流程。

**立项红线**：折算预期 < 0.5%（decode）/ < 1.0%（prefill）的直接不立项；
理论值一律先乘折算系数 **×0.41~0.45**（由 TileN 0.411 与 packet 0.447 两实测样本反推）；
低于 MDD 0.433%（decode）/ 0.481%（prefill）的项"做了也测不出"，只存档。

---

## 一、目标

**在已合并基线上继续寻找可验证的性能提升，验证后合并进主分支。**
prefill 与 decode 两个指标都算，各自有门槛。

主仓库：`C:/Users/lwb6/Documents/git/NInfer-V100-Duo-win`
本机：Windows 11 + 2 × Tesla V100-SXM2-16GB（sm_70），TP2，PCIe 3.0 x4。

---

## 二、当前状态（= 新基线，2026-10-26 final-confirm 确认）

| | |
|---|---|
| **主分支 HEAD** | **`f9e5ce82`**（packet collective `df1f2958` + prefill-down-cutlass 已合并），tag `result/final-confirm` |
| **新基线实测**（2 rep 均值） | **prefill 1187.077**（1188.778/1185.376）· **decode 135.7862**（135.7612/135.8111） |
| oracle | **A 类逐位一致**：sha1 `06287348665d3648`、acc/drf/rd `406/421/106`、al `4.8301887` 两 rep 全等 |
| 累积收益（vs 原始基线 `34412f4f` = 945.799/129.727） | prefill **+25.511%**、decode **+4.671%** |
| 回滚 | `git reset --hard 34412f4f` |

**当前门槛**（judge 已输出，相对新基线）：

```
prefill ≥ 1198.948   (+1.0%)
decode  ≥ 136.4651   (+0.5%)
```

GDN chunked 已否决（−24.230%），封存于 lab 分支 `port/gdn-chunked-s1`（`c01799dd` 止），
**不要合并、不要重测**；复活条件见候选池 C3。

---

## 三、判据（v3，不变，已固化进 `judge.py`）

| 角色 | 判据 |
|---|---|
| 目标指标 | `delta ≥ 该项门槛` |
| 非目标指标 | `delta ≥ −MDD`（落在 MDD 内视为无变化） |

噪声地板（`noise-floor.md`，已成 judge.py 常数）：2-rep MDD decode **0.433%** / prefill **0.481%**；
噪声几乎全部来自运行之间 → 加运行次数才有效。**低于 MDD 的差异本协议测不出来。**
B 类判据（若 oracle 变类）：accepted/drafted/rounds 严格相等 + acceptance_length 相等
+ 差异位 ≤2% 且每差异位 margin <0.05（queue-v2 裁决 #1）。

---

## 四、已完成项的结论（不要重做）

| 项 | 结果 |
|---|---|
| packet 两 epoch 邮箱 collective | ✅ decode +4.523%，已合并 |
| prefill-down-cutlass（相位归因发现 down 投影两 rank 差 4 倍） | ✅ prefill +25.274%，已合并 |
| TileN=256 / GDN split-10 / argmax 合并 / shard-winner | ❌ 全部低于门槛或 MDD |
| 融合 RMSNorm（站点 A/B） | ❌ decode −2.748%（归约几何退化），连带否决第 7 项 |
| W8 scale 32B 超块 | ❌ oracle FAIL（草稿坏），已回退 |
| **GDN chunked Volta 移植（S1-S4）** | ❌ oracle A 类但 prefill **−24.230%**：ema 仿真层（bf16→4×f16、tf32→3×f16+shfl、ldmatrix→LDS.32）吃光 tensor core 优势；chunked 7.57ms/调用 vs recurrent 5.33ms。**败于性能，非正确性** |

 mixer 勘察定论：prefill 路由级收益已挖尽——GEMM 两侧 ~74 TFLOPS（60% MFU，cuBLAS 水位）、
attention 已是 VoltaFlash、通信/norm/conv 无油水。**剩余空间只在深水区（见候选池）。**

---

## 五、候选池（按理论空间排序，全部先尽调后动手）

> 参考源码位置：上游快照 `_scratch/ref-v100x2/ninfer-V100X2-master` 与 `_ref/`（只读）；
> 外部参考：llama.cpp（ggml-cuda GDN/quant epilogue）、vLLM + FLA（chunked GDN）、CUTLASS sm70 examples。
> 尽调卡格式：预期（折算后）→ 机制 → 参考源码证据 → 风险 → **kill 条件**。

### C1 · prefill 大消息 collective（D1 线索，理论空间最大）

- **线索**：`inventory-retained.md` D1 行——上游 TP4 实测 prefill ≈**+34.2%**（RS/AG 替代 one-shot
  allreduce）；prefill 的 26 MB collective 目前走 one-shot。
- **尽调要点**：① 数清 prefill 每 chunk 的 collective 次数与字节数（`mixer-survey.md` 的 trace 数据可复用）；
  ② TP2 下 RS+AG 与 one-shot 的理论流量差（TP2 时 RS+AG 总流量 = 2M，与 one-shot 相同，
  **收益只能来自流水/重叠，不是流量**——这是尽调必须先回答的问题）；③ 上游源码里 D1 的实现形态
  与它依赖的拓扑假设（TP4 NVLink？本机 PCIe x4 是否成立）。
- **kill 条件**：若上游收益本质是 NVLink 下的带宽倍增（PCIe x4 复制不出来），或本机 collective
  占 prefill 墙钟 < 5%，出队。

### C2 · GEMM 融合反量化（上限 ~5% prefill）

- **尽调要点**：① fork 的 `nvfp4_cutlass_sm70` epilogue 还有多少挂钩点；② 参考项目里
  dequant-fused-GEMM 的实现（llama.cpp mul_mat 的 dequant 策略、CUTLASS EVT）在 sm70 的寄存器
  压力证据；③ 理论上限 = GEMM 时间的 ~7%（两侧已 61% MFU，能省的只是"反量化那次额外读写"）。
- **kill 条件**：若融合需改 tile 调度或寄存器翻倍导致占用率下降，或折算后 < 1.0% prefill，出队。

### C3 · GDN chunked 复活（三条路，`gdn-chunked-volta.md` §6）

前提：先把 **−42%/调用**（7.57 vs 5.33ms）翻正，再谈 +2% prefill。
- state_passing fp16 预缩放（最大项 2.90ms，MM1/MM2 是 tf32 路径）：尽调核心问题是
  **K/W 预缩放能否静态完成**（编译期或 per-model 常数）——若需运行时逐 chunk 动态范围检查
  （host sync）直接 kill；oracle 将变 B 类。
- prepare 的 T_inv 求逆同路（2.54ms）。
- ldmatrix 替代改 LDS.64/128（当前逐 lane LDS.32）：尽调 = 数一下 fragment 收集的 smem 事务数。
- **kill 条件**：尽调后若三项合计仍翻不正 −42%，永久封存该方向。

### C4 · `Streaming`/`Fp32Scale`（decode"地基"，fork 整块缺失）

- 上游阶梯第一行 `Parent FP32 epilogue + shape-specific streaming`（157.312）不是增量、是起点；
  fork 的 `nvfp4_volta_qpn_gemm.cuh` 无这些模板参数，scale 在 K 归约前乘。
- **尽调要点**：读上游源码确认它改了什么（scale 乘法时机？FP32 累加路径？）、为什么铺在最底层；
  理论上它影响的是**精度路径**而非吞吐——若只是精度改进、无性能机制，出队（本项目 oracle 必须 A 类，
  精度路径改动连 B 类都危险）。
- **kill 条件**：无明确吞吐机制，或会改 oracle 类。

### C5 · vector16 SASS 探针（唯一可直接做的"快速验证"）

- 尽调**已完成**（STATE.md 第 2/3 节）：key_split 是 8 维/lane，改 16B 要 16 维/lane，
  会瓦解 warp↔(token,dim) 归属、`key = warp + 8i` 不变式与 smem PStride。
- 剩余动作本身就是快速验证：临时拼 16B 加载编译一次，`cuobjdump -sass` 查 `LDG.E.128`。
  没生成 → 记"技术不成立"结案（不占测量轮次）；生成了才谈测量。**越界风险：跨页 int4 必须留边界分支。**
- **kill 条件**：SASS 无 128-bit → 永久结案。

### 曾被连带否决（尽调需先推翻否决理由才可复活）

- **#7 projection epilogue 发布 peer packet**：否决理由 = 消费端与 rmsnorm 融合是同一内核同一几何
  （T=5~6 归约退化）。尽调若能证明 prefill 侧（宽 T）几何不同、不受该退化影响，可重新评估；
  否则维持否决。折算 +0.64~0.70% decode 过线但余量薄（贴 MDD）。

---

## 六、工具（都在 `_scratch/port-bat/`）

```bash
# 构建（lab worktree，固定 -j2；只有改 CMakeLists 才传 configure）
JOBS=2 bash _scratch/port-bat/build-port.sh <feat> build

# 测量（约 4 分钟；自动前置检查 GPU 空载 → bench → judge）
bash _scratch/port-bat/measure.sh <feat> <ref>      # ref=baseline 用原始基线

# 单独重判
python _scratch/port-bat/judge.py <feat> --ref <ref> --target decode|prefill

# SASS 检查（C5 用）
_toolchain/cuda-12.8/bin/cuobjdump -sass <.exe 或 .obj> | grep -E "LDG.E.128"
```

lab worktree：`.agents/worktrees/port-packet-allreduce`（**目录名不要改**，Ninja 烧了绝对路径）。
分支在它内部切换；**主分支代码验证用 `git checkout --detach f9e5ce82`**（master 被主仓库占用，
worktree 里 checkout master 会报错）。

---

## 七、必须遵守的纪律（都是踩过的坑）

1. **shell cwd 每条命令重置回主仓库根**：git/构建命令一律 `cd <worktree> && ...` 单链或 `git -C`。
   本战役已三次险情（两次误操作落主仓库、一次 commit 落空）。
2. **测量独占**：measure.sh 运行期间不派代理、不编译、不做全仓搜索。
   曾因测量窗口起只读代理，decode rep0 掉 −4.5%。
3. **先提交再切分支**；不删 lab worktree、不 `git clean -xdf`（连热 build 一起删）。
4. **`-j2` 是安全上限**：`-j4` 两次 WHEA 风暴 + 无 dump 硬挂（`incident-20261025-freeze.md`）。
5. 中断三形态判别（硬件硬挂/WHEA 预警/后台任务回收）见 `STATE.md`——别混判，回收态直接重跑 ninja 续。
6. 子代理会被回收，关键工作主代理自己做；文档先落盘骨架、逐节追加。

---

## 八、文件索引（`_scratch/port-log/`）

| 文件 | 内容 |
|---|---|
| `STATE.md` | **权威入口**（已置收官横幅），先读这个 |
| `campaign-close.md` | 上一阶段收官报告：11 项全记录 + 方法学 + 剩余方向盘点 |
| `gdn-chunked-volta.md` | GDN 战役全程：S1-S4、归因链、复活三条件（§6） |
| `mixer-survey.md` | prefill 拆解与"路由级收益已挖尽"的定论 |
| `noise-floor.md` / `judge.py` | 噪声地板与判据实现 |
| `LAB-WORKFLOW.md` | lab worktree 规程 + 测量独占规则 |
| `inventory-retained.md` | 快照 retained 改动 × fork 现状（D1 行是 C1 线索） |
| `candidates.md` | 折算系数（×0.41~0.45）与优先级重校准 |
| `plan-prep-roadmap.md` 等 plan-* | 历史规划（多数已执行或否决） |

快照只读参考：`_scratch/ref-v100x2/ninfer-V100X2-master`、`_ref/`（**不要改任何东西**）。

---

## 九、一句话现状

**基线已是 prefill 1187 / decode 135.8（比战役起点 +25.5%/+4.7%），路由级收益挖尽。
下一阶段：先纸面尽调（理论 + 参考源码）筛掉不合适项，幸存者才进快速验证——
别再让 GDN chunked 这种"正确但更慢"的工程重演：它花了 S1-S4 四个阶段才在测量时发现方向性错误。**
