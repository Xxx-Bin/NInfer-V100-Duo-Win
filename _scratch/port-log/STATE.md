# 战役状态快照（冻结/重启后从这里恢复）

> **⚠️⚠️ 2026-10-26 09:00 撤回与更正（先读这段，它推翻下面第二条横幅的部分内容）⚠️⚠️**
>
> 1. **C3 量化前提已撤回**：`gdn_w_stats.log` 的"W/T_inv 静态 2 的幂预缩放成立、全配置
>    span ≤25~28 bits < fp16 29 bits"**只对合成输入成立**。真实 prefill chunk（T=2560，
>    64 层）实测：Af/expG/T_inv/W 的 subnormal 占比 **16.4%~30.7%**（合成 ≤6.8%），
>    state `max|S|` 到 **61.2**、U/v_new 到 43/40（合成 0.3~2.0），名义 span 到 double 下溢区。
>    根因：真实 g 的 64-token 累积和能压到 −10 以下（合成最狠只到 −4.7），`exp(G[r]−G[c])`
>    深衰减把尾部打进次正规。**判决见 `c3-real-check.md`**；静态缩放不能直接开工程，
>    必须先做 host 侧 fp16 误差传播模拟（A 全局静态 / B per-(head,chunk) 静态 / C bf16 基线）。
>    **✅ 模拟已完成（14:27）**：f16 精度**不低于**现状 bf16（v_new rel 1.296e-2 vs 1.949e-2，
>    state 7.506e-3 vs 7.404e-3）；**per-chunk 缩放无必要**（A/B 指标逐位相同，因 `max|T_inv|`
>    每 chunk 恒为 1.0）；**T_inv 留 fp32 不亏**（A0 与 C 的 state rel 相同）。⇒ 精度前提不构成否决，
>    C3 只剩性能问题（7.57 → <5.33ms/调用）。详见 `c3-real-check.md` 附二。
>    **裁决（17:35）**：预登记判据 P1（绝对容差 9.2e-4 / 7.8e-3）**作废**——现状 bf16 路径在同数据下
>    同样超标 8~21 倍，判据不可分辨；改 P2（相对现状增量 ≤+10%）后 A 过、B 过但零收益。
>    ⇒ **C3 立项，方案取 A（f16 + 全局静态 2 的幂缩放，T_inv 留 fp32，不走 per-chunk）**；
>    唯一否决面是性能；未验项 = `out` 张量与 decode acceptance 漂移（真机 S4 必测）。附三。
>    **★ 最终判决（18:17）：C3 封存。** road#1 已在内核里落地实测（非估值）：
>    state_passing 2.918 → **2.231ms（1.31×，不是微基准的 5.12×）**，chunked 6.488 → **5.800ms**，
>    仍比 recurrent 5.33ms 慢 0.47ms。反推 **mma 只占 state_passing 的 35%**，内核不是 mma-bound；
>    剩余 road#3 即使给满也只能刚好打平 → 折算 +0~1.5% prefill，压在门槛/MDD 上。
>    **纪律新增：微基准倍率不得线性外推到内核实测，必须先量目标块在真实内核里的时间占比。**
>    证据与现场见 `c3-real-check.md` 附五。
> 2. **lab worktree 已不存在**：`.agents/worktrees/port-packet-allreduce` 于 2026-10-26 07:01 前后
>    被删除（热 build 一并丢失）；`port-gdn-chunked-volta` / `port-gdn-split10` / `port-kv-vec16`
>    三个目录都是**孤儿（无 `.git` 链接）**。⇒ 在这些目录里跑 git 会**静默落到主仓**
>    （本会话已被咬一次：`git checkout --detach` 改到主仓 HEAD，环境已回滚）。
>    `git worktree list` 只剩 `bench-qwen3-8b-nvfp4` 是真的。
> 3. **"复用 build-win 增量"证伪**：旧对象（10-25 02:15）+ 新源码混编的二进制不带改动也崩
>    （`cudaErrorIllegalAddress`）。必须 `ninja -t clean` + 重配 + 冷编。本轮冷编 08:18 完成可用
>    （`-j2`，314 单元；期间 `cicc` 两次 ACCESS_VIOLATION，重试即过）。
> 4. **07:54 机器硬挂重启**（uptime 归零，重启前 1 小时 WHEA id=19 共 34 条）。
> 5. **2026-10-26 19:00 分支清零（按用户决定：C3 无复用可能）**：`feat/prefix-registry`、
>    `port/prefill-down-cutlass`、`port/rmsnorm-wiring`、`port/rmsnorm-fusion`、`port/gdn-split10`、
>    `port/gdn-chunked-s1`、`port/gdn-road1-f16` 全部删除，`s5-prof` worktree 已移除，
>    `git branch` 只剩 `master`。可达性：`prefill-down-cutlass` 与 `feat/prefix-registry` 的提交
>    在 master 历史里；`rmsnorm-wiring` / `gdn-split10` 仍由 `result/*` tag 固定；
>    **`gdn-chunked-s1`（S1-S4）、`gdn-road1-f16`（road#1）、`rmsnorm-fusion` 的代码已不可恢复**
>    （reflog 过期后回收）。⇒ 本文及 `c3-real-check.md` / `campaign-close.md` / `gdn-*.md` 中
>    对这三个分支名的引用，一律视为**历史指针**，只能读到结论与数字，代码需重做。
> 5. **2026-10-26 19:00 分支清零（按用户决定：C3 无复用可能）**：`feat/prefix-registry`、
>    `port/prefill-down-cutlass`、`port/rmsnorm-wiring`、`port/rmsnorm-fusion`、`port/gdn-split10`、
>    `port/gdn-chunked-s1`、`port/gdn-road1-f16` 全部删除，`s5-prof` worktree 已移除，
>    `git branch` 只剩 `master`。可达性：`prefill-down-cutlass` 与 `feat/prefix-registry` 的提交
>    在 master 历史里；`rmsnorm-wiring` / `gdn-split10` 仍由 `result/*` tag 固定；
>    **`gdn-chunked-s1`（S1-S4）、`gdn-road1-f16`（road#1）、`rmsnorm-fusion` 的代码已不可恢复**
>    （reflog 过期后回收）。⇒ 本文及 `c3-real-check.md` / `campaign-close.md` / `gdn-*.md` 中
>    对这三个分支名的引用，一律视为**历史指针**，只能读到结论与数字，代码需重做。
>
> **★★ 第二阶段尽调已出判决（2026-10-26 凌晨→清晨）★★**：见 `dd-cards-20261026.md`——
> C1/C2/C4/#7 **出队或维持否决**（各有量化依据）；C5 **永久结案**（探针 SASS 实测 key_split 0 条
> `LDG.E.128`，int4 被 ptxas 拆成 2×64 位；探针 diff 存档 `c5-vec16-probe.diff`，现场已还原）；
> **C3 立项条件成立**（W/T_inv fp16 静态预缩放指数统计全配置 span ≤25~28 bits < 29 bits 可用域，
> 证据 `port/gdn-chunked-s1` @ `de76e0b2` `_scratch/gdn_w_stats.log`；立项后第一动作 = 真实 chunk
> W/T_inv 实测分布复核）。接手先读 `dd-cards-20261026.md`，再读 `HANDOFF.md`。
>
> **★ 战役已收官（2026-10-26）★★**：最终确认 `final-confirm` 通过——master `f9e5ce82`
> 实测 prefill **1187.077（+25.511%）** / decode **135.7862（+4.671%）** vs 原始基线，
> oracle A 类逐位一致，tag `result/final-confirm`。**主分支无需再合并任何东西**
> （GDN chunked 已否决，封存于 lab 分支 `port/gdn-chunked-s1`）。总结见 `campaign-close.md`。
> 剩余候选方向盘点亦在该文档第四节——全部有结论，无待办。
>
> 每次冻结或重启后，读这个文件即可知道下一步，不要重新做源码分析。
> **接手请先读 `HANDOFF.md`** —— 已按用户指令重写为第二阶段文案（2026-10-26 00:45）：
> 以合并后版本为基线 + 尽调前置（理论+参考源码筛候选）+ 幸存者才进快速验证，候选池与 kill 条件见其第五节。

## 固定事实

- 仓库 `C:/Users/lwb6/Documents/git/NInfer-V100-Duo-win`
- **主分支 HEAD = `f9e5ce82`（packet collective + prefill-down-cutlass）**；原始基线 tag `baseline/port-study-34412f4` = `34412f4f`
- 原始基线实测：prefill 945.799 tok/s、decode 129.727 tok/s，oracle sha1 `06287348665d3648`
- **当前基线实测：prefill 1184.617、decode 135.808**（prefill-down-cutlass 合并后）
- **当前门槛：prefill ≥ 1196.463、decode ≥ 136.487**（+1%/+0.5% 相对新基线；旧门槛已作废）
- 累积收益：**decode +4.523%（129.727→135.594）+ prefill +25.274%（945.623→1184.617）**，oracle 全程逐位一致
- **最新战果：prefill-down-cutlass（`prefill-down-cutlass.md`）** —— 相位归因发现 TP2 down 投影 rank0/1 同形状差 4 倍（cutlass 1.9ms vs wide-T mma 8.1ms），plain rank 改走 `nvfp4_cutlass_sm70`（T≥33，+15 行）。swiglu 路由 A/B 为负结果（VoltaCutlass 已是最优）。下一格已修正：GEMM 两侧均 ~74 TFLOPS（60% MFU），融合反量化上限仅 ~5% prefill，降级；**mixer 勘察完毕（mixer-survey.md）：attention 已在 VoltaFlash、GEMM 已 60% MFU，路由级收益挖尽；剩余是 GDN 核心 kernel 深水区（5.33ms/层@88GB/s），投入前先精读 gated_delta_net kernel**
- 编译脚本 `_scratch/port-bat/build-port.sh <feat> [configure]`，**并行度锁定 `-j2`**（`JOBS=` 可覆盖，但不建议）
- **测量脚本** `bash _scratch/port-bat/measure.sh <feat> [ref]`：GPU 空载前置检查 → 按协议跑 bench → 自动调判定
- **判定脚本** `_scratch/port-bat/judge.py <feat> [--ref baseline]`：oracle 三段式 + 双门槛 + ladder 下一档
- 本地编译说明：仓库根 `BUILD-LOCAL.md`（已加入 `.git/info/exclude`，不进版本库，`git status` 不可见）
- 测量输出统一落 `_scratch/port-log/<feat>.json`

## 判定流程 v2（2026-10-25 用户调整）

测完 → 写 `port-log/<feat>.md` → **commit** → `tag result/<feat>` → **保留 lab worktree**（不再删）。
**不当场删分支**；整条队列跑完后统一裁决。第一步 commit 不能省（踩过：packet 改动未提交即被
`worktree remove --force` 删除）。收藏后必须核对 `git rev-parse result/<feat>` ≠ `34412f4f`。

## 构建流程 v3（2026-10-25 06:25）· lab worktree

**不再每项新建 worktree** —— 冷编译 222 个编译单元要 30 分钟，而每项只改 2~4 个文件。
统一在 `.agents/worktrees/port-packet-allreduce` 里迭代（目录名是历史遗留，**不要改名**，
Ninja 烧了绝对路径），分支在它内部切换。规程见 **`LAB-WORKFLOW.md`**。

- 每项：`git checkout -- .` → 确认 `git status` 空 → `git checkout -b port/<feat>` → 改 →
  `build-port.sh <feat> build`（**不加 configure**）→ `measure.sh <feat> <ref>` → commit + tag
- **不执行 `git worktree remove`**；**不要 `git clean -xdf`**（会删掉热 build）
- 新 API 优先放新头文件，避免触发 `allreduce.h`(15 个 include) / `linear_add.h`(7 个) 重编

## ★ 主分支状态（2026-10-25 08:55）

**`df1f2958` = packet collective 已合并进主分支**（fast-forward，2 文件 +132/−6）。
依据：oracle A 类 + 判据 v3（目标 decode +4.523% ≥ +0.5%；非目标 prefill −0.019% 在 MDD 0.481% 内）。
详见 `merge-packet.md`。回滚：`git reset --hard 34412f4f`。

主分支新基线门槛（继续叠加时用）：prefill ≥ **955.079**、decode ≥ **136.272**。

## 判据 v3（2026-10-25 08:50 生效，已固化进 `judge.py --target`）

| 角色 | 判据 |
|---|---|
| 目标指标 | delta ≥ 该项门槛 |
| 非目标指标 | delta ≥ −MDD（落在 MDD 内视为无变化） |

替代原合取判据。原判据要求两指标同时上涨，对结构性只影响一侧的改动物理上不可能成立
（TileN 的目标是 prefill，decode 本就不会动）——**这个缺陷在 packet 之前就已存在**。
v3 不放宽门槛：TileN（+0.281% vs +1.0%）与 GDN（+0.369% vs +0.5%）在 v3 下仍不合格。

## 噪声地板（2026-10-25 08:28 实测，已成常数写进 `judge.py`）

同一二进制 4 次运行 / 8 样本（`noise-floor.md`）：

| 指标 | 运行间 sd | 2-rep A/B 的 MDD(95%) |
|---|---:|---:|
| decode | 0.153% | **0.433%** |
| prefill | 0.170% | 0.481% |

**噪声几乎全部来自运行之间，不是 rep 之间**（rep 内只差 0.01~0.06%）→ 加 reps 帮助有限，
加**运行次数**才有效。两次运行 MDD 降到 0.306%。

对既有结论的重读：packet +4.523%（远高于 MDD，**成立**）；GDN +0.369%（**低于 MDD，本来
就测不出来**）；TileN +0.473%（勉强可分辨）。门槛 +0.5% 与 MDD 0.433% 只差 0.07 个点 ——
这是判据与测量能力不匹配。



## ★ 当前方向（2026-10-26 收官：无待办）

~~2026-10-25 17:45 启动的 GDN chunked Volta 移植已于 S4 否决~~（prefill −24.230%，
归因与复活路径见 `gdn-chunked-volta.md` §6；正确性 A 类、封存于 lab 分支）。
**下一步若重启优化**：先读 `campaign-close.md` 第四节的剩余方向盘点
（vector16 SASS 探针 / #7 epilogue publish / GDN 复活三条路，均有预评估结论）。

**S1 仿真层已完成并真机验证 PASS（2026-10-25 深夜，lab commit `d2fb2323`）**：
`src/ops/common/gdn_volta_mma.cuh` 提供签名兼容的 `ema_mma_bf16` / `ema_mma_tf32(_bits)`
（Ampere fragment 进出，内部复用 volta_mma_qk），64 trial 对拍 CPU 全过。
实测笔记与误差模型见 `gdn-s1-lab-notes.md`。

**S2 prepare_wy_wu 已完成 PASS（lab commit `3bbb3792`）**：fp64 oracle 对拍 worst rel
g 1.5e-7 / W 3.9e-3 / U 3.8e-3。注意 W 设备侧 d 维交错 `{0,4,1,5,2,6,3,7}`
（比较需 kWDeinterleave `{0,2,4,6,1,3,5,7}`）。

**S3 state_passing + output 已完成 PASS（lab commit `9e97b54d`）**：三段式验证
（kT=128、2 chunks、k 按 token L2 归一化模拟真实管线）：
- Round A 端到端：v_new bad=0（max_abs 7.8e-3）、state max_rel 9.2e-4
- Round B clean-input state：**h_chunk 逐位一致**、state_out 9.2e-4
- Round C clean-input output：bad=0、max_abs 7.8e-3（bf16 量化 + 64 步 A@v_new
  累积舍入；ema 层严格优于 sm_80 硬件 tf32）
S3 调试要点（防重蹈）：① 全流水线 NaN 的根因是 harness 输入 k 未归一化导致
T_inv=(I+Af)^-1 数值发散（Af 元素可达 ±30），**不是 ema 层 bug**——归一化后全消；
② 比较判据必须 NaN 感知（`e>tol` 对 NaN 为 false、std::max 不传播 NaN，
S2 的 PASS 曾被此盲区掩盖）；③ bf16 输出与 fp64 oracle 比较时应先把 ref round
到 bf16，消减小值元素用绝对容差（row_max 的 ~1%）而非相对容差。
harness：`_scratch/validate_gdn_pipeline.cu`（A/B/C 三轮）。

**S4 已否决（2026-10-25/26）**：trap 解除后全量实测 prefill 897.587（−24.230%），
oracle 却是 A 类逐位一致（优于预期的 B 类）。归因：ema 仿真层（bf16→4×f16 mma、
tf32→3×f16 + shfl、ldmatrix→逐 lane LDS.32）吃光 tensor core 优势，
chunked 7.57ms/调用 vs recurrent 5.33ms。按 S5 规则：不合并、不 tag、trap 还原，
S1-S4 全部工作封存于 `port/gdn-chunked-s1`（`c01799dd` 止）。

## 队列进度

| # | 功能 | 分支 | 状态 | 结论 |
|---|---|---|---|---|
| 1 | NVFP4 CUTLASS TileN=256 | port/tilen256 | **完成·不保留** | prefill +0.473%（A/B 同 binary 对照），未过 +1.0% 门槛；oracle 逐位一致。见 `tilen256.md` |
| 2 | five-query packed KV vector16 | — | **否决（已核源码）→ 仅做探针** | 目标核 `qpn_verify5` 本 fork 不存在；T=5 真路径是 key_split。按裁决只做 SASS 探针：能否生成 `LDG.E.128`，不生成即结案 |
| 3 | KV 寄存器预取 | — | **否决（机制已在）** | key_split 的 `load_tile()` 已是"整 tile 进寄存器再消费" |
| 4 | 两 epoch packet 邮箱 collective | port/packet-allreduce | **完成·已存档** | 提交 `df1f2958` = tag `result/packet-allreduce`。**复测确认**：decode 129.727→**135.594（+4.523%）**，prefill 945.623（−0.019%，结构性为 0），oracle 逐位一致。首测 135.804，两次差 0.155%。tier1 门槛：prefill ≥955.079、decode ≥136.272 |
| 5 | 跨卡 argmax 合并（argmax_with_value + allreduce_argmax） | — | **出队（2026-10-25 06:20）** | 上游 TP2 受控实测 `parallel shard argmax` 仅 **+0.046%**，文档自述 "within end-to-end noise"；400+ 行换噪声。依据见 `candidates.md` 第三节 |
| 6 | 融合 RMSNorm 进 collective（站点 A） | port/rmsnorm-wiring | **完成·否决** | tag `result/rmsnorm-wiring`。oracle **A 类逐位一致**，但 decode **−2.748%**（两 rep 131.914/131.821）。原因：融合把归约从 80 个 CTA 压成每列 1 个（T=5~6），每线程串 10 次 publish→spin→retire，延迟暴露 10 倍。**连带否决站点 B 与第 7 项**（同内核同几何）。见 `rmsnorm-wiring.md` |
| 7 | projection epilogue 发布 peer packet | — | 备选（若 6 超额达标再评估） | 阶梯 +1.55% → 折算 +0.64~0.70%，过线但**最贵**（改 CUTLASS epilogue + 协议） |
| 8 | fused shard-winner collective | — | **出队（2026-10-25 06:20）** | 阶梯 +1.37%，但 TP2 受控实测只 **+0.73%**（82.396/81.797）→ 折算 +0.30~0.33%，不过门槛 |
| 9 | GDN controls 融合（split-10 精确保序） | port/gdn-split10 | **完成·低于门槛** | tag `result/gdn-split10`。oracle **A 类**（四层求和顺序逐项相同，输出还是 FP32）。decode **+0.369%**，低于门槛 +0.5%；且低于噪声 MDD 0.433% → **本来就在测不出来**。1 文件 +104 行。见 `gdn-split10.md` |
| 10 | prefill 相位归因 + prefill-down-cutlass | port/prefill-down-cutlass | **完成·已合并** | `f9e5ce82`。prefill **+25.274%**（945.623→1184.617），oracle 逐位一致。见 `prefill-down-cutlass.md` |
| 11 | GDN chunked prefill Volta 移植（S1-S4） | port/gdn-chunked-s1 | **完成·否决（性能）** | oracle A 类逐位一致，但 prefill **−24.230%**（ema 仿真层开销）。不合并、不 tag，封存于 lab 分支（S1-S4 五提交）。见 `gdn-chunked-volta.md` §6 |
| — | **收官确认 final-confirm** | master `f9e5ce82` | **完成·通过** | prefill **1187.077（+25.511%）** / decode **135.7862（+4.671%）** vs 原始基线，oracle A 类，tag `result/final-confirm`。见 `campaign-close.md` |

> **收益数字的可信度分级**：`candidates.md` 第四节。
> 阶梯值（`performance.md:550-565`）是**顺序测量、非独立归因**，且底数 157 tok/s 与本机 129.7 不同配置；
> TP2 受控 campaign（`performance-v100.md:140-148`）可信度最高。折算系数 ×0.41~0.45 由 TileN（0.411）
> 与 packet（0.447）两个实测样本反推。

## 第 2/3 项否决依据（2026-10-25 05:32 核过源码，勿再复议）

结论：**跳过成立**，且理由比"快照文件不存在"更硬。三条独立证据：

1. **快照侧机制边界**：16B KV 加载（`uint4 __ldcs` → 一次拆两次 `int2` dequant）只出现在
   `.../src/ops/kernel/gqa_attention_decode_i8_qpn_verify5_volta.cuh:154-163`。
   fork 的 `src/ops/kernel/` 无任何 `qpn_verify*` 文件。
2. **fork 的真路径是 key_split，不是 tc_volta**：`Gqa27Tp2Geometry = <12,2,2>`（GroupSize=6，
   `gqa_attention_geometry.cuh:27`）；`launcher/gqa_attention_decode.cu:169-187` 的 key_split 条件是
   `QHeads==12 && KVHeads==2 && TokenTile*GroupSize<=32` → **5×6=30 命中**。
   故 60K 上下文的 T=5 验证走 `gqa_attention_decode_i8_key_split_volta.cuh`
   （launcher:64 的 `tokens==5 && window 128..512` 是 split 数策略，不是换核，别混判）。
3. **第 3 项机制已存在**：同文件 `load_tile()` 把整个 tile 先读进寄存器
   `int2 k_codes[ChunksPerThread] / v_codes / half k_scales[] / v_scales[]`（263-303 行），
   再逐块消费 —— 即"寄存器暂存 + 之后才算"，与快照 PrefetchKv 同旨。

第 2 项即使不做"移植"、改成"重写"也不划算：key_split 的 lane↔dim 映射是 **8 维/lane**
（`lane*8`，块步长 `i*8*kGqaKvQuantHeadDim`）。改 16B 就要 **16 维/lane**，会同时改动
warp↔(token,dim) 归属、`key = warp + 8i` 不变式、smem PStride、
以及 `static_assert(Bc==kPagedKVPageSize && Threads==256 && D/8==32)` 的前提。
收益上限只是全局加载指令数减半 → 风险/收益比不成立。

**折损因子（新增判据，适用于所有剩余微移植项）**：第 1 项实测 +0.473% vs 文档 +1.1466%
= **×0.41**。凡"快照内循环微优化"搬过来，先按文档值 ×0.4 预估；达不到 0.5% 的直接不立项。

## 恢复步骤（当前断点）

1. `git worktree list` → `.agents/worktrees/port-packet-allreduce` 应在 `df1f2958`
2. 续编：`JOBS=2 bash _scratch/port-bat/build-port.sh packet-allreduce build`
   （日志 `_scratch/port-log/packet-rebuild.log`；`[n/291]` 起步，因为前面已编完 159 个）
3. 出 exe 后：`bash _scratch/port-bat/measure.sh packet-allreduce baseline`
   期望 oracle `06287348665d3648`、decode ≈135.8
4. 复测通过后：`git tag result/packet-allreduce` → 核对 `git rev-parse result/packet-allreduce` ≠ `34412f4f`
   → **保留 worktree 作 lab**（不删）→ 在第 6 项里 `git checkout -b port/rmsnorm-fusion`

## 硬件风险（重要）

本机在 `-j4` 满载编译下已两次无 dump 硬挂，WHEA `Bus/Interconnect` 可纠正错误风暴。
详见 `incident-20261025-freeze.md`。两块 V100 只协商到 PCIe 3.0 x4。
⇒ 一律 `-j2` 或更低；测量前后检查 `nvidia-smi`；挂死后先看本文件。

### 中断的三种形态，别混判

| 形态 | 判据 | 处置 |
|---|---|---|
| 硬件硬挂 | 有重启记录（`Kernel-General 12`）、`Uptime` 归零、崩溃前 WHEA ≥ 30 条/分钟 | 查 BIOS/链路，见 incident 文档 |
| WHEA 预警 | 无重启，但 WHEA > 10 条/分钟 | 停手，等风暴过去再续 |
| 后台任务被回收 | 无重启、WHEA 稀少、日志 mtime 停滞且 **无 ninja/cl/cicc 进程** | 直接重跑 `JOBS=2 ... build`，`ninja` 会从断点续 |

## 队列 v2（2026-10-25 追加 6 项，规划见 `queue-v2.md`）

| # | 项 | 类 | 落点 | 预期 | 状态 |
|---|---|---|---|---|---|
| 5 | 跨卡 argmax 合并（argmax_with_value + allreduce_argmax + runtime 改线） | A | 存在，约 350–400 行 / 4 文件 | +0.7% decode（压线） | 待开工（不依赖 packet） |
| 6 | 融合 RMSNorm 进 collective | A | 存在，需 packet 在位 | +0.69% | 阻塞：依赖第 4 项 |
| 7 | projection epilogue 直接发布 peer packet | A | 存在，需 packet + 改 GEMM epilogue | +1.55% | 阻塞：依赖第 4 项 |
| 8 | fused shard-winner collective | A | 存在，需 packet + argmax | **≈0（TP2 噪声内）** | 可选 |
| 9 | GDN controls 融合（split-10 精确保序） | **B** | 只剩 `shard_exact` 一个核，融合部分 fork 已有 | +1.76% | **默认跳过**：改求和顺序 → oracle 必变 |
| — | five-query packed KV vector16 | — | 需改 KV cache 布局，非移植 | +1.99% | 跳过 |

**三个待裁决项（已裁决 2026-10-25，细则见 `queue-v2.md` 第八节）**：

1. B 类（GDN `shard_exact`）**放行**，oracle 改为三段式：accepted/drafted/rounds 严格相等
   （硬否决）+ acceptance_length/per-position 相等 + 513 IDs 差异位 ≤2% 且每个差异位 margin <0.05。
2. **R1**：packet 通过后 `keep/packet-allreduce` 成为 6/7/8 的新基线，门槛改为相对该版 +1.0%/+0.5%，
   报告同时列 `vs packet 版` 与 `vs 原始基线` 两档。
3. vector16 → **只做探针实验**：不改 KV 布局，先用 `cuobjdump -sass` 确认能否生成 `LDG.E.128`，
   没生成就直接记为"技术不成立"，不占测量轮次。
