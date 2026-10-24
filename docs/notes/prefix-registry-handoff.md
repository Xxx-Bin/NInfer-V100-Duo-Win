# prefix registry 交接（feat/prefix-registry）

更新时间 2026-10-24。分支 `feat/prefix-registry`，HEAD（本文档所在提交）。

## 一句话

`--prefix-registry-slots N` 让被挤出 lane 的 retained 序列 detach 进 registry，之后能被任意 lane
重新 attach。默认 `0` 逐位保持原行为。功能已完成、三条缺陷已修、速率零回归；剩余工作是测试固化
和 C=2 并发验证。

## 提交

| commit | 内容 |
|---|---|
| `0eca192b` | 主体：detach/attach、registry 槽位 `[2C, 2C+2R)`、准入记账、容量治理 |
| `58dedde4` | 修 `start_prefill` 过早清 `retained`；displacement 改 best-effort |
| `f87e7915` | 修准入与回收不一致（hang 的真因）、attach 抢占不再牺牲别的会话、`--no-prefix-reuse` 互斥 |
| `d66b3e1f` / `ba9537de` | 文档：容量规则按形状分开写、槽位边界实测、撤回被否证的结论、hang 写成完整问题陈述 |
| 本提交 | 结构修复：`AdmissionPlan` 成为唯一页账，executor 照单执行，删除回收循环与 `throw` 兜底 |

## 必须知道的四条结论

1. **`N` 容纳的是「你想同时保住的不同会话数」，不是「N 个请求」。** 侧请求（标题、摘要）
   属于当前会话的流量，不额外占槽，但它在落地时会把当前会话挤进 registry。
   - **C=1、单会话 + 侧请求**：`N=1` 够——**前提是只要求会话命中**。A.turn1 → 侧请求落地 detach A
     → A.turn2 attach 回来；attach 挤掉侧请求时 registry 已满（A 在里面且受保护），侧请求被丢弃，
     它本来就是可牺牲的。实测：`slots=1/2` 会话命中；`slots=3/4` 连侧请求也能复用（见「验证现状」）。
   - **`N=1` 只保得住一个会话。** 第二个无关会话进来必然淘汰前一个（`evict_registry_entry` 按
     `execution_frontier` 排序）。这不是缺陷，是 1 槽的定义。
   - **C≥2 或多会话轮转、要并存 k 个**：`N ≥ k+1`。多出那槽是给 attach 落地时被挤下的那个会话
     留的。实测 C=2 + 双 side：capacity=2 只保一个，capacity=4 两个都保。

   早期交接把最后一条的 `+1` 写成了通用规则，会让 C=1 白配 2 槽（+146 MiB）。
2. **KV 不足时缓存一定让位。** 池子装不下新请求时，registry entry 会被 evict。收益上限就是
   「池子装得下缓存」的那部分场景，别指望无穷收益。
3. **显存成本两笔账，别只算第一笔。** 固定预留 146 MiB/卡/entry（2 个线性注意力状态槽），
   之外 detached 序列一直占着 KV 页（32k 会话 = 32765 token 的页），后者是大头。
4. **`--no-prefix-reuse` 与 `--prefix-registry-slots` 互斥**，启动即拒绝。前者说不用缓存，
   后者还扣着页不放，是矛盾配置。

## 缺陷与改动清单

- **attach 抢占会牺牲别人**：attach 时 displace 触发的 detach 发现 registry 满就去 evict，
  等于为了接回一个会话销毁另一个。改成 `detach_sequence(..., allow_evict=false)`：满就丢被挤出的
  那个，不动已有 entry。
- **试过并撤回：给 `FullReset` 分支的 detach 也加 `allow_evict=false`。** 动机是"不牺牲别人"
  （`58dedde4` 的原则看着该统一），但实测直接打掉目标：cohort 顺序 C→A 时 registry 里还留着
  C 的 entry，A 的侧请求落地时 1 槽已满、不允许淘汰，于是 A 的序列被销毁，**A.turn2 `cached`
  从 961 掉到 0（`full_reset`）**。`evict_registry_entry` 本来就按 `execution_frontier` 排序，
  淘汰不是"乱牺牲"，是容量治理本身。已在 `program_impl.h` 恢复 `allow_evict=true` 并留注释。
- **准入说能腾、执行不腾导致引擎半死**：见下一节，这条是本分支唯一的生产级故障。
- **矛盾配置**：见结论 4。

## 问题与解决：KV 耗尽时引擎半死（hang）

### 现象

不是死锁。某个请求在 KV 预留阶段失败后，**该请求连同所有在飞和排队的请求一起失败，worker 线程
退出，但 HTTP 端口仍在 accept**。客户端连得上、发得出去、既收不到响应也不报错，只能等到超时，
看起来就是永久卡住。

区分三种表现：返回 500 是健康的，连不上也是健康的，只有「连上但永远没回应」是这个问题。

### 成因链

1. `can_admit_lane_after_retained_eviction` 把当前 lane 的 retained 页计入可回收——这是分支前的
   账法，当时成立。
2. 执行侧的回收循环 `if (retained_lane != lane && ...)` **排除当前 lane**——这行也是分支前的。
3. 本分支把 `start_prefill_lane` FullReset 分支的 `sequence.kv.reset()` 改成 `detach_sequence()`：
   页不再还给池子，而是跟着 entry 留在了 registry。
4. 准入按第 1 步的账放行 → `reserve_sequence_kv` 找不到页 → `std::bad_alloc`。
5. `admit_planned_request` 的 catch 先 `complete_error`，然后 **rethrow**（注释写明是故意的）→
   `worker_loop` 的 `catch (...)` → `fail_all` → worker 退出；HTTP 监听不受影响。

### 为什么算本分支引入

不是新逻辑写错，是**老代码的前提被换掉了**：

| | 准入记账 | 执行动作 | 账 |
|---|---|---|---|
| 分支前 | credit 当前 lane 的 `old_pages` | `reset()` 真还页 | 平 |
| 本分支 | credit `old_pages` | detach，页不还 | 差 `old_pages` |

`0eca192b` 给 `can_admit_lane` 补了新账法（`lane_bundle_stays_entitled`：detach 的页不再 credit），
但漏了 `can_admit_lane_after_retained_eviction` 这一条路径。执行侧那行 `retained_lane != lane` 因此
从「对的」变成「错的」。

判定成老 bug 会得出错误结论：分支前那个形状能正常返回，`evict_retained_lane` 走 `clear_lane`，页
确实回来了。

### 第一段修复（`f87e7915`，已合入）

1. `can_admit_lane` 增加 `lane_bundle_stays_entitled`：detach 的序列其页仍被占用，不再计入可回收。
2. executor 的回收循环不再排除当前 lane，并在 evict retained 之后调用
   `reclaim_registry_for`——它按 `execution_frontier` 真的淘汰 entry、把页还给池子，且永不淘汰本
   plan 即将 attach 的那个。
3. 回收判定与执行对同一条账，不再出现「说腾了、没腾」。

这一段是**行为修复**：让执行侧多腾一点，把旧账兑现掉。账本身还是两套。

### 第二段修复（本提交，结构修复）

第一段之后仍是「判定侧按旧账 credit，执行侧靠多腾兑现，兑不出来就 throw」。第二段把账收敛成一份：

- 新增 `runtime::AdmissionPlan{feasible, reclaim[]}` 契约，`reclaim` 有序且已最小化。
- Program 侧删掉 `can_admit_lane` / `can_admit_lane_after_retained_eviction` /
  `reclaim_registry_for` / `has_retained_lane` / `evict_retained_lane`，换成
  `plan_lane_admission`（唯一页账：`required <= page_group_count - (entitled - credit) + reclaimed`）
  与 `apply_lane_admission`（照单执行，`noexcept`）。
- executor 的 `LaneChoice` 直接携带计划，整段回收循环和第 866 行 `throw` 删除。准入通过 ⇔ 同一列表
  被执行，判定与执行不可能再分叉。

顺带纠正一个前提：`can_admit_lane_after_retained_eviction` 里「其他 lane 的 retained 可回收」只在
`registry_slots == 0` 成立；有 registry 时 `evict_retained_lane` 是 detach、页一页不腾，老代码是靠
「detach 完再把这个新 entry 淘汰掉」的往返拿到页——事故就出在这个往返上。

### 残余风险（不在本分支范围）

- 「引擎级错误即停服」的产品行为没动。准入这条通道已经不会产生引擎级失败，但别的路径仍然会。
  要改需要给 Engine 加对外状态，跨层改动。
- 准入被拒（`QueueTimeout`）的边界没有实测。用 `--pending-timeout-ms 100` 即可构造，不需要把
  KV 池填到边缘。

### 证据缺口

修复的复现配方目前只存在于 `_scratch/bench/kv_pressure.py`（未跟踪，不进 git）。默认
`--prefix-registry-slots 0` 时所有现有测试都走老路径，改坏了不会变红。建议落点：
`tests/targets/qwen3_6_27b/test_engine_v100-duo_kv_pressure.cpp`，与
`test_engine_v100-duo_prefix_real.cpp` 同样用 `NINFER_V100_DUO_ARTIFACT` 门控、`SKIP_RETURN_CODE 77`，
断言「准入放行 ⇒ `start_prefill` 不因页不足失败；失败也必须是 request-scope」。

## 验证现状

| 项 | 结果 |
|---|---|
| 主测试（graph/eager 输出、logits、接受率、frontier） | PASS，`cold_teacher_force_mismatches` 0/0 |
| registry 用例 | PASS，`reused=3305/3335`，与 cold 对照一致 |
| 速率（slots=0 vs 1，同窗口 81920） | prefill 4k/8k/32k、decode 512/8k 全部 ≤0.5% |
| KV 压力（3×17.6k，32k 窗口，slots=1） | 修复前 B=500 / C=503 引擎死；修复后三个全 200 |

### C=1 端到端（分支的目标形状，32768 ctx，会话 965 token，slots=1）

三段对照，同一台机器、同一 artifact（`qwen3_8_27b_nvfp4`，TP2）：

| 组 | 旧二进制 | 新二进制 flag 关 | 新二进制 `registry=1` |
|---|---:|---:|---:|
| 对照 turn2 `cached` | 961 | 961 | 961 |
| 典型标题（42 token）标题请求 `cached` | 0 | 0 | 0 |
| **典型标题后 turn2 `cached`** | **0** | **0** | **961**（`restore_turn_checkpoint`） |
| postfix 标题（1033 token）turn2 `cached` | 0 | 0 | 0 |

- 目标成立：B（本地的小请求）之后 A 的 turn2 从 `cached=0/full_reset` 回到 `961/restore_turn_checkpoint`。
- **复跑（2026-10-24，撤回 `allow_evict=false` 改动、重新编译后）**：C `961`、A `961`、B `0`、D `961`，
  与基线逐字段一致；`p0_result-registry1.json` 为本次快照，`p0_result-baseline.json` 为改动前快照，
  `p0_result.json` 是随后 `slots=2` 那一轮的覆盖结果（同样 C/A/D `961`、B `0`）。
- **槽位边界实测（C=1，会话 961 token，侧请求 315 token）**，`_scratch/p0/ab_two_slot.py`：

  | slots | 会话 turn2 | 重复的侧请求 |
  |---:|---:|---:|
  | 1 | 961 命中 | 0（丢） |
  | 2 | 957 命中 | 0（丢） |
  | 3 | 957 命中 | **311 命中** |
  | 4 | 957 命中 | **311 命中** |

  A→侧→A→侧 这条链上同时要活 **3** 个序列：A 的 turn1、侧请求、以及侧请求回来时要被换下
  的那个序列。2 槽时淘汰策略按 frontier 把侧请求的 entry 踢掉（它最短），所以第 4 跳退回
  `full_reset`。**只要 A 命中，1 槽就够；要让侧请求也能复用，要 3 槽。**
  两种侧请求长度（42 / 315 token）都测过：42 token 在不同槽数下都是 0，是因为它自己长度内没有
  可复用边界；315 token 才能体现槽位差异。
- 默认关闭逐字段与改动前一致，`runtime` 预留同为 1.65 GiB；开启后 1.79 GiB，+146 MiB = 2×73.4 MiB 状态槽/卡。
- postfix 组不变是设计预期：该形状没有 detach 发生，旁路请求在原地覆盖 checkpoint，属方案 2（`PreserveExisting`）的职责。
- 原始记录：`_scratch/p0/p0_result-registry1.json`（slots=1）、`p0_result.json`（slots=2）。

速率明细（slots0 / slots1）：prefill-4k 1121.1/1120.6，8k 1105.6/1105.1，32k 1029.2/1031.1；
decode-512 149.3/149.5，decode-8k 141.7/141.8。

## 没做完的

1. **KV 压力场景只有脚本，没有用例**（见「证据缺口」）。这是防 hang 回归的唯一证据。
2. **C=2 没有端到端并发验证。** 只跑过串行 cohort 看恢复率，没做真正的并发多会话。
3. **bench 口径失效。** 之前的对比用 `--no-prefix-reuse` + `slots=1` 隔离缓存，现在被互斥规则
   禁了。要重测得改成「reuse 开 + 每次不同 nonce 保证不命中」。
4. **「引擎失败即停服」的设计没动。** `worker_loop` 的 catch-all 仍会让整个引擎退出，只堵住了
   KV 那条触发路径。要让别类引擎级错误也变成可上报的 503，需要给 Engine 加对外状态，跨层改动。
5. **准入被拒（`QueueTimeout`）那条边界没实测。** 不需要把池子填到差一两页：把
   `--pending-timeout-ms` 设成 100 就能让它自然发生。（早先"必须压满 KV"的判断是错的，
   那种做法确实会挂驱动，但没必要。）

## 环境与命令

机器：2×Tesla V100-SXM2-16GB（sm_70）+ 30.9GB RAM，Windows + git-bash。
模型：`F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer`。

构建（改过头文件必须删 obj，否则 ninja 会说 "no work to do"）：

```bash
cd /c/Users/lwb6/Documents/git/NInfer-V100-Duo-win
export MSVC='C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130'
export SDK='C:/Program Files (x86)/Windows Kits/10'
export INCLUDE="$MSVC/include;$SDK/Include/10.0.22621.0/ucrt;$SDK/Include/10.0.22621.0/um;$SDK/Include/10.0.22621.0/shared;$SDK/Include/10.0.22621.0/winrt"
export LIB="$MSVC/lib/x64;$SDK/Lib/10.0.22621.0/ucrt/x64;$SDK/Lib/10.0.22621.0/um/x64"
export PATH="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.38.33130/bin/Hostx64/x64:$PATH"
export TMP='C:/Users/lwb6/Documents/git/NInfer-V100-Duo-win/build-win-tmp'
export TEMP="$TMP"
cd build-win && find . -name "*.obj" ! -name "*.cu.obj" -delete && cd ..
.tools/venv/Scripts/ninja.exe -C build-win -j2 ninfer-serve ninfer_qwen3_8_27b_v100-duo_prefix_real_test
```

测试（约 6.5 分钟，三次加载模型）：

```bash
export NINFER_V100_DUO_ARTIFACT=F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer
./build-win/tests/ninfer_qwen3_8_27b_v100-duo_prefix_real_test.exe
```

起服务做压力验证：

```bash
./build-win/apps/ninfer-serve.exe F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18888 --max-context 32768 --kv-capacity 32768 \
  --tp 2 --devices 0,1 --max-concurrency 1 --prefix-registry-slots 1
no_proxy='127.0.0.1,localhost' python _scratch/bench/kv_pressure.py 200
```

## 踩过的坑（别重复）

- **`--max-context 184320` 在本机装不下**：带 `--lm-head-draft` 后权重 20.92 GiB/卡，只剩
  5.09 GiB，而 184k 要 8.33 GiB，直接 fail-fast。用 65536/81920。
- **改头文件后 ninja 可能不重编**，报 "no work to do"。先删 obj。
- **并发请求的日志不能按索引对齐**：用 `result.prompt_tokens` 匹配，索引对齐会把一个请求的
  timings 记到另一个头上（我因此误判过一次）。
- **KV 池被占满会把 GPU 驱动挂死**：进程杀不掉、`nvidia-smi` 无响应，需要重启机器。别在
  KV 边缘做压力实验。
- **`taskkill` 报「成功」不代表进程死了**：僵尸态时 PID 仍在 `tasklist` 里。

## 测量卫生

- 速率口径：`prefill_tok_s = computed_prefill_tokens / timings.prefill`，
  `decode_tok_s = (completion_tokens - 1) / timings.decode`。
- 16 token 输出的 decode 数字不要引用，那是采样噪声。
- 每个点看 rep 间散布，<0.5% 才可信。

## 脚本与记录

- `_scratch/bench/RESULT.md` — 全部实测数据、复现配方
- `_scratch/bench/kv_pressure.py` — KV 压力脚本（待固化成用例）
- `_scratch/bench/bench_points.py` — 按 SUMMARY 口径测点
