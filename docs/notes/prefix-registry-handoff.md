# prefix registry 交接（feat/prefix-registry）

更新时间 2026-10-24。分支 `feat/prefix-registry`，HEAD（本文档所在提交）。

## 一句话

`--prefix-registry-slots N` 让被挤出 lane 的 retained 序列 detach 进 registry，之后能被任意 lane
重新 attach。默认 `0` 逐位保持原行为。功能已完成、三条缺陷已修、速率零回归；剩余工作是测试固化
和 C=2 并发验证。

## 三个 commit

| commit | 内容 |
|---|---|
| `0eca192b` | 主体：detach/attach、registry 槽位 `[2C, 2C+2R)`、准入记账、容量治理 |
| `58dedde4` | 修 `start_prefill` 过早清 `retained`；displacement 改 best-effort |
| `f87e7915` | 修准入与回收不一致（hang 的真因）、attach 抢占不再牺牲别的会话、`--no-prefix-reuse` 互斥 |
| 本文档提交 | 文档：容量规则按 C=1 / C≥2 两种形状分开写，补 C=1 三段实测；记录被撤回的 `allow_evict=false` 改动 |

## 必须知道的四条结论

1. **`N` 容纳的是「你想同时保住的不同会话数」，不是「N 个请求」。** 侧请求（标题、摘要）
   属于当前会话的流量，不额外占槽，但它在落地时会把当前会话挤进 registry。
   - **C=1、单会话 + 侧请求**：`N=1` 够。A.turn1 → 侧请求落地 detach A → A.turn2 attach 回来；
     attach 挤掉侧请求时 registry 已满（A 在里面且受保护），侧请求照旧被丢弃——它本来就是可牺牲的。
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

## 已修的三条缺陷（+ 一条被实测否证后撤回的改动）

- **attach 抢占会牺牲别人**：attach 时 displace 触发的 detach 发现 registry 满就去 evict，
  等于为了接回一个会话销毁另一个。改成 `detach_sequence(..., allow_evict=false)`：满就丢被挤出的
  那个，不动已有 entry。
- **试过并撤回：给 `FullReset` 分支的 detach 也加 `allow_evict=false`。** 动机是"不牺牲别人"
  （`58dedde4` 的原则看着该统一），但实测直接打掉目标：cohort 顺序 C→A 时 registry 里还留着
  C 的 entry，A 的侧请求落地时 1 槽已满、不允许淘汰，于是 A 的序列被销毁，**A.turn2 `cached`
  从 961 掉到 0（`full_reset`）**。`evict_registry_entry` 本来就按 `execution_frontier` 排序，
  淘汰不是"乱牺牲"，是容量治理本身。已在 `program_impl.h` 恢复 `allow_evict=true` 并留注释。
- **准入说能腾、执行不腾**（hang 真因）：`can_admit_lane` 的注释写着「lane 自己的 retained 是
  detach 不是 release，不能计入可回收」，代码照扣；executor 的回收循环又用
  `retained_lane != lane` 把当前 lane 排除。结果准入放行、`start_prefill_lane` 预留 KV 时抛
  `bad_alloc` → `worker_loop` catch-all → 引擎停服、HTTP 端口仍监听 → 客户端永久 hang。
  现在 executor 对当前 lane 也执行 `evict_retained_lane`，再用 `reclaim_registry_for` 真正释放页。
- **矛盾配置**：见结论 4。

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
  与基线逐字段一致；`_scratch/p0/p0_result.json` 即本次结果，`p0_result-baseline.json` 为改动前快照。
- 默认关闭逐字段与改动前一致，`runtime` 预留同为 1.65 GiB；开启后 1.79 GiB，+146 MiB = 2×73.4 MiB 状态槽/卡。
- postfix 组不变是设计预期：该形状没有 detach 发生，旁路请求在原地覆盖 checkpoint，属方案 2（`PreserveExisting`）的职责。
- 原始记录：`_scratch/p0/p0_result.json`、`_scratch/p0/HANDOFF-prefix-registry.md`。

速率明细（slots0 / slots1）：prefill-4k 1121.1/1120.6，8k 1105.6/1105.1，32k 1029.2/1031.1；
decode-512 149.3/149.5，decode-8k 141.7/141.8。

## 没做完的

1. **KV 压力场景只有脚本，没有用例。** `_scratch/bench/kv_pressure.py`，重启机器就丢。
   这是目前唯一能防止 hang 回归的证据，应该搬进 `tests/`。
2. **C=2 没有端到端并发验证。** 只跑过串行 cohort 看恢复率，没做真正的并发多会话。
3. **bench 口径失效。** 之前的对比用 `--no-prefix-reuse` + `slots=1` 隔离缓存，现在被互斥规则
   禁了。要重测得改成「reuse 开 + 每次不同 nonce 保证不命中」。
4. **「引擎失败即停服」的设计没动。** `worker_loop` 的 catch-all 仍会让整个引擎退出，我只堵住了
   KV 那条触发路径。要让别类引擎级错误也变成可上报的 503，需要给 Engine 加对外状态，跨层改动。
5. **准入被拒（QueuedTimeout）那条边界没实测**，因为要把池子精确填到差一两页，容易再次把 GPU
   驱动搞挂（已经发生过一次，需要重启机器）。

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
