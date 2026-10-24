# 旁路请求打掉会话前缀缓存：决策与现状

> **状态：现行 · 本主题的唯一权威**
> 记录：2026-10-24 · 角色：note · 归属：active
> 适用条件：Windows 11 · 2×Tesla V100-SXM2-16GB · `sm_70` · CUDA 12.8 · TP2 · `qwen3.8-27b`
> 取代：`kv-cache-title-request.md`（问题分析与方案枚举）、`kv-cache-title-request-p0.md`（首轮实测）
> 结论一句话：C=1 下用**detached prefix registry**（方案 3）保住会话；已实现并合入 `master`（`f7a284a`）。

---

## 0. 一句话

`--max-concurrency 1` 时，旁路请求（标题、摘要）必须落在唯一的 lane 上，把会话的
retained 序列连同支点一起销毁，下一轮 `cached` 归零。已实现的做法是：会话与其 lane 解绑，
状态与 KV 页留在 registry 里，之后任意 lane 可重新 attach。

---

## 1. 问题（压缩版）

现象与机制细节见已归档的两份笔记，这里只留判据需要的部分：

| 事实 | 证据 |
|---|---|
| 正常多轮靠 `restore_turn_checkpoint` 命中，不是 `append_at_frontier` | 对照组 turn2 `cached=961`，路径字面值 |
| 一个 42 token 的模板标题把 turn2 从 961 打到 0 | `prefix_reuse_path=full_reset` |
| postfix 形状（标题 prompt 以会话为前缀）能让标题自己命中，但救不了下一轮 | 标题 `cached=962` → turn2 `0` |
| 提高 `--max-concurrency` 单独无效 | C=2 下两种形状 turn2 同样为 0（tie-break，非 lane 数） |

关键区分：**页存活 ≠ 支点存活**。两个销毁点（`sequence.kv.reset()` 与 checkpoint 被覆盖）
相互独立，只修一个不成立。

---

## 2. 选了什么，为什么

方案枚举（1 postfix / 2 `PreserveExisting` / 3 detached registry / 4 lane 选择键）在旧笔记 §6。

**选定方案 3**，理由只有一条：**标题 prompt 的构造权不在我们手里**。典型客户端在 token 0
就分叉，入口即 `full_reset`，页当场回池 —— 方案 1+2 的生效前提（postfix 形状 + 页未释放）
两个都不满足。C=1 又是产品约束（单 lane 串行），所以方案 4 的前提也不存在。

方案 3 的代价是三条不变量要重新证明（detach 后状态槽不被 lane 复用、页不被二次释放、
attach 时 frontier 与状态一致），这也是它被放到最后才做的原因。

---

## 3. 现状（已实现并合入）

| 提交 | 内容 |
|---|---|
| `0eca192b` | 主体：detach / attach、registry 槽位 `[2C, 2C+2R)`、准入记账、容量治理 |
| `58dedde4` | 修 `start_prefill` 过早清 `retained`；displacement 改 best-effort |
| `f87e7915` | 修准入与回收不一致（hang 的真因）；attach 抢占不再牺牲别的会话 |
| `88730c28` | 准入收敛为一个 `plan_lane_admission` + 一个执行器，删除第二本账 |
| `f7a284a` | `--prefix-registry-drop-current`：entry 可只存 checkpoint；日志加两个复用事实字段 |
| `abc3fb64`/`e1f07998`/`d66b3e1f` | 容量规则按 C=1 / C≥2 两种形状分开写，并撤回一条错结论 |
| `ba9537de` | hang 写成单一问题陈述（成因链与证据缺口） |

运行期结构：

```
lane 槽  [0, 2C)          lane i: current=i, checkpoint=C+i
entry 槽 [2C, 2C+2R)      entry e: current=2C+2e, checkpoint=2C+2e+1   （keep-current，默认）
         [2C, 2C+R)       entry e: checkpoint=2C+e                      （--prefix-registry-drop-current）
```

两个日志字段是排障入口：`prefix_reuse_path`（走了哪条复用路径）、`attach_registry_entry`
（复用基是不是 registry entry）。缺了后者无法区分「lane 自带命中」与「registry 命中」。

---

## 4. 实测锚点

### 4.1 C=1 端到端（32768 ctx，会话 965 token，`slots=1`）

| 组 | 旧二进制 | flag 关 | `registry=1` |
|---|---:|---:|---:|
| 对照 turn2 `cached` | 961 | 961 | 961 |
| 典型标题后 turn2 `cached` | 0 | 0 | **961**（`restore_turn_checkpoint`） |
| postfix 标题后 turn2 | 0 | 0 | 0（该形状无 detach，不在本方案职责内） |

### 4.2 容量：槽数不是「会话数」，是「容器数 × 2」

`slots=N` = N 个 entry，每个 entry 2 槽（drop-current 时 1 槽）。C=1 的实测边界：

| 配置 | 总槽数 | 显存/卡 | 会话 turn2 | 侧请求重发 |
|---|---:|---:|---:|---:|
| `registry=0` | 2 | 147 MiB | 0 | 0 |
| `registry=1` | 4 | 294 MiB | **961** | 0 |
| `registry=2` | 6 | 440 MiB | 957 | 0 |
| `registry=3` | 8 | 587 MiB | 957 | **311** |

要点：**保会话只要 1 个 entry**；要让侧请求自己也能复用，需要 3 个（第 4 跳还要一个空容器
接住被换下的序列）。`registry=2` 是纯浪费 147 MiB。

### 4.3 `--prefix-registry-drop-current`（省 1 槽/entry）

| 项 | 数值 |
|---|---|
| `runtime reservation` | 1,920,778,496 → 1,843,806,464 B（**−73.41 MiB/卡/entry**） |
| 常规形状 `A1→B→A2` | 两配置逐字段相同（`cached 979`，restore） |
| 决定性形状 `A1→B→[u1,a1]` | `cached 1035/9 append` → `979/65 restore`，TTFT +260 ms |
| 无旁路 `A1→[u1,a1]` | 两配置相同（append，attach=false） |
| 生成文本 | 两配置 **sha256 相同** |
| `--spec mtp --draft-tokens 3` | 同上；accept rate 414/278/0.6715 两配置完全一致 |

降级的代价严格等于 **checkpoint 边界 → frontier 的间隔**（= 上一轮输出长度 + 模板尾），
不是常数：Δ=51 时 +270 ms，按 1110 tok/s 线性放大。

### 4.4 状态槽为什么这么贵

单槽 73.4 MiB/卡 = recurrent（48 层 × 24 head × 128×128 × FP32 = 72.0 MiB）+ conv（1.4 MiB）。
它与上下文长度无关，是 GDN 的常数代价；交叉点约 4.3k token —— 会话更长时状态比 KV 便宜，
更短时比 KV 贵得多（32k KV = 544 MiB，256k = 4.25 GiB）。

---

## 5. 被撤回的结论（读旧笔记时会遇到）

| 旧结论 | 状态 | 修正 |
|---|---|---|
| 「提高 `--max-concurrency` 能缓解」 | 部分撤回 | 单独提无效（tie-break 严格大于 + 下标优先）；它是方案 4 的前提，不是方案本身 |
| 「`N ≥ k+1` 是通用容量规则」 | 撤回 | 只在 C≥2 成立；C=1 单会话 1 槽就够，多配 2 槽白花 147 MiB |
| 「第二个 slot 买不到东西」 | **已撤回**（`d66b3e1f`） | 错因是把「被丢弃」与「不可用」混为一谈；侧请求需要 3 槽 |
| 「registry entry 的 current 槽是死重，可删」 | **已撤回** | 可达性实测为真：prompt 以 assistant 消息结尾时走 `attach + AppendAtFrontier`（2/2 复现，`cached 1032 / computed 9`）；删掉会静默算错，不是变慢 |

最后一条的教训值得单独记：**选择顺序（append 优先）能证明「如果可用就会赢」，不能证明
「不可用」**。判断可达性必须拿真实 token 流，样本里缺一种形状就会得出反向结论。

---

## 6. 已知边界

- **KV 池是上限**。池子装不下时 registry entry 一定被淘汰；收益上限就是「池子装得下」的那部分。
- **detached 期间 KV 页一直占着**（32k 会话 = 32765 token 的页），不还给池子。这是方案 3 的
  真实显存成本，比 73 MiB 状态槽大一个量级。
- **`--no-prefix-reuse` 与 `--prefix-registry-slots` 互斥**，启动即拒绝。
- **MTP/DFlash 的 drop-current 分支只测了 MTP**；DFlash 在 TP2 不支持，本 fork 无法覆盖。
- **greedy 不等于跨后端可复现**：`temperature 0` 下 `spec=mtp` 与 `spec=none` 输出不同
  （87 vs 92 字符，个别词不同），各自内部可复现。未判定是数值末位差还是接受逻辑问题。

---

## 7. 还没做的

1. **KV 压力场景只有脚本没有用例**（`_scratch/bench/kv_pressure.py`），是防 hang 回归的唯一证据。
2. **C=2 没有端到端并发验证**，只跑过串行 cohort。
3. **准入被拒（`QueueTimeout`）边界未实测**（把 `--pending-timeout-ms` 设 100 即可构造）。
4. **`admit_planned_request` 的 throw 仍走 `fail_all`**：现在只是触发窗口变窄，不是不会发生。
5. **bench 口径失效**：`--no-prefix-reuse` 被互斥规则禁用后，隔离缓存要用「每次不同 nonce」。

---

## 8. 复现命令

```bash
# 起服务（C=1，1 个 registry entry）
./build-win/apps/ninfer-serve.exe F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18888 --max-context 32768 --kv-capacity 32768 \
  --tp 2 --devices 0,1 --max-concurrency 1 --prefix-registry-slots 1 \
  --request-log-jsonl _scratch/append_probe/requests.jsonl

# 只读 checkpoint（省 73.41 MiB/卡/entry）
#   追加 --prefix-registry-drop-current

# 行为测试（约 6.5 分钟，三次加载）
export NINFER_V100_DUO_ARTIFACT=F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer
./build-win/tests/ninfer_qwen3_8_27b_v100-duo_prefix_real_test.exe
```

探针与对比脚本：`_scratch/append_probe/`（`compare_drop_current.py`、`probe.py`、各组
`requests-*.jsonl`）。这些目录不进 git，结论以本笔记第 4 节为准。
