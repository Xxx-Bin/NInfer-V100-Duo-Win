# P0 观测：旁路标题请求对前缀缓存的影响（两次实测）

> **状态：已归档 · 首轮观测，结论已被后续实测细化**
> 归档：2026-10-24 · 归属：superseded · 角色：evidence
> 原文适用条件：Windows 11 · 2×V100-SXM2-16GB · sm_70 · CUDA 12.8 · TP2 · C=1 与 C=2 各一轮
> 现行替代：[prefix-reuse-decision.md](prefix-reuse-decision.md)（§4 有当前口径的实测锚点）
> 保留理由：四个 cohort 的原始读数与两处测量陷阱（metrics 发布周期、本地 http_proxy），仍然有效

> 配套设计笔记：`kv-cache-title-request.md`（机制与方案）。本文只记录实测，不改设计笔记的结论。
> 两次运行分别落在 `--max-concurrency 2` 与 `--max-concurrency 1` 上，因此也顺带回答了
> 设计笔记里「提高并发能不能缓解」这个问题。

---

## 0. 结论先行

| # | 结论 | 证据 |
|---|---|---|
| 1 | 正常多轮对话**靠 turn checkpoint 恢复**，不是靠 append | 对照组 turn2 路径恒为 `restore_turn_checkpoint` |
| 2 | 一个 **42 token** 的模板标题请求，把 turn2 从 93.8% 命中打到 0 | A 组：`cached=0`、`full_reset` |
| 3 | postfix 标题请求**自身命中 962**，但 turn2 仍 0 | B 组：`title=cached 962` → `turn2=cached 0` |
| 4 | 页存活 ≠ 支点存活；两处销毁点相互独立 | 同上，B 组页明显还在 |
| 5 | **C=2 不能保护会话** | 第一轮：C=2 下 A/B 组 turn2 同样归零 |
| 6 | 设计笔记 §6.1 的判据表**命中了第 1、2 两行**，取决于标题模板形状 | A→第 1 行，B→第 2 行 |

一句话：**典型客户端的标题模板（方案 3 / 方案 5 二选一）；若愿意把标题请求改成 postfix，
则落到方案 1+2（约 30 行）。** 两条路都实测支撑，不再是推断。

---

## 1. 环境

共同部分：

| 项 | 值 |
|---|---|
| 硬件 | 2 × Tesla V100-SXM2-16GB（sm_70），TP2，NVLink |
| 载荷 | `F:\models\ninfer\qwen3_8_27b_nvfp4.ninfer`（`qwen3.8-27b` / `nvfp4`） |
| 引擎 | `version 0.1.1-win.5` |
| KV | `int8_g64` |
| `prefix_reuse` | `true`（两次都是） |
| 采样 | `temperature 0`、`reasoning_effort: "none"`（关 thinking）、`stream: false` |
| 每组会话长度 | 约 965 prompt token |

差异部分：

| 运行 | 服务来源 | `max_concurrency` | `max_context` / kv tokens | 观测通道 |
|---|---|---:|---|---|
| ① | 已存在的 `ninfer-serve.exe`（PID 8764，`C:\Users\Public\ninfer-v100-duo-win\test\`，:8888），未改配置，未开请求日志 | **2** | 192512 / 192512 | `usage.cached_tokens` + `/metrics` 差值 |
| ② | 新起 `build-win/apps/ninfer-serve.exe`（PID 15156，:18888） | **1** | 32768 / 32768 | 同上 **＋ `--request-log-jsonl` 的 `prefix_reuse_path`** |

②的启动命令：

```bash
build-win/apps/ninfer-serve.exe F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18888 \
  --max-context 32768 --kv-capacity 32768 \
  --tp 2 --devices 0,1 --max-concurrency 1 \
  --request-log-jsonl _scratch/p0/requests.jsonl
```

加载 70.7 s，启动后每卡 12342 MiB，`free-after-startup=3.81 GiB`。

---

## 2. 方法

四个 cohort，每组使用**互不相同的会话文本**，避免上一组的 retained state 被记到下一组头上：

| cohort | 形状 | 请求序列 |
|---|---|---|
| **C** | 对照 | `turn1` → `turn2` |
| **A** | 典型客户端标题 | `turn1` → `title`（token 0 即与会话不同）→ `turn2` |
| **B** | postfix 标题 | `turn1` → `title`（= 会话 + assistant + 起标题指令）→ `turn2` |
| **D** | 对照复测 | 同 C，放在最后跑 |

具体消息形态：

- `turn1`：`[user: “[会话<tag>] …<约 900 token 运维文本>… 请用一句话总结上述记录的主题。”]`
- `turn2`：`[user: 同 turn1, assistant: turn1 的实际输出, user: “基于刚才的总结，再补充一句你认为最关键的运维风险。”]`
  ——**assistant 用的是模型真实输出**，与真实客户端一致。
- A 的 `title`：`[user: “请为以下对话生成一个不超过六个字的标题，只输出标题本身。”]`（42 token）
- B 的 `title`：`turn1` 的 messages + `assistant: a1` + `[user: 同上起标题指令]`（1033 token）

测量细节：

- 每个请求之间等 `/metrics` 的 `live.running == 0`，全串行。
- 每请求后 `sleep 1.3 s` 再读累计量（原因见 §5 陷阱 1）。
- `cached_tokens` 与 `prompt_tokens_reused` 累计差值**逐请求完全相等**，且满足
  `computed + reused == prompt`；②还有日志里的 `computed_prefill_tokens` 作为第三方对齐。
- 对照组首末各跑一次，两次数值逐字段相同，排除漂移与预热。

---

## 3. 数据

### 3.1 运行②（`max-concurrency 1`，含 reuse path 字面值）

| 组 | 请求 | prompt | cached | computed | `prefix_reuse_path` | wall |
|---|---|---:|---:|---:|---|---:|
| **C** | turn1 | 965 | 0 | 965 | `full_reset` | 2.07 s |
| | turn2 | 1030 | **961** | 69 | `restore_turn_checkpoint` | 1.53 s |
| **A** | turn1 | 965 | 0 | 965 | `full_reset` | 1.91 s |
| | title | 42 | 0 | 42 | **`full_reset`** | 0.42 s |
| | turn2 | 1025 | **0** | 1025 | `full_reset` | 2.07 s |
| **B** | turn1 | 966 | 0 | 966 | `full_reset` | 2.04 s |
| | title | 1033 | **962** | 71 | **`restore_turn_checkpoint`** | 0.47 s |
| | turn2 | 1031 | **0** | 1031 | `full_reset` | 2.25 s |
| **D** | turn1 | 965 | 0 | 965 | `full_reset` | 2.04 s |
| | turn2 | 1030 | **961** | 69 | `restore_turn_checkpoint` | 1.53 s |

### 3.2 运行①（`max-concurrency 2`，仅数量）

| 组 | 请求 | prompt | cached | 命中率 | wall |
|---|---|---:|---:|---:|---:|
| C | turn1 / turn2 | 965 / 1025 | 0 / **961** | 93.8% | 1.43 / 0.96 s |
| A | title / turn2 | 42 / 1031 | 0 / **0** | 0 / 0 | 0.41 / 1.68 s |
| B | title / turn2 | 1028 / 1026 | **962** / **0** | 93.6% / 0 | 0.42 / 1.55 s |
| D | turn1 / turn2 | 965 / 1025 | 0 / **961** | 93.8% | 1.37 / 0.85 s |

两轮的 A/B 组行为完全一致，只是①拿不到路径字面值（未开请求日志），其 `full_reset` 判定
是由 `cached == 0` 反推的。

---

## 4. 读法与判据映射

### 4.1 对照组：turn2 走的是 checkpoint 恢复

C 与 D 的 turn2 路径都是 `restore_turn_checkpoint`，不是 `append_at_frontier`。这一点很关键：

- 正常多轮对话本来就不是「在 frontier 上追加」，而是回到 turn 边界处的 checkpoint 再继续。
- 所以设计笔记 §2.3 里「`Drop` / `CaptureNew` 会清空 checkpoint」不是次要风险，**它就是主销毁路径**。

### 4.2 A 组：页在入口即回池

`title` 只有 42 token、token 0 就与会话分叉，路径是 `full_reset`。按设计笔记 §2.2，
`FullReset` 分支第一件事就是 `sequence.kv.reset()`。⇒ 页在标题请求入口已经回池，
下游 turn2 无从恢复，只能整段重算。

对应设计笔记 §6.1 判据表 **第 1 行** → 方案 3（C=1）或方案 5（C≥2）。

### 4.3 B 组：页活着，支点死了

B 的 `title` 命中 962、路径 `restore_turn_checkpoint` —— 入口**没有**走 `reset()`，页当时肯定还在。
紧接着 `turn2` 的 prompt 比它只短 2 个 token（1031 vs 1033），却 `full_reset` + `cached=0`。

⇒ 页的存活与支点的存活是两个独立事实。支点（checkpoint / ledger / `prefix_identity`）被
旁路请求整个换掉，页再多也接不上。

对应设计笔记 §6.1 判据表 **第 2 行** → 方案 2 有生效空间，但必须配方案 1（postfix），
因为方案 2 要求「旁路 prompt 以会话为前缀」且「入口未释放页」——B 形状恰好两个条件都满足。

### 4.4 成本侧的数字

- 命中一次 ≈ 省 961 token 的计算：turn2 从 2.07 s（重算）降到 1.53 s（命中），约 **0.5 s / 1k token**。
  按 `docs/performance.md` 的 1,110 tok/s 折算，丢 3 万 token ≈ 27 s，与设计笔记的估算一致。
- **postfix 的额外开销几乎为零**：B.title 只算了 71 个 token（不是 1033），因为它吃到了 962 的命中。
  所以方案 1 的真正短板只有一个——它救不了下一轮。

### 4.5 并发：C=2 不保护会话

运行①的对照组证明顺序请求能稳定命中同一份 retained；A/B 组证明中间插一个旁路请求就归零。
⇒ 旁路请求落到了会话所在的 lane 并就地重置了它。

这与设计笔记 §6.3 方案 5 的判断一致：**单靠提高并发无效，必须先改 lane 选择键（tie-break）**。
本文没有验证「改了选择键之后 C=2 是否真能保住会话」——那需要改代码，不在 P0 范围内。

---

## 5. 两个测量陷阱

**① `/metrics` 的发布周期会吞掉短请求。**
第一轮首次运行时 `A.title` 显示 `computed_delta=0` 而 `prompt=42`。原因是 `RuntimeStats` 约 1 s
发布一次，0.42 s 的请求在差值窗口里读不到；下一轮请求的窗口里会把它补进来（所以 A.turn2
的差值一度是 1073 = 42 + 1031）。修法：每请求后 settle 1.3 s，并以响应体
`usage.cached_tokens` 为准、累计差值只作交叉验证。

**② 本机 shell 有 `http_proxy`，`curl` 打本地端口会被代理转一手。**
表现为 `upstream connect failed: os error 10061`，看起来像「服务没起来」。修法：
`curl --noproxy '*'`，Python 侧设 `no_proxy=127.0.0.1,localhost`。

---

## 6. 局限与未覆盖

- 单轮约 1k token。绝对秒数按 1,110 tok/s 折算，未在 3 万 token 会话上直接测。
- 未覆盖：连续多个旁路请求、`stream=true` 的旁路请求、旁路请求与 turn2 并发抢占。
- 未覆盖 C≥2 下的 lane 选择键改动（需改代码）。
- 运行①所在实例未开请求日志，其 `prefix_reuse_path` 为反推值；②为实测字段。
- 两轮都在 `nvfp4` artifact 上做，未在 `groupwise-int` 上复测；复用逻辑与量化格式无关，
  但若要声称覆盖全部已注册 artifact，需补测。

---

## 7. 复现

```bash
# 1) 起服务（C=1，开请求日志）
build-win/apps/ninfer-serve.exe F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 18888 --max-context 32768 --kv-capacity 32768 \
  --tp 2 --devices 0,1 --max-concurrency 1 \
  --request-log-jsonl _scratch/p0/requests.jsonl

# 2) 跑四组
cd _scratch/p0
no_proxy='127.0.0.1,localhost' python p0_probe.py 10

# 3) 产物
#    p0_result.json  结构化结果（含 engine 配置块）
#    requests.jsonl  服务端逐请求记录（含 prefix_reuse_path）
```

`p0_probe.py` 用响应体的 `usage.cached_tokens` 作为主观测，`/metrics` 累计差值作交叉验证，
日志里的 `computed_prefill_tokens` 作第三重对齐。三者都写进 `p0_result.json`（字段
`cached_tokens` / `reused_delta` / `log_cache_hit`），本文的两张表逐行核对过三者一致；
脚本本身不做断言，取值依赖人工比对。
