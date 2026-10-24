# `--prefix-registry-drop-current`：registry entry 只保留 checkpoint

分支 `feat/prefix-registry` @ `88730c28`（工作区另有只读探针改动，未提交）。
目标：给前缀 registry 的每个 entry 加一个"只存 rewrite checkpoint"的模式，省一个
linear-attention 状态槽（**73.41 MiB / 卡 / entry**），代价是本该走 append-at-frontier 的
请求形状降级为 checkpoint restore，重放 Δ 个 token。

## 1. 开关语义

```
--prefix-registry-drop-current          # 布尔，出现即为真，默认关闭
```

| | 默认（开关不出现） | `--prefix-registry-drop-current` |
|---|---|---|
| 每个 entry 的槽位 | 2 个：`current` + `rewrite_checkpoint` | 1 个：`rewrite_checkpoint` |
| 槽位公式 | `2C + 2i`（current）、`2C + 2i + 1`（checkpoint） | `2C + i`（checkpoint），没有 current |
| `registry_state_slot(i)` | 返回 `2C + 2i` | **抛 `std::logic_error("registry keeps no current slot")`** |
| entry 能被 append 命中 | 是 | 否（pass 0 整趟跳过） |
| 总槽数 | `2C + 2R` | `2C + R` |

校验：`--prefix-registry-slots 0` 时出现该开关 → 解析期报错
`--prefix-registry-drop-current requires --prefix-registry-slots`（serve 层与 target 层各一处，
后者防直接构造 `EngineOptions` 的调用方）。

## 2. 改动点

| # | 文件:行 | 内容 |
|---|---|---|
| A1 | `src/targets/qwen3_6/impl/runtime/linear_state_slots.h:23` | `state_slot_count(C, R, drop_current)`：`2C + (drop?1:2)R` |
| A2 | `.../linear_state_slots.h:75` | `registry_state_slot(i, C, drop_current)`：drop 时抛 `logic_error`，绝不返回别名索引 |
| A3 | `.../linear_state_slots.h:83` | `registry_checkpoint_state_slot(i, C, drop_current)`：drop 时 `2C + i` |
| A4 | `.../linear_state_slots.h:90` | `checked_entry_offset(entry, stride)`，stride 1 或 2 |
| B1 | `.../layouts.h:78` | `SequencePlanningInputs::prefix_registry_drop_current` |
| B2 | `.../layouts.h:120` | `SequencePlanImpl::prefix_registry_drop_current` |
| B3 | `.../layouts_impl.h:103` | `state_slot_count` 传第三参 |
| B4 | `.../layouts_impl.h:662` | `validate_target_options` 互斥校验 |
| B5 | `.../layouts_impl.h:762` | planner inputs → plan 透传 |
| B6 | `.../layouts_impl.h:876` | `EngineOptions` → planner inputs |
| B7 | `include/ninfer/types.h:127` | `EngineOptions::prefix_registry_drop_current` |
| C1 | `src/serve/serve_options.h:47` | `ServeOptions` 字段 |
| C2 | `src/serve/serve_options.cpp:240` | 解析分支（仿 `--no-prefix-reuse`，无值） |
| C3 | `src/serve/serve_options.cpp:402` | 与 `--prefix-registry-slots` 的互斥校验 |
| C4 | `src/serve/serve_options.cpp:126`、`~150` | usage 增加一行说明 |
| C5 | `src/serve/generation_service.cpp:240` | 透传到 `EngineOptions` |
| D1 | `program.h:346`、`program_impl.h:252` | `const bool registry_drops_current`，构造期冻结 |
| D2 | `program_impl.h:1620-1650`（`detach_sequence`） | current→registry 的拷贝仅在 `!registry_drops_current` 时执行；checkpoint 那份始终拷到 `2C + i`；peer 卡同样 |
| D3 | `program_impl.h:1735-1780`（`attach_registry_sequence`） | registry→lane.current 仅在 `!registry_drops_current` 时执行；drop 模式下在 checkpoint 拷入后**追加一次 `lane.current ← lane.checkpoint`**（同 stream，两卡各一次），不依赖后续 restore 是否自带 current 初始化 |
| D4 | （不变）`ordered_reset` | 仍只清 current |
| E | `request_plan_impl.h:222-226` | drop 时 pass 0 整趟跳过（从 1 开始），pass 1 的 checkpoint 候选逻辑不变 |
| F | `request_log.cpp:602`、`generation_service.cpp:402`、`generation_service.h:42` | `request_done.result` 增加 `"registry_keep_current": true\|false`（`!drop_current`），紧挨 `attach_registry_entry` |

未改：`ordered_reset`、registry 淘汰/保护逻辑、KV 页归属、任何非 registry 路径。

## 3. 对比测试

同一份二进制（本次构建），只换 CLI；每次换配置重启服务。`spec=none`、`temperature 0`、
`reasoning_effort none`、`stream false`、`max-concurrency 1`、`prefix-registry-slots 1`、
`max-context 32768`、`kv-capacity 32768`、`tp 2`、`devices 0,1`。
消息两边逐字节相同（生成的 a1 由第一轮落盘到 `compare_messages.json`，第二轮原样重放，
脚本内校验两次 a1 一致）。

| 组 | 形状 | 配置 | prompt | cached | computed | reuse path | attach | TTFT (s) |
|---|---|---|---|---|---|---|---|---|
| G1 | `A1 → B → A2`（user 结尾） | keep | 1054 | 979 | 75 | restore_turn_checkpoint | true | 0.356 |
|   |   | drop | 1054 | 979 | 75 | restore_turn_checkpoint | true | 0.360 |
| G2 | `A1 → B → [u1,a1]`（assistant 结尾） | keep | 1044 | **1035** | **9** | **append_frontier** | true | **0.062** |
|   |   | drop | 1044 | **979** | **65** | **restore_turn_checkpoint** | true | **0.322** |
| G3 | 同 G2，换文本 | keep | 1039 | 1030 | 9 | append_frontier | true | 0.059 |
|   |   | drop | 1039 | 979 | 60 | restore_turn_checkpoint | true | 0.342 |
| G4 | `A1 → [u1,a1]`（无旁路） | keep | 1044 | 1035 | 9 | append_frontier | false | 0.057 |
|   |   | drop | 1044 | 1035 | 9 | append_frontier | false | 0.063 |

- **G1 / G4 两种配置逐字段相同**（G1 的 TTFT 差 4 ms、G4 差 6 ms，为噪声；cached/computed/path/attach 完全一致）。
- G2 / G3 按设计降级：append → restore，base 从 1035/1030 退回 979，多算 56 / 51 个 token。
- `attach && append_frontier` 在 drop 配置下 **0 次**（4 组 12 条请求全查）；keep 配置下 2 次（G2/G3），
  与工作区既有探针结论一致。

### 输出等价（temp 0，逐 token）

| 组 | keep 文本 | drop 文本 | sha256 前 16 位 | gen tokens |
|---|---|---|---|---|
| G1 | 同 | 同 | 一致 | 39 = 39 |
| G2 | 93 字符 | 93 字符 | `5091b6f2c173acaa` = `5091b6f2c173acaa` | 53 = 53 |
| G3 | 88 字符 | 88 字符 | `3ce35a185c7266a3` = `3ce35a185c7266a3` | 48 = 48 |
| G4 | 同 | 同 | 一致 | 53 = 53 |

即：降级路径（checkpoint restore + 本地 current 播种）与原 append 路径产生**逐 token 相同**的
输出，且生成长度相同——attach 之后第一条 prefill 的状态是对齐的。

## 4. 预留差值

`server_start` 记录中的 `memory` 块（两次配置、同一二进制）：

| 字段 | keep | drop | 差值 |
|---|---|---|---|
| `runtime_reservation_bytes` | 1,920,778,496 (1.789 GiB) | 1,843,806,464 (1.717 GiB) | **76,972,032 B = 73.41 MiB** |
| `sequence.capacity_bytes` | 1,394,136,832 | 1,317,164,800 | 76,972,032 B |
| `available_after_startup_bytes` | 3,936,616,448 | 4,012,113,920 | +76,972,032 B |
| `planned_slack_bytes` | 3,951,836,928 | 4,028,808,960 | +76,972,032 B |
| `weights / workspace / graph_*` | — | — | 完全相同 |

即一个 entry 的 current 状态槽 = 73.41 MiB/卡，与预期一致（1.79 GiB → 1.72 GiB）。

### 开关不出现时的回归证据

keep 配置的整份 `memory` 块（14 个字段：`runtime_reservation_bytes`、`sequence/weights/workspace/
request_transient` 的 capacity/peak/used、graph 观测与余量、headroom、slack、KV payload）与
**改动前二进制**留下的 `_scratch/append_probe/requests.jsonl`（`88730c28` + 只读探针，同参数）
**逐字段相同**。请求侧：G1 与旧记录的 T1、G2/G3 与旧记录的 D2 是同一形状，`prefix_reuse_path`
与 `attach` 判定一致（cached 相差 ≤ 3 token，来自 filler 里会话标签文本长度不同，不是布局差异）。
改动对新路径是加法，旧路径每个判断都带 `!registry_drops_current` 守卫。

## 5. 降级边界与代价

降级条件：**prompt 以 assistant 消息结尾**、命中 registry entry、且只在前缀
`execution_frontier` 上匹配（旧 pass 0 的 append 候选）。

- 只影响 registry entry 的 append 候选；lane 自有序列的 append（G4）不受影响——
  那条路径读的是 lane 自己的 current 槽，与 entry 布局无关。
- 常规对话（`A1 → B → A2`，user 结尾）本来就走 `restore_turn_checkpoint`，两种配置完全一致（G1）。
- 若某个 entry 连 checkpoint 也不匹配，则 drop 模式下无候选 → `full_reset`（keep 模式下可能仍能
  append）。这是该开关的第二个降级点：append-only 可复用的 entry 在 drop 模式下完全不可复用。

实测 Δ（重放的 token 数 = 上一轮 assistant 输出 + 模板尾）：

| 组 | Δ token | computed 增量 | TTFT 代价 | 端到端代价 |
|---|---|---|---|---|
| G2 | 56（a1 53 + 尾 3） | 9 → 65 | 0.062 → 0.322 s（**+260 ms**） | 1.357 → 1.611 s（+254 ms） |
| G3 | 51 | 9 → 60 | 0.059 → 0.342 s（**+283 ms**） | 1.229 → 1.507 s（+278 ms） |

即：约 **0.26–0.28 s 的 prefill 代价 / 51–56 token**，换 **73.4 MiB/卡/entry**。
Δ 与上一轮生成长度线性相关，长回答场景代价按比例增长。

## 6. 风险点

1. **`registry_state_slot` 抛异常是刻意的**：drop 模式下该索引会指向另一个 entry 的 checkpoint。
   任何新增调用点若在 drop 模式下取 current 索引，运行期会立刻 `logic_error` 而不是静默串状态。
   代价是这条路径不可用于"返回 0 号槽"之类的兜底。
2. **attach 后额外一次 `lane.current ← lane.checkpoint`**（每卡 73.4 MiB D2D 拷贝）。放在同一
   stream 的 checkpoint 拷贝之后，靠 stream 顺序保证；没有引入新的同步点。它的存在是为了不依赖
   restore 实现是否顺手初始化 current——若将来 restore 改为跳过 current，这段就变成唯一保证。
3. **`request_plan` 里 pass 0 被整趟跳过**，因此 drop 模式下 registry 的复用率可能低于 keep；
   对本测的形状只损失 append 那一档，对"只有 append 候选"的 entry 则完全失去复用。
4. 日志新增 `registry_keep_current` 是**纯加法**（schema_version 不变，符合本文件既有约定），
   但旧的消费脚本若断言字段集合严格相等需同步。
5. 本次只验证 `max-concurrency 1` + 1 entry + `spec=none` + tp2。MTP/DFlash 的
   `reuse_tail_hidden_valid` / `mtp_kv_valid` 判断未在 drop 模式下专门跑过；从代码看它们只在
   append 候选上生效，drop 模式下这些候选根本不产生，故预期不受影响。
   **MTP 已补测，见 §7**（结论：降级停在 restore，没有二次掉到 full_reset）；DFlash 仍未实测
   （`--tp 2` 不支持 `--spec dflash`，需要换单卡配置）。

## 7. MTP（`--spec mtp --draft-tokens 3`）对比

补测 MTP：`request_plan_impl.h:279-293` 的 `append_ready` / `checkpoint_ready` 要求
`reuse_mtp_kv_valid >= reuse_base - 1`，不满足会把 `append` **和** `restore` 一起推成 `full_reset`。
也就是说 drop 模式在 MTP 下有可能比 spec=none 再降一档。本节是实测结论。

命令与 §3 差异仅三处：`--spec mtp --draft-tokens 3`、日志文件加 `-mtp` 后缀、第二轮加
`--prefix-registry-drop-current`。脚本 `_scratch/append_probe/compare_drop_current_mtp.py`
（复用 `compare_drop_current.py`，只把 `LOGS/OUTS/STORE` 换成 `-mtp` 后缀，a1 种子单独落
`compare_messages-mtp.json`，两配置的 messages 仍逐字节相同）。四组形状与 §3 完全相同。

| 组 | 形状 | 配置 | prompt | cached | computed | reuse path | attach | keep_cur | TTFT (s) | gen |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | `A1 → B → A2` | keep | 1059 | 979 | 80 | restore_turn_checkpoint | true | true | 0.360 | 47 |
|   |   | drop | 1059 | 979 | 80 | restore_turn_checkpoint | true | false | 0.368 | 47 |
| G2 | `A1 → B → [u1,a1]` | keep | 1039 | **1030** | **9** | **append_frontier** | true | true | **0.074** | 48 |
|   |   | drop | 1039 | **979** | **60** | **restore_turn_checkpoint** | true | false | **0.342** | 48 |
| G3 | 同 G2，换文本 | keep | 1039 | 1030 | 9 | append_frontier | true | true | 0.069 | 48 |
|   |   | drop | 1039 | 979 | 60 | restore_turn_checkpoint | true | false | 0.342 | 48 |
| G4 | `A1 → [u1,a1]`（无旁路） | keep | 1039 | 1030 | 9 | append_frontier | false | true | 0.063 | 48 |
|   |   | drop | 1039 | 1030 | 9 | append_frontier | false | false | 0.073 | 48 |

（G1 的 prompt 是 1059 而 G2 只有 1039，因为 G1 多一条 `FOLLOW_UP`。）

### 结论：MTP 没有把降级再推一档

- **drop 下 G2/G3 落在 `restore_turn_checkpoint`，cached=979，不是 `full_reset`。** 与 spec=none
  完全同档，`mtp_kv_valid` 条件在本次四组里都成立：detach 时
  `entry.mtp_kv_valid = sequence.mtp_kv_valid = execution_frontier = 1039`，而 checkpoint
  frontier 是 979，`1039 >= 979 - 1` 通过，`checkpoint_ready == true`。
- 反过来说，只有当 entry 的 `mtp_kv_valid` 落后 checkpoint frontier 一个 token 以上时才会掉
  `full_reset`——例如 entry 由**未 `prepare_mtp`** 的请求 detach、或 checkpoint 落在 entry
  的 `execution_frontier` 之后。本次形状不触发，但这仍是一条**未覆盖**的边界，代码上确实存在。
- **drop 下 `attach && append_frontier` 出现 0 次**（4 组 11 条请求全查）；keep 下 2 次（G2/G3），
  符合 pass 0 被整趟跳过的设计。
- G1 / G4 两配置逐字段相同：G1 是本来就走 restore 的形状，G4 的 append 读的是 lane 自有 current
  槽，与 entry 布局无关。TTFT 差 4–10 ms 为噪声。

### 预留差值（`server_start.memory`）

| 字段 | keep | drop | 差值 |
|---|---|---|---|
| `runtime_reservation_bytes` | 2,069,444,864 (1.928 GiB) | 1,992,472,832 (1.856 GiB) | **76,972,032 B = 73.41 MiB** |
| `sequence.capacity_bytes` | 1,467,057,408 | 1,390,085,376 | 76,972,032 B |
| `planned_slack_bytes` | 3,576,678,144 | 3,653,650,176 | +76,972,032 B |
| `available_after_startup_bytes` | 3,623,747,584 | 3,701,342,208 | +77,594,624 B |
| `weights` / `workspace` / `kv_payload` / graph 观测 | — | — | 完全相同（11,050,488,320 / 516,404,224 / 1,140,981,760） |

**与 spec=none 的 73.41 MiB 逐字节一致**：一个 entry 的 current 状态槽不随 `--spec` 变化。
MTP 自身抬高的是基线（同配置 `runtime_reservation` 1,920,778,496 → 2,069,444,864，+141.78 MiB
来自 MTP 头与 MTP 预填充 workspace），与开关无关。

### 投机统计（`request_done.speculative`，11 条请求求和）

| 配置 | drafted | accepted | rounds | fallback_steps | accept rate | gen |
|---|---|---|---|---|---|---|
| keep | 414 | 278 | 138 | 0 | 0.6715 | 400 |
| drop | 414 | 278 | 138 | 0 | **0.6715** | 400 |

逐请求也一致：G2/G3 的 end 请求两配置都是 `3.77 tok/round (92.3%)`（服务端 stderr 的
`speculative=` 字段）。原因：restore 重放的 51 个 token 走 prefill 通道，不进 speculative decode
的计数；而 decode 起始状态（KV + MTP 缓存）在 restore 后是对齐的，所以 accept 分布不变。
即 **drop 模式在 MTP 下没有投机质量损失**。

### 输出等价（temp 0，逐 token）

| 组 | keep sha256/16 | drop sha256/16 | 字符数 | gen tokens |
|---|---|---|---|---|
| G1 | `6a4bb4f57410859c` | `6a4bb4f57410859c` | 78 = 78 | 47 = 47 |
| G2 | `3ce35a185c7266a3` | `3ce35a185c7266a3` | 88 = 88 | 48 = 48 |
| G3 | `3ce35a185c7266a3` | `3ce35a185c7266a3` | 88 = 88 | 48 = 48 |
| G4 | `3ce35a185c7266a3` | `3ce35a185c7266a3` | 88 = 88 | 48 = 48 |

四组 end 输出逐 token 相同，生成长度相同。**MTP 路径上的 restore 也是状态对齐的。**

### 代价

| 组 | Δ token | computed 增量 | TTFT 代价 |
|---|---|---|---|
| G2 | 51（a1 48 + 尾 3） | 9 → 60 | 0.074 → 0.342 s（**+268 ms**） |
| G3 | 51 | 9 → 60 | 0.069 → 0.342 s（**+273 ms**） |

与 spec=none 的 0.26–0.28 s 同一量级，MTP 没有放大它。

### 一个不属于本开关的观察

两轮各自的 a1（983 token 同一 prompt、`temperature 0`、服务端 `sampler=[greedy]`）**文本不同**：
spec=none 出 87 字符（"…通过引入**精细化**统计采样…"），MTP 出 92 字符（"…通过引入**周期性**统计采样…"）。
所以本表的 prompt 比 §3 小 5（G2 1039 vs 1044）。两种 MTP 配置内部都完全可复现（第二轮 a1
重放校验通过），差异只在 backend 之间。它不影响本节 keep/drop 的任何结论（两边同 backend、
messages 逐字节相同），但"greedy 下 MTP 与 none 输出不一致"本身值得单独查——不在本开关范围内。

## 8. 产物

| 文件 | 内容 |
|---|---|
| `_scratch/append_probe/compare_drop_current_mtp.py` | MTP 版 A/B 驱动（只换输出路径） |
| `_scratch/append_probe/report_mtp.py` | 结果 jsonl + result json 对齐，出 TTFT/投机/预留 |
| `_scratch/append_probe/requests-{keep,drop}-mtp.jsonl` | 两配置的 request log（含 `server_start`） |
| `_scratch/append_probe/result-compare-{keep,drop}-mtp.json` | 两配置的逐请求记录 |
| `_scratch/append_probe/compare_messages-mtp.json` | MTP 轮复用的 a1 种子 |
| `_scratch/append_probe/server-{keep,drop}-mtp.err` | 服务端 stderr（含每请求 accept rate） |
| `_scratch/append_probe/{requests-*,result-*}.jsonl/json` | spec=none 那轮，未改动 |
