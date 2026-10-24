# prefix registry：entry 的 current 状态槽会被读吗

结论先行：**会。不可删。** 只要请求以 assistant 消息结尾、且复用基来自 registry entry，
`attach_registry_index >= 0 && reuse == AppendAtFrontier` 就会出现；该路径用的是 entry 的
current 状态槽，不是它的 rewrite-checkpoint 槽。

## 问题

`attach_registry_sequence` 会把 entry 的两个线性注意力槽都拷进 lane：
`registry_state_slot` → `current_state_slot`，`registry_checkpoint_state_slot` →
`rewrite_checkpoint_state_slot`。若 attach 之后只会走 checkpoint restore，那么第一份拷贝在
被覆盖前从未被读过，entry 的 current 槽就是死数据，可以删。反之则不行。

## 方法

只读探针（不改变任何分支条件），在 `--request-log-jsonl` 的 `request_done` 里多记一个事实位：

| 文件:行 | 改动 |
|---|---|
| `include/ninfer/types.h:449` | `GenerationResult::attached_from_registry` |
| `src/runtime/contract/types.h:93` | `BeginSummary::attached_from_registry` |
| `src/targets/qwen3_6/impl/runtime/program.h:249` | `RequestControl::Prefill::attached_from_registry` |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:1027` | 填充：`request_plan.attach_registry_index >= 0` |
| `src/targets/qwen3_6/impl/runtime/program_impl.h:2701` | 进 `BeginSummary` |
| `src/runtime/engine/concurrent_executor.h:477` | 进 `GenerationResult` |
| `src/serve/generation_service.h:42` / `.cpp:400` | 进 `GenerationMetrics` |
| `src/serve/request_log.cpp:422`, `:601` | `attach=entry|lane` / `"attach_registry_entry"` |

服务：`ninfer-serve`，`--max-concurrency 1 --prefix-registry-slots 1 --max-context 32768
--kv-capacity 32768 --tp 2`，`speculative_backend=none`，采样 `temperature 0`、
`reasoning_effort none`、非流式。全部串行，每次等 `live.running == 0`。
脚本与原始日志：`_scratch/append_probe/`（`probe.py`、`probe3_d2.py`、`requests.jsonl`）。

## 五组结果

| # | 形状 | prompt | cached | prefix_reuse_path | attach_registry_entry |
|---|---|---|---|---|---|
| T1 | A1 → B → A2 | 1055 | 980 | restore_turn_checkpoint | **true** |
| T2 | A1 → B → A2 → 续写 | 1059 / 1121 | 979 / 1055 | restore_turn_checkpoint | **true** / false |
| T3 | A1 → B → A2 全文 + a2 + 尾部空白 | 1055 / 1106 | 980 / 1051 | restore_turn_checkpoint | **true** / false |
| T4 | A1 → B → C → A2 | 1055 | 0 | full_reset | false |
| T5 | A1 → A1 + a1 + 继续（阴性对照） | 1043 | 979 | restore_turn_checkpoint | false |

规定的五组里 `attach && append` 是 0，T5 的 `!attach && append` 也是 0——按预定判据这只说明
append 在这几个形状里没被走到，不能支持"可删"。

## 追加的两组（决定性）

| # | 形状 | prompt | cached | prefix_reuse_path | attach |
|---|---|---|---|---|---|
| D1 | A1 → `[u1, a1]`（assistant 结尾） | 1044 | 1035 | append_frontier | false |
| D3 ×2 | 同上，另两个文本 | 1041 | 1032 | append_frontier | false |
| **D2 ×2** | A1 → B → `[u1, a1]` | 1041 / 1046 | 1032 / 1037 | **append_frontier** | **true** |

聚合（41 条 `request_done`）：`attach=true & append_frontier` = **2**。

同一份 a1、同一 lane，只改尾巴（D3）：`[u1, a1]` → append（base 1032）；
`[u1, a1, 继续]` → restore_turn_checkpoint（base 981）。模板在最后一条 assistant 消息之后的
编码与被拒形状在 `[981, 1032)` 上 token 不同，逐 token 前缀校验因此失败。这是客户端编码形状的
属性，不是某个开关。

## 判决

`attach && AppendAtFrontier` 可达 ⇒ entry 的 current 槽被读 ⇒ **不可删**。
`attach_registry_sequence`（`program_impl.h:1731`）拷进 lane 的正是这份状态，append 分支随后
从它继续 prefill。

## 天花板

只覆盖这 5 + 3 种形状、6 段不同填充文本、单机单并发、`speculative_backend=none`；
不证明其它形状下的事实，也不回答"多常出现"。以 user 消息结尾的常规对话在 7 次采样里一次都
没走到 append，这只能说明该形状下 append 少见，不能说明它不存在。
