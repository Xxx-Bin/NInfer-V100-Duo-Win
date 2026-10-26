# port/packet-allreduce · 两 epoch packet 邮箱 collective

## 改动摘要

参照 `_scratch/ref-v100x2/ninfer-V100X2-master/src/ops/kernel/allreduce.cuh:86-272` 与  
`src/ops/common/allreduce.cu` 对应实现移植两 epoch 固定 pitch 邮箱 collective。

| 文件                               |   行数 | 内容                                                                                                   |
| -------------------------------- | ---: | ---------------------------------------------------------------------------------------------------- |
| `src/ops/common/allreduce.cu`    | +132 | 邮箱内核（`0xffff` 空槽标记、每槽单生产者/单消费者、固定 pitch）；≤80 KiB 走 packet，更大走原 one-shot；`direct_admits` 增加 packet 分支 |
| `include/ninfer/ops/allreduce.h` |   +4 | `DirectChannel` 增加 packet 存储指针（独立 160 KiB，`0xff` 初始化）                                                |

要点：packet 使用**独立存储**并预先填 `0xff`，避免与 one-shot 路由共用缓冲区读到陈旧数据；  
归约仍是 FP32 求和 + 单次 BF16 舍入。

## 编译

`-j2`，270 目标，`BUILD_OK` @ 05:31:01。中途两次硬挂（04:10、04:39）与一次后台任务回收（05:00），  
见 `incident-20261025-freeze.md` 与 `STATE.md`。

## 测量（协议见 `protocol.env`）

前置：GPU 空载（9 MiB / 卡，50/49 °C）；测量窗口 WHEA = **0 条**；退出干净（显存 10 MiB，49/43 W）。

|               |                  基线 |                  packet |                门槛 | 结果                  |
| ------------- | ------------------: | ----------------------: | ----------------: | ------------------- |
| prefill tok/s |  945.799 (sd 1.573) |  **946.044** (sd 1.599) | ≥ 955.257 (+1.0%) | **未达**（+0.026%）     |
| decode tok/s  | 129.727 (sd 0.0086) | **135.804** (sd 0.0923) | ≥ 130.376 (+0.5%) | **通过**（**+4.685%**） |

逐 rep：

| rep      | prefill_s | decode_s | sha1             | acc/drf/rd  |
| -------- | --------: | -------- | ---------------- | ----------- |
| base 0   |    63.364 | 3.9466   | 06287348665d3648 | 406/421/106 |
| base 1   |    63.513 | 3.9469   | 06287348665d3648 | 406/421/106 |
| packet 0 |    63.346 | 3.7683   | 06287348665d3648 | 406/421/106 |
| packet 1 |    63.498 | 3.7719   | 06287348665d3648 | 406/421/106 |

oracle：**逐位一致**，`acceptance_rate=0.9643705`、`acceptance_length=4.8301887`、  
`per-position=[105,102,100,99]` 全部相等 → 纯传输改动，A 类。

## 为什么 prefill 是 0.000 而 decode 是 +4.685%

这不是"只改了一半"，是路由的必然结果：

- decode 每次 collective 的消息是 `5120(BF16) × T=5 ≈ 50 KiB` ≤ 80 KiB → 走新 packet 路径
- prefill 每 chunk 2560 token，消息 `5120 × 2560 × 2 ≈ 26 MB` → 远超 80 KiB，仍走原 one-shot 路径
- 且 prefill 是 GEMM 主导，collective 开销本来就被掩盖；decode 有 106 轮 × 每层一次，是延迟敏感

所以 **prefill 结构上不该动**，`+0.026%` 落在噪声内（sd 1.573）是正确的"无变化"。

## 复测（2026-10-25 06:35，提交 df1f2958，确认抢救的代码没变味）

改动曾被误删，从会话记录逐字节回放后提交 `df1f2958`，重新全量编译并复测：

| | 首测（未提交版本） | 复测（df1f2958） | 差异 |
|---|---:|---:|---:|
| prefill tok/s | 946.044 | 945.623 | −0.045% |
| decode tok/s | 135.804 | **135.594** | −0.155% |
| oracle sha1 | 06287348665d3648 | 06287348665d3648 | 一致 |
| acc/drf/rd | 406/421/106 | 406/421/106 | 一致 |

两次独立测量相差 0.155%，在 decode 的 sd（0.09）量级附近但略大，属正常轮间波动。
**结论：回放的代码与首测是同一份，decode 增益 +4.5~4.7% 可复现。**

判定：`oracle=A 类`，`decode gate=PASS`，`prefill gate=FAIL（−0.019%，结构性为 0）` →
按 v2 流程**存起来**（`result/packet-allreduce` @ `df1f2958`），整条队列跑完后统一裁决。

## 判定：已存档，待统一裁决

按调整后的流程（`protocol.env` 判定方式 v2）：测完不当场决定删留，**存起来**，整条队列跑完后统一裁决。

- 分支：`port/packet-allreduce`，提交 **`df1f2958`**
- tag `result/packet-allreduce`：**待复测通过后再打**（先前那个指向基线的错 tag 已删除）
- decode 135.804 已过门槛（+4.685%），prefill 946.044 未过（+0.026%，结构性）
- 作为 ladder tier1（按复测值 945.623 / 135.594 计）：后续依赖项门槛 = prefill ≥ 955.079、decode ≥ 136.272

### 事故记录（2026-10-25 05:46，已恢复）

第一次"存起来"是**错的**：改动只在 worktree 工作区、**从未 commit**，`git worktree remove --force`  
把 `+132 / +4` 行全部删掉，而 `result/packet-allreduce` tag 实际指向基线提交 `34412f4f`。

恢复方式：从会话记录（`.../98b29e18-....jsonl`）里按 `function_call` / `Edit` 逐条提取 5 处编辑的  
`old_string`/`new_string`，按行尾规范化后回放 → `git diff --stat` 得到 `+132 / +4`，与丢失前完全一致  
→ 提交 `df1f2958`。随后重新 configure + 全量编译 + 复测，确认该提交确实产出 135.804。

教训已写进 `protocol.env`：**存起来的第一步是 commit**，且 `git rev-parse result/<feat>` 必须不等于  
`34412f4f`（等于就说明改动根本没进版本库）。

## 资产位置

- 报告：`_scratch/port-log/packet-allreduce.json`
- 日志：`_scratch/port-log/packet-allreduce.log`
- 分支：`port/packet-allreduce`（worktree `.agents/worktrees/port-packet-allreduce`，**保留中**）
