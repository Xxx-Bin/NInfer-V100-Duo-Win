# 基线测量结果（唯一分母）

- 采集时间：2026-10-25 02:16–02:20
- 代码：`baseline/port-study-34412f4`（= `perf/v100x2-port-study` HEAD，工作区干净）
- 原始报告：`baseline.json`（schema_version 12）；stdout：`baseline.log`
- 命令（即 `protocol.env` 的全部字段）：
  ```
  ninfer_bench --weights F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer \
    --corpus bench/fixtures/bench_corpus.ids -pg 60000,512 \
    --prefill-chunk 2560 --kv-dtype int8 --mtp-draft-tokens 4 --lm-head-draft \
    --tp 2 --devices 0,1 -r 2 --warmup 1 --capture-generation -o json \
    --output-file _scratch/port-log/baseline.json
  ```

## 数值

| 指标 | 均值 | 标准差 | 相对抖动 |
|---|---:|---:|---:|
| prefill | **945.799 tok/s** | 1.573 | 0.166% |
| decode（output） | **129.727 tok/s** | 0.0086 | 0.0066% |
| prefill 耗时 | 63.439 s | 0.105 | — |
| decode 耗时 | 3.947 s | 0.00026 | — |
| 加载 | 73.88 s（H2D 22.46 GB） | — | — |

投机解码：`draft_window=4`，接受率 **0.96437**，平均接受长度 **4.830**，
accepted/drafted = 406/421，per-position [105,102,100,99]。
`decode_path = mtp_cuda_graph`，`decode_graph_prime.primed = true`。

显存（单卡）：weights 11.23 GB、sequence 1.28 GB、workspace 容量 755 MB、
runtime reservation 2.12 GB；`max_context=60520`，`kv_capacity=60544`，
KV 峰值 13.0 GB/卡（跑 prefill 时），温度峰值 71°C。

## 判定门槛（由本基线推出，写死）

| 门槛 | 值 | 说明 |
|---|---:|---|
| prefill 最小增益 | **+9.458 tok/s（+1.0%）** | 主导项；2×SD 仅 3.15，故用 1% |
| decode 最小增益 | **+0.649 tok/s（+0.5%）** | 2×SD 仅 0.017，故用 0.5% |

两项都需 **≥2 次 cold request** 且两次都过门槛；只过一次算否。

## oracle 指纹（前置硬否决）

任一次测量若下列任一项与基线不同，**直接判定该移植为否**，不看性能：

```
token_ids(513)  sha1 = 06287348665d3648
accepted_tokens     = 406
drafted_tokens      = 421
rounds              = 106
acceptance_rate     = 0.9643705
acceptance_length   = 4.8301887
accepted_per_position = [105, 102, 100, 99]
```

注：两次重复的输出与计数**完全一致**（同 sha1），说明本基线在
`mtp_cuda_graph` 下是确定性的 —— 所以这条门槛可以用「严格相等」，不需要容差。
