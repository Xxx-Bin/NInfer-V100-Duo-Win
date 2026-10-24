# eval/results

本目录存放评测脚本的**输出**，不是手写文档。路径由 `eval/run_*.sh` 的 `results_dir` 固定，
因此目录本身不可移动，见 [文档适用性治理规范](../../docs/doc-policy.md)。

## 当前状态

| 子目录 | 内容 | 状态 |
|---|---|---|
| `cross-engine-nvfp4/` | NInfer vs vLLM 长上下文对照（500 W/GPU） | inherited-5090，不适用 |
| `final-rerun/` | 575 W 功耗条件下的复测 | inherited-5090，不适用 |
| `mtp3-1m/` | MTP3 @ 1M 上下文的贪婪流一致性 | inherited-5090，不适用 |
| `needle-1m/` | 653k / 1,046k needle 检索 | inherited-5090，不适用 |
| `soak-1m/` | 1,048,576-token 浸泡 | inherited-5090，不适用 |

这些结果都在**单卡 RTX 5090 / Linux 主机**上采集，上下文规模超出 2 × 16 GB V100 的能力。
脚本仍保留（可复用于其他目标），重新运行会在同目录覆盖同名文件。

适用于当前构建的测量在 [docs/performance.md](../../docs/performance.md) 的 V100 Duo 段。
