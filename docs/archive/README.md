# 归档索引

本目录存放**不适用于当前构建**但仍需保留的文档。判定规则、横幅格式与保留期见
[文档适用性治理规范](../doc-policy.md)。

当前构建：Windows 11 · 2×V100-SXM2 16 GB · NVLink · `sm_70` · CUDA 12.8 · TP2。

## 布局

```text
docs/archive/
├── README.md                       # 本文件
├── inherited-5090/                 # 上游单卡 5090 / sm_120a 场景
├── inherited-3060/                 # 上游双 3060 / sm_86 场景
├── superseded/<yyyy-mm>-<topic>/   # 被取代的文件组
└── records/<yyyy-mm>/              # 一次性记录
```

## 已归档（2026-10-24）

| 文件 | 归属 | 原位置 | 替代 |
|---|---|---|---|
| [README_KO.md](inherited-3060/README_KO.md) | inherited-3060 | 仓库根目录 | [README.md](../../README.md) |
| [performance/qwen3.6-27b.md](inherited-5090/performance/qwen3.6-27b.md) | inherited-5090 | `docs/performance/` | [performance.md](../performance.md) 的 V100 Duo 段 |
| [performance/qwen3.6-35b-a3b.md](inherited-5090/performance/qwen3.6-35b-a3b.md) | inherited-5090 | `docs/performance/` | 同上 |
| [performance/qwen3.8-27b.md](inherited-5090/performance/qwen3.8-27b.md) | inherited-5090 | `docs/performance/` | 同上 |
| [performance/methodology.md](inherited-5090/performance/methodology.md) | inherited-5090 | `docs/performance/` | 同上（仅"公共剖面"不适用，规则本身仍沿用） |
| [serving-5090.md](inherited-5090/serving-5090.md) | inherited-5090 | `docs/performance.md` 拆分 | 同上 |
| [v0.1.0-win.1.md](superseded/2026-10-release-notes/v0.1.0-win.1.md) | superseded | `docs/Releases/` | [v0.1.1-win.5.md](../Releases/v0.1.1-win.5.md) |
| [v0.1.1-win.1.md](superseded/2026-10-release-notes/v0.1.1-win.1.md) | superseded | `docs/Releases/` | 同上 |

`serving-5090.md` 是从 `docs/performance.md` 切出的 866 行：Inherited RTX 5090 campaigns、单请求
方法、corpus makespan、并发饱和、TP2 + YaRN 1M、跨引擎对照与三份分模型 run records。原文件保留
V100 Duo 段，末尾有一段指针。

## 就地标注（未移动）

这些文件仍在原位，因为入链数 ≥ 2、是脚本写入路径、或本身就是目录入口。看文件顶部横幅。

| 文件 | 归属 | 为什么不动 |
|---|---|---|
| [docs/performance.md](../performance.md) | inherited-5090 段已切出 | 入链 12，且 README 大量引用其 V100 锚点 |
| [docs/maintainer/tp2-yarn-1m.md](../maintainer/tp2-yarn-1m.md) | inherited-5090 | 入链 3；transport 设计仍现行 |
| [docs/v100.md](../v100.md) | inherited-5090（部分） | 唯一的 `sm_70` 数值策略入口 |
| [tests/data/tp1-golden/MANIFEST.md](../../tests/data/tp1-golden/MANIFEST.md) | inherited-5090 | 说明同行数据被脚本读取 |
| [model-cards/](../../model-cards/) 全部 5 张 | inherited-5090 | 官方 artifact 卡片的仓库内副本 |
| [eval/results/](../../eval/results/README.md) 全部结果 | inherited-5090 | `eval/run_*.sh` 把 `results_dir` 固定在这些路径 |
| [docs/maintainer/softmax-attention.md](../maintainer/softmax-attention.md) | planned | 计划态，不是归档 |
| [docs/maintainer/model-weight-execution.md](../maintainer/model-weight-execution.md) | planned | 计划态，不是归档 |
| [AGENTS.md](../../AGENTS.md) | inherited（部分） | 仓库规则文件，用顶部"偏离"小节标注 |

## 保留期

`inherited-*` 与 `superseded/` 永久保留（上游同步与 release 审计都需要）。
`records/` 在归档满 6 个月后评估是否删除。
