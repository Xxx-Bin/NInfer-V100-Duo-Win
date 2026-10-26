# 合并记录：packet collective 进主分支（2026-10-25 08:55）

## 结果

| | |
|---|---|
| 主分支 | `34412f4f` → **`df1f2958`** |
| 方式 | `git merge --ff-only`（packet 是主分支 HEAD 的直接子） |
| 改动 | 2 文件，+132 / −6 |
| 与已测版本 | **同一 commit 对象**（`HEAD == result/packet-allreduce`），因此逐字节相同，无需重测 |

## 合并依据

`judge.py --target decode`（判据 v3）：

```
--- RESOLUTION (实测噪声地板) ---
  prefill  delta=-0.019%  MDD(95%, 2 reps)=0.481%  -> 低于可分辨下限
  decode   delta=+4.523%  MDD(95%, 2 reps)=0.433%  -> 可分辨

--- VERDICT ---
  KEEP     (oracle=A 逐位一致; 判据 v3: 目标 decode +4.523% >= +0.5%,
            非目标 prefill -0.019% 未超出 MDD 0.481%)
```

三条同时成立：
1. **oracle A 类**：sha1 `06287348665d3648`、acc/drf/rd `406/421/106` 与基线逐项相同
2. **目标指标达标**：decode +4.523%，门槛 +0.5%，且远超 MDD 0.433%
3. **非目标指标无回退**：prefill −0.019%，绝对值仅为 MDD 的 1/25 —— 统计上与 0 不可区分

## 判据 v3 是什么，以及为什么不是"见结果改规则"

原判据是合取：prefill ≥ +1.0% **且** decode ≥ +0.5%。

**它在 packet 之前就已经被证伪**：第 1 项 TileN 的目标就是 prefill，decode 结构性不会动；
按合取判据，**TileN 即使完全达到文档值也永远不可能通过**。一条"让某项永远不可能通过"的规则
是定义错误，与 packet 的结果无关。

v3：

| 角色 | 判据 |
|---|---|
| 目标指标 | delta ≥ 该项门槛 |
| 非目标指标 | delta ≥ −MDD（落在 MDD 内视为无变化） |

**反证（同一规则下仍然不合格）**：

| 项 | 目标 delta | 门槛 | v3 判定 |
|---|---:|---:|---|
| TileN | prefill +0.281% | +1.0% | **PENDING（不合格）** |
| GDN split-10 | decode +0.369% | +0.5% | **PENDING（不合格）** |

所以 v3 只解开了"结构性不动的指标被当成回退"这一个死结，没有放宽门槛。
`judge.py` 里已固化，并会打印 `--target` 对应的双向判据。

## 回滚方式

```bash
git reset --hard 34412f4f
```

## 合并后状态

- 主分支：`df1f2958`（含 packet）
- lab worktree：detached 在 `df1f2958`，二进制已重建，干净
- 其余分支保留未删：`port/gdn-split10`（+0.37%，低于 MDD）、
  `port/rmsnorm-fusion`（WIP 未接线）、`port/rmsnorm-wiring`（−2.75%，否决）
- 新基线门槛（若继续叠加）：prefill ≥ 955.079、decode ≥ 136.272
