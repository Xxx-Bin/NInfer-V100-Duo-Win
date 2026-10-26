# port/rmsnorm-wiring · 融合 RMSNorm 进 collective（站点 A）· **负结果**

分支 `port/rmsnorm-wiring`，基线 `result/packet-allreduce`（`df1f2958`）。

## 判决

**oracle 正确，但性能稳定退化 −2.75%。本项否决，并连带否决站点 B 与第 7 项的同几何变体。**

## 两次测量（都在 GPU 空载、无并行任务下）

### 第一次：功能接线错误（已修）

| | 值 | 正常值 |
|---|---|---|
| sha1 | `1a870d67bc0cec66` | `06287348665d3648` |
| accepted/drafted/rounds | **6/2014/505** | 406/421/106 |
| acceptance_length | **1.0119** | 4.8302 |
| decode | 26.9 tok/s | ~136 |

**根因：消费端先于生产端上线。** 工人实现了 `input_prenormalized` 这个消费侧开关
（`attn_mix_tp2` 跳过自己的 `rmsnorm(x, input_norm)`、直接把 `norm_hidden` 当 `h`），
并在调用处传 `fuse_norm && layer > 0` —— 对第 1~47 层恒为真。
但**站点 B 本轮没实现**，没有任何东西产出"已归一化的 x"，于是这些层把
**上一层的 `post_attn_norm` 结果**当成自己 `input_norm` 的结果用掉了。

修法：新增命名常量 `kNextInputNormalized = false`（`text_context.h:107`），
两个调用点改传它，注释写明"只在实现站点 B 时一起翻"。

### 第二次：数值正确，性能退化

参照 `packet-allreduce`（135.594 / 945.623）。

| rep | prefill tok/s | decode tok/s | sha1 | acc/drf/rd |
|---|---:|---:|---|---|
| 0 | 943.898 | **131.914** | 06287348665d3648 | 406/421/106 |
| 1 | 943.645 | **131.821** | 06287348665d3648 | 406/421/106 |
| 均值 | 943.771 | **131.867** | **A 类逐位一致** | 逐项相同 |

| 指标 | delta | 门槛 |
|---|---:|---:|
| prefill | −0.196% | ≥ −0.33%（噪声下限） → 过 |
| decode | **−2.748%** | ≥ +0.5% → **不过** |

两 rep 相差 0.07%，远小于 2.75% 的退化量 ⇒ **是稳定退化，不是噪声**。

## 为什么退化：融合把归约的并行度砍掉了

| | packet 单独跑 | 融合后 |
|---|---|---|
| 归约 kernel 的 CTA 数 | **80**（每 block 1 KiB，64 线程，每线程 1 个 pack） | **T**（每列一个 CTA，256 线程，每线程 10 个 pair） |
| 占用 SM | 80 | 5~6 |
| 每线程的 publish→spin→retire 次数 | 1 | **10，串行** |

decode 每层一次 collective、64 层/轮，而列数就是 T（MTP4 时 5~6）。
融合后每次交换只剩 5~6 个 CTA 参与，而 spin 等待是延迟敏感操作；每线程串 10 次
等于把延迟暴露 10 倍。省下的一次 kernel launch（3~5 µs）远不足以补偿。

**这不是"折算系数偏保守"，是符号反了**：上游阶梯给它 +0.69%（TP4 的 ladder），
本机实测 −2.75%。

## 由此外推的三条结论

1. **站点 B（down 归约 → 下一层 input_norm）不做**。同内核、同列数几何，预期同为负。
   仅节省的 launch 数相同，而交换延迟惩罚相同。
2. **第 7 项（projection epilogue 发布 peer packet）的前提被削弱**。
   它的消费端就是 `allreduce_published_sum_rmsnorm_to`，即同一个融合内核；
   即使 epilogue 侧零成本，消费侧的 −2.75% 也不会消失。
   → 建议在第 6 项否决后**一并出队**，除非改成"发布但不融合 norm"的新设计（那需要新增消费端）。
3. **凡是"把宽归约塞进按列切分的 CTA"的融合，在本机都是负的。**
   packet 之所以有效，正是因为它用 80 个细粒度 CTA 摊开了交换延迟。
   以后评估类似融合，先看 CTA 数会不会从 O(packs) 掉到 O(columns)。

## 仍有价值的两处产出

- **`kNextInputNormalized` 这个显式开关**：把"消费端"与"生产端"解耦成一个具名常量，
  避免下次再出现"消费先上线、生产缺席"的静默错误。
- **`norm_admits` 补了 `norm.output[r].data == nullptr` 检查**（`allreduce.cu:530`）。
  原先只校验 shape 与对齐，shape 正确但 data 为空的 output 会让融合内核写空指针。

## 资产

- 报告：`_scratch/port-log/rmsnorm-wiring.json`（功能错误版）、`rmsnorm-wiring-r2.json`（有效）
- 分支：`port/rmsnorm-wiring`
- 前置：`port/rmsnorm-fusion` @ `8bb59c64`（API + 内核，未接线）
- 设计：`plan-rmsnorm-fusion.md`、`plan-rmsnorm-wiring.md`
