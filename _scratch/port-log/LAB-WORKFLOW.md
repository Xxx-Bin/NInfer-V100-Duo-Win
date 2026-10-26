# Lab worktree 规程（2026-10-25 06:25 起生效）

替代原战役「每项新建 worktree」的流程。**存档形态不变**（每项仍是一个 `port/<feat>` 分支 +
`result/<feat>` tag），只是构建缓存不再随项重建。

## 为什么改

实测：冷编译 **222 个编译单元**（129 nvcc + 93 cl）、-j2 下 **28~35 分钟**；
而每项实际只改 2~4 个文件，真正需要重编的上限是：

| 头文件 | include 者数 |
|---|---:|
| `include/ninfer/ops/allreduce.h` | 15 |
| `include/ninfer/ops/linear_add.h` | 7 |
| `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | 3 |

即 20 来个 TU 够用。冷编译里多出来的 200 个，全是"新 worktree = 新 build 目录"的代价。
租用同一个热 build 目录后，每项编译从 30 min 降到 10~12 min。

## lab worktree 的位置

```
.agents/worktrees/port-packet-allreduce
```

**目录名保持不动。** Ninja 的 `build.ninja` 里烧的是绝对路径，改目录名等于再付一次冷编译
（30 分钟）。所以这个目录从第 6 项起只作为"lab"，名字是历史遗留，**不代表当前项**。
分支在它里面切换，`git worktree list` 里看到的名字与当前分支不一致是正常的。

## 每项的标准动作

```bash
LAB="$MAIN/.agents/worktrees/port-packet-allreduce"
cd "$LAB"

# 1) 回到本项的基线（通常是上一项的 result tag）
git checkout -- .
git status --porcelain            # 必须为空
git log --oneline -1               # 确认在预期提交

# 2) 建本项分支
git checkout -b port/<feat>

# 3) 改代码 → 增量编译（不要重新 configure）
JOBS=2 bash "$MAIN/_scratch/port-bat/build-port.sh" <feat> build
#    注意：build-port.sh 里的 "configure" 参数只在 CMakeLists 变更时才用

# 4) 测量
bash "$MAIN/_scratch/port-bat/measure.sh" <feat> <ref>

# 5) 存档（与旧流程相同）
git add -A && git commit -m "..."
git tag result/<feat>
git checkout <下一项的基线>
```

**不再执行 `git worktree remove`。** 原流程删 worktree 是为了省磁盘，现在它是热缓存，
删掉就要重付冷编译。

## 三条硬约束

1. **每项开始前 `git status` 必须为空**。lab 里若残留上一项的改动，A/B 立刻失效。
2. **不要在 lab 里 `git clean -xdf`**，那会连同 `build/` 一起删掉热缓存。
3. **改动前先看影响面**：能不碰 `allreduce.h` / `linear_add.h` 就不碰。
   新 API 优先放新头文件（如 `allreduce_norm.h`），只让真正使用它的 TU 重编。

## 测量独占：连"只读"的准备代理也不行（2026-10-25 07:0x 踩过）

第 9 项第二轮测量里，decode 的 rep0 掉到 **129.948 tok/s**，几乎等于未优化的基线 129.7。
排查结果：两个 rep 的总时长几乎相同（67.490 / 67.452 s），但 rep0 的 decode 慢 0.176 s、
prefill 快 0.133 s —— **不是 GPU 变慢，是 host 侧争用**。

decode 每步有 128+ 次跨卡 collective，对 launch 延迟极其敏感；当时我在测量窗口里派了一个
做全仓 Grep 的只读准备代理。**"只读"只保证不写文件，不保证不占 CPU。**

规则：
- `measure.sh` 运行期间（约 4.5 分钟）**不派任何代理、不编译、不做全仓搜索**。
- 要并行准备资料，等测量结束后再派。
- 每项测量前记录 `nvidia-smi` 空载，测完再记一次；两次都要干净。

## 数值预检（可选，省一轮 E2E）

`tests/ops/test_allreduce.cpp`、`test_rmsnorm.cpp` 已存在，但默认 `BUILD_TESTING=OFF`。
启用它会触发重新 configure —— **只在确认 `build/CMakeFiles/rules.ninja` 前后一致后才用**，
否则编译选项可能变化、测量作废。

## 已知代价（如实记录）

- 项之间切换要 `git checkout -- .`，那 20 个 TU 会重编一次；再打下一项改动又重编一次。
  所以每项实际是 **2 × 5~6 min**，而不是 1 × 5 min。仍然远好于 30 min。
- 若某项的基线不是上一项（例如退回原始基线），切换成本相同。
