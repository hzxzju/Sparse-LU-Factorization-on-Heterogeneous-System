# 批量运行矩阵

脚本只依赖 Python 3 标准库。先在仓库根目录编译最新求解器，再批量运行：

```sh
make -C src MAIN
python3 src/run_batch.py
```

默认递归查找 `src/matrix` 下全部 `.mtx`，按路径顺序逐个运行。默认使用
`right-looking` 和 `level`，不开启扰动。初次求解后自动检查绝对残差
`||Ax-b||₂ <= 1e-6`，未达标时复用 LU 因子迭代校正，默认最多 n 次
（n 为矩阵阶数）。程序统计各阶段耗时，不执行逐 kernel profiler 采集。
矩阵目录和求解器默认路径相对于脚本，
因此从其他工作目录启动也可以；显式指定的相对路径则相对于当前工作目录。
脚本设置 `$HPCC_PATH/lib` 的运行库路径，`HPCC_PATH` 默认是 `/opt/hpcc`。

结果写入 `src/batch_results/<运行时间>/`：

- `summary.csv`：每个矩阵一行，包含矩阵信息、运行配置、各阶段耗时、总求解时间、三种残差和运行状态，共 23 列。
- `0001_<矩阵名>/solver.log`：该矩阵的完整标准输出和错误输出。
- `0001_<矩阵名>/x.dat`：该矩阵的解向量，由求解器生成。

`total_solve_ms` 精确提取 `Task 1 total solve time (T_gpu)`，单位为毫秒，
包括输入预处理、符号分析、GPU 初始化和传输、数值分解、首次 CPU 三角求解、
初始残差检查及全部迭代校正。
不使用 `Total GPU time`，也不叠加程序输出的各阶段时间。

各阶段耗时直接提取求解器输出，单位均为毫秒：

| CSV 列 | 阶段 |
| --- | --- |
| `preprocessing_time` | 输入与预处理 |
| `symbolic_time` | 符号分析 |
| `csr_time` | CSR 构建 |
| `predictlu_time` | PredictLU |
| `leveling_time` | 层级划分；同步自由调度时为 0 |
| `symbolic_preparation_time` | 符号准备总耗时，包含符号分析、CSR、PredictLU 和层级划分 |
| `numeric_lu_time` | 数值 LU 分解，包含 GPU 初始化和传输 |
| `triangular_solve_time` | 首次 CPU 三角求解；不含迭代校正 |

`symbolic_preparation_time` 与其子阶段有重叠，不能一起累加。

`residual_l1`、`residual_l2`、`residual_linf` 对应 `||Ax-b||` 的 1、2、无穷范数。
CSV 不再计算或保存 `relative_residual_l2`。这些残差不能替代与参考解比较的
`||x-x_ref||₂ / ||x_ref||₂`。残差基于原始矩阵和最终保留的解，使用 CPU
`long double` 累加（额外精度取决于平台），日志保留至少 17 位有效数字。
`x.dat` 使用科学计数法和至少 17 位有效数字，避免固定小数位截断。

## 计时范围与执行方式

所有 RL、LL、LL-SF kernel 均直接启动，不再查询 occupancy、kernel 资源属性或
CE/RM 计数器，也不逐次测量 kernel 耗时或写入采集文件。RL 小层继续使用多流执行，
仅保留算法和结果正确性所需的同步，不再为逐 kernel 采集额外串行化启动。

求解器日志仍输出 `Numeric LU compute time (wall-clock)`，统计数值计算阶段的
启动、执行和同步耗时，不含分配、初始化及主机/设备传输；CSV 中的 `numeric_lu_time`
覆盖整个 GPU 数值分解调用，两者范围不同。日志中的 `Total GPU time` 也继续保留。
`total_solve_ms` 包含残差检查和迭代校正，不包含最终 `x.dat` 写入和日志输出
（`--progress` 的实时日志除外）。`Iterative refinement time (CPU)` 单独记录残差检查、
校正求解和更新耗时；CSV 保持 23 列，这项耗时、迭代次数、初始/逐轮残差及
最终状态保存在 `solver.log`，其耗时已包含在 `total_solve_ms` 中。

单独运行求解器会直接输出阶段耗时，例如：

```sh
./src/lu_cmd -i src/nicslu/demo/ASIC_100k.mtx -u left-looking \
  -s synchronization-free
```

`--profile-kernels` 和 `--no-profile-kernels` 已移除，无需传入。
新运行不生成 `kernel_metrics.csv` 或 `kernel_profiles.jsonl`，也不在 `summary.csv`
中添加 kernel 统计列。已有运行目录中的历史文件保持原样，其旧耗时可能包含采集开销。
修改源码后须先运行 `make -C src MAIN` 重建求解器。

## 运行参数

选择 LL 与同步自由调度，并设置每个矩阵最多运行 600 秒：

```sh
python3 src/run_batch.py --update-strategy left-looking \
  --scheduling synchronization-free --timeout 600
```

指定输入目录和输出目录，开启静态对角扰动：

```sh
python3 src/run_batch.py --matrix-dir src/matrix \
  --output-dir /tmp/glu-batch-results --perturb
```

输出目录必须尚不存在，避免覆盖以前的结果。可用 `--solver /path/to/lu_cmd`
指定其他已编译的求解器；`--timeout 0`（默认）表示不限制运行时间。

失败、超时、缺失指标或非有限指标都会记入 CSV，并继续运行下一个矩阵。
CSV 每完成一个矩阵就写入磁盘。全部矩阵的求解和指标检查通过时脚本返回 0，
任何矩阵求解失败、残差不达标、超时或指标检查失败时返回 1，
Ctrl+C 中断时返回 130。`status=ok` 表示进程正常退出、所有指标有效且最终 residualL2 不超过指定阈值，
但不代表参考解相对误差也达标。求解器退出码 2 或残差超标记为
`status=residual_not_converged`，批量运行继续；保留的最好解仍写入 `x.dat`。

## 残差检查与迭代改进

命令行和批量脚本均支持以下参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--residual-tol T` | `1e-6` | 有限正数；绝对 residualL2 的达标阈值 |
| `--max-refine-iters N` | `n` | 最大校正次数；`n` 表示矩阵阶数，或指定 0～4294967295 的整数 |

```sh
# 最多校正 10 次，目标为绝对 residualL2 <= 1e-6
python3 src/run_batch.py --residual-tol 1e-6 --max-refine-iters 10

# 使用默认的矩阵阶数作为上限
./src/lu_cmd -i src/matrix/ASIC_680ks_csr.mtx --max-refine-iters n

# 对照初次求解结果；仍会检查阈值，超标返回 2
./src/lu_cmd -i src/matrix/twotone_csr.mtx --max-refine-iters 0
```

首次求解达标时不做额外三角求解。否则使用原始矩阵计算 `r=b-Ax`，调用现有
`solve_CSR` 解出校正量 `δx`（内部处理置换、缩放），再更新 `x←x+δx`。
GPU LU 因子保持复用，不重新分解。每轮检查候选解，达到阈值或用完上限即停止。
残差不下降时提前停止（`stagnated`），非有限数值时停止（`nonfinite`）；拒绝这些
候选更新并保留此前最好解。最终状态还包括 `converged` 和 `max_iterations`。
未达标返回 2，不能保证任意病态矩阵都能通过增加次数达到目标。

迭代增加 CPU 残差计算、三角求解、向量更新耗时及 O(n) 临时内存。
这里的自动迭代适用于 `lu_cmd` 与批量脚本，Python `pyglu` API 不自动启用。
`-p`/`--perturb` 是主元扰动，不是提高原方程精度的开关。

## SF 超时排查

`--timeout` 限制整个求解器进程，包括 CPU 准备阶段，不能仅凭超时判断 GPU 死锁。
默认诊断输出会先缓存在内存中；超时杀死进程后，这部分输出可能丢失。
`--progress` 可实时刷新阶段标记和 GPU 配置到 `solver.log`，不启用 profiler。
诊断输出本身有开销，正式性能比较时去掉该参数。

先在单个矩阵上观察最后停留的阶段：

```sh
make -C src MAIN
timeout 60s ./src/lu_cmd -i src/matrix/ASIC_100k_csr.mtx \
  -u left-looking -s synchronization-free --progress
```

再用一个 worker 对比，排除多个 worker 之间的依赖等待：

```sh
timeout 60s ./src/lu_cmd -i src/matrix/ASIC_100k_csr.mtx \
  -u left-looking -s synchronization-free --sf-workers 1 --progress
```

`--sf-workers 0`（默认）自动选择；正数表示请求的 worker 数，实际数量仍受内存、
矩阵规模和设备限制，见日志 `LL-SF workers`。非零值只能与 SF 一起使用。
单 worker 用于诊断，不能据此判断多 worker 的同步正确性或并行性能。

- 停在 `symbolic fill-in` 或 `numeric value preparation`：仍在 CPU 准备阶段。
- 停在 `GPU numeric LU started`，没有出现 kernel 启动消息：检查设备初始化、上传和工作区分配。
- 停在 `LL-SF kernel launch; waiting for completion...`：kernel 启动/同步尚未返回，
  需要区分计算慢、等待不推进及设备错误；先按 [tests/README.md](../tests/README.md)
  运行小规模原子发布测试。
- 单 worker 完成而自动 worker 超时：重点排查跨 worker 发布/等待、调度及轮询开销，
  不能只靠增大 timeout 判定修复。

批量脚本也支持 `--sf-workers 1 --progress`。各阶段统计与 23 列 CSV 格式保持一致；
请求的 worker 数及实际数量可从命令参数和每个矩阵的日志确认。


## 带宽利用率检查
occupancy使用ht-smi采样，里面有gpu-util。每10ms采样一个数据，要求在循环求解阶段达到60%利用率。按时间加权。
ht-smi -l 100 -o tmp.csv --show-usage
