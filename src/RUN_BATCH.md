# 批量运行矩阵

脚本只依赖 Python 3 标准库。先在仓库根目录编译最新求解器，再批量运行：

```sh
make -C src MAIN
python3 src/run_batch.py
```

默认递归查找 `src/matrix` 下全部 `.mtx`，按路径顺序逐个运行。默认使用
`right-looking` 和 `level`，不开启扰动。矩阵目录和求解器默认路径相对于脚本，
因此从其他工作目录启动也可以；显式指定的相对路径则相对于当前工作目录。
脚本设置 `$HPCC_PATH/lib` 的运行库路径，`HPCC_PATH` 默认是 `/opt/hpcc`。

结果写入 `src/batch_results/<运行时间>/`：

- `summary.csv`：每个矩阵一行，包含各阶段耗时、总求解时间、三种残差和运行状态。
- `0001_<矩阵名>/solver.log`：该矩阵的完整标准输出和错误输出。
- `0001_<矩阵名>/x.dat`：该矩阵的解向量，由求解器生成。

`total_solve_ms` 精确提取 `Task 1 total solve time (T_gpu)`，单位为毫秒，
包括输入预处理、符号分析、GPU 初始化和传输、数值分解以及 CPU 三角求解。
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
| `triangular_solve_time` | CPU 三角求解 |

`symbolic_preparation_time` 与其子阶段有重叠，不能一起累加。

`residual_l1`、`residual_l2`、`residual_linf` 对应 `||Ax-b||` 的 1、2、无穷范数。
CSV 不再计算或保存 `relative_residual_l2`。这些残差不能替代与参考解比较的
`||x-x_ref||₂ / ||x_ref||₂`。数值精度沿用求解器的日志输出精度。

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
CSV 每完成一个矩阵就写入磁盘。全部正常结束时脚本返回 0，有失败时返回 1，
Ctrl+C 中断时返回 130。`status=ok` 表示进程正常退出且所有指标有效，
不代表通过某个误差阈值；请根据残差列判断精度。
