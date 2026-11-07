# CPU KLU 参考解与耗时

从仓库根目录运行（也可以在任意工作目录使用脚本的绝对路径）：

```bash
python3 src/KLU_standalone/test/run_reference.py
```

脚本先调用 `make` 构建 KLU，然后按文件名顺序，逐个求解 `src/matrix/*.mtx`。
仅需 Python 3.6+、C 编译器和 make，无需 GPU 或 Python 第三方库。
右端项为 `b = ones(n)`，使用 CPU double 精度和 KLU 默认参数，不做迭代精化。
直接使用文件中存储的矩阵元素，不展开 symmetric 标签，与当前 GLU 的输入保持一致。

默认输出到 `src/KLU_standalone/test_result/`：

| 文件 | 内容 |
|---|---|
| `<矩阵名>_x.txt` | 解向量，每行一个数，17 位有效数字；例如 `add32_csr_x.txt` |
| `<矩阵名>_timing.json` | 该矩阵的耗时、残差、状态和输出路径 |
| `<矩阵名>.log` | KLU 原始输出和错误信息 |
| `summary.csv` | 本次所有选中矩阵的汇总记录 |
| `total_time.json` | 本次成功数、失败数及累加耗时 |

所有耗时单位均为秒，测量的是单调时钟的实际经过时间：

- `analyze_s`：KLU 符号分析，包括 BTF 和排序。
- `factor_s`：KLU 数值 LU 分解。
- `solve_s`：一次右端项的三角求解。
- `total_s`：以上三项之和，不含编译、读入、CSC 组装、残差验算、输出和释放内存。
- `process_wall_s`：单个 KLU 程序完整运行时间，包含进程启动、读入、组装、求解、验算、输出和清理。
- `batch_wall_s`：整批运行时间，包含 Python 汇总开销，不含编译，仅在 `total_time.json` 中提供。

`total_time.json` 的 `total_s` 只累加成功矩阵；`process_wall_s` 累加所有尝试的运行时间。
这些是一次完整求解的直接测量，不使用 GLU 的 `0.005 * once_s + solve_s` 加权公式。
`residual_l2` 是 `||Ax-b||₂`，不是与另一个求解器的解向量误差。
`status=ok` 表示 KLU 成功并输出有限残差，不表示残差达到某个阈值。

只运行部分矩阵，或指定输入、输出路径和每个矩阵的超时：

```bash
python3 src/KLU_standalone/test/run_reference.py --matrix add32_csr --matrix bcircuit_csr
python3 src/KLU_standalone/test/run_reference.py --matrix-dir /path/to/matrices --output-dir /tmp/klu_results --timeout 300
```

重复运行会覆盖本次选中矩阵的结果和本次汇总表，未选中的矩阵文件保留。
矩阵失败或超时后继续运行其余矩阵，只保留日志和状态，不保留失败解；任何失败使脚本返回 1。
