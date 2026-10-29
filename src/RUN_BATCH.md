# 批量运行矩阵

本机单卡 Docker 环境的进入方式与当前 GPU 验收状态见
[GPU_Docker_Experiment.md](../docs/GPU_Docker_Experiment.md)。
可在主机用 `bash src/gpu_docker.sh shell` 进入容器后执行下面的命令。

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

- `summary.csv`：每个矩阵一行，包含矩阵信息、运行配置、各阶段耗时、总求解时间、三种残差和运行状态，共 25 列；`symbolic_backend` 记录请求的符号后端，实际回退见日志；`numeric_sm_padding` 为实验开关（0/1，默认 0）。
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
最终日志中的 L2 范数转为 `double` 输出，收敛判断使用转换前的高精度值。
`x.dat` 使用科学计数法和 `long double` 的 `max_digits10` 精度，保留高精度解；
重新校验时也需以 `long double` 或更高精度读取，转回 `double` 可能丢失校正效果。

## 计时范围与执行方式

默认模式下所有 RL、LL、LL-SF kernel 均直接启动，不再查询 occupancy、kernel 资源属性或
CE/RM 计数器，也不逐次测量 kernel 耗时或写入采集文件。RL 小层继续使用多流执行，
仅保留算法和结果正确性所需的同步，不再为逐 kernel 采集额外串行化启动。
显式开启下述 `--numeric-sm-padding` 实验时例外：启动前查询一次 kernel 属性和
occupancy，并改用 cooperative launch。

求解器日志仍输出 `Numeric LU compute time (wall-clock)`，统计数值计算阶段的
启动、执行和同步耗时，不含分配、初始化及主机/设备传输；CSV 中的 `numeric_lu_time`
覆盖整个 GPU 数值分解调用，两者范围不同。日志中的 `Total GPU time` 也继续保留。
`total_solve_ms` 包含残差检查和迭代校正，不包含最终 `x.dat` 写入和日志输出
（`--progress` 的实时日志除外）。`Iterative refinement time (CPU)` 单独记录残差检查、
校正求解和更新耗时；CSV 共 25 列，这项耗时、迭代次数、初始/逐轮残差及
最终状态保存在 `solver.log`，其耗时已包含在 `total_solve_ms` 中。

单独运行求解器会直接输出阶段耗时，例如：

```sh
./src/lu_cmd -i src/nicslu/demo/ASIC_100k.mtx -u left-looking \
  -s synchronization-free --symbolic-backend gpu-sf
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

## 数值分解 SM 空转实验（默认关闭）

新增 `--numeric-sm-padding`，仅支持 `left-looking + synchronization-free`。
求解器和批量脚本均可使用；不加此参数时保持原有启动、工作区预算和计算路径，
不查询该实验的 occupancy，也不分配完成计数器。与 RL 或 level 调度组合时直接拒绝。

```sh
make -C src MAIN

# 普通对照：显式固定计算 worker 上限，便于与实验组比较。
./src/lu_cmd -i src/matrix/ASIC_680ks_csr.mtx -u left-looking \
  -s synchronization-free --sf-workers 49 --symbolic-backend cpu

# 实验组：保留相同的计算 worker 上限，增加其余 SM 的空转 block。
./src/lu_cmd -i src/matrix/ASIC_680ks_csr.mtx -u left-looking \
  -s synchronization-free --sf-workers 49 --symbolic-backend cpu --numeric-sm-padding

# 批量实验，开关写入 summary.csv 的 numeric_sm_padding 列。
python3 src/run_batch.py --update-strategy left-looking \
  --scheduling synchronization-free --sf-workers 49 --numeric-sm-padding \
  --timeout 600
```

Python 扩展重建后也支持：
`pyglu.splu(A, update_strategy="left-looking", scheduling="synchronization-free", numeric_sm_padding=True)`；
`spsolve` 支持同一 keyword-only 参数，默认 `False`。

### 布局、工作区与退出

- grid block 数等于所选设备的 SM/MP 数，不固定为 104，也不固定为 256。
- 计算 worker 的上限仍由 `--sf-workers`（0 为自动）、n 和原有工作区预算决定，
  实验模式额外封顶为 SM 数；分配失败时仍减半重试。自动模式可能比普通 LL-SF 少，
  因此性能对照建议显式设置相同的 `--sf-workers`。
- 只为实际计算 worker 分配 `compute_workers × n × sizeof(REAL)` 字节列工作区。
  计算 block 使用紧凑的 `[0, compute_workers)` 工作区编号；空转 block 不访问列工作区、
  不领取列任务。计算 worker 复用原 LL-SF 的同一个 device 函数，列任务领取、前驱等待、
  算术及完成发布顺序不变。额外设备状态只有一个每次分解清零的 unsigned 完成计数器。
- 每 block 预留 `sharedMemPerMultiprocessor / 2 + 256` 字节动态共享内存，连同静态共享
  内存检查设备限制，再查询 `hcOccupancyMaxActiveBlocksPerMultiprocessor`，必须恰为 1。
  X201 的 64 KiB/SM 对应 33,024 字节动态预留。然后通过
  `hcLaunchCooperativeKernel` 启动共同驻留的 grid；共享内存预留不用于列数据。
- 计算 worker 耗尽任务后，先由 block barrier 汇合所有线程，再用 device-scope
  acquire/release 原子操作递增完成计数；随后也进入等待。所有 block 的 leader
  acquire 轮询，计数等于实际计算 worker 数时全体退出。停止通知由 GPU 发出，
  不依赖主机在 `hcDeviceSynchronize()` 之后发通知。
- 不支持 cooperative launch、共享内存不足、occupancy 不是 1 或启动失败时明确报错，
  不静默退回普通启动，避免把失败的实验当成成功。日志中的 `Numeric SM padding:`
  记录实际 grid、计算/空转 worker 数、动态共享内存、驻留数和列工作区字节数。

例如设备有 104 个 SM、实际计算 worker 为 49 时，启动 104 个 block，其中 49 个
计算、55 个空转；列工作区仍是 49 份。若所有 SM 都分配到计算 worker，则空转 block
数为 0，先完成的计算 worker 仍等待其余 worker 完成后一起退出。

这是主动增加设备活动的实验，会影响 util 观测；不代表有效计算量增加，也不保证
运行时间不变。需同时比较 `Numeric LU compute time (wall-clock)`、`numeric_lu_time`、
`total_solve_ms` 和残差。计算计时包含空转、完成通知和全 grid 退出，分配、初始化及
资源查询计入完整数值分解调用。采样仍按 [Kernel_Util_Profiling.md](../docs/Kernel_Util_Profiling.md)
执行，短窗口及硬件指标刷新限制仍然存在。

不访问 GPU 的资源配置测试，以及需要 GPU 的数值回归：

```sh
make -C src test-numeric-sm-padding-host numeric_strategies_test
timeout -k 3s 180s ./src/numeric_strategies_test --sm-padding
```

GPU 回归将实验的 1/3/49/自动 worker 配置与 CPU 分解和普通 RL/LL/LL-SF 对照，
覆盖空转 block、多列依赖、扰动，以及同一符号结构的重复数值分解。

2026-10-30 已在 `glu-gpu3`（物理 GPU 3，容器逻辑设备 0）通过上述 17 组 GPU
回归。实际 `add32_csr.mtx` 求解也通过：104 个 block、49 个计算 worker、55 个空转
worker，列工作区 1,944,320 字节，最终残差 L2 约 `1.60e-13`，退出码 0。
主机资源配置测试通过；Python 回归共 54 项，50 项通过，4 项因缺少 NumPy/原生
扩展而跳过。尚未进行 util 增幅或运行时间不变的性能验收。

## GPU 符号分析 SF（2026-10-26）

按照 [SF 实施说明](../docs/GLU_GPU_Symbolic_SF_Codex_Implementation_Spec.md)
新增 `gpu-sf`，保留 `cpu` 与旧图搜索 `gpu` 供回归和性能对照。
符号后端独立于数值 LU 的 `-u/-s`；例如符号 SF 也能接 RL-level。
后续 `csr()`、`predictLU()`、`leveling()` 与数值 LU 继续使用原来的主机 CSC 布局。
Python `pyglu`、C++ 配置、`lu_cmd` 和批量脚本均可选择 `auto/cpu/gpu/gpu-sf`，默认 `auto`。

```sh
make -C src MAIN symbolic_gpu_test symbolic_gpu_numeric_test DEBUG=0 -j2

# 首次正确性验收：逐元素比较 CPU/GPU 的三份 CSC 数组，禁用静默回退
./src/lu_cmd -i src/matrix/add32_csr.mtx -u left-looking \
  -s synchronization-free --symbolic-backend gpu-sf \
  --symbolic-verify --symbolic-no-fallback --progress

# 批量运行；每个矩阵 600 秒上限
python3 src/run_batch.py --update-strategy left-looking \
  --scheduling synchronization-free --symbolic-backend gpu-sf \
  --symbolic-no-fallback --timeout 600
```

| 参数（求解器与批量脚本均支持） | 默认值 | 含义 |
| --- | --- | --- |
| `--symbolic-backend auto\|cpu\|gpu\|gpu-sf` | `auto` | 按输入结构选择 CPU/SF，或显式指定 CPU、旧图搜索、新 SF |
| `--symbolic-sf-workers N` | `0` | SF worker 数，0 按 `2 × MP` 自动选择，并受 n、显存和设备限制 |
| `--symbolic-block-size N` | `256` | 64、128 或 256，遵守原生 64-lane wave |
| `--symbolic-pool-entries N` | 自动 | 每个稀疏行号 arena 的初始容量，单位 unsigned 条目；正数可用于小池重试测试 |
| `--symbolic-max-pool-entries N` | 显存预算/32-bit 上限 | 每个行号 arena 的容量上限，正数；不得小于显式初始容量 |
| `--symbolic-verify` | 关闭 | 运行 CPU golden 并逐元素比较 `sym_c_ptr/sym_r_idx/l_col_ptr` |
| `--symbolic-no-fallback` | 关闭 | GPU 初始化、资源或调度错误直接失败；默认显式记录后回退 CPU |

workers/pool 参数可用于 `auto`、`gpu-sf`；verify/no-fallback 可用于 `auto` 或 GPU 后端。
`auto` 选中 CPU 时这些 GPU 参数不生效，选中 CPU 本身不属于 fallback。
求解器的 `--symbolic-batch-cols` 仍仅控制旧 `gpu` 的 workspace slot 上限，
新 SF 使用独立的 `--symbolic-sf-workers`。它与数值 LU 的 `--sf-workers` 含义不同，
无需为了调整符号 worker 而修改数值调度。

默认 `auto` 在预处理后、`fill_in` 开始时，只扫描实际传入的列指针。
令 `m_j = ap[j+1] - ap[j]`、`B = Σ m_j × max(0, m_j - (n-j))`，`j` 从 0 编号。
同时满足 `n >= 20000`、`max(m_j)/n >= 0.10`、
`max(m_j)/(nnz/n) >= 64`、`B >= 10^9` 时使用 GPU-SF，否则使用 CPU。
复杂度为 O(n) 时间、O(1) 额外空间；64 位整数比较保留全部阈值边界，判断不初始化 GPU。
CLI 此处将重排后的 CSR 当作转置矩阵的 CSC，因此使用预处理后的实际顺序进行判断。
显式传 `cpu`、`gpu`、`gpu-sf` 可覆盖自动选择。

日志保留 `Symbolic requested backend: auto`、`Symbolic auto selected backend`，
以及 `symbolic_auto_input_nnz`、`symbolic_auto_max_column_nnz`、
`symbolic_auto_copy_lower_bound`、`symbolic_auto_ms`。
自动判断计入完整 `Symbolic time`；`summary.csv` 的 `symbolic_backend` 记录请求值 `auto`，
实际使用的后端和可能的 GPU 失败回退见日志。CPU 选择路径不做 GPU 设备探测。

### 算法与发布协议

每个 worker 有 `work_bitmap/dirty_bitmap`，共
`2 × W × ceil(n/32) × 4` 字节。新列只读一次原始 CSC，并补入结构对角；
thread 0 从 bitmap 中按 `cursor` 找到最小的未处理前驱，整个 block 连续读取
已发布前驱的 `[diag,end)` 行号并协作做集合并集，继续处理合并中新出现的前驱。
这里的区间包含前驱对角，严格对应 CPU `set_union` 的语义。

`symbolic_sf.cuh` 中 `sym_done=0` 表示可运行/恢复，`1` 表示等待；完成以独立的
`FINISHED` 状态表示，与数值 SF 的 `done=1` 表示完成的协议分开。
未完成前驱不会占住 worker 自旋：只保存 dirty 中新增的行及单调递增 cursor，
然后以 CAS 登记到该前驱的 waiter 链，并让 worker 取其他任务。
恢复时重建 bitmap，不重读已合并过的前驱；每列的 checkpoint 行号只保存一次，
总条目不超过最终结构 nnz。

使用设备作用域 acquire/release 原子操作发布完成列、checkpoint 和队列项；
publisher 将 waiter 链原子交换为 CLOSED 后逐个唤醒。
若登记时已看到 CLOSED，则撤销等待并在本 worker 继续，避免丢失唤醒与重复入队。
ready queue 是带逐 slot sequence 的有界 MPMC ring，容量至少为
`next_power_of_two(n+W+1)`，不依赖 block 编号或全 grid 同时驻留。
队列空时只有有界退避，完成计数达到 n 或出现 error code 后退出。

完成列先写只读 final arena；最后复用 64-bit 分层 GPU prefix scan，并按列号 pack
成 CSC，`l_col_ptr` 转成全局索引。只在下载、边界验证及可选 CPU 比较全部成功后，
才提交三个结构数组并清空旧 CSR、数值、levels 和 debug 缓存。

### 显存、错误及输入约束

不建立 `n×n` unsigned 或浮点工作区。初始行号容量按输入 nnz 和 n 估计，
final/checkpoint 行号与 chunk 元数据各自做 CAS 有界预留。
超额会设置设备错误并使 workers 退出，释放整次尝试的 buffer 后扩容重跑；
chunk 元数据可以独立增长。不会在已发布状态上替换池指针继续运行。
预算按选择设备当时空闲显存的 50% 控制，包含输入、位图、队列、元数据、arenas、
scan 临时数据和最终 packed 输出。分配失败优先减少 worker，初始池也可缩小。
`symbolic_peak_workspace_bytes` 是该分配预算的保守峰值估计，不是硬件实测显存水位。

GPU 接受无重复、行号合法、列指针单调且从 0 开始的 CSC；列内可以无序，原始
`ai/ax` 不会被重排。重复坐标和非法输入直接报错。现有数值代码使用 signed 列号，
因此要求 n 不超过 `INT_MAX`；CSC/pool 使用 32-bit unsigned。
queue capacity 上限为 `2^29`，ticket 在 `INT_MAX-capacity` 前停止并报错，
避免序号绕回/ABA；结构 nnz 超过 unsigned 容量会拒绝，不用可能溢出的 CPU 结果替代。
普通资源失败的日志含 `GPU symbolic fallback: ...` 与
`symbolic_cpu_fallback_count: 1`；这类结果不能当作 GPU 性能成功。

### 计时、访存代理与性能对照

`solver.log` 新增以下 SF 指标，`summary.csv` 共 25 列，含请求的 `symbolic_backend` 和 `numeric_sm_padding`：

| 指标 | 范围/含义 |
| --- | --- |
| `symbolic_cpu_ms` | 选择 CPU 或开启 verify 时的 CPU golden 时间 |
| `symbolic_gpu_total_ms` | 符号调用的输入验证、设备初始化、分配、H2D、worker、scan/pack、D2H、输出检查及提交；扣除可选 CPU golden/比较时间 |
| `symbolic_gpu_compute_ms` | GPU event 测得的 worker kernel 时间，不含初始化和 pack |
| `symbolic_gpu_pack_ms` | 输出分配、GPU scan、nnz 回读、narrow/pack 与同步的主机墙钟时间 |
| `symbolic_gpu_h2d_ms / symbolic_gpu_d2h_ms` | 输入上传与结果/统计下载墙钟时间；nnz 小量回读计入 pack |
| `symbolic_workers / symbolic_bitmap_bytes` | 最终实际 worker 数与两份 bitmap 总字节数 |
| `symbolic_input_nnz / symbolic_output_nnz / symbolic_fill_ratio` | 原结构、填充结构及比值；输入 nnz=0 时比值记 0 |
| `symbolic_checkpoint_pool_used / capacity` | 增量 checkpoint 行号条目数与容量 |
| `symbolic_checkpoint_chunks_used` | 非空差量 chunk 数 |
| `symbolic_final_pool_used / capacity` | 最终 arena 条目数与容量 |
| `symbolic_task_parks / symbolic_task_wakeups` | 成功挂起与唤醒次数 |
| `symbolic_pred_merges / symbolic_predecessor_rows_read` | 实际前驱合并数与连续读取的 L 行号数，后者为结构访存量代理 |
| `symbolic_duplicate_rows_eliminated` | bitmap OR 发现已存在的行号数，包括原输入中已存在的结构对角 |
| `symbolic_checkpoint_rows_restored / symbolic_input_rows_read` | 恢复时行号读取数、原始输入行号读取数 |
| `symbolic_empty_queue_polls / symbolic_closed_wait_races` | scheduler 空轮询与登记时发现前驱已关闭的次数 |
| `symbolic_oom_retries / symbolic_cpu_fallback_count` | arena/chunk 扩容及分配重试、是否 CPU fallback |

有重试时 compute、传输与结构访问计数包含失败尝试；arena used/capacity 是最终成功尝试。
指标输出、验证及调度均有开销，正式测量关闭 `--symbolic-verify/--progress`，
先用严格正确性测试单独验收。若 n=0，直接返回 `{0}` 列指针，不启动 GPU。

性能对照脚本依次运行 CPU、旧 GPU 图搜索与新 SF，统一使用 LL-SF 数值路径，
GPU 后端禁用 fallback。默认每种后端预热 1 次、测量 5 次，报告 median/P90，
保留每次完整求解器日志与解，不把不可用后端算作加速：

```sh
python3 src/benchmark_symbolic_sf.py src/matrix/add32_csr.mtx \
  --output-dir /tmp/glu-symbolic-comparison --workers 0 --repeats 5 --timeout 600
```

输出 `symbolic.csv`；公平的端到端比较使用各后端完整 `Symbolic time`，
另列 SF worker/pack 时间、worker 数、workspace、predecessor rows 和 parks。
数值残差目标未达标（退出码 2）仍保留已完成的符号时间，但行状态标记 incomplete。
目前没有真实 GPU 性能数据，不能声称新路径更快；后续应根据 empty polls、checkpoint
恢复量与前驱读取量判断 queue、bitmap 扫描/原子 OR 或 checkpoint 是否成为瓶颈。

### 本次检查结果与后续 GPU 验收

2026-10-26 在当前受限环境完成：

- HTHPCC 的 `MAIN`、`symbolic_gpu_test`、`symbolic_gpu_numeric_test` 编译及链接通过。
- `test-symbolic-host`：66,066 个穷举结构的 CPU 语义检查、输入验证和复用检查通过。
- `test-symbolic-sf-host`：直接执行生产 SF helpers 的 1,000 组随机并发调度通过；
  包含倒序派发、多 waiter、1/2/8/32/64 workers、动态前驱、checkpoint 去重、
  已合并前驱只读一次、最终 pack、池/chunk/queue/ticket 错误安全退出；
  64/128/256 个主机线程模拟的协作 bitmap/scan/export 屏障也通过。
- 批量脚本 10 项测试、性能脚本 2 项测试通过。
- `symbolic_gpu_test --sf` 返回 **77（SKIP）**：无可访问 GPU，回退与严格失败检查通过。
  因而 GPU 跨 CTA 原子发布、实际求解正确性、显存水位及性能尚未验收。
- 当前 GCC 安装缺少 ASan/UBSan/TSan 链接库，未运行 sanitizer。
  主机原子/屏障模拟不能替代目标 GPU 的 memory-scope 和 wave 验证。

设备恢复后，执行：

```sh
make -C src test-symbolic-sf-host test-symbolic-host DEBUG=0
# 先运行 tests/README.md 中的 sf_sync.cu 设备原子发布测试
make -C src test-symbolic-sf-gpu DEBUG=0
timeout -k 3s 600s ./src/symbolic_gpu_test --sf --stress
timeout -k 3s 600s ./src/symbolic_gpu_numeric_test --sf
python3 -m unittest discover -s tests -p test_run_batch.py -v
python3 -m unittest discover -s tests -p test_symbolic_benchmark.py -v
```

GPU 测试覆盖逐元素 CSC、CSR/values/levels、三种 CTA 大小、小池扩容/回退、
不同 pattern 连续复用、共享前驱、动态前驱、大 n 稀疏结构，以及请求 1024 workers
的超驻留 grid。`--stress` 做 1,000 组随机输入、每组两种 worker 配置。
带 numeric 的目标还比较 RL-level/LL-level/LL-SF 因子，并检查正向与转置三角求解。
退出 77 是未验收，不是通过；所有设备用例必须在 fallback 禁用时返回 0。

主要文件：`include/symbolic.h`（配置）、`src/symbolic_sf.cuh`（设备阶段）、
`src/symbolic_gpu.cu`（host 资源、计时和提交）、`src/lu_cmd.cpp`、
`src/run_batch.py`、`src/benchmark_symbolic_sf.py`、`src/Makefile`，
及 `tests/symbolic_gpu.cpp`、`tests/symbolic_sf_host.cpp` 和脚本回归测试。

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
`solve_CSR` 解出 double 校正量 `δx`（内部处理置换、缩放），再以 `long double`
保存解向量并执行 `x←x+δx`，避免小于 double 表示间距的校正量在更新时丢失。
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

- 停在 `symbolic fill-in`：查看 `auto` 选择结果；CPU 路径在主机准备，GPU 路径也可能在设备初始化、符号 worker 或 pack。
- 停在 `numeric value preparation`：在 CPU `predictLU`。
- 停在 `GPU numeric LU started`，没有出现 kernel 启动消息：检查设备初始化、上传和工作区分配。
- 停在 `LL-SF kernel launch; waiting for completion...`：kernel 启动/同步尚未返回，
  需要区分计算慢、等待不推进及设备错误；先按 [tests/README.md](../tests/README.md)
  运行小规模原子发布测试。
- 单 worker 完成而自动 worker 超时：重点排查跨 worker 发布/等待、调度及轮询开销，
  不能只靠增大 timeout 判定修复。

批量脚本也支持 `--sf-workers 1 --progress`。各阶段统计写入 25 列 CSV；
请求的 worker 数及实际数量可从命令参数和每个矩阵的日志确认。


## 符号分析接口一致性

`Symbolic_Matrix::fill_in(ai, ap)` 保留原 CPU 入口；三参数重载使用同一
`SymbolicConfig` 校验器，并要求所有后端输入 CSC 的坐标合法、无重复（允许列内无序）。
原始数组不被修改；Python 仍先经 NICSLU 预处理，再按既有顺序执行
`fill_in -> csr -> predictLU -> 按数值调度条件 leveling -> LUonDevice`。

`pyglu.splu`、`spsolve` 和原生 `GLUFactorization` 的新增参数均为 keyword-only，
现有 positional 参数顺序不变。C++ 仍使用 `(ai, ap, config)`，没有交换行号/列指针顺序。

| CLI / run_batch.py | Python keyword | SymbolicConfig 字段 | 默认值 |
| --- | --- | --- | --- |
| `--symbolic-backend` | `symbolic_backend` | `backend` | `auto` |
| `--symbolic-batch-cols` | `symbolic_batch_cols` | `batch_cols` | 128，正数，作用于旧 graph GPU |
| `--symbolic-block-size` | `symbolic_block_size` | `block_size` | 256；支持 64/128/256 |
| `--symbolic-verify` | `symbolic_verify` | `verify` | false |
| `--symbolic-no-fallback` | `symbolic_no_fallback` | `!fallback_to_cpu` | false，默认允许 CPU 回退 |
| `--symbolic-sf-workers` | `symbolic_sf_workers` | `sf_workers` | 0，自动 |
| `--symbolic-pool-entries` | `symbolic_pool_entries` | `pool_entries` | 0，自动 |
| `--symbolic-max-pool-entries` | `symbolic_max_pool_entries` | `max_pool_entries` | 0，显存预算/索引上限 |

数值 `--sf-workers` 对应 Python `sf_workers`，独立于符号 worker 配置；非零时要求
数值 `synchronization-free` 调度。符号 worker/pool 非零时要求 `auto` 或 `gpu-sf`。
所有计数参数范围为 unsigned 32-bit；worker/pool 显式传 0 与默认值等价。
校验和结构比对失败不会静默回退。GPU 提交与 CPU 重建共用派生缓存清理函数，
包括 CSR、val、level 和 debug 缓存；严格模式失败不改变上次已提交的结果及后端标识。

`lu.symbolic_backend` 返回实际产生结构的后端（发生回退时为 `cpu`），
`lu.diagnostics` 保留符号 compute/total/pack、显存和结构访问计数以及数值日志。
Python 对象持有日志流和 NICSLU/符号对象的 RAII 所有权；构造途中失败也会清理。
GPU 失败后回退的日志还区分 `symbolic_gpu_failed_ms` 和 `symbolic_cpu_ms`；
零阶 SF 在 C++ 入口返回 `{0}` 列指针并输出零 compute/资源计数，不初始化设备。

构建扩展会同时编译 `numeric.cu` 和 `symbolic_gpu.cu`，采用与 Makefile 一致的
HTHPCC 工具链、`__sync_threads=__syncthreads` 和 PIC。重新安装后可验证：

```python
lu = pyglu.splu(A, symbolic_backend="gpu-sf", symbolic_sf_workers=2,
                symbolic_verify=True, symbolic_no_fallback=True)
print(lu.symbolic_backend)
print(lu.diagnostics)
```

PEP 517 入口使用 `setuptools.build_meta`；扩展的源码/依赖使用相对路径，
`MANIFEST.in` 将 CUDA/HTHPCC 源文件、头文件、NICSLU 源码和 Makefile 纳入源码包。

benchmark 对缺失、负数、非有限的时间/计数及不同的矩阵/nnz 直接记为 incomplete；
CPU 基线或 GPU 样本不完整时不计算加速比。维度/nnz 比较不能替代逐元素 CSC 验证。

本轮接口补齐的验证（2026-10-26）：

- HTHPCC 编译/链接 `MAIN`、`symbolic_gpu_test`、`symbolic_gpu_numeric_test` 成功；
  符号 GPU 源文件另以 `-fPIC` 编译成功。
- `symbolic_sf_host_test` 通过 1,000 次调度模拟、64/128/256 host lanes 协作、
  增量 checkpoint、精确前驱读取数、池/队列/ticket 边界与安全退出检查。
- `symbolic_gpu_test --sf --host-only` 通过 66,066 个结构枚举用例及接口/复用检查；
  `GLU_DEBUG=1` 独立构建同样通过，含 debug 缓存清理。
- Python unittest 共 35 项，31 项通过、4 项因缺少原生扩展或未启用 GPU 而跳过。
  参数/构建命令测试使用 mocks，不表示 NumPy ABI 或 Python 原生模块构建验证成功。
- `symbolic_gpu_test --sf` 验证了不可用设备的 CPU 回退和严格失败，随后以 77 跳过
  GPU 执行；设备探针返回 status 3（initialization error）。当前 Python 为 3.7，
  缺少 NumPy/pybind11/Python 开发头文件，完整 Python 扩展安装尚未执行。

因此尚无 GPU compute/e2e、真实峰值显存和性能加速数据；设备并发、sanitizer
及 GPU 数值 LU/solve 验收仍需在可用设备和 Python >=3.9 环境执行上述测试。

## 带宽利用率检查
occupancy使用ht-smi采样，里面有gpu-util。每10ms采样一个数据，要求在循环求解阶段达到60%利用率。按时间加权。
ht-smi -l 100 -o tmp.csv --show-usage
