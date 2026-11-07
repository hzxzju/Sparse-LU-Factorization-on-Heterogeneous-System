# GPU 活跃窗口的连续采样与事后加权

在仓库根目录执行：

```sh
make -C src MAIN
bash src/profile_kernel_util.sh --gpu 3 --output util-run-01 -- \
  ./src/lu_cmd -i src/matrix/add32_csr.mtx -u left-looking -s synchronization-free
```

`--gpu` 是 `ht-smi` 的物理卡号或 PCI BDF，例如 `0000:d6:00.0`。
它不一定等于 runtime 的逻辑设备号；每个数值分解阶段仍通过 runtime BDF
核对物理卡，选错时直接报错。`--output` 必须尚不存在；省略时自动建立
`util_sampling/YYYYMMDD_HHMMSS/`，目录名使用北京时间。

依赖 Bash、Python 3、C++11 编译器、HPCC 头文件、`ht-smi`、`hcTracer`
以及 `libhcToolsExt.so`。默认 SDK 路径为 `/opt/hpcc`；可设置 `HPCC_PATH`、
`CXX`、`GCC_TOOLCHAIN`、`HT_SMI`、`HC_TRACER`。已包含阶段回调的求解器
无需因本次采样修改重新编译；更早版本仍须重新构建，Python 扩展同理。

请求周期统一由 `src/kernel_util_monitor.py` 的 `INTERVAL_MS = 1` 控制。
修改该值时，加权保持上限（两倍周期）和 `below_sampling_interval` 判定自动同步；
无需修改 `run_sequentially_util.sh` 或重新编译求解器。

## 采集顺序与统计窗口

1. 在启动目标程序之前启动一次 `ht-smi -i <物理卡号或BDF> -l 1 -o tmp.csv --show-usage`。
   默认请求每 1 ms 采样一次，只查询所选卡，减少轮询其他卡的开销。
   等所选物理卡出现有效样本，并校验样本时间与主机时间一致后，才允许目标启动。
2. 使用 `hcTracer --hctx` 执行目标命令；同一个 `ht-smi` 进程持续覆盖初始化、
   符号分解、全部数值分解阶段和程序收尾，不在阶段之间重启。
3. 每次数值分解的原有阶段回调检查 BDF，并在首尾发出独立命名的 HCTX 标记。
   它们限定阶段范围、提供时钟锚点，不作为最终利用率窗口。
4. 目标命令及 hcTracer 收尾完成后停止 `ht-smi`，读取落盘的 GPU trace，
   完成时钟校验、目标 kernel 筛选、执行区间合并和时间加权。

最终窗口定义为数值分解阶段内目标进程的 **GPU kernel 实际执行区间的并集**：

```text
W_active = union([kernel_i.GPU_start, kernel_i.GPU_end))
```

- 使用 GPU 执行记录的 `ts/dur`，不使用 CPU 上 `hcLaunchKernel` 的调用时间。
- 通过原进程 PID、`co_id` 和阶段范围关联 CPU 调用与 GPU 执行记录；
  GPU 记录必须包含 `device_id`、`grid`、`block` 等执行属性。
- 多 stream 重叠区间合并，只计算一次；分离区间之间的空隙不计入活跃时间。
- 初始化、符号分解、拷贝、memset、CPU 调度、同步返回延迟和清理不作为活跃窗口。
- 普通 LL-SF 路径的窗口通常只有一个 kernel 区间；LL/RL 分层路径可有多个区间。
- 不拦截 launch，不插入逐 kernel event 或额外 GPU 同步；保留原有异步和多 stream 调度。

`host_window_ms` 仍保留原来完整数值分解计算阶段的墙钟时间，供比较。
`gpu_span_ms` 是第一个目标 kernel 开始到最后一个结束的跨度，包含中间空隙；
**加权分母使用 `gpu_active_ms`，不是这两个时间。**

## 时钟校验与对齐

本机 hcTracer 3.8.0.0 的真实 JSON 已验证：`ts` 和 `dur` 都使用整数纳秒，
不能按普通 Chrome trace 的微秒惯例直接转换。读取时避免把 Unix 纳秒转成浮点数。

阶段首尾各调用一次 `hctxMarkA`，并在调用前后记录 `CLOCK_REALTIME`。
标记名称为 `GLU_NUMERIC_UTIL:<pid>:<phase_id>:begin/end`。若标记在 trace 中
的时间是 `T`，主机调用前后时间是 `B/A`，则时钟转换偏移必须满足：

```text
B - T <= offset_ns <= A - T
host_unix_ns = trace_ns + offset_ns
```

取首尾两组偏移范围的交集；交集包含 0 时使用 0，否则使用距 0 最近的可行偏移。
交集为空、标记丢失/重复、时间倒退或单次标记括号超过 1 ms 时，拒绝该阶段的利用率。
`clock_uncertainty_ns` 记录所选偏移相对可行范围端点的最大误差。
另比较阶段和整次运行的 realtime/monotonic 耗时，差值超过 1 ms 时拒绝输出有效利用率。

`ht-smi` 时间按目标与采样器共有的主机本地时区解释。采样器启动后第一个样本必须
落在主机采样启动时间到准备完成时间之间（上界允许 1 ms），结束时再次检查范围；
采样时间倒退也会报错。这样同时检查 `ht-smi` 的时区/时间格式以及 trace 的时钟域。

沿用本机 `ht-smi 2.3.4` 的格式：`.29031` 是未补零的整数微秒，表示
`.029031` 秒。只使用 `dieId=-` 的整卡行；更换工具版本需重新核对格式。

时钟证据保存在 `clock_alignment.json`，包含样本启动检查、主机时区、运行时钟差值、
首尾锚点、偏移、误差界和每阶段校验状态。校验失败时不会退回 launch 时间或阶段墙钟窗口。

## 时间加权

保留前值向右保持策略，每个样本最多保持两倍请求周期。对观测 `(t_i, u_i)`：

```text
I_i = [t_i, min(t_(i+1), t_i + 2 ms))
Δt_i = duration(I_i intersect W_active)
observed_util_pct = sum(u_i * Δt_i) / sum(Δt_i)
coverage_pct = 100 * sum(Δt_i) / duration(W_active)
```

最后一个样本最多保持到最后一个活跃区间结束。不向采样点左侧扩展，不把缺样部分
当成 0%；长时间缺样不跨越保持上限。活跃窗口之前的样本可在保持时限内参与计算。
`numeric_util_pct` 在无运行、时钟或采样错误且存在覆盖观测时等于上述加权估计；
没有可用覆盖时留空。覆盖不完整仍保留估计，并输出诊断标记。

`below_sampling_interval` 表示活跃总时长小于 1 ms；`no_in_window_sample`
表示没有样本时间点落在活跃并集内，此时仍可能有之前的样本向右覆盖；
`single_sample_estimate` 表示并集内只有一个采样点；`partial_coverage` 表示覆盖不足；
`no_covered_observation` 表示完全没有可计算的覆盖；`no_gpu_intervals` 表示没有目标
GPU 执行区间。成功的 launch 缺失对应 GPU 执行记录会报错。

## 输出与退出码

每次运行只生成一份连续采样原始文件，`summary.csv` 仍然每次数值分解一行：

```text
<output>/
  tmp.csv                         # 整次运行的 ht-smi 原始采样
  ht-smi.log
  hcTracer.log                    # hcTracer 与目标命令的输出
  hc-config/                      # 本次运行独立的 hcTracer 配置
  trace/numeric-<pid>.json         # 可直接交给 Viewer 的 GPU/CPU/HCTX 时间线
  phases.jsonl                    # 原有计算阶段范围与时钟括号
  target_status.json              # 实际目标命令的退出码
  clock_alignment.json
  <pid>_<phase>_active_intervals.csv  # 合并后的活跃窗口，Unix 纳秒
  <pid>_<phase>_kernels.json          # 选中的 kernel、设备、stream/queue 与关联 ID
  summary.csv
```

主要汇总列：

- `pid`、`phase_id`、`phase=numeric_lu_compute`、`gpu`、`bdf`。
- `window_basis=gpu_kernel_execution_union`。
- `start_unix_ns`、`end_unix_ns`、`host_window_ms`：原有计算阶段。
- `gpu_start_unix_ns`、`gpu_end_unix_ns`、`gpu_active_ms`、`gpu_span_ms`。
- `kernel_count`、`active_interval_count`、`samples_in_window`、`coverage_pct`。
- `observed_util_pct`、`gpu_util_pct`、`numeric_util_pct`、`clock_status`、`clock_offset_ns`、`clock_uncertainty_ns`。
- `status`、`numeric_status`、`raw_csv`、`trace_json`、`active_intervals_csv`。

`raw_csv` 现在统一指向运行根目录的 `tmp.csv`，不再每阶段新建采样目录。
目标命令通过小型 runner 启动并记录退出码，因为 hcTracer 本身不会可靠透传目标退出码；
无采样故障时保留目标退出码。没有数值分解阶段且目标成功时返回 3；采样或 trace/时钟
故障返回 125；中断返回 130。中断只清理本次启动的目标/tracer 进程组与采样进程。

hcTracer 的本机 gRPC 通信需要本地 socket 权限。启动器自动把本机地址追加到
`no_grpc_proxy`，避免环境 HTTP 代理导致 RPC 连接超时。工具自身仍可能维护主目录中的
端口登记文件，需确保正常运行环境的主目录可写。采集日志可在运行时查看：

```sh
tail -f util-run-01/hcTracer.log
```

## 限制与验证

这是 **GPU 活跃区间内的整卡 util 时间加权估计**，不是 SM/CU occupancy 或 FLOPS
效率。同卡其他进程会进入观测。1 ms 是请求轮询周期，实际间隔以 CSV 时间戳为准，
硬件指标的刷新和统计周期也不保证为 1 ms。若实际间隔超过 2 ms，保持上限会使
`coverage_pct` 降低，不会把缺失时间补成 0%。按 trace 裁剪无法消除硬件指标自身跨越
kernel 边界的影响，微秒级或短毫秒级 kernel 可能完全没有可用样本。

hcTracer 采集和首尾标记会增加开销，因此常规求解耗时比较仍应使用不带 trace 的运行。
当前支持单 GPU、顺序的数值分解阶段，包括批量命令的顺序子进程；不支持阶段并发、
单阶段多 GPU 或 fork 后复用已初始化的 GPU runtime。

不初始化 GPU 的测试覆盖连续采样生命周期、异步提交、多 stream 重叠合并、空隙排除、
加权保持上限、时钟错误拒绝、物理卡核对、目标退出码和无采样器运行：

```sh
python3 -m unittest discover -s tests -p 'test_kernel_util.py' -v
```

测试需要本地 Unix socket 权限，使用模拟 ht-smi、hcTracer 和 runtime。

SM 空转实验已移除；采样仅用于实际数值 LU 工作。
