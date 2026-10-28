# 逐 kernel 的 ht-smi 利用率采样

在仓库根目录运行（`--gpu` 必须与程序实际使用的物理 GPU 对应）：

```sh
bash src/profile_kernel_util.sh --gpu 3 --output util-run-01 -- \
  ./src/lu_cmd -i src/matrix/add32_csr.mtx -u left-looking -s synchronization-free
```

也可以使用 PCI BDF，例如 `--gpu 0000:d6:00.0`。`--output` 必须是尚不存在的目录；
省略时自动建立 `kernel-util.XXXXXX`。运行环境必须能正常执行 GPU 程序和 `ht-smi`。
单卡容器内 runtime 的设备 0 可能对应主机的物理 GPU 3，不能直接把逻辑序号填入
`--gpu`。拦截器用 `hcGetDevice`/`hcDeviceGetPCIBusId` 获取当前卡 PCI 地址，
与选中采样行的 BDF 核对；不匹配时停止分析，防止统计错卡。

依赖 Bash、Python 3、C++11 编译器和 HPCC 头文件。默认使用 `/opt/hpcc`，
优先选择 `$GCC_TOOLCHAIN/bin/g++` 或本机 `$HOME/local/gcc-7.3.0/usr/bin/g++`；
可用 `HPCC_PATH`、`CXX`、`HT_SMI`（可执行文件路径）覆盖。
脚本不会修改 runtime 库和求解器源码，也不需要重新编译求解器。

## 工作流程

普通 shell 无法观察进程内部的异步 kernel 起止，因此 `.sh` 启动器编译并通过
`LD_PRELOAD` 加载 `kernel_util_hook.cpp`，由 `kernel_util_monitor.py` 管理采样进程。
本项目 `numeric.o`、`symbolic_gpu.o` 中的 `<<<...>>>` 调用均引用 `hcLaunchKernel`。

每次拦截的操作顺序：

1. 持有进程内互斥锁，并调用 `hcDeviceSynchronize`，排空当前 context 先前的任务。
2. 开启独立采样进程：`ht-smi -l 100 -o <pid_launch>/tmp.csv --show-usage`。
   等到指定 GPU 的首个有效样本写入 CSV，并核对 PCI 地址，再允许 launch；准备超时为 5 秒。
3. 在原 stream 记录起始 event，记录主机 realtime 和 monotonic 时间，调用原始 `hcLaunchKernel`。
4. 在相同 stream 记录结束 event，并 `hcEventSynchronize` 等待真实执行完成。
   记录结束主机时间和 GPU event 间隔。
5. 向采样进程发 SIGINT，等待文件落盘；2 秒内不退出则终止该采样进程。
6. 解析 CSV、立即打印并刷新 `summary.csv`，再允许下一次 launch。

采样启动等待、采样关闭和计算时间不计入 `host_window_ms`。
`host_window_ms` 是从调用 launch 前到结束 event 同步返回的主机包围窗口，
包含提交和同步返回的延迟；它并非 GPU 内核的精确活动区间。
`gpu_event_ms` 是同 stream 的 event 时间间隔，供对照，不把它与主机时间戳强行对齐。
停止采样、落盘和计算存在实际延迟，“立即”指结束通知到达后立刻执行这些步骤，
不会等整个求解结束才统计。

## 时间加权

对选中 GPU 的观测 `(t_i, u_i)`，采用前值保持：`u_i` 代表
`[t_i, t_(i+1))`，并与主机窗口 `[S,E)` 相交。
最后一个值最多延续到 `E`；任何值最多保持 200 ms，避免采样中断时无限外推。

```text
Δt_i = max(0, min(E, t_(i+1), t_i + 200 ms) - max(S, t_i))
observed_util_pct = Σ(u_i × Δt_i) / ΣΔt_i
coverage_pct = 100 × ΣΔt_i / (E - S)
```

启动前的最后一个样本可以覆盖窗口左边界，但窗口内完全没有新样本时，
`kernel_util_pct` 留空。覆盖不足、时钟跳变、runtime 错误或采样进程异常时也留空；
`observed_util_pct` 保留有覆盖部分的数值，便于排查。
例如 20% 保持 100 ms，80% 保持 200 ms，加权平均是 60%。

CSV 使用本机 `ht-smi 2.3.4` 实测格式：
`timestamp,deviceId,dieId,deviceName,bdfId,utilization.GPU [%],...`。
时间如 `2026/10/27 19:56:42.29031` 的最后一段是未补零的整数微秒，
对应 `.029031` 秒。解析器按整数微秒处理；更换工具版本后应核对格式。
CSV 时间按采样进程和目标进程共有的主机本地时区解释。只使用 `dieId=-`
的整卡行，不擅自合并 die 指标。

## 输出及边界

每次 launch 保存 `<pid>_<launch_id>/tmp.csv` 和 `ht-smi.log`，汇总在 `summary.csv`。
汇总包含 kernel 名称或入口地址、grid/block、stream、PCI 地址、起止时间、
主机窗口时间、GPU event 时间、窗口内样本数、覆盖率、加权利用率、错误码和状态。
动态符号表没有 kernel 名称时显示地址；可在构建程序时添加 `-rdynamic` 改善名称解析。

- 这是 **整卡 GPU util 的时间加权估计**，不等于 SM/CU occupancy、显存占用率或
  此 kernel 独占的资源比例。同卡其他进程的活动也会进入观测。
- 100 ms 是请求的轮询周期，实际间隔以 CSV 为准；硬件利用率指标自身的统计窗口
  可能跨越 kernel 边界，单靠 ht-smi 无法恢复其瞬时利用率。
- 小于 100 ms 的 kernel 标记 `below_sampling_interval`；零个窗口内样本时报告
  `no_in_window_sample` 和 N/A；一个样本时明确标记 `single_sample_estimate`。
  即便有数值也不能把短 kernel 的结果当成精确利用率。
- 该模式会串行化 kernel，并插入 event、同步、进程启动和 IPC，影响缓存、功耗状态
  与并发调度。不要将本次运行时间作为原程序性能数据。
- 当前拦截范围为动态链接的 host `hcLaunchKernel` 调用。没有承诺覆盖 graph replay、
  module/batch/cooperative launch、设备端动态 launch、静态链接或绕过动态符号绑定的库。
  未拦截任何 launch 时返回 3；目标程序失败时保留其非零退出码；采样故障返回 125。
- 目标是单个 GPU 程序，可有多 CPU 线程；不支持 fork 后复用 runtime、多卡混用、
  被 kernel 依赖的其他 CPU 线程还必须提交 GPU 工作等运行方式。
- 中断时清理本次启动的目标进程组和采样进程，不使用 `pkill ht-smi`。

可用不依赖 GPU 的模拟 runtime 验证流程和数学计算：

```sh
python3 -m unittest discover -s tests -p 'test_kernel_util.py' -v
```
