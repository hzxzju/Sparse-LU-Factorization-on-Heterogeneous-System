# 数值分解计算阶段的 ht-smi 利用率采样

在仓库根目录先重新编译求解器，再启动采样（脚本保留原名称）：

```sh
make -C src MAIN
bash src/profile_kernel_util.sh --gpu 3 --output util-run-01 -- \
  ./src/lu_cmd -i src/matrix/add32_csr.mtx -u left-looking -s synchronization-free
```

`--gpu` 是 `ht-smi` 的物理卡号，也可以使用 PCI BDF，例如
`--gpu 0000:d6:00.0`。单卡容器的 runtime 逻辑设备 0 可能对应主机物理 GPU 3，
不能据此把采样参数改成 0。采样开始前会用当前 runtime GPU 的 BDF 核对选中的卡。
`--output` 必须尚不存在；省略时自动在仓库根目录建立
`util_sampling/YYYYMMDD_HHMMSS/`，目录名使用本次启动的北京时间。

依赖 Bash、Python 3、C++11 编译器、HPCC 头文件和 `ht-smi`。
默认使用 `/opt/hpcc`，优先选择 `$GCC_TOOLCHAIN/bin/g++` 或
`$HOME/local/gcc-7.3.0/usr/bin/g++`；可以设置 `HPCC_PATH`、`CXX`、`HT_SMI`。

## 统计范围

采样窗口与 `src/numeric.cu` 中 `Numeric LU compute time (wall-clock)` 的
计时边界对应：临时内存初始化和扰动参数准备完成后、计算开始前启动；
数值分解的全部 GPU 工作通过原有同步完成后、结果拷回主机前结束。
边界回调的时间戳与计时器调用之间只有少量主机调用开销。

- 包含 RL、LL 分层循环或 LL-SF 计算，以及窗口内 CPU 调度、kernel 间空隙和原有同步等待。
- 排除符号分解、内存分配与初始化、主机/设备数据传输、CPU 三角求解和清理。
- 每次数值分解连续采样一次、输出一条汇总，不再统计每个 kernel。
- 不拦截 `hcLaunchKernel`，不插入逐 kernel event 或同步，不串行化原有多 stream 调度。

`include/numeric_util.h` 在计算边界提供可选回调；不使用采样启动器时这些回调为空，
求解器只需链接系统 `libdl`（构建脚本已配置）。启动器通过 `LD_PRELOAD` 加载回调实现，
`kernel_util_monitor.py` 在收到阶段开始消息后执行：

```sh
ht-smi -l 10 -o <pid_phase>/tmp.csv --show-usage
```

等到所选 GPU 的第一个有效样本且 BDF 核对通过后，允许计算开始。
采样准备超时为 5 秒。数值分解结束时先记录结束时间，再通知采样器停止、落盘和汇总。
采样进程启动等待、停止、落盘、统计和结果传输都在计算窗口外。
采样仍会产生边界通信及轮询开销，也可能影响设备状态。

旧求解器没有阶段回调，因此必须重新编译；Python 扩展也须重新构建。
没有收到任何数值分解阶段时返回 3，并提示重新编译。

## 时间加权

保留前值向右保持的策略，不向采样点左侧扩展。对观测 `(t_i, u_i)`：

```text
Δt_i = max(0, min(E, t_(i+1), t_i + 2 × interval) - max(S, t_i))
observed_util_pct = Σ(u_i × Δt_i) / ΣΔt_i
coverage_pct = 100 × ΣΔt_i / (E - S)
```

`[S,E)` 是整个数值分解计算窗口。当前请求采样周期为 10 ms，每个样本最多保持
20 ms；末尾样本最多保持到 E。周期统一由 `kernel_util_monitor.py` 的
`INTERVAL_MS` 控制，实际间隔以 CSV 时间戳为准。
开始前的最后一个样本可向右覆盖窗口起点；长时间缺样的部分不当作 0%。
例如 20% 保持 5 ms，80% 保持 15 ms，加权平均为 65%。

`numeric_util_pct` 直接输出有样本覆盖部分的加权平均值，不因覆盖不完整或窗口内
没有新样本而留空；开始前的样本仍可在保持时限内参与计算。有阶段错误、时钟跳变、
采样器错误或完全没有可计算的观测时仍留空。正常情况下它与 `observed_util_pct`
相同；`coverage_pct` 和 `partial_coverage` 仅作诊断，不限制输出。计算窗口小于 10 ms 时标记
`below_sampling_interval`；无窗口内样本时标记 `no_in_window_sample`；
只有一个样本时标记 `single_sample_estimate`。

CSV 按本机 `ht-smi 2.3.4` 格式解析。时间如
`2026/10/27 19:56:42.29031` 的小数段是未补零的整数微秒，即 `.029031` 秒。
更换工具版本后需核对格式。时间按目标与采样进程共有的主机本地时区解释；
只使用 `dieId=-` 的整卡行，不合并 die 指标。

## 输出与限制

每次数值分解保存 `<pid>_<phase_id>/tmp.csv` 和 `ht-smi.log`，
`summary.csv` 每个阶段一行：

- `pid`、`phase_id`、`phase=numeric_lu_compute`、`gpu`、`bdf`。
- `start_unix_ns`、`end_unix_ns`、`host_window_ms`。
- `samples_in_window`、`coverage_pct`、`observed_util_pct`、`numeric_util_pct`。
- `status`、`numeric_status`、`raw_csv`。

旧的 kernel 名称、grid/block、stream、`gpu_event_ms`、`kernel_util_pct` 等列已移除。
`numeric_status=0` 表示阶段完成；提前返回或运行时错误会使该值非零，利用率留空。
目标失败时保留其退出码；采样故障返回 125。

这是整个计算阶段的整卡 GPU util 时间加权估计，不是 SM/CU occupancy 或 FLOPS 效率。
同卡其他进程的活动会进入观测；硬件利用率指标本身的统计窗口可能跨越阶段边界。
10 ms 轮询不能保证硬件指标以相同周期刷新，短阶段仍可能得不到有效结果。
当前面向单 GPU、顺序执行的数值分解，不支持并发的多个分解阶段或 fork 后复用 runtime。
中断时只清理本次启动的目标进程组和采样进程。

不初始化 GPU 的模拟测试覆盖加权、阶段边界、异步提交、多 stream、原有同步、
错误和无采样器运行：

```sh
python3 -m unittest discover -s tests -p 'test_kernel_util.py' -v
```
