# 全 SM 空转五秒的 GPU-util 测试

默认在物理 GPU 3 对应的既有容器运行：

```sh
GPU_INDEX=3 bash run_idle_sm_util.sh
```

脚本编译独立程序 `src/gpu_idle_spin.cu`，不运行矩阵读取、符号分解或数值分解。
它复用 `profile_kernel_util.sh`：先持续采样 `ht-smi`（请求间隔 1 ms），通过
hcTracer/HCTX 校验时钟，再按 `SpinAllSms` 的真实 GPU 执行区间加权。

程序读取所选设备的 `multiProcessorCount`，以此作为 grid 大小。使用 cooperative
launch，动态共享内存预留超过每 SM 容量的一半，并检查 occupancy 必须为每 SM
一个 block；不支持时明确失败。这样全部 block 同时开始且保持驻留时，覆盖全部
runtime 可用 SM。不会把 SM 数写死为 104，也不会只用多个 block 的总数冒充 SM 数。

每个 block 启动后向映射的主机内存写一个独立确认标志。主机确认所有 block 都已启动后，
用 steady clock 计时五秒，再用系统作用域原子协议发布停止标志。同步结束后检查所有
block 都正常退出，最后结束采样阶段。五秒指全体 block 就绪后的空转保持时间；
初始化、编译和 hcTracer 收尾还需额外时间。GPU 另有宽松的设备时钟保护期限，
不是用 GPU 标称频率决定正常五秒时长。

默认 `leader` 模式与现有 LU 等待方式相近：每个 block 的线程 0 轮询停止标志，
其他线程等待 block barrier。循环包含标志读取、分支和保护计时检查，不执行矩阵或
浮点算术负载；空转本身仍需执行控制指令，不能理解为硬件完全不执行指令。
停止标志位于映射的主机内存，与 LU 的显存 `done` 标志访存路径不同，且循环包含
保护计时检查。因此读数是本空转模式的结果，不能直接作为 LU 等待循环的精确性能基线。
所有线程同时轮询可作为独立对照：

```sh
GPU_INDEX=3 bash run_idle_sm_util.sh --all-threads
```

短时调试可使用 `--seconds 0.2`；默认五秒，允许 0.01 至 60 秒。
`--device` 是容器内 runtime 逻辑设备号，默认 0；`GPU_INDEX` 是主机物理卡号，
通常不相同。单卡容器不应把主机 GPU 3 直接当作 runtime 设备 3。

主机直接构建、采样的示例（四张卡都可见时，逻辑设备 3 对应物理卡 3）：

```sh
make -C src gpu_idle_spin
bash src/profile_kernel_util.sh --gpu 3 --output spin-util-01 -- \
  ./src/gpu_idle_spin --device 3
```

不采样只运行空转程序：`./src/gpu_idle_spin --device 3`。

输出使用现有采样目录 `util_sampling/YYYYMMDD_HHMMSS/`。查看：

- `summary.csv`：`phase=gpu_idle_spin`，`gpu_util_pct` 为通过校验的加权 GPU-util。
  `numeric_util_pct` 对空转阶段留空；数值分解阶段仍保留该兼容列。
- `hcTracer.log`：设备/BDF、SM 数、block 数、occupancy、模式、实际保持秒数以及
  `started_blocks`、`finished_blocks`。两者应等于设备报告的 SM 数。
- `trace/numeric-<pid>.json`：原始 GPU 执行时间线，文件前缀保留现有启动器命名。
- `clock_alignment.json`、`*_active_intervals.csv`、`*_kernels.json`：对齐及窗口证据。
- `tmp.csv`：整次运行的连续原始采样。

`gpu_active_ms` 是完整 kernel 执行时间，包含短暂的启动/停止及保护计时开销。
1 ms 是请求周期，实际间隔看 CSV；超过两倍周期的缺口仍不外推，需结合
`coverage_pct` 查看。轮询全部 SM 不保证 GPU-util 达到 100%，该实验正是测量
这种驻留空转在本机硬件指标下的读数。同卡其他进程会影响整卡 GPU-util。

编译检查和现有采样回归：

```sh
make -C src gpu_idle_spin
./src/gpu_idle_spin --help
python3 -m unittest discover -s tests -p 'test_kernel_util.py' -v
```
