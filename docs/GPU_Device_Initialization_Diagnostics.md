# GPU 逐卡选择与初始化诊断

记录日期：2026-10-21。以下是当次现场状态，不代表 GPU 的永久健康结论。

## 当前选卡行为

`include/gpu_device.h` 提供 Symbolic 与 Numeric 共用的选择逻辑。首次按 runtime 返回的逻辑设备号 `0..count-1` 依次尝试，每张卡检查：

1. `hcSetDevice(id)`；
2. `hcGetDeviceProperties`，要求 64-thread Wave，并满足当前 kernel 的 block/grid 限额；
3. 申请 4 bytes，执行 `hcMemset`、`hcDeviceSynchronize` 并释放，用于发现延迟到资源初始化时才暴露的错误。

任一步**返回错误**则记录设备号、API、错误码及错误字符串，清理 probe 并继续下一张。全部失败会汇总原因；不会重置设备或终止其他进程。成功设备保存在 `Symbolic_Matrix::gpu_device`，Numeric 和后续 refactor 优先复用它；若它后来失败，再尝试其他卡。所有符号及数值输入仍由 host vector 持有，因此在数值 kernel 启动前可以切换设备。没有改动 Numeric 算术 kernel。

这不是运行中 kernel 的自动故障恢复：数值计算已经启动后的错误仍按原有路径返回，不会将可能部分更新的因子继续提交到下一张卡。

**边界：`hcGetDeviceCount` 必须先成功。** 如果它返回初始化错误，程序报告“枚举阶段失败”，不会猜测 sysfs/PCI 编号等于 runtime 的逻辑设备号。如果它在 SDK 内一直不返回，后续逐卡循环没有执行机会；不能安全地从另一个线程强行中断 GPU API 后继续复用该进程。

## 定位结果

### 受限执行环境：设备节点未暴露

当前受限执行环境中 `/dev/htcd` 和 `/dev/dri` 均不存在，`/dev` 被独立 tmpfs 覆盖；设备 cgroup 列表是 `a *:* rwm`。`ldd` 检查未发现缺失的 runtime 动态库。

在这个环境的独立进程中，`hcGetDeviceCount` 以及作为首次 runtime 调用的 `hcSetDevice(0/1/2/3)` 都返回 `status=3 (initialization error)`。因此，之前这里的 status 3 不能作为某张 GPU 硬件故障的证据：环境自身已缺少必要设备入口。

### 主机执行环境：枚举等待 GPU 2 的 CE 队列

在经过批准的沙箱外只读诊断中，`strace` 确认 runtime 成功打开 `/dev/htcd` 和 `/dev/dri/renderD128..131`，随后 `hcGetDeviceCount` 长时间反复打开：

```text
/sys/devices/virtual/htcd/htcd/layout/nodes/6/status
```

该节点对应 `ht-smi` 的 GPU 2、PCI `0000:c1:00.0`。状态快照：

```text
                 ori bitmap          used bitmap         free bitmap         hang bitmap(phy)
ce status        0x000000000000ffff  0x000000000000ffff  0x0000000000000000  0x000000000000ffff
```

可直接确认 CE 队列全部占用、无空闲，且驱动标记全部 hang。结合 SDK 在此处持续轮询，枚举阻塞与这张卡的队列状态相关。尚未确认造成该状态的根本原因：可能涉及残留上下文、任务挂起或驱动/设备问题，不能直接判断为硬件损坏。

其他只读观测：

- `ht-smi --show-unavailable-reason` 仍将 4 张卡全部标为 available；这个结果不足以证明计算队列可用。
- 沙箱外的 `ht-smi -i 2 --show-process` 未列出进程，不能据此确认驱动队列已被正确回收。
- GPU 2 的 RAS 可纠正/不可纠正错误计数及 PCIe fatal 计数在当次读取中为 0；这些计数不能排除队列或驱动异常。
- `HPCC_VISIBLE_DEVICES=1` 的限时测试仍超时。相应 `strace` 显示 SDK 虽改变检查顺序，仍检查 GPU 2 并在上述节点持续轮询。**在当次 SDK/现场状态下，该变量没有隔离这一步全局初始化检查，不能将其作为已验证的绕过方案。**

现场只做读取和小型初始化探测；已停止本次启动的阻塞诊断进程，没有 reset GPU、卸载驱动、修改 sysfs、清除故障状态或终止其他任务。进一步恢复异常队列需管理员/厂商确认影响范围后处理。

完整临时跟踪文件为 `/tmp/glu-gpu-init-host.strace` 和 `/tmp/glu-gpu-init-mask1.strace`；它们不属于仓库交付文件，可能被系统清理。上面保留了关键路径和状态，下面命令可重新采集。

## 构建、测试与复现

```sh
make -C src MAIN symbolic_gpu_test gpu_device_selection_test gpu_device_probe DEBUG=0 -j2
make -C src test-gpu-device-selection test-symbolic-host DEBUG=0
```

模拟 runtime 回归不需要 GPU，覆盖选卡、属性、分配、memset、同步及释放各阶段返回错误后的跳过/清理；同时检查设备复用、设备要求、全部失败、空设备列表以及枚举初始化错误。`--host-only` 的符号测试现在不会调用 `hcGetDeviceCount`，避免主机测试被故障卡的枚举等待卡住。

在有设备节点访问权限的普通 GPU 终端，分别运行独立进程；超时退出 124 表示调用未返回，不等于 API 返回的错误码：

```sh
timeout 15s ./src/gpu_device_probe --count
timeout 15s ./src/gpu_device_probe --set 0
timeout 15s ./src/gpu_device_probe --set 1
timeout 15s ./src/gpu_device_probe

/opt/htdriver/bin/ht-smi --show-unavailable-reason
cat /sys/class/htcd/htcd/layout/nodes/6/status

timeout 10s strace -f -e trace=openat,ioctl \
  -o /tmp/glu-gpu-init.strace ./src/gpu_device_probe --count
```

实际 GPU 逐卡切换与数值集成仍需在可正常完成枚举的环境验收；当前通过的是 HTHPCC 编译/链接、模拟错误处理及 CPU 符号回归，不能将这些结果视为真实故障卡恢复测试。
