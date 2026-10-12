# hcTracer-Viewer-win32-x64 使用说明

这个文件夹是 **Windows 64 位的 hcTracer 时间线查看器**。把整个
`hcTracer-Viewer-win32-x64/` 文件夹复制到 Windows，双击其中的
`hcTracer-Viewer.exe` 即可启动，无需安装。

完整流程是：**Linux GPU 服务器采集 → 将生成的 JSON 复制到 Windows → Viewer 打开分析**。

本说明依据 [hcTracer 使用手册 V05](X系列_通用GPU_hcTracer使用手册_CN_V05.pdf)：
第 1 章介绍采集的数据类型；第 2.2 节介绍采集命令和输出；第 3.1、3.2 节及图 3.1
介绍 Windows Viewer。手册正文第 1–4 页对应 PDF 第 5–8 页。
下面的 LU 示例结合本仓库源码，快捷操作核对了当前 Viewer 包内的帮助代码。

## 1. 先区分两个工具

| 工具 | 在哪里运行 | 用途 |
| --- | --- | --- |
| `/opt/hpcc/bin/hcTracer` | 安装 HTHPCC 的 Linux GPU 服务器 | 启动或附加目标程序，采集活动事件，生成 JSON |
| `hcTracer-Viewer-win32-x64/hcTracer-Viewer.exe` | Windows 64 位电脑 | 打开采集结果，查看 CPU API 与 GPU 核函数时间线 |

查看器是绿色软件。复制时保留整个目录，包括 `resources/`、`locales/`、DLL 和
其他配套文件；其中 `resources/app.asar.unpacked/trace_processor_shell.exe` 也是配套文件。
在当前 Linux 终端中，采集使用 `hcTracer`；Windows 的 `.exe` 在 Windows 上启动。

当前工作环境已确认存在 `/opt/hpcc/bin/hcTracer` 和 `/opt/hpcc/lib/libhcToolsExt.so`。
后者是手册说明的采集依赖。实际采集工具的 `--help` 显示版本为
`3.5.3.18-ef9e10e`，仓库中的 Viewer 压缩包文件名标注 `3.5.3.27`；两者的兼容性尚未实测。

## 2. Linux 上采集本项目的 LU 程序

以下命令均在**仓库根目录**执行。先按照 [BUILD_AND_RUN.md](../BUILD_AND_RUN.md)
完成构建，确认相同参数在普通运行时能正常结束。
如果刚修改过源码，先重新构建，确保 `src/lu_cmd` 包含所需的策略和调度参数。

```bash
export HPCC_PATH=/opt/hpcc  # SDK 安装在其他位置时修改这里
export LD_LIBRARY_PATH="$HPCC_PATH/lib:${LD_LIBRARY_PATH:-}"

# 检查采集工具支持的选项；本机版本打印帮助后会以非零状态退出
"$HPCC_PATH/bin/hcTracer" --help

# 普通运行：先确认程序及矩阵路径有效
./src/lu_cmd -i src/nicslu/demo/ASIC_100k.mtx
```

最简单的采集命令：

```bash
"$HPCC_PATH/bin/hcTracer" ./src/lu_cmd -i src/nicslu/demo/ASIC_100k.mtx
```

正常结束后，当前目录会出现：

```text
tracer_out_<timestamp>/
└── tracer_out-<pid>.json
```

`<timestamp>` 是实际时间戳，`<pid>` 是被追踪程序的进程号。
后续在 Viewer 中打开这个 `.json` 文件。
`x.dat` 是 LU 程序生成的解向量，供求解结果使用。

### 分别采集 RL、LL 和同步自由 LL

为了方便比较，可以为三次运行指定不同的输出目录和文件名前缀：

```bash
# 以下三次运行使用同一个矩阵；可替换为自己的矩阵路径
TRACE_TAG=$(date +%Y%m%d_%H%M%S)

# Right-looking，按 level 调度
"$HPCC_PATH/bin/hcTracer" --odname "tracer_rl_$TRACE_TAG" --name rl \
  ./src/lu_cmd -i src/nicslu/demo/ASIC_100k.mtx \
  -u right-looking -s level

# Left-looking，按 level 调度
"$HPCC_PATH/bin/hcTracer" --odname "tracer_ll_$TRACE_TAG" --name ll \
  ./src/lu_cmd -i src/nicslu/demo/ASIC_100k.mtx \
  -u left-looking -s level

# Left-looking，同步自由调度；此调度要求使用 left-looking
"$HPCC_PATH/bin/hcTracer" --odname "tracer_ll_sf_$TRACE_TAG" --name ll_sf \
  ./src/lu_cmd -i src/nicslu/demo/ASIC_100k.mtx \
  -u left-looking -s synchronization-free

# 列出本次生成的 JSON
find "tracer_rl_$TRACE_TAG" "tracer_ll_$TRACE_TAG" "tracer_ll_sf_$TRACE_TAG" \
  -type f -name '*.json' -print
```

`--odname` 指定输出文件夹，`--name` 指定文件名前缀；以实际生成的 JSON 文件名为准。
若比较时需要 `-p` 静态对角扰动，三次运行均添加相同的 `-p`，保持配置一致。
同步自由路径的 SDK 编译与设备验证要求见 [tests/README.md](../tests/README.md)。

## 3. 将 JSON 复制到 Windows 并打开

1. 使用 IDE 的远程文件下载、SFTP 或 WinSCP，将采集目录里的 `.json` 下载到 Windows。
   可以放在例如 `D:\hcTracer-data\`；数据文件可以放在 Viewer 程序目录之外。
2. 将完整的 `hcTracer-Viewer-win32-x64/` 目录放到 Windows 本地磁盘。
   如果拿到的是 `.tar.gz` 压缩包，先完整解压到能看到 `.exe` 的目录。
3. 双击 `hcTracer-Viewer.exe`。
4. 在左侧 **Navigation → Open trace file** 中选择刚下载的 `.json`。
   如果左侧栏收起，点击左上角菜单按钮展开。
5. 等待文件解析完成，展开 CPU、GPU 和 Stream 轨道查看时间线。
   可以通过左侧 **Show timeline** 回到时间线页面。

## 4. 时间线怎么看

以下位置对应手册图 3.1。

| 界面位置 | 表示什么 | 在本项目中关注什么 |
| --- | --- | --- |
| CPU / TID | CPU 线程调用的 HTHPCC Runtime API | `hcLaunchKernel`、`hcMemcpy`、`hcDeviceSynchronize` 等调用的时间与频率 |
| GPU / Stream | GPU 上实际执行的核函数，按 Stream 展示 | 核函数耗时、执行间隙、不同 Stream 是否重叠 |
| Current Selection / Details | 选中事件的详细信息 | `Name`、`Start time`、`Duration`，以及可用的参数信息 |
| 顶部 Search | 搜索事件名称 | 搜索核函数名或 API 名，定位相关事件 |

点击一个彩色事件块后，在下方 **Current Selection** 中查看 `Duration`。
评估某个核函数的设备执行时间时，选择 **GPU Stream 上的核函数事件**。
CPU 上的 `hcLaunchKernel` 耗时描述主机端调用过程；GPU 事件描述设备执行过程。
`hcDeviceSynchronize` 的 CPU 时间可能包含等待 GPU 完成的时间。

当前源码中的主要核函数如下，Viewer 里的名字可能还带有参数签名：

| 运行配置 | 可以搜索的核函数名 | 可以观察的现象 |
| --- | --- | --- |
| RL + level | `RL`、`RL_perturb`、`RL_onecol_factorizeCurrentCol`、`RL_onecol_updateSubmat`、`RL_onecol_cleartmpMem`；开启扰动时还有 `RL_onecol_factorizeCurrentCol_perturb` | 不同 level 的调用批次；小 level 路径在多个 Stream 上启动核函数 |
| LL + level | `LL_factorizeColumns` | 每个 level 可能分成多个批次，源码在每个 level 结束后调用 `hcDeviceSynchronize` |
| LL + synchronization-free | `LL_factorizeSyncFree` | 当前源码的一次数值分解启动一次该核函数，随后等待完成；核函数内部处理列任务 |

轨道上的间隙和调用分布可以帮助定位进一步分析的区间。
单凭时间线无法确定内存带宽、缓存命中率或核函数内部依赖等待的具体原因。
本项目输出的 `Total GPU time` 包含设置和数据传输等阶段，比较时同时查看
GPU 核函数事件、传输事件和整个数值分解区间。

### 常用操作

这些操作已核对当前 Viewer 包内的快捷键帮助代码；先将焦点放到时间线区域。

| 操作 | 功能 |
| --- | --- |
| 单击事件块 | 查看该事件的详情 |
| `W` / `S` | 放大 / 缩小时间线 |
| `A` / `D` | 向左 / 向右平移 |
| `Ctrl` + 鼠标滚轮 | 放大 / 缩小 |
| `Shift` + 鼠标拖动 | 水平平移 |
| 鼠标拖动 | 选择时间区域 |
| 选中事件后按 `F` | 定位并缩放到所选事件 |
| `Q` | 恢复初始时间线范围 |
| `?` | 查看快捷键帮助 |

## 5. 其他采集选项

手册第 2.2 节还给出了以下选项：

| 选项 | 用途 |
| --- | --- |
| `--debug` | 打印 hcTracer 自身的调试信息，用于排查采集问题 |
| `--hctx` | 追踪程序通过 HCTX API 标记的自定义区间 |
| `--attach <pid>` | 附加到正在运行的目标进程，采集一个运行区间 |

当前 LU 源码未发现 HCTX 标记调用。添加 `--hctx` 后，要由目标程序调用
相应 HCTX API 才会产生自定义区间；具体 API 用法需查对应 SDK 文档。

attach 模式使用两个 Linux 终端：一个运行目标程序，另一个运行采集工具。

```bash
# 第二个终端：将 12345 替换为目标程序的真实 PID
"$HPCC_PATH/bin/hcTracer" --attach 12345
```

在 **hcTracer 所在终端**按 `Ctrl+T` 结束本次 attach 采集。手册说明支持多次 attach。
对较快结束的 `lu_cmd`，优先使用前面的直接启动采集方式，以便覆盖完整运行过程。

## 6. 常见问题

| 现象 | 检查方法 |
| --- | --- |
| Linux 提示 `hcTracer: command not found` | 使用完整路径 `$HPCC_PATH/bin/hcTracer`，或将 `$HPCC_PATH/bin` 加入 `PATH` |
| 提示缺少 `libhcToolsExt.so` | 检查 SDK 安装及 `$HPCC_PATH/lib`，并设置前面给出的 `LD_LIBRARY_PATH` |
| 找不到采集结果 | 检查启动命令的当前目录、`--odname` 指定的目录和采集日志；确认目标程序正常结束 |
| Viewer 提示缺少 DLL 或配套文件 | 重新完整解压或复制 Viewer 文件夹，保留 `resources/` 等目录 |
| 时间线中未看到 GPU 核函数 | 检查 GPU / Stream 轨道是否展开、缩放范围，以及目标程序是否成功执行 GPU 路径；结合采集日志排查 |
| 最新调度选项无法使用 | 按构建说明重新编译 `src/lu_cmd`，核对当前可执行文件的帮助输出 |
| JSON 无法导入 | 确认复制完整、文件来自正常结束的采集，并核对采集端和 Viewer 的版本；当前版本组合尚未实测 |

本说明已核对本地手册、Viewer 包内的界面与快捷键代码，以及采集工具的帮助输出。
尚未实际运行 Windows Viewer，也未执行上述 LU 性能采集。
