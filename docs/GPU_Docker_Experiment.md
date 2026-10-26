# X201 单卡 Docker 实验环境

2026-10-24 已在本机 rootless Docker 中启动 `glu-gpu1`，使用已有镜像
`publiccase:arm64`（镜像 ID `f886f65722c5`），保留主机 GPU 1：

| 主机卡号 | PCI 地址 | render 节点 | 容器内计算设备号 |
| --- | --- | --- | --- |
| 1 | `0000:ab:00.0` | `/dev/dri/renderD129` | 0 |

只挂载 `/dev/htcd` 和 `/dev/dri/renderD129`，未挂载其他三个 render 节点、
整个 `/dev` 或 Docker socket，未启用 privileged。容器默认无网络。
主机项目目录映射到 `/workspace`，源码和实验结果会直接保存在主机仓库；
SDK、驱动用户态库和 GCC 7.3 工具链只读挂载。容器停止后文件仍保留。

## 当前验收结果

- `hcGetDeviceCount` 成功，返回 **1**；容器中 `/dev/dri` 只有 `renderD129`。
- `make -C src MAIN gpu_device_probe DEBUG=0 -j2` 编译/链接通过。
- **GPU 计算尚不可用**：GPU 1 的小型初始化探测停在队列创建，日志为
  `htkwCreateQueueBlock ... timeout, gpu_id:63712 type:21`，限时探测结束。
- 对 GPU 0、2、3 分别使用只挂载该卡的临时容器做同样的小型探测，
  均在队列创建处超时（20 秒超时，退出码 124）。临时容器已自动删除。
  四张卡的现场 CE `hang bitmap` 均非零。

因此，单卡可见性和编译环境已经验证，但未通过 GPU 初始化及实际求解验收。
隔离其他卡不能修复被保留卡的异常队列；尚需管理员/厂商恢复至少一张卡，
恢复前不建议启动批量实验。本次没有 reset GPU、卸载驱动、修改 sysfs 或终止其他用户任务。

这里验证的是正常实验程序的设备可见性隔离。主机 rootless Docker 的
`CgroupDriver=none`，未提供设备 cgroup 强制隔离；`/dev/htcd` 是共享控制节点，
sysfs/管理工具仍可能显示整机拓扑。这不是针对恶意程序的独占或安全隔离，
也不会阻止其他主机用户使用同一张卡。

## 进入和运行

在主机仓库根目录执行：

```sh
bash src/gpu_docker.sh status
bash src/gpu_docker.sh shell
```

进入后工作目录为 `/workspace`。管理员恢复目标卡后，先验证初始化：

```sh
# 在主机执行；有超时保护，失败时返回非零
bash src/gpu_docker.sh probe
```

只有 probe 显示 `device 0 ready` 后，再在容器内编译、运行一个矩阵并确认
残差和退出码通过，最后启动批量实验：

```sh
make -C src MAIN
timeout -k 3s 60s ./src/lu_cmd -i src/matrix/add32_csr.mtx \
  -u left-looking -s synchronization-free
python3 src/run_batch.py --update-strategy left-looking \
  --scheduling synchronization-free --timeout 600
```

批量结果仍在主机 `src/batch_results/`。容器内逻辑 GPU 0 就是主机 GPU 1，
无需设置 `HPCC_VISIBLE_DEVICES=1`。

也可以在主机直接将命令交给容器：

```sh
bash src/gpu_docker.sh exec make -C src MAIN
bash src/gpu_docker.sh exec python3 src/run_batch.py \
  --update-strategy left-looking --scheduling synchronization-free --timeout 600
```

退出交互 shell 用 `exit`，容器继续运行。停止及再次启动：

```sh
bash src/gpu_docker.sh stop
bash src/gpu_docker.sh start
```

脚本默认明确使用 Docker 的 `rootless` context。在其他环境需用
`GLU_DOCKER_CONTEXT` 指定 context，并核对设备映射。本机已核实主机 GPU
0/1/2/3 分别对应 `renderD128/129/130/131`；该编号规则不能直接套用到其他机器。
若管理员恢复的是另一张卡，可用 `GPU_INDEX=0 bash src/gpu_docker.sh start`
创建 `glu-gpu0`，后续 `shell`、`probe`、`exec` 等命令也须带同一个 `GPU_INDEX`。
