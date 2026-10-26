#!/usr/bin/env bash
# HTHPCC GPU smoke check. Usage: bash src/check_gpu.sh
# Exit: 0 = ready, 1 = failed, 124 = runtime timed out.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
probe="$repo_root/src/gpu_device_probe"
export HPCC_PATH=${HPCC_PATH:-/opt/hpcc}
export LD_LIBRARY_PATH="$HPCC_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if ! command -v timeout >/dev/null 2>&1; then
    echo '[FAIL] 缺少 timeout 命令（GNU coreutils）。' >&2
    exit 1
fi

if [[ ! -e /dev/htcd || ! -d /dev/dri ]]; then
    echo '[WARN] /dev/htcd 或 /dev/dri 不存在，当前环境可能未挂载 GPU。'
fi

if [[ ! -x "$probe" ]]; then
    echo '编译 GPU 探测程序……'
    make -C "$repo_root/src" gpu_device_probe
fi

run_probe() {
    local status
    if timeout -k 3s 15s "$probe" "$@"; then
        return 0
    else
        status=$?
    fi
    if [[ "$status" == 124 || "$status" == 137 ]]; then
        echo '[FAIL] GPU 探测超时（15 秒），runtime 可能阻塞。' >&2
        exit 124
    fi
    echo "[FAIL] 当前环境的 GPU 探测失败（进程退出码 $status），请查看上方错误。" >&2
    exit 1
}

echo '[1/2] 枚举当前可见 GPU'
run_probe --count
echo '[2/2] 检查设备选择、属性、显存分配、清零、同步与释放'
run_probe
echo '[PASS] 至少一张 GPU 通过基础检查；完整求解仍需另行运行。'
