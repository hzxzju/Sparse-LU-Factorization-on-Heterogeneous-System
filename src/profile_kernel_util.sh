#!/usr/bin/env bash
# Profile this project's numeric LU compute phase (historical script name).
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ${1:-} == -h || ${1:-} == --help || $# == 0 ]]; then
    cat <<'EOF'
Usage: bash src/profile_kernel_util.sh --gpu DEVICE [--output DIR] -- COMMAND [ARGS...]

Default output: <repo>/util_sampling/YYYYMMDD_HHMMSS (Asia/Shanghai).
DEVICE is an ht-smi physical index or PCI BDF, NOT a runtime logical index.
Each numeric LU compute phase starts ht-smi -l 10 -o .../tmp.csv --show-usage.
Rebuild the solver first: make -C src MAIN
Sampling covers the compute loop and existing waits, without serializing kernels.
Requires Python 3, a C++ compiler, HPCC headers and ht-smi.
See docs/Kernel_Util_Profiling.md for measurement limits and output columns.

Environment: HPCC_PATH, CXX, GCC_TOOLCHAIN, HT_SMI (executable path).
EOF
    exit 0
fi

gpu= output=
while (($#)); do
    case "$1" in
        --gpu|--output)
            (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
            if [[ $1 == --gpu ]]; then gpu=$2; else output=$2; fi
            shift 2 ;;
        --) shift; break ;;
        *) echo "Unknown option: $1 (use -- before COMMAND)" >&2; exit 2 ;;
    esac
done
[[ -n $gpu && $# -gt 0 ]] || { echo 'Specify --gpu DEVICE and -- COMMAND.' >&2; exit 2; }
export HPCC_PATH=${HPCC_PATH:-/opt/hpcc}
if [[ -z ${CXX:-} ]]; then
    if [[ -x ${GCC_TOOLCHAIN:-$HOME/local/gcc-7.3.0/usr}/bin/g++ ]]; then
        CXX=${GCC_TOOLCHAIN:-$HOME/local/gcc-7.3.0/usr}/bin/g++
    else
        CXX=g++
    fi
fi
command -v "$CXX" >/dev/null
command -v python3 >/dev/null
command -v "${HT_SMI:-ht-smi}" >/dev/null
if [[ -z $output ]]; then
    mkdir -p -- "$script_dir/../util_sampling"
    output="$script_dir/../util_sampling/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
# Refuse reuse: every run owns its files, including interrupted runs.
mkdir -- "$output"
output=$(cd -- "$output" && pwd)
compiler_lib=$(cd -- "$(dirname -- "$(command -v "$CXX")")/../lib64" 2>/dev/null && pwd || true)
export LD_LIBRARY_PATH="$HPCC_PATH/lib${compiler_lib:+:$compiler_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$CXX" -std=c++11 -O2 -Wall -Wextra -shared -fPIC -pthread \
    -I"$HPCC_PATH/include" -I"$HPCC_PATH/include/hcr" \
    "$script_dir/kernel_util_hook.cpp" -ldl -o "$output/kernel_util_hook.so"
exec python3 "$script_dir/kernel_util_monitor.py" \
    --gpu "$gpu" --output "$output" --smi "${HT_SMI:-ht-smi}" -- "$@"
