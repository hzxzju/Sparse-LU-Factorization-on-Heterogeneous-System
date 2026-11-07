#!/usr/bin/env bash
# Run the sequential LL-SF block-update batch with GPU utilization profiling.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$repo_root"
export GPU_INDEX=${GPU_INDEX:-3}

exec bash src/gpu_docker.sh exec bash -c '
set -euo pipefail
exec bash src/profile_kernel_util.sh --gpu "$GPU_INDEX" -- \
    python3 src/run_batch.py \
        --update-strategy left-looking \
        --scheduling synchronization-free \
        --sf-update block \
        --timeout 0
'
