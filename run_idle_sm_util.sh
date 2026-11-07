#!/usr/bin/env bash
# Run the bounded all-SM spin experiment in the selected single-GPU container.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$repo_root"
export GPU_INDEX=${GPU_INDEX:-3}
exec bash src/gpu_docker.sh exec bash -c '
set -euo pipefail
make -C src gpu_idle_spin
exec bash src/profile_kernel_util.sh --gpu "$GPU_INDEX" -- ./src/gpu_idle_spin "$@"
' gpu-idle-spin "$@"
