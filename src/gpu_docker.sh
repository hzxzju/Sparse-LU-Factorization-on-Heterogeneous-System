#!/usr/bin/env bash
# Host-side helper for the X201 machine's existing rootless Docker daemon.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
gpu_index=${GPU_INDEX:-3}
case "$gpu_index" in
    0|1|2|3) ;;
    *) echo 'GPU_INDEX must be 0, 1, 2 or 3' >&2; exit 2 ;;
esac
container="glu-gpu${gpu_index}"
render="/dev/dri/renderD$((128 + gpu_index))"
docker_cmd=(docker --context "${GLU_DOCKER_CONTEXT:-rootless}")
# Apply this environment on every exec, including already-created containers.
docker_exec=("${docker_cmd[@]}" exec --workdir /workspace
    --env "GPU_INDEX=$gpu_index" --env "HT_SMI=${HT_SMI:-/opt/htdriver/bin/ht-smi}")
container_cmd=("$container" bash -c '
set -euo pipefail
export HPCC_PATH=${HPCC_PATH:-/opt/hpcc}
export PATH="${GCC_TOOLCHAIN:+$GCC_TOOLCHAIN/bin:}$HPCC_PATH/bin:$HPCC_PATH/htgpu_llvm/bin:/opt/htdriver/bin:$PATH"
export LD_LIBRARY_PATH="${GCC_TOOLCHAIN:+$GCC_TOOLCHAIN/lib64:}$HPCC_PATH/lib:/opt/htdriver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"
' gpu-docker)
action=${1:-shell}
if (($#)); then shift; fi

usage() {
    cat <<'EOF'
Usage: bash src/gpu_docker.sh {start|shell|exec COMMAND [ARGS...]|probe|status|stop}

GPU_INDEX selects the host GPU and container (default: 3), and is exported inside.
Commands run in /workspace with the SDK, GCC and ht-smi environment configured.
Use exec bash -c 'COMMAND ...' for shell syntax, or exec bash with stdin for a script.
HT_SMI may override the ht-smi executable path inside the container.

Examples (run from the host repository root):
  GPU_INDEX=3 bash src/gpu_docker.sh start
  GPU_INDEX=3 bash src/gpu_docker.sh exec bash -c 'bash src/profile_kernel_util.sh --gpu "$GPU_INDEX" -- ./src/lu_cmd -i src/matrix/add32_csr.mtx -u left-looking -s synchronization-free'

Profiling output is saved under /workspace and is also visible in the host repository.
EOF
}

verify_container() {
    # Refuse to reuse a same-named container with different devices/workspace.
    "${docker_cmd[@]}" inspect "$container" | python3 -c '
import json, sys
config = json.load(sys.stdin)[0]
expected = {("/dev/htcd", "/dev/htcd", "rw"), (sys.argv[1], sys.argv[1], "rw")}
actual = {(d["PathOnHost"], d["PathInContainer"], d["CgroupPermissions"])
          for d in config["HostConfig"].get("Devices", [])}
workspace = any(m["Source"] == sys.argv[2] and m["Destination"] == "/workspace" and m["RW"]
                for m in config["Mounts"])
if actual != expected or not workspace or config["HostConfig"].get("Privileged"):
    sys.exit("Container configuration differs from the expected single-GPU workspace")
' "$render" "$repo_root"
}

case "$action" in
    start)
        if "${docker_cmd[@]}" container inspect "$container" >/dev/null 2>&1; then
            verify_container
            "${docker_cmd[@]}" start "$container"
        else
            gcc_toolchain=${GCC_TOOLCHAIN:-$HOME/local/gcc-7.3.0/usr}
            for path in "$render" /dev/htcd /opt/hpcc /opt/htdriver "$gcc_toolchain"; do
                if [[ ! -e "$path" ]]; then echo "Missing host path: $path" >&2; exit 1; fi
            done
            "${docker_cmd[@]}" run -d --name "$container" --hostname "$container" \
                --network none --cap-drop ALL --security-opt no-new-privileges \
                --shm-size 2g \
                --device /dev/htcd:/dev/htcd:rw --device "$render:$render:rw" \
                --mount "type=bind,src=$repo_root,dst=/workspace" \
                --mount "type=bind,src=$gcc_toolchain,dst=$gcc_toolchain,readonly" \
                --mount type=bind,src=/opt/hpcc,dst=/opt/hpcc,readonly \
                --mount type=bind,src=/opt/htdriver,dst=/opt/htdriver,readonly \
                --env HPCC_PATH=/opt/hpcc --env "GCC_TOOLCHAIN=$gcc_toolchain" \
                --env GCC_VERSION=7.3.0 --workdir /workspace \
                publiccase:arm64 sleep infinity
        fi
        ;;
    shell)
        verify_container
        exec "${docker_exec[@]}" -it "${container_cmd[@]}" bash
        ;;
    exec)
        if (($# == 0)); then echo 'exec requires a command' >&2; exit 2; fi
        verify_container
        exec "${docker_exec[@]}" -i "${container_cmd[@]}" "$@"
        ;;
    probe)
        verify_container
        "${docker_exec[@]}" "${container_cmd[@]}" make -C src gpu_device_probe
        "${docker_exec[@]}" "${container_cmd[@]}" timeout -k 3s 20s ./src/gpu_device_probe --count
        exec "${docker_exec[@]}" "${container_cmd[@]}" timeout -k 3s 30s ./src/gpu_device_probe
        ;;
    status)
        verify_container
        "${docker_cmd[@]}" inspect "$container" --format '{{.Name}}: {{.State.Status}} devices={{json .HostConfig.Devices}}'
        ;;
    stop)
        verify_container
        "${docker_cmd[@]}" stop "$container"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
