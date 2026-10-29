#!/usr/bin/env python3
"""Sample the numeric LU compute phase; integrate actual ht-smi timestamps."""
import argparse
import csv
import datetime
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

INTERVAL_MS = 10
INTERVAL_NS = INTERVAL_MS * 1_000_000
FIELDS = [
    "pid", "phase_id", "phase", "gpu", "bdf",
    "start_unix_ns", "end_unix_ns", "host_window_ms",
    "samples_in_window", "coverage_pct", "observed_util_pct", "numeric_util_pct",
    "status", "numeric_status", "raw_csv",
]


def timestamp_ns(value):
    # ht-smi 2.3.4 prints tv_usec as an UNPADDED integer: .29031 means
    # 29,031 microseconds, not the usual decimal fraction .290310 seconds.
    whole, microseconds = value.strip().rsplit(".", 1)
    dt = datetime.datetime.strptime(whole, "%Y/%m/%d %H:%M:%S")
    usec = int(microseconds)
    if len(microseconds) > 6 or not 0 <= usec < 1_000_000:
        raise ValueError("invalid ht-smi microsecond field")
    return int(dt.timestamp()) * 1_000_000_000 + usec * 1000


def read_samples(path, gpu):
    """Read aggregate rows for one physical GPU. Ignore incomplete/invalid rows."""
    result = []
    try:
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                row = {k.strip(): (v.strip() if v is not None else "")
                       for k, v in row.items() if k is not None}
                device, bdf = row.get("deviceId", ""), row.get("bdfId", "")
                if gpu.lower() not in (device.lower(), bdf.lower()):
                    continue
                # Do not average per-die rows into an invented aggregate.
                if row.get("dieId", "-") not in ("", "-", "N/A"):
                    continue
                try:
                    t = timestamp_ns(row["timestamp"])
                    u = float(row["utilization.GPU [%]"].rstrip("%"))
                    if math.isfinite(u) and 0 <= u <= 100:
                        result.append((t, u, bdf))
                except (KeyError, ValueError, OverflowError):
                    continue
    except FileNotFoundError:
        pass
    # A repeated timestamp is replaced by the final complete row.
    return [(t, *value) for t, value in sorted({t: (u, bdf) for t, u, bdf in result}.items())]


def integrate(samples, start, end):
    """Left hold each observation, clipped to the phase and at most two intervals."""
    if end <= start:
        return 0, 0.0, None
    count = sum(start <= t < end for t, _, _ in samples)
    covered, area = 0, 0.0
    for i, (t, util, _) in enumerate(samples):
        following = samples[i + 1][0] if i + 1 < len(samples) else end
        left, right = max(start, t), min(end, following, t + 2 * INTERVAL_NS)
        width = max(0, right - left)
        covered += width
        area += util * width
    return count, covered / (end - start) * 100, area / covered if covered else None


def stop_process(process, first_signal=signal.SIGINT):
    if process.poll() is None:
        try:
            os.killpg(process.pid, first_signal)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


class Monitor:
    def __init__(self, output, gpu, smi, child, writer, summary):
        self.output, self.gpu, self.smi = output, gpu, smi
        self.child, self.writer, self.summary = child, writer, summary
        self.completed = 0

    def line(self, connection, buffer):
        while b"\n" not in buffer:
            if self.child.poll() is not None:
                raise RuntimeError("target exited while numeric LU was being profiled")
            try:
                data = connection.recv(4096)
            except socket.timeout:
                continue
            if not data:
                raise RuntimeError("target disconnected before completing numeric LU")
            buffer += data
            if len(buffer) > 65536:
                raise RuntimeError("oversized hook request")
        line, buffer = buffer.split(b"\n", 1)
        return line.decode().split("\t"), buffer

    def handle(self, connection):
        connection.settimeout(0.2)
        begin, buffer = self.line(connection, b"")
        if len(begin) != 5 or begin[0] != "BEGIN":
            raise RuntimeError("invalid BEGIN request")
        _, pid, phase_id, runtime_bdf, name = begin
        if name != "numeric_lu_compute":
            raise RuntimeError("unsupported profiling phase")
        # Only numeric IDs can form directory names received over the socket.
        pid, phase_id = str(int(pid)), str(int(phase_id))
        directory = self.output / (pid + "_" + phase_id.zfill(6))
        directory.mkdir()
        raw = directory / "tmp.csv"
        command = [self.smi, "-l", str(INTERVAL_MS), "-o", str(raw), "--show-usage"]
        with (directory / "ht-smi.log").open("w") as log:
            sampler = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not read_samples(raw, self.gpu):
                    if sampler.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("ht-smi has no valid sample for " + self.gpu +
                                           "; inspect " + str(directory / "ht-smi.log"))
                    if self.child.poll() is not None:
                        raise RuntimeError("target exited before sampler became ready")
                    time.sleep(0.01)
                sample_bdf = read_samples(raw, self.gpu)[-1][2]
                if sample_bdf.lower() != runtime_bdf.lower():
                    raise RuntimeError(f"selected {self.gpu} has BDF {sample_bdf}, "
                                       f"but numeric LU uses {runtime_bdf}; select the matching physical GPU")
                connection.sendall(b"O")
                end, _ = self.line(connection, buffer)
                if len(end) != 5 or end[0] != "END":
                    raise RuntimeError("invalid END request")
                _, start_ns, end_ns, duration_ns, numeric_status = end
                start_ns, end_ns, duration_ns = map(int, (start_ns, end_ns, duration_ns))
                # Numeric completion was observed BEFORE stopping/waiting for ht-smi.
            finally:
                stop_process(sampler)

        samples = read_samples(raw, self.gpu)
        count, coverage, observed = integrate(samples, start_ns, end_ns)
        status = []
        if int(numeric_status):
            status.append("runtime_error")
        if abs((end_ns - start_ns) - duration_ns) > 5_000_000:
            status.append("clock_jump")
        if sampler.returncode not in (0, -signal.SIGINT):
            status.append("sampler_error")
        if count == 0:
            status.append("no_in_window_sample")
        elif count == 1:
            status.append("single_sample_estimate")
        if coverage < 99.999:
            status.append("partial_coverage")
        if duration_ns < INTERVAL_NS:
            status.append("below_sampling_interval")
        valid = (observed is not None and
                 not any(s in status for s in ("runtime_error", "clock_jump", "sampler_error")))
        row = dict(pid=pid, phase_id=phase_id, phase=name, gpu=self.gpu,
                   bdf=samples[0][2] if samples else "")
        row.update(start_unix_ns=start_ns, end_unix_ns=end_ns,
                   host_window_ms=f"{duration_ns / 1e6:.6f}",
                   samples_in_window=count, coverage_pct=f"{coverage:.6f}",
                   observed_util_pct=f"{observed:.6f}" if observed is not None else "",
                   numeric_util_pct=f"{observed:.6f}" if valid else "",
                   status=";".join(status) or "sampled_estimate",
                   numeric_status=numeric_status,
                   raw_csv=str(raw.relative_to(self.output)))
        self.writer.writerow(row)
        self.summary.flush()
        self.completed += 1
        print(f"[numeric-util] {pid}/{phase_id} {name}: util={row['numeric_util_pct'] or 'N/A'}% "
              f"window={row['host_window_ms']} ms samples={count} coverage={coverage:.1f}% "
              f"{row['status']}", file=sys.stderr, flush=True)
        connection.sendall(b"O")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smi", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command")
    gpu = "GPU#" + args.gpu if args.gpu.isdigit() else args.gpu
    output = args.output.resolve()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    with tempfile.TemporaryDirectory(prefix="glu-util-", dir="/tmp") as temporary:
        # Keep socket and LD_PRELOAD paths short and free of spaces.
        socket_path = str(Path(temporary) / "monitor.sock")
        hook = Path(temporary) / "hook.so"
        hook.symlink_to(output / "kernel_util_hook.so")
        with socket.socket(socket.AF_UNIX) as server, (output / "summary.csv").open("w", newline="") as summary:
            server.bind(socket_path)
            server.listen(16)
            server.settimeout(0.2)
            writer = csv.DictWriter(summary, fieldnames=FIELDS)
            writer.writeheader()
            summary.flush()
            env = dict(os.environ, GLU_NUMERIC_UTIL_SOCKET=socket_path)
            env["LD_PRELOAD"] = str(hook) + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
            print("[numeric-util] output: " + str(output), file=sys.stderr, flush=True)
            child = subprocess.Popen(command, env=env, start_new_session=True)
            monitor = Monitor(output, gpu, args.smi, child, writer, summary)
            try:
                while child.poll() is None:
                    try:
                        connection, _ = server.accept()
                    except socket.timeout:
                        continue
                    with connection:
                        monitor.handle(connection)
                code = child.wait()
                if not monitor.completed:
                    print("[numeric-util] No numeric LU compute phases were reported; rebuild the solver with numeric_util.h.",
                          file=sys.stderr)
                    return code if code else 3
                return code if code >= 0 else 128 - code
            finally:
                stop_process(child, signal.SIGTERM)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, ValueError) as error:
        print("[numeric-util] ERROR: " + str(error), file=sys.stderr)
        sys.exit(125)
