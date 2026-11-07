#!/usr/bin/env python3
"""Continuously sample ht-smi; weight observations over hcTracer GPU intervals."""
import argparse
import bisect
import csv
import datetime
from decimal import Decimal
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

INTERVAL_MS = 1
INTERVAL_NS = INTERVAL_MS * 1_000_000
CLOCK_TOLERANCE_NS = 1_000_000
FIELDS = [
    "pid", "phase_id", "phase", "gpu", "bdf", "window_basis",
    "start_unix_ns", "end_unix_ns", "host_window_ms",
    "gpu_start_unix_ns", "gpu_end_unix_ns", "gpu_active_ms", "gpu_span_ms",
    "kernel_count", "active_interval_count", "samples_in_window", "coverage_pct",
    "observed_util_pct", "gpu_util_pct", "numeric_util_pct", "clock_status", "clock_offset_ns",
    "clock_uncertainty_ns", "status", "numeric_status", "raw_csv", "trace_json",
    "active_intervals_csv",
]


def timestamp_ns(value):
    # ht-smi 2.3.4 prints tv_usec as an UNPADDED integer, not a decimal fraction.
    whole, microseconds = value.strip().rsplit(".", 1)
    dt = datetime.datetime.strptime(whole, "%Y/%m/%d %H:%M:%S")
    usec = int(microseconds)
    if len(microseconds) > 6 or not 0 <= usec < 1_000_000:
        raise ValueError("invalid ht-smi microsecond field")
    return int(dt.timestamp()) * 1_000_000_000 + usec * 1000


def read_samples(path, gpu):
    """Read one physical card's aggregate rows, ignoring unfinished CSV rows."""
    result = []
    try:
        with path.open(newline="") as file:
            for row in csv.DictReader(file):
                row = {k.strip(): (v.strip() if v is not None else "")
                       for k, v in row.items() if k is not None}
                device, bdf = row.get("deviceId", ""), row.get("bdfId", "")
                if gpu.lower() not in (device.lower(), bdf.lower()):
                    continue
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
    if any(right[0] < left[0] for left, right in zip(result, result[1:])):
        raise RuntimeError("ht-smi clock moved backwards")
    return [(t, *value) for t, value in sorted({t: (u, bdf) for t, u, bdf in result}.items())]


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def integrate_windows(samples, intervals):
    """Left hold, cap at two polling intervals, intersect with the active union."""
    windows = merge_intervals(intervals)
    duration = sum(end - start for start, end in windows)
    if not duration:
        return 0, 0.0, None
    starts = [start for start, _ in windows]
    covered, area, count, cursor = 0, 0.0, 0, 0
    for i, (t, util, _) in enumerate(samples):
        index = bisect.bisect_right(starts, t) - 1
        if index >= 0 and t < windows[index][1]:
            count += 1
        following = samples[i + 1][0] if i + 1 < len(samples) else windows[-1][1]
        right = min(following, t + 2 * INTERVAL_NS)
        while cursor < len(windows) and windows[cursor][1] <= t:
            cursor += 1
        j = cursor
        while j < len(windows) and windows[j][0] < right:
            width = max(0, min(right, windows[j][1]) - max(t, windows[j][0]))
            covered += width
            area += util * width
            j += 1
    return count, covered / duration * 100, area / covered if covered else None


def integrate(samples, start, end):
    return integrate_windows(samples, [(start, end)])


def trace_int(value):
    # Never round epoch nanoseconds through binary float (about 256 ns ULP).
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)) or int(value) != value:
        raise ValueError("hcTracer timestamps must be integer nanoseconds")
    return int(value)


def align_clock(events, phase):
    """Intersect the offset bounds supplied by two bracketed HCTX markers."""
    prefix = "GLU_NUMERIC_UTIL:{}:{}:".format(phase["pid"], phase["phase_id"])
    anchors, lower, upper = [], None, None
    for suffix in ("begin", "end"):
        matches = [e for e in events if e.get("name") == prefix + suffix and e.get("ph") == "X"]
        if len(matches) != 1:
            raise ValueError("missing or duplicate HCTX clock marker: " + prefix + suffix)
        stamp = trace_int(matches[0]["ts"])
        before, after = phase[suffix + "_before"], phase[suffix + "_after"]
        if not 0 <= after - before <= CLOCK_TOLERANCE_NS:
            raise ValueError("HCTX clock bracket exceeds 1 ms or clock moved backwards")
        lo, hi = before - stamp, after - stamp
        lower = lo if lower is None else max(lower, lo)
        upper = hi if upper is None else min(upper, hi)
        anchors.append(dict(marker=prefix + suffix, trace_ns=stamp,
                            realtime_before_ns=before, realtime_after_ns=after))
    if anchors[1]["trace_ns"] <= anchors[0]["trace_ns"] or lower > upper:
        raise ValueError("HCTX clock anchors disagree: timestamp units, offset, or clock drift")
    if abs(phase["end_ns"] - phase["start_ns"] - phase["duration_ns"]) > CLOCK_TOLERANCE_NS:
        raise ValueError("CLOCK_REALTIME changed relative to CLOCK_MONOTONIC")
    offset = min(max(0, lower), upper)
    uncertainty = max(offset - lower, upper - offset)
    return dict(status="aligned", trace_time_unit="ns", offset_ns=offset,
                uncertainty_ns=uncertainty, anchors=anchors)


def gpu_intervals(events, phase, alignment):
    """Identify GPU kernels by execution attributes and CPU correlation, not API bars."""
    lo, hi = [anchor["trace_ns"] for anchor in alignment["anchors"]]
    calls = {e["args"]["co_id"] for e in events
             if e.get("ph") == "X" and "co_id" in e.get("args", {})
             and str(e["args"].get("process_id")) == phase["pid"]
             and lo <= trace_int(e["ts"]) < hi}
    launches = {e["args"]["co_id"] for e in events
                if e.get("name") in ("hcLaunchKernel", "hcLaunchCooperativeKernel")
                and e.get("args", {}).get("co_id") in calls
                and str(e["args"].get("process_id")) == phase["pid"]}
    kernels = []
    for event in events:
        args = event.get("args", {})
        # Verified hcTracer 3.8 schema. memcpy/memset and CPU launch records
        # do not contain this combination of execution attributes.
        if (event.get("ph") != "X" or "process_id" in args or
                not all(k in args for k in ("device_id", "grid", "block"))):
            continue
        start, duration = trace_int(event["ts"]), trace_int(event["dur"])
        if start >= hi or start + duration <= lo:
            continue
        if duration <= 0 or args.get("co_id") not in calls:
            raise ValueError("GPU kernel lacks a matching CPU process/correlation record")
        if start < lo or start + duration > hi:
            raise ValueError("GPU kernel execution crosses its synchronized phase markers")
        kernels.append(dict(start_ns=start + alignment["offset_ns"],
                            end_ns=start + duration + alignment["offset_ns"],
                            name=event.get("name", ""), device_id=args["device_id"],
                            stream_id=args.get("stream_id", args.get("queue_id", event.get("tid"))),
                            correlation_id=args["co_id"]))
    if len({k["device_id"] for k in kernels}) > 1:
        raise ValueError("multiple GPU devices in one numeric phase are unsupported")
    if not phase["numeric_status"] and launches - {k["correlation_id"] for k in kernels}:
        raise ValueError("hcTracer is missing GPU execution records for numeric kernel launches")
    return kernels, merge_intervals((k["start_ns"], k["end_ns"]) for k in kernels)


def stop_process(process, first_signal=signal.SIGINT):
    if process is None:
        return
    # Also kill leftover descendants if the group leader has already exited.
    try:
        os.killpg(process.pid, first_signal)
    except ProcessLookupError:
        pass
    if process.poll() is None:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def forward_output(process, log):
    """Keep the command's live console output as well as an auditable trace log."""
    console = True
    with process.stdout:
        for line in process.stdout:
            log.write(line)
            log.flush()
            if console:
                try:
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                except BrokenPipeError:
                    console = False


class Monitor:
    def __init__(self, gpu, bdf, child, sampler, phases_file):
        self.gpu, self.bdf, self.child, self.sampler = gpu, bdf, child, sampler
        self.phases_file, self.phases = phases_file, []

    def line(self, connection, buffer):
        while b"\n" not in buffer:
            if self.sampler.poll() is not None:
                raise RuntimeError("ht-smi exited before the target completed")
            try:
                data = connection.recv(4096)
            except socket.timeout:
                if self.child.poll() is not None:
                    raise RuntimeError("hcTracer exited during a numeric phase")
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
        if (len(begin) != 5 or begin[0] != "BEGIN" or
                begin[4] not in ("numeric_lu_compute", "gpu_idle_spin")):
            raise RuntimeError("invalid BEGIN request")
        _, pid, phase_id, runtime_bdf, name = begin
        if self.bdf.lower() != runtime_bdf.lower():
            raise RuntimeError(f"selected {self.gpu} has BDF {self.bdf}, "
                               f"but numeric LU uses {runtime_bdf}; select the matching physical GPU")
        phase = dict(pid=str(int(pid)), phase_id=str(int(phase_id)), bdf=runtime_bdf, phase=name)
        connection.sendall(b"O")
        end, _ = self.line(connection, buffer)
        if len(end) != 9 or end[0] != "END":
            raise RuntimeError("invalid END request; rebuild the profiling hook")
        phase.update(zip(("start_ns", "end_ns", "duration_ns", "numeric_status",
                          "begin_before", "begin_after", "end_before", "end_after"),
                         map(int, end[1:])))
        self.phases.append(phase)
        self.phases_file.write(json.dumps(phase) + "\n")
        self.phases_file.flush()
        # No stopping, disk postprocessing, or GPU synchronization in the phase.
        connection.sendall(b"O")


def summarize(output, gpu, samples, phases, sampler_error, run_clock_error, run_clocks=None):
    clock_report = dict(smi_clock="CLOCK_REALTIME/local timezone", smi_interval_ms=INTERVAL_MS,
                        trace_time_unit="ns", run=run_clocks, phases=[])
    errors, cache = [], {}
    with (output / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for phase in phases:
            label = phase["pid"] + "_" + phase["phase_id"].zfill(6)
            row = dict(pid=phase["pid"], phase_id=phase["phase_id"], phase=phase["phase"],
                       gpu=gpu, bdf=phase["bdf"], window_basis="gpu_kernel_execution_union",
                       start_unix_ns=phase["start_ns"], end_unix_ns=phase["end_ns"],
                       host_window_ms=f"{phase['duration_ns'] / 1e6:.6f}",
                       numeric_status=phase["numeric_status"], raw_csv="tmp.csv",
                       clock_status="unverified", gpu_util_pct="", numeric_util_pct="")
            status, detail = [], dict(pid=phase["pid"], phase_id=phase["phase_id"])
            if phase["numeric_status"]:
                status.append("runtime_error")
            if sampler_error:
                status.append("sampler_error")
            if run_clock_error:
                status.append("clock_jump")
            try:
                paths = list((output / "trace").glob("*-" + phase["pid"] + ".json"))
                if len(paths) != 1:
                    raise ValueError("expected one hcTracer JSON for target PID " + phase["pid"])
                path = paths[0]
                row["trace_json"] = str(path.relative_to(output))
                if path not in cache:
                    document = json.loads(path.read_text(), parse_float=Decimal)
                    events = document["traceEvents"]
                    if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
                        raise ValueError("invalid hcTracer traceEvents")
                    cache[path] = events
                events = cache[path]
                alignment = align_clock(events, phase)
                detail.update(alignment)
                row.update(clock_status="aligned", clock_offset_ns=alignment["offset_ns"],
                           clock_uncertainty_ns=alignment["uncertainty_ns"])
                kernels, windows = gpu_intervals(events, phase, alignment)
                interval_path = output / (label + "_active_intervals.csv")
                with interval_path.open("w", newline="") as intervals_file:
                    intervals_writer = csv.writer(intervals_file)
                    intervals_writer.writerow(["start_unix_ns", "end_unix_ns", "duration_ns"])
                    intervals_writer.writerows((s, e, e - s) for s, e in windows)
                # Keep individual kernels for auditing stream overlap and selection.
                (output / (label + "_kernels.json")).write_text(json.dumps(kernels, indent=2) + "\n")
                row["active_intervals_csv"] = interval_path.name
                active_ns = sum(e - s for s, e in windows)
                count, coverage, observed = integrate_windows(samples, windows)
                row.update(kernel_count=len(kernels), active_interval_count=len(windows),
                           gpu_active_ms=f"{active_ns / 1e6:.6f}", samples_in_window=count,
                           coverage_pct=f"{coverage:.6f}",
                           observed_util_pct=f"{observed:.6f}" if observed is not None else "")
                if windows:
                    row.update(gpu_start_unix_ns=windows[0][0], gpu_end_unix_ns=windows[-1][1],
                               gpu_span_ms=f"{(windows[-1][1] - windows[0][0]) / 1e6:.6f}")
                    if active_ns < INTERVAL_NS:
                        status.append("below_sampling_interval")
                    if count == 0:
                        status.append("no_in_window_sample")
                    elif count == 1:
                        status.append("single_sample_estimate")
                    if coverage < 99.999:
                        status.append("partial_coverage")
                    if observed is None:
                        status.append("no_covered_observation")
                else:
                    status.append("no_gpu_intervals")
                if observed is not None and not (phase["numeric_status"] or sampler_error or run_clock_error):
                    row["gpu_util_pct"] = f"{observed:.6f}"
                    if phase["phase"] == "numeric_lu_compute":
                        row["numeric_util_pct"] = row["gpu_util_pct"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                status.append("trace_or_clock_error")
                detail.update(status="error", error=str(error))
                errors.append(label + ": " + str(error))
            row["status"] = ";".join(status) or "sampled_estimate"
            clock_report["phases"].append(detail)
            writer.writerow(row)
            print(f"[numeric-util] {label}: GPU active={row.get('gpu_active_ms', 'N/A')} ms "
                  f"util={row['gpu_util_pct'] or 'N/A'}% {row['status']}", file=sys.stderr)
    clock_report["status"] = ("error" if errors or run_clock_error or sampler_error else
                              "aligned" if phases else "no_phases")
    (output / "clock_alignment.json").write_text(json.dumps(clock_report, indent=2) + "\n")
    if errors:
        raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smi", required=True)
    parser.add_argument("--tracer", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing command")
    gpu = "GPU#" + args.gpu if args.gpu.isdigit() else args.gpu
    output = args.output.resolve()
    # Create reviewable files even if startup fails.
    with (output / "summary.csv").open("w", newline="") as file:
        csv.DictWriter(file, fieldnames=FIELDS).writeheader()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    sampler, child, output_thread = None, None, None
    raw = output / "tmp.csv"
    run_start_ns, run_start_mono = time.time_ns(), time.monotonic_ns()
    print("[numeric-util] output: " + str(output), file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix="glu-util-", dir="/tmp") as temporary:
        socket_path = str(Path(temporary) / "monitor.sock")
        hook = Path(temporary) / "hook.so"
        hook.symlink_to(output / "kernel_util_hook.so")
        config = output / "hc-config"
        config.mkdir()
        with socket.socket(socket.AF_UNIX) as server, (output / "ht-smi.log").open("w") as smi_log, \
                (output / "hcTracer.log").open("wb") as tracer_log, \
                (output / "phases.jsonl").open("w") as phases_file:
            server.bind(socket_path)
            server.listen(16)
            server.settimeout(0.1)
            try:
                smi_gpu = gpu[4:] if gpu.upper().startswith("GPU#") else gpu
                sampler = subprocess.Popen([args.smi, "-i", smi_gpu, "-l", str(INTERVAL_MS),
                                            "-o", str(raw), "--show-usage"],
                                           stdout=smi_log, stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.monotonic() + 5
                samples = []
                while not samples:
                    samples = read_samples(raw, gpu)
                    if sampler.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("ht-smi has no valid sample for " + gpu + "; inspect ht-smi.log")
                    if not samples:
                        time.sleep(0.01)
                ready_ns = time.time_ns()
                if not run_start_ns <= samples[0][0] <= ready_ns + CLOCK_TOLERANCE_NS:
                    raise RuntimeError("ht-smi timestamp does not match host CLOCK_REALTIME/local timezone")
                bdf = samples[-1][2]
                first_sample_ns = samples[0][0]
                env = dict(os.environ, GLU_NUMERIC_UTIL_SOCKET=socket_path,
                           GLU_NUMERIC_UTIL_HOOK=str(hook), HPCC_ETC_PATH=str(config))
                # gRPC otherwise sends local hcTracer traffic through HTTP proxies.
                env["no_grpc_proxy"] = ",".join(filter(None, [env.get("no_grpc_proxy", ""),
                                                          "localhost,127.0.0.1,0.0.0.0,::1"]))
                target_status = output / "target_status.json"
                runner = Path(__file__).with_name("kernel_util_target.py")
                # hcTracer 3.8 prefixes cwd even when --odname is absolute.
                trace_directory = os.path.relpath(output / "trace", Path.cwd())
                child = subprocess.Popen([args.tracer, "--hctx", "--odname", trace_directory,
                                          "--name", "numeric", sys.executable, str(runner),
                                          str(target_status), *command], env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
                output_thread = threading.Thread(target=forward_output, args=(child, tracer_log), daemon=True)
                output_thread.start()
                monitor = Monitor(gpu, bdf, child, sampler, phases_file)
                while child.poll() is None:
                    if sampler.poll() is not None:
                        raise RuntimeError("ht-smi exited before the target completed")
                    try:
                        connection, _ = server.accept()
                    except socket.timeout:
                        continue
                    with connection:
                        monitor.handle(connection)
                tracer_code = child.wait()
            finally:
                stop_process(child, signal.SIGTERM)
                stop_process(sampler)
                if output_thread is not None:
                    output_thread.join(timeout=2)
    run_end_mono, run_end_ns = time.monotonic_ns(), time.time_ns()
    run_clock_error = abs((run_end_ns - run_start_ns) - (run_end_mono - run_start_mono)) > CLOCK_TOLERANCE_NS
    sampler_error = sampler.returncode not in (0, -signal.SIGINT)
    samples = read_samples(raw, gpu)
    if any(sample[2].lower() != bdf.lower() for sample in samples):
        raise RuntimeError("ht-smi physical GPU identity changed during sampling")
    if not samples or samples[-1][0] > run_end_ns + CLOCK_TOLERANCE_NS:
        raise RuntimeError("ht-smi timestamps do not match the completed host sampling interval")
    run_clocks = dict(start_realtime_ns=run_start_ns, sampler_ready_realtime_ns=ready_ns,
                      first_smi_timestamp_ns=first_sample_ns, end_realtime_ns=run_end_ns,
                      last_smi_timestamp_ns=samples[-1][0],
                      realtime_minus_monotonic_elapsed_ns=(run_end_ns - run_start_ns) -
                                                         (run_end_mono - run_start_mono),
                      host_timezone=time.tzname, tolerance_ns=CLOCK_TOLERANCE_NS)
    summarize(output, gpu, samples, monitor.phases, sampler_error, run_clock_error, run_clocks)
    if tracer_code:
        raise RuntimeError("hcTracer failed (exit {}); inspect hcTracer.log".format(tracer_code))
    if not target_status.is_file():
        raise RuntimeError("hcTracer did not finish the target; inspect hcTracer.log")
    code = json.loads(target_status.read_text())["returncode"]
    code = code if code >= 0 else 128 - code
    if sampler_error or run_clock_error:
        raise RuntimeError("sampling or host clock failed; inspect summary.csv")
    if not monitor.phases:
        print("[numeric-util] No numeric LU compute phases were reported; rebuild the solver with numeric_util.h.",
              file=sys.stderr)
        return code if code else 3
    return code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, RuntimeError, ValueError) as error:
        print("[numeric-util] ERROR: " + str(error), file=sys.stderr)
        sys.exit(125)
