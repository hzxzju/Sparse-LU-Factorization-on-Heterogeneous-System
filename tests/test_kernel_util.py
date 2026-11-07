"""Numerical and asynchronous mock-runtime checks; no GPU is initialized."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kernel_util_monitor", ROOT / "src/kernel_util_monitor.py")
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


class WeightingTests(unittest.TestCase):
    def test_overlapping_streams_are_merged_and_idle_gaps_excluded(self):
        interval = monitor.INTERVAL_NS
        samples = [(0, 20, ""), (interval, 80, ""), (2 * interval, 0, ""),
                   (3 * interval, 40, ""), (4 * interval, 100, "")]
        windows = [(0, interval), (interval // 2, 2 * interval),
                   (3 * interval, 4 * interval)]
        self.assertEqual(monitor.merge_intervals(windows), [(0, 2 * interval),
                                                         (3 * interval, 4 * interval)])
        count, coverage, util = monitor.integrate_windows(samples, windows)
        self.assertEqual(count, 3)
        self.assertEqual(coverage, 100)
        self.assertAlmostEqual(util, (20 + 80 + 40) / 3)

    def test_sample_before_kernel_can_contribute_with_hold_limit(self):
        interval = monitor.INTERVAL_NS
        self.assertEqual(monitor.integrate_windows([(0, 30, "")],
                         [(interval, interval + 1000)]), (0, 100, 30))
        self.assertEqual(monitor.integrate_windows([(0, 30, "")],
                         [(3 * interval, 4 * interval)]), (0, 0, None))

    def test_irregular_intervals_and_clipped_boundaries(self):
        # Scale to the configured interval: 20% for half, then 80% for 1.5 intervals.
        interval = monitor.INTERVAL_NS
        count, coverage, util = monitor.integrate(
            [(0, 20, ""), (interval, 80, ""), (3 * interval, 0, "")],
            interval // 2, 5 * interval // 2)
        self.assertEqual(count, 1)
        self.assertEqual(coverage, 100)
        self.assertEqual(util, 65)

    def test_long_sampling_gap_is_uncovered(self):
        _, coverage, util = monitor.integrate([(0, 80, "")], 0, 1_000_000_000)
        self.assertAlmostEqual(coverage, 2 * monitor.INTERVAL_NS / 1_000_000_000 * 100)
        self.assertEqual(util, 80)

    def test_no_sample_and_zero_duration(self):
        self.assertEqual(monitor.integrate([], 0, 10), (0, 0, None))
        self.assertEqual(monitor.integrate([], 10, 10), (0, 0, None))

    def test_unpadded_microseconds(self):
        left = monitor.timestamp_ns("2026/10/27 19:56:42.29031")
        right = monitor.timestamp_ns("2026/10/27 19:56:42.129031")
        self.assertEqual(right - left, 100_000_000)

    def test_csv_gpu_filter_invalid_rows_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tmp.csv"
            path.write_text(
                "timestamp,deviceId,dieId,bdfId,utilization.GPU [%]\n"
                "2026/10/27 19:56:42.1000,GPU#0,-,0000:96:00.0,100\n"
                "2026/10/27 19:56:42.1000,GPU#3,-,0000:d6:00.0,20\n"
                "2026/10/27 19:56:42.1000,GPU#3,-,0000:d6:00.0,80\n"
                "2026/10/27 19:56:42.2000,GPU#3,-,0000:d6:00.0,NaN\n"
                "2026/10/27 19:56:42.3000,GPU#3,0,0000:d6:00.0,99\n"
                "2026/10/27 19:56:42.4000,GPU#3,-,0000:d6:00.0,\n")
            samples = monitor.read_samples(path, "GPU#3")
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0][1], 80)
            self.assertEqual(samples, monitor.read_samples(path, "0000:d6:00.0"))


class TraceClockTests(unittest.TestCase):
    def setUp(self):
        base, duration = 1_790_000_000_000_000_000, 10_000_000
        self.phase = dict(pid="42", phase_id="1", numeric_status=0,
                          start_ns=base + 100, end_ns=base + duration - 100,
                          duration_ns=duration - 200,
                          begin_before=base - 10, begin_after=base + 10,
                          end_before=base + duration - 10, end_after=base + duration + 10)
        self.events = [
            dict(ph="X", name="GLU_NUMERIC_UTIL:42:1:begin", ts=base, dur=0),
            dict(ph="X", name="GLU_NUMERIC_UTIL:42:1:end", ts=base + duration, dur=0),
            dict(ph="X", name="hcLaunchKernel", ts=base + 1000, dur=5_000_000,
                 args=dict(process_id=42, co_id=1, device_id=0, grid={}, block={})),
            dict(ph="X", name="LL_factorizeSyncFree()", ts=base + 2000, dur=100_000,
                 args=dict(co_id=1, device_id=0, grid={}, block={}, queue_id=3)),
            dict(ph="X", name="memcpy HTOD", ts=base + 2000, dur=500_000,
                 args=dict(co_id=2, device_id=0, stream_id=0)),
        ]

    def test_epoch_nanoseconds_and_gpu_execution_are_preserved_exactly(self):
        alignment = monitor.align_clock(self.events, self.phase)
        self.assertEqual(alignment["offset_ns"], 0)
        self.assertEqual(alignment["uncertainty_ns"], 10)
        kernels, windows = monitor.gpu_intervals(self.events, self.phase, alignment)
        self.assertEqual(len(kernels), 1)
        self.assertEqual(windows, [(self.events[3]["ts"], self.events[3]["ts"] + 100_000)])
        self.assertEqual(kernels[0]["stream_id"], 3)

    def test_stable_different_clock_origin_can_be_mapped(self):
        for event in self.events:
            event["ts"] -= 1_000_000_000
        alignment = monitor.align_clock(self.events, self.phase)
        self.assertLessEqual(abs(alignment["offset_ns"] - 1_000_000_000), 10)
        _, windows = monitor.gpu_intervals(self.events, self.phase, alignment)
        self.assertLessEqual(abs(windows[0][0] - (self.phase["start_ns"] + 1900)), 10)

    def test_microseconds_are_rejected_instead_of_assumed(self):
        for event in self.events:
            event["ts"] //= 1000
            event["dur"] //= 1000
        with self.assertRaisesRegex(ValueError, "units, offset, or clock drift"):
            monitor.align_clock(self.events, self.phase)

    def test_missing_or_duplicate_anchor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            monitor.align_clock(self.events[1:], self.phase)
        with self.assertRaisesRegex(ValueError, "missing or duplicate"):
            monitor.align_clock(self.events + [self.events[0]], self.phase)

    def test_missing_gpu_execution_record_is_rejected(self):
        alignment = monitor.align_clock(self.events, self.phase)
        with self.assertRaisesRegex(ValueError, "missing GPU execution"):
            monitor.gpu_intervals(self.events[:3], self.phase, alignment)

    def test_missing_correlation_does_not_fall_back_to_launch_time(self):
        alignment = monitor.align_clock(self.events, self.phase)
        self.events[3]["args"]["co_id"] = 99
        with self.assertRaisesRegex(ValueError, "correlation record"):
            monitor.gpu_intervals(self.events, self.phase, alignment)


FAKE_RUNTIME = r'''
#include <hc_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <thread>
using Clock = std::chrono::steady_clock;
static Clock::time_point cursors[2] = {Clock::now(), Clock::now()};
static int sync_calls = 0;
static int correlation = 0;
static long long now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}
static FILE *trace() {
    const char *path=std::getenv("FAKE_TRACE_EVENTS");
    return path ? std::fopen(path,"a") : nullptr;
}
extern "C" void hctxMarkA(const char *name) {
    long long t=now_ns();
    if (std::getenv("FAKE_CLOCK_DRIFT") && std::string(name).find(":end") != std::string::npos)
        t+=5000000;
    if (FILE *f=trace()) {
        std::fprintf(f,"{\"pid\":%d,\"event\":{\"ph\":\"X\",\"name\":\"%s\",\"ts\":%lld,\"dur\":0}}\n",
                     getpid(),name,t);
        std::fclose(f);
    }
}
extern "C" int fake_sync_calls() { return sync_calls; }
extern "C" hcError_t hcDeviceSynchronize() {
    ++sync_calls;
    std::this_thread::sleep_until(std::max(cursors[0], cursors[1]));
    return hcSuccess;
}
extern "C" hcError_t hcGetDevice(int *d) { *d=0; return hcSuccess; }
extern "C" hcError_t hcDeviceGetPCIBusId(char *p, int n, int) {
    std::snprintf(p,n,"0000:d6:00.0"); return hcSuccess;
}
extern "C" hcError_t hcLaunchKernel(const void *f, dim3, dim3, void **, size_t, hcStream_t stream) {
    if (reinterpret_cast<size_t>(f)==3) return static_cast<hcError_t>(7);
    auto now=Clock::now();
    long long host=now_ns();
    auto &cursor = cursors[stream ? 1 : 0];
    auto start=std::max(now,cursor);
    int milliseconds=reinterpret_cast<size_t>(f)==1 ? 350 : 200;
    cursor=start+std::chrono::milliseconds(milliseconds);
    long long gpu=host+std::chrono::duration_cast<std::chrono::nanoseconds>(start-now).count();
    if (FILE *log=trace()) {
        int id=++correlation;
        std::fprintf(log,"{\"pid\":%d,\"event\":{\"ph\":\"X\",\"name\":\"hcLaunchKernel\",\"ts\":%lld,\"dur\":100,\"args\":{\"process_id\":%d,\"co_id\":%d}}}\n",
                     getpid(),host,getpid(),id);
        std::fprintf(log,"{\"pid\":%d,\"event\":{\"ph\":\"X\",\"name\":\"numeric_kernel\",\"ts\":%lld,\"dur\":%lld,\"args\":{\"device_id\":0,\"queue_id\":%d,\"grid\":{},\"block\":{},\"co_id\":%d}}}\n",
                     getpid(),gpu,(long long)milliseconds*1000000,stream ? 1 : 0,id);
        std::fclose(log);
    }
    return hcSuccess; // Deliberately asynchronous; the target owns its waits.
}
'''

FAKE_TARGET = r'''
#include <hc_runtime_api.h>
#include "numeric_util.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
extern "C" int fake_sync_calls();
int main(int argc, char **argv) {
    // Symbolic/setup work must not create its own measurement.
    hcLaunchKernel(reinterpret_cast<void*>(2), dim3(1), dim3(64), nullptr, 0, nullptr);
    hcDeviceSynchronize();
    if (argc > 1 && std::strcmp(argv[1], "no-phase") == 0) return 0;
    NumericUtilScope phase;
    phase.start(0, argc > 1 && std::strcmp(argv[1], "spin-phase") == 0 ? "gpu_idle_spin" : nullptr);
    if (argc > 1 && std::strcmp(argv[1], "abort") == 0) return 9;
    auto start=std::chrono::steady_clock::now();
    hcLaunchKernel(reinterpret_cast<void*>(1), dim3(2), dim3(64), nullptr, 0, nullptr);
    hcLaunchKernel(reinterpret_cast<void*>(2), dim3(1), dim3(64), nullptr, 0,
                   reinterpret_cast<hcStream_t>(1));
    // An added per-launch wait would block for at least 350 ms.
    if (std::chrono::steady_clock::now()-start > std::chrono::milliseconds(200)) return 90;
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    hcDeviceSynchronize();
    // Host-side work between kernels/barriers is part of the phase.
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    double measured=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start).count();
    phase.finish();
    std::printf("compute_ms=%.6f\n", measured);
    if (fake_sync_calls() != 2) return 91;
    // Download/cleanup work must be outside the phase.
    hcLaunchKernel(reinterpret_cast<void*>(2), dim3(1), dim3(64), nullptr, 0, nullptr);
    hcDeviceSynchronize();
    NumericUtilScope short_phase;
    short_phase.start(0);
    short_phase.finish();
    NumericUtilScope failed_phase;
    failed_phase.start(0);
    int status=static_cast<int>(hcLaunchKernel(reinterpret_cast<void*>(3), dim3(1),
                                             dim3(64), nullptr, 0, nullptr));
    failed_phase.finish(status);
    return status;
}
'''

FAKE_SMI = r'''#!/usr/bin/env python3
import csv, datetime, os, signal, sys, time
running=True
def stop(*args):
    global running
    running=False
signal.signal(signal.SIGINT,stop)
with open(os.environ['FAKE_SMI_STARTS'],'a') as starts:
    starts.write('start\n')
with open(sys.argv[sys.argv.index('-o')+1],'w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['timestamp','deviceId','dieId','bdfId','utilization.GPU [%]'])
    i=0
    while running:
        t=datetime.datetime.now()
        stamp=t.strftime('%Y/%m/%d %H:%M:%S')+'.'+str(t.microsecond)
        w.writerow([stamp,'GPU#0','-','0000:96:00.0',100])
        w.writerow([stamp,'GPU#3','-','0000:d6:00.0',20 if i%2==0 else 80])
        f.flush()
        i+=1
        if i == int(os.environ.get('FAKE_SMI_EXIT_AFTER', '-1')):
            sys.exit(4)
        interval_ms = os.environ.get('FAKE_SMI_INTERVAL_MS', sys.argv[sys.argv.index('-l')+1])
        time.sleep(float(interval_ms)/1000)
'''

FAKE_TRACER = r'''#!/usr/bin/env python3
import json, os
from pathlib import Path
import subprocess, sys
args=sys.argv[1:]
assert args.pop(0)=='--hctx'
assert args.pop(0)=='--odname'
trace=Path(args.pop(0)).resolve()
assert args.pop(0)=='--name'
prefix=args.pop(0)
trace.mkdir()
assert len((trace.parent/'tmp.csv').read_text().splitlines()) >= 2
(trace.parent/'sampler_started_before_target').touch()
events_path=trace.parent/'fake_events.jsonl'
env=dict(os.environ, FAKE_TRACE_EVENTS=str(events_path))
subprocess.call(args,env=env)
grouped={}
if events_path.exists():
    for line in events_path.read_text().splitlines():
        record=json.loads(line)
        grouped.setdefault(record['pid'],[]).append(record['event'])
for pid,events in grouped.items():
    (trace/(prefix+'-'+str(pid)+'.json')).write_text(json.dumps({'traceEvents':events}))
# Like real hcTracer, return 0 even if the target failed.
'''


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hpcc = Path(os.environ.get("HPCC_PATH", "/opt/hpcc"))
        cls.compiler = os.environ.get("CXX") or shutil.which("g++")
        local = Path.home() / "local/gcc-7.3.0/usr/bin/g++"
        if not cls.compiler and local.is_file():
            cls.compiler = str(local)
        if not cls.compiler or not (cls.hpcc / "include/hcr/hc_runtime_api.h").is_file():
            raise unittest.SkipTest("HPCC headers and C++ compiler required for integration test")
        cls.temporary = tempfile.TemporaryDirectory(prefix="glu-util-test-", dir="/tmp")
        cls.directory = Path(cls.temporary.name)
        for name, content in [("runtime.cpp", FAKE_RUNTIME), ("target.cpp", FAKE_TARGET),
                              ("ht-smi", FAKE_SMI), ("hcTracer", FAKE_TRACER)]:
            (cls.directory / name).write_text(content)
        (cls.directory / "ht-smi").chmod(0o755)
        (cls.directory / "hcTracer").chmod(0o755)
        cls.env = dict(os.environ, CXX=cls.compiler, HPCC_PATH=str(cls.hpcc),
                       HT_SMI=str(cls.directory / "ht-smi"), HC_TRACER=str(cls.directory / "hcTracer"))
        lib = str(Path(cls.compiler).resolve().parent.parent / "lib64")
        cls.env["LD_LIBRARY_PATH"] = lib + ":" + cls.env.get("LD_LIBRARY_PATH", "")
        common = [cls.compiler, "-std=c++11", "-pthread", "-I" + str(cls.hpcc / "include"),
                  "-I" + str(cls.hpcc / "include/hcr"), "-I" + str(ROOT / "include")]
        for command in [common + ["-shared", "-fPIC", str(cls.directory / "runtime.cpp"),
                                   "-o", str(cls.directory / "libfake.so")],
                        common + [str(cls.directory / "target.cpp"), "-L" + str(cls.directory),
                                  "-lfake", "-ldl", "-Wl,-rpath," + str(cls.directory),
                                  "-o", str(cls.directory / "target")]]:
            subprocess.run(command, env=cls.env, check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_profile(self, gpu, name, target=None, target_args=(), smi_interval_ms=None, **extra_env):
        output = self.directory / name
        env = dict(self.env, FAKE_SMI_STARTS=str(self.directory / (name + "_smi_starts")), **extra_env)
        if smi_interval_ms is not None:
            env["FAKE_SMI_INTERVAL_MS"] = str(smi_interval_ms)
        result = subprocess.run(["bash", str(ROOT / "src/profile_kernel_util.sh"),
                                 "--gpu", gpu, "--output", str(output), "--",
                                 target or str(self.directory / "target"), *target_args],
                                env=env, capture_output=True, text=True, timeout=20)
        with (output / "summary.csv").open() as file:
            rows = list(csv.DictReader(file))
        return result, rows

    def test_gpu_active_union_preserves_async_launches_and_existing_waits(self):
        result, rows = self.run_profile("3", "good")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(len(rows), 3, result.stderr)
        first = rows[0]
        self.assertEqual(first["phase"], "numeric_lu_compute")
        self.assertGreaterEqual(float(first["host_window_ms"]), 375)
        log = (self.directory / "good/hcTracer.log").read_text()
        measured = float(log.split("compute_ms=")[1].split()[0])
        self.assertAlmostEqual(float(first["host_window_ms"]), measured, delta=10)
        self.assertAlmostEqual(float(first["gpu_active_ms"]), 350, delta=10)
        self.assertLess(float(first["gpu_active_ms"]), float(first["host_window_ms"]) - 15)
        self.assertEqual(first["window_basis"], "gpu_kernel_execution_union")
        self.assertEqual(first["kernel_count"], "2")
        self.assertEqual(first["active_interval_count"], "1")
        self.assertEqual(first["clock_status"], "aligned")
        self.assertEqual(first["clock_offset_ns"], "0")
        self.assertGreaterEqual(int(first["samples_in_window"]), 2)
        self.assertTrue(20 <= float(first["observed_util_pct"]) <= 80)
        self.assertEqual(first["numeric_util_pct"], first["observed_util_pct"])
        self.assertTrue(20 <= float(first["numeric_util_pct"]) <= 80)
        self.assertIn("no_gpu_intervals", rows[1]["status"])
        self.assertEqual(rows[1]["numeric_util_pct"], "")
        self.assertIn("runtime_error", rows[2]["status"])
        self.assertEqual(rows[2]["numeric_status"], "7")
        self.assertEqual(rows[2]["numeric_util_pct"], "")
        self.assertTrue(all((self.directory / "good" / row["raw_csv"]).is_file() for row in rows))
        self.assertEqual(len(list((self.directory / "good").rglob("tmp.csv"))), 1)
        self.assertEqual((self.directory / "good_smi_starts").read_text().splitlines(), ["start"])
        self.assertTrue((self.directory / "good/sampler_started_before_target").exists())
        alignment = json.loads((self.directory / "good/clock_alignment.json").read_text())
        self.assertEqual(alignment["status"], "aligned")

    def test_bad_clock_alignment_rejects_utilization(self):
        result, rows = self.run_profile("3", "clock-drift", FAKE_CLOCK_DRIFT="1")
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertEqual(rows[0]["numeric_util_pct"], "")
        self.assertEqual(rows[0]["clock_status"], "unverified")
        self.assertIn("trace_or_clock_error", rows[0]["status"])
        alignment = json.loads((self.directory / "clock-drift/clock_alignment.json").read_text())
        self.assertEqual(alignment["status"], "error")

    def test_spin_phase_uses_the_same_clock_and_gpu_window_weighting(self):
        result, rows = self.run_profile("3", "spin-phase", target_args=("spin-phase",))
        self.assertEqual(result.returncode, 7, result.stderr)
        first = rows[0]
        self.assertEqual(first["phase"], "gpu_idle_spin")
        self.assertEqual(first["clock_status"], "aligned")
        self.assertEqual(first["numeric_util_pct"], "")
        self.assertEqual(first["gpu_util_pct"], first["observed_util_pct"])
        self.assertTrue(20 <= float(first["gpu_util_pct"]) <= 80)
        self.assertAlmostEqual(float(first["gpu_active_ms"]), 350, delta=10)

    def test_sampler_failure_stops_the_run(self):
        result, rows = self.run_profile("3", "sampler-failure", smi_interval_ms=10,
                                        FAKE_SMI_EXIT_AFTER="3")
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertEqual(rows, [])
        self.assertIn("ht-smi exited before the target completed", result.stderr)

    def test_one_continuous_sampler_covers_sequential_solver_subprocesses(self):
        batch = self.directory / "batch.py"
        batch.write_text("import subprocess,sys\nfor _ in range(2):\n"
                         "    if subprocess.call([sys.argv[1]]) != 7: sys.exit(90)\nsys.exit(6)\n")
        result, rows = self.run_profile("3", "batch", sys.executable,
                                        target_args=(str(batch), str(self.directory / "target")))
        self.assertEqual(result.returncode, 6, result.stderr)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({row["trace_json"] for row in rows}), 2)
        self.assertTrue(all(row["clock_status"] == "aligned" for row in rows))
        self.assertEqual((self.directory / "batch_smi_starts").read_text().splitlines(), ["start"])
        status = json.loads((self.directory / "batch/target_status.json").read_text())
        self.assertEqual(status["returncode"], 6)

    def test_partial_coverage_still_outputs_numeric_utilization(self):
        result, rows = self.run_profile("3", "partial", smi_interval_ms=50)
        self.assertEqual(result.returncode, 7, result.stderr)
        first = rows[0]
        self.assertIn("partial_coverage", first["status"])
        self.assertLess(float(first["coverage_pct"]), 100)
        self.assertTrue(20 <= float(first["numeric_util_pct"]) <= 80)
        self.assertEqual(first["numeric_util_pct"], first["observed_util_pct"])

    def test_early_return_reports_failed_phase(self):
        result, rows = self.run_profile("3", "abort", target_args=("abort",))
        self.assertEqual(result.returncode, 9, result.stderr)
        self.assertEqual(len(rows), 1, result.stderr)
        self.assertIn("runtime_error", rows[0]["status"])
        self.assertEqual(rows[0]["numeric_util_pct"], "")

    def test_target_runs_without_profiler(self):
        result = subprocess.run([str(self.directory / "target")], env=self.env,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 7, result.stderr)

    def test_kernel_launches_outside_numeric_phase_are_not_profiled(self):
        result, rows = self.run_profile("3", "outside", target_args=("no-phase",))
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(rows, [])
        self.assertIn("No numeric LU compute phases", result.stderr)

    def test_wrong_physical_gpu_is_rejected(self):
        result, rows = self.run_profile("0", "wrong-gpu")
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertIn("numeric LU uses 0000:d6:00.0", result.stderr)
        self.assertEqual(rows, [])

    def test_missing_phase_markers_is_explicit(self):
        result, rows = self.run_profile("3", "no-launch", "/bin/true")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
