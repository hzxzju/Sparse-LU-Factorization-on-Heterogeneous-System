"""Numerical and asynchronous mock-runtime checks; no GPU is initialized."""
import csv
import importlib.util
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


FAKE_RUNTIME = r'''
#include <hc_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
using Clock = std::chrono::steady_clock;
static Clock::time_point cursors[2] = {Clock::now(), Clock::now()};
static int sync_calls = 0;
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
    auto &cursor = cursors[stream ? 1 : 0];
    cursor=std::max(Clock::now(), cursor)+std::chrono::milliseconds(
        reinterpret_cast<size_t>(f)==1 ? 350 : 200);
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
    phase.start(0);
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
        interval_ms = os.environ.get('FAKE_SMI_INTERVAL_MS', sys.argv[sys.argv.index('-l')+1])
        time.sleep(float(interval_ms)/1000)
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
        for name, content in [("runtime.cpp", FAKE_RUNTIME), ("target.cpp", FAKE_TARGET), ("ht-smi", FAKE_SMI)]:
            (cls.directory / name).write_text(content)
        (cls.directory / "ht-smi").chmod(0o755)
        cls.env = dict(os.environ, CXX=cls.compiler, HPCC_PATH=str(cls.hpcc),
                       HT_SMI=str(cls.directory / "ht-smi"))
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

    def run_profile(self, gpu, name, target=None, target_args=(), smi_interval_ms=None):
        output = self.directory / name
        env = dict(self.env)
        if smi_interval_ms is not None:
            env["FAKE_SMI_INTERVAL_MS"] = str(smi_interval_ms)
        result = subprocess.run(["bash", str(ROOT / "src/profile_kernel_util.sh"),
                                 "--gpu", gpu, "--output", str(output), "--",
                                 target or str(self.directory / "target"), *target_args],
                                env=env, capture_output=True, text=True, timeout=20)
        with (output / "summary.csv").open() as file:
            rows = list(csv.DictReader(file))
        return result, rows

    def test_entire_numeric_phase_preserves_async_launches_and_existing_waits(self):
        result, rows = self.run_profile("3", "good")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(len(rows), 3, result.stderr)
        first = rows[0]
        self.assertEqual(first["phase"], "numeric_lu_compute")
        self.assertGreaterEqual(float(first["host_window_ms"]), 375)
        measured = float(result.stdout.split("compute_ms=")[1].split()[0])
        self.assertAlmostEqual(float(first["host_window_ms"]), measured, delta=10)
        self.assertGreaterEqual(int(first["samples_in_window"]), 2)
        self.assertTrue(20 <= float(first["observed_util_pct"]) <= 80)
        self.assertEqual(first["numeric_util_pct"], first["observed_util_pct"])
        self.assertTrue(20 <= float(first["numeric_util_pct"]) <= 80)
        self.assertIn("below_sampling_interval", rows[1]["status"])
        self.assertEqual(rows[1]["numeric_util_pct"], rows[1]["observed_util_pct"])
        self.assertIn("runtime_error", rows[2]["status"])
        self.assertEqual(rows[2]["numeric_status"], "7")
        self.assertEqual(rows[2]["numeric_util_pct"], "")
        self.assertTrue(all((self.directory / "good" / row["raw_csv"]).is_file() for row in rows))
        self.assertEqual(len(list((self.directory / "good").glob("*/tmp.csv"))), 3)

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
