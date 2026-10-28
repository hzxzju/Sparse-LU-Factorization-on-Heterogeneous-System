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
        # Window [50,250) ms: 20% for 50ms, then 80% for 150ms -> 65%.
        count, coverage, util = monitor.integrate(
            [(0, 20, ""), (100_000_000, 80, ""), (300_000_000, 0, "")],
            50_000_000, 250_000_000)
        self.assertEqual(count, 1)
        self.assertEqual(coverage, 100)
        self.assertEqual(util, 65)

    def test_long_sampling_gap_is_uncovered(self):
        _, coverage, util = monitor.integrate([(0, 80, "")], 0, 1_000_000_000)
        self.assertEqual(coverage, 20)
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
#include <chrono>
#include <cstdio>
#include <thread>
using Clock = std::chrono::steady_clock;
static Clock::time_point cursor = Clock::now();
struct FakeEvent { Clock::time_point time; };
extern "C" hcError_t hcDeviceSynchronize() {
    std::this_thread::sleep_until(cursor); cursor = Clock::now(); return hcSuccess;
}
extern "C" hcError_t hcGetDevice(int *d) { *d=0; return hcSuccess; }
extern "C" hcError_t hcDeviceGetPCIBusId(char *p, int n, int) {
    std::snprintf(p,n,"0000:d6:00.0"); return hcSuccess;
}
extern "C" hcError_t hcEventCreate(hcEvent_t *e) {
    *e=reinterpret_cast<hcEvent_t>(new FakeEvent); return hcSuccess;
}
extern "C" hcError_t hcEventDestroy(hcEvent_t e) {
    delete reinterpret_cast<FakeEvent*>(e); return hcSuccess;
}
extern "C" hcError_t hcEventRecord(hcEvent_t e, hcStream_t) {
    if (cursor < Clock::now()) cursor = Clock::now();
    reinterpret_cast<FakeEvent*>(e)->time=cursor; return hcSuccess;
}
extern "C" hcError_t hcEventSynchronize(hcEvent_t e) {
    std::this_thread::sleep_until(reinterpret_cast<FakeEvent*>(e)->time); return hcSuccess;
}
extern "C" hcError_t hcEventElapsedTime(float *m, hcEvent_t a, hcEvent_t b) {
    *m=std::chrono::duration<float,std::milli>(reinterpret_cast<FakeEvent*>(b)->time -
         reinterpret_cast<FakeEvent*>(a)->time).count(); return hcSuccess;
}
extern "C" hcError_t hcLaunchKernel(const void *f, dim3, dim3, void **, size_t, hcStream_t) {
    if (reinterpret_cast<size_t>(f)==3) return static_cast<hcError_t>(7);
    cursor=Clock::now()+std::chrono::milliseconds(reinterpret_cast<size_t>(f)==1 ? 350 : 1);
    return hcSuccess; // Deliberately asynchronous: the event wait must complete it.
}
'''

FAKE_TARGET = r'''
#include <hc_runtime_api.h>
int main() {
    if (hcLaunchKernel(reinterpret_cast<void*>(1), dim3(2), dim3(64), nullptr, 0, nullptr)) return 10;
    if (hcLaunchKernel(reinterpret_cast<void*>(2), dim3(1), dim3(64), nullptr, 0, nullptr)) return 11;
    return static_cast<int>(hcLaunchKernel(reinterpret_cast<void*>(3), dim3(1), dim3(64), nullptr, 0, nullptr));
}
'''

FAKE_SMI = r'''#!/usr/bin/env python3
import csv, datetime, signal, sys, time
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
        time.sleep(.1)
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
                  "-I" + str(cls.hpcc / "include/hcr")]
        for command in [common + ["-shared", "-fPIC", str(cls.directory / "runtime.cpp"),
                                   "-o", str(cls.directory / "libfake.so")],
                        common + [str(cls.directory / "target.cpp"), "-L" + str(cls.directory),
                                  "-lfake", "-Wl,-rpath," + str(cls.directory),
                                  "-o", str(cls.directory / "target")]]:
            subprocess.run(command, env=cls.env, check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_profile(self, gpu, name, target=None):
        output = self.directory / name
        result = subprocess.run(["bash", str(ROOT / "src/profile_kernel_util.sh"),
                                 "--gpu", gpu, "--output", str(output), "--",
                                 target or str(self.directory / "target")],
                                env=self.env, capture_output=True, text=True, timeout=20)
        with (output / "summary.csv").open() as file:
            rows = list(csv.DictReader(file))
        return result, rows

    def test_async_completion_weighting_short_kernel_and_error_exit(self):
        result, rows = self.run_profile("3", "good")
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual(len(rows), 3, result.stderr)
        first = rows[0]
        self.assertGreaterEqual(float(first["host_window_ms"]), 350)
        self.assertGreaterEqual(int(first["samples_in_window"]), 2)
        self.assertEqual(first["coverage_pct"], "100.000000")
        self.assertTrue(20 <= float(first["kernel_util_pct"]) <= 80)
        self.assertAlmostEqual(float(first["gpu_event_ms"]), 350, delta=2)
        self.assertIn("below_sampling_interval", rows[1]["status"])
        if rows[1]["samples_in_window"] == "0":
            self.assertEqual(rows[1]["kernel_util_pct"], "")
        self.assertIn("runtime_error", rows[2]["status"])
        self.assertEqual(rows[2]["kernel_util_pct"], "")
        self.assertTrue(all((self.directory / "good" / row["raw_csv"]).is_file() for row in rows))

    def test_wrong_physical_gpu_is_rejected(self):
        result, rows = self.run_profile("0", "wrong-gpu")
        self.assertEqual(result.returncode, 125, result.stderr)
        self.assertIn("kernel uses 0000:d6:00.0", result.stderr)
        self.assertEqual(rows, [])

    def test_zero_launch_is_explicit(self):
        result, rows = self.run_profile("3", "no-launch", "/bin/true")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
