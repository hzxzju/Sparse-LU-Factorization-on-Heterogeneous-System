"""Exercise the real CLI/preprocessing with a CPU numeric backend (no GPU)."""

import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = r'''
#include "numeric.h"
#include <cstdlib>
#include <cmath>

bool ShutdownGPUDevice(int, std::ostream &) {
    return !std::getenv("GLU_FIXTURE_SHUTDOWN_FAIL");
}

void Symbolic_Matrix::fill_in(unsigned *ai, unsigned *ap, const SymbolicConfig &) {
    m_out << "fixture symbolic analysis\n";
    fill_in(ai, ap);
}

void LUonDeviceRepeated(Symbolic_Matrix &a, std::ostream &out, std::ostream &, bool,
                       UpdateStrategy update, SchedulingStrategy scheduling, unsigned, SFUpdateStrategy sf_update,
                       unsigned repeat, const std::function<void()> &solve) {
    ValidateNumericConfiguration(update, scheduling, sf_update);
    out << "fixture sf_update=" << SFUpdateStrategyName(sf_update) << '\n';
    const std::vector<REAL> original = a.val;
    unsigned cycle = 0;
    RunNumericIterations(repeat, [&]() {
        ++cycle;
        out << "fixture numeric cycle=" << cycle << '\n';
        if (std::getenv("GLU_FIXTURE_FAIL")) throw std::runtime_error("numeric failure");
        {
            glu_timing::Scope phase(GLU_NUMERIC_FACTOR);
            // A deterministic marker verifies that only the last cycle counts.
            glu_timing::active()->seconds[GLU_NUMERIC_FACTOR] += 0.125 * cycle;
            std::vector<double> dense(size_t(a.n) * a.n, 0);
            for (unsigned col = 0; col < a.n; ++col)
                for (unsigned p = a.sym_c_ptr[col]; p < a.sym_c_ptr[col + 1]; ++p)
                    dense[size_t(a.sym_r_idx[p]) * a.n + col] = original[p];
            for (unsigned k = 0; k < a.n; ++k)
                for (unsigned i = k + 1; i < a.n; ++i) {
                    dense[size_t(i) * a.n + k] /= dense[size_t(k) * a.n + k];
                    for (unsigned j = k + 1; j < a.n; ++j)
                        dense[size_t(i) * a.n + j] -=
                            dense[size_t(i) * a.n + k] * dense[size_t(k) * a.n + j];
                }
            for (unsigned col = 0; col < a.n; ++col)
                for (unsigned p = a.sym_c_ptr[col]; p < a.sym_c_ptr[col + 1]; ++p)
                    a.val[p] = dense[size_t(a.sym_r_idx[p]) * a.n + col];
            if (std::getenv("GLU_FIXTURE_NAN")) a.val[a.l_col_ptr[0]] = NAN;
        }
        solve();
    });
    glu_timing::Scope cleanup(GLU_CLEANUP);
    glu_timing::active()->seconds[GLU_CLEANUP] += 0.25;
}
'''


class StageTimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="glu-cli-build-", dir="/tmp")
        directory = Path(cls.build.name)
        backend = directory / "numeric.cpp"
        backend.write_text(BACKEND)
        subprocess.run(["make", "-C", str(ROOT / "src"), "preprocess/preprocess.o",
                        "nicslu/lib/nicslu.a", "nicslu/util/nicslu_util.a"],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        toolchain = Path.home() / "local/gcc-7.3.0/usr"
        compiler = os.environ.get("CXX", str(toolchain / "bin/g++"))
        cls.solver = directory / "lu_cmd"
        subprocess.run([
            compiler, "-std=c++11", "-O2", "-Wall", "-Wextra", "-DGLU_DEBUG=0",
            "-I" + str(ROOT / "include"), "-I" + str(ROOT / "src/preprocess"),
            "-I" + str(ROOT / "src/nicslu/include"),
            str(ROOT / "src/lu_cmd.cpp"), str(ROOT / "src/symbolic.cc"),
            str(ROOT / "src/Timer.cpp"), str(ROOT / "src/solution_error.cpp"),
            str(backend), str(ROOT / "src/preprocess/preprocess.o"),
            str(ROOT / "src/nicslu/lib/nicslu.a"),
            str(ROOT / "src/nicslu/util/nicslu_util.a"),
            "-Wl,-rpath," + str(toolchain / "lib64"), "-pthread", "-lm", "-lrt",
            "-o", str(cls.solver)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="glu-cli-run-", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.matrix = self.directory / "matrix.mtx"
        self.matrix.write_text("%%MatrixMarket matrix coordinate real general\n"
                               "3 3 7\n1 1 4\n1 2 1\n2 1 -2\n2 2 5\n"
                               "2 3 2\n3 2 3\n3 3 6\n")

    def run_solver(self, *arguments, **environment):
        env = os.environ.copy()
        env.update(environment)
        result = subprocess.run([str(self.solver), "-i", str(self.matrix)] + list(arguments),
                                cwd=str(self.directory), env=env, universal_newlines=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        self.assertRegex(result.stdout, r"^once_s=\d+\.\d{6} solve_s=\d+\.\d{6} post_s=\d+\.\d{6}\n$")
        return result

    def read_solution(self):
        data = (self.directory / "x.bin").read_bytes()
        self.assertEqual(len(data), 8 + 8 * 3)
        self.assertEqual(data[:8], b"\x03\x00\x00\x00\x00\x00\x00\x00")
        n, = struct.unpack_from("<q", data)
        x = struct.unpack_from("<" + "d" * n, data, 8)
        self.assertTrue(all(math.isfinite(value) for value in x))
        self.assertAlmostEqual(4 * x[0] + x[1], 1, places=12)
        self.assertAlmostEqual(-2 * x[0] + 5 * x[1] + 2 * x[2], 1, places=12)
        self.assertAlmostEqual(3 * x[1] + 6 * x[2], 1, places=12)
        return data, x

    def test_default_single_cold_run_and_binary_layout(self):
        (self.directory / "x.bin").write_bytes(b"obsolete output" * 100)
        result = self.run_solver()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.read_solution()
        self.assertEqual(result.stderr.count("fixture numeric cycle="), 1)
        self.assertNotIn("CPU reference file:", result.stderr)
        self.assertFalse((self.directory / "x.dat").exists())
        once, solve, _ = [float(word.split("=")[1]) for word in result.stdout.split()]
        self.assertGreaterEqual(once, 0.25)  # Device cleanup marker is in once_s.
        self.assertGreaterEqual(solve, 0.125)
        self.assertLess(solve, 0.2)
        total = float(re.search(r"total_solve_ms=([\d.]+)", result.stderr).group(1))
        self.assertAlmostEqual(total, (0.005 * once + solve) * 1000, places=6)

    def test_repeat_one_matches_default(self):
        self.assertEqual(self.run_solver().returncode, 0)
        expected, _ = self.read_solution()
        result = self.run_solver("--repeat", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.count("fixture numeric cycle="), 1)
        self.assertEqual(self.read_solution()[0], expected)
        result = self.run_solver("--repeat")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.count("fixture numeric cycle="), 1)
        self.assertEqual(self.read_solution()[0], expected)

    def test_sf_update_selection_preserves_final_output_and_repeat(self):
        default = self.run_solver()
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertIn("fixture sf_update=block", default.stderr)
        expected, _ = self.read_solution()
        for mode in ("block", "wave"):
            for repeat in ("1", "2"):
                result = self.run_solver("--sf-update", mode, "--repeat", repeat)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("fixture sf_update=" + mode, result.stderr)
                self.assertEqual(self.read_solution()[0], expected)
                self.assertEqual(result.stderr.count("fixture numeric cycle="),
                                 1 if repeat == "1" else 3)
                self.assertFalse((self.directory / "x.dat").exists())

    def test_invalid_sf_update_and_removed_padding_fail_before_numeric(self):
        for options in (("--sf-update",), ("--sf-update", "invalid"),
                        ("--sf-update", "wave", "-s", "level"),
                        ("--sf-update", "wave", "-u", "right-looking", "-s", "level"),
                        ("--numeric-sm-padding",)):
            result = self.run_solver(*options)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("fixture numeric cycle=", result.stderr)
            self.assertFalse((self.directory / "x.bin").exists())

    def test_repeat_warmup_and_final_cycle_accounting(self):
        self.assertEqual(self.run_solver().returncode, 0)
        expected, _ = self.read_solution()
        result = self.run_solver("--repeat", "3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr.count("fixture symbolic analysis"), 1)
        self.assertEqual(result.stderr.count("fixture numeric cycle="), 4)
        self.assertEqual(self.read_solution()[0], expected)
        factor = float(re.search(r"numeric_factor_s=([\d.]+)", result.stderr).group(1))
        self.assertGreaterEqual(factor, 0.5)
        self.assertLess(factor, 0.6)  # Not the sum of all four marker durations.

    def test_invalid_repeat_and_multiple_inputs_fail(self):
        for value in ("0", "-1", "1.5", "abc", "4294967296", "18446744073709551616"):
            result = self.run_solver("--repeat", value)
            self.assertNotEqual(result.returncode, 0, value)
        self.assertNotEqual(self.run_solver("-i", str(self.matrix)).returncode, 0)
        self.assertFalse((self.directory / "x.bin").exists())

    def test_output_failure_never_returns_zero(self):
        (self.directory / "x.bin").mkdir()
        result = self.run_solver()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to write x.bin", result.stderr)

    def test_shutdown_failure_never_returns_zero(self):
        result = self.run_solver(GLU_FIXTURE_SHUTDOWN_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.read_solution()

    def test_nonfinite_solution_and_numeric_failure(self):
        for flag in ("GLU_FIXTURE_NAN", "GLU_FIXTURE_FAIL"):
            result = self.run_solver(**{flag: "1"})
            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertFalse((self.directory / "x.bin").exists())

    def test_relative_cpu_comparison_with_large_and_small_solutions(self):
        reference = self.directory / "cpu.txt"
        for diagonal, offset, expected_status in ((1e-8, 1.0, 0), (1e8, 1e-10, 2)):
            self.matrix.write_text("3 3 3\n1 1 {0}\n2 2 {0}\n3 3 {0}\n".format(diagonal))
            reference.write_text("{0}\n{1}\n{1}\n".format(1 / diagonal + offset, 1 / diagonal))
            result = self.run_solver("--cpu-reference", str(reference))
            self.assertEqual(result.returncode, expected_status, result.stderr)
            absolute = float(re.search(r"GPU-CPU absolute L2 error: (\S+)", result.stderr).group(1))
            relative = float(re.search(r"GPU-CPU relative L2 error: (\S+)", result.stderr).group(1))
            self.assertEqual(absolute > 1e-6, expected_status == 0)
            self.assertEqual(relative > 1e-6, expected_status == 2)
            self.assertIn("GPU-CPU relative L2 tolerance:", result.stderr)

    def test_explicit_cpu_comparison_and_legacy_text(self):
        self.assertEqual(self.run_solver().returncode, 0)
        _, x = self.read_solution()
        reference = self.directory / "cpu.txt"
        reference.write_text("\n".join(map(repr, x)) + "\n")
        result = self.run_solver("--cpu-reference", str(reference))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GPU-CPU comparison status: converged", result.stderr)
        self.assertTrue((self.directory / "x.dat").is_file())
        reference.write_text("0\n0\n0\n")
        result = self.run_solver("--cpu-reference", str(reference))
        self.assertEqual(result.returncode, 2, result.stderr)
        self.read_solution()


if __name__ == "__main__":
    unittest.main()
