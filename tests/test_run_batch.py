"""Exercise batch stage timing output through an independent solver subprocess."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOLVER = '''#!/usr/bin/env python3
import argparse
from pathlib import Path
import time
import struct

parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True)
parser.add_argument('-u', required=True)
parser.add_argument('-s', required=True)
parser.add_argument('-p', action='store_true')
parser.add_argument('--sf-workers', type=int, default=0)
parser.add_argument('--sf-update', choices=['block', 'wave'], default='block')
parser.add_argument('--progress', action='store_true')
parser.add_argument('--residual-tol', type=float, default=1e-6)
parser.add_argument('--max-refine-iters', default='n')
parser.add_argument('--cpu-reference', required=True)
parser.add_argument('--symbolic-backend', choices=['auto', 'cpu', 'gpu', 'gpu-sf'], default='auto')
parser.add_argument('--symbolic-sf-workers', type=int, default=0)
parser.add_argument('--symbolic-pool-entries', type=int, default=0)
parser.add_argument('--symbolic-max-pool-entries', type=int, default=0)
parser.add_argument('--symbolic-block-size', type=int, default=256)
parser.add_argument('--symbolic-batch-cols', type=int, default=128)
parser.add_argument('--symbolic-verify', action='store_true')
parser.add_argument('--symbolic-no-fallback', action='store_true')
args = parser.parse_args()
print('cpu_reference={}'.format(args.cpu_reference), flush=True)
print('symbolic_batch_cols={}'.format(args.symbolic_batch_cols), flush=True)
mode = Path(args.i).stem
print('tolerance={} limit={}'.format(args.residual_tol, args.max_refine_iters), flush=True)
print('strategy={} scheduling={} perturb={}'.format(args.u, args.s, args.p), flush=True)
print('workers={} progress={}'.format(args.sf_workers, args.progress), flush=True)
print('sf_update={}'.format(args.sf_update), flush=True)
print('symbolic={} workers={} pool={} maxpool={} threads={} verify={} strict={}'.format(
    args.symbolic_backend, args.symbolic_sf_workers, args.symbolic_pool_entries,
    args.symbolic_max_pool_entries, args.symbolic_block_size,
    args.symbolic_verify, args.symbolic_no_fallback), flush=True)
if mode == 'timeout':
    time.sleep(10)
if mode == 'failed':
    raise SystemExit(7)
print('Matrix Row: 4')
print('Original nonzero: 8')
print('Input/preprocessing time: 1.25 ms')
print('Symbolic time: 2.5 ms')
print('CSR time: 0.25 ms')
print('PredictLU time: 0.75 ms')
print('Leveling time: 0 ms')
print('Symbolic preparation time (total): 3.5 ms')
print('Numeric LU time (including GPU setup/transfers): 8.5 ms')
print('Numeric LU compute time (wall-clock): 5 ms')
print('Total GPU time: 7 ms')
print('Triangular solve time (CPU): 0.5 ms')
print('Task 1 total solve time (T_gpu): 13.75 ms')  # Old wall time must be ignored.
print('total_solve_ms=999')  # Derive weighting from the terminal summary instead.
print('GPU-CPU absolute L2 error: {}'.format('23' if mode == 'large_absolute' else 'nan' if mode == 'invalid_error' else '0.0000002'))
if mode != 'missing_comparison':
    print('GPU-CPU relative L2 error: {}'.format('0.000002' if mode in ('unconverged', 'exit2') else '0.000001' if mode == 'at_tolerance' else 'nan' if mode == 'invalid_relative' else '0.0000001'))
print('Ax-b (1-norm): 0.000003')
print('Ax-b (2-norm): {}'.format('10' if mode == 'large_axb' else '0.0000002'))
if mode != 'missing':
    print('Ax-b (infinite-norm): 0.000001')
Path('x.dat').write_text('1\\n' * 4)
Path('x.bin').write_bytes(struct.pack('<q4d', 4, 1, 1, 1, 1))
if mode == 'no_solution':
    Path('x.bin').unlink()
    Path('x.dat').unlink()
if mode != 'missing_summary':
    once = 'nan' if mode == 'invalid' else '-0.010000' if mode == 'negative' else '0.010000'
    solve = 'inf' if mode == 'infinite_solve' else '0.004000'
    post = 'nan' if mode == 'invalid_post' else '0.002000'
    summary = 'once_s={} solve_s={} post_s={}'.format(once, solve, post)
    print(summary)
    if mode == 'duplicate': print(summary)
if mode == 'exit2':
    raise SystemExit(2)
'''


class BatchTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="glu-batch-tests-", dir="/tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.matrices = self.root / "matrices"
        self.matrices.mkdir()
        self.solver = self.root / "solver"
        self.solver.write_text(SOLVER.replace("#!/usr/bin/env python3", "#!" + sys.executable, 1))
        self.solver.chmod(0o755)
        self.output = self.root / "results"

    def run_batch(self, names, *options):
        for name in names:
            (self.matrices / (name + ".mtx")).write_text("solver fixture\n")
        result = subprocess.run(
            [sys.executable, str(ROOT / "src/run_batch.py"),
             "--matrix-dir", str(self.matrices), "--solver", str(self.solver),
             "--output-dir", str(self.output)] + list(options),
            cwd=str(self.root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=20)
        with (self.output / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            self.assertIn("symbolic_backend", reader.fieldnames)
        self.assertEqual({p.name for p in self.output.iterdir()},
                         {"summary.csv"} | {"{:04d}_{}".format(i, name)
                                           for i, name in enumerate(sorted(names), 1)})
        for row in rows:
            run_dir = (self.output / row["log_path"]).parent
            expected = {"solver.log", "x.dat", "x.bin"} if row["solution_path"] else {"solver.log"}
            self.assertEqual({p.name for p in run_dir.iterdir()}, expected)
        return result, rows

    def test_default_stage_timings_and_residuals(self):
        result, rows = self.run_batch(["small"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        row = rows[0]
        self.assertEqual(row["symbolic_backend"], "auto")
        self.assertEqual(row["sf_update"], "block")
        self.assertIn("sf_update=block", (self.output / row["log_path"]).read_text())
        self.assertEqual((row["status"], row["exit_code"]), ("ok", "0"))
        self.assertEqual((row["update_strategy"], row["scheduling"], row["perturb"]),
                         ("left-looking", "synchronization-free", "0"))
        expected = dict(n=4, nnz=8, preprocessing_time=1.25, symbolic_time=2.5,
                        csr_time=0.25, predictlu_time=0.75, leveling_time=0,
                        symbolic_preparation_time=3.5, numeric_lu_time=8.5,
                        triangular_solve_time=0.5, once_s=0.01, solve_s=0.004, post_s=0.002, total_solve_ms=4.05,
                        solution_error_l2=2e-7, relative_solution_error_l2=1e-7,
                        axb_residual_l1=3e-6, axb_residual_l2=2e-7, axb_residual_linf=1e-6)
        for field, value in expected.items():
            self.assertEqual(float(row[field]), value, field)
        self.assertEqual((self.output / row["solution_path"]).read_bytes(),
                         struct.pack('<q4d', 4, 1, 1, 1, 1))

    def test_solution_comparison_controls_target_and_requires_both_norms(self):
        result, rows = self.run_batch(["invalid_error", "large_axb", "missing_comparison", "unconverged"])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual([row["status"] for row in rows],
                         ["invalid_metrics", "ok", "missing_metrics", "residual_not_converged"])
        self.assertEqual(float(rows[1]["axb_residual_l2"]), 10)
        self.assertIn("GPU-CPU relative L2", rows[3]["note"])

    def test_relative_target_accepts_large_absolute_error_and_equality(self):
        result, rows = self.run_batch(["at_tolerance", "large_absolute"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([row["status"] for row in rows], ["ok", "ok"])
        self.assertEqual(float(rows[0]["relative_solution_error_l2"]), 1e-6)
        self.assertEqual(float(rows[1]["solution_error_l2"]), 23)

    def test_nonfinite_relative_error_is_rejected(self):
        result, rows = self.run_batch(["invalid_relative"])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(rows[0]["status"], "invalid_metrics")
        self.assertIn("relative_solution_error_l2", rows[0]["note"])

    def test_existing_cpu_reference_path_reaches_solver(self):
        reference_dir = self.root / "CPU results"
        result, rows = self.run_batch(["small"], "--cpu-reference-dir", str(reference_dir))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("cpu_reference=" + str(reference_dir / "small_x.txt"),
                      (self.output / rows[0]["log_path"]).read_text())

    def test_success_requires_binary_solution(self):
        result, rows = self.run_batch(["no_solution"])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(rows[0]["status"], "missing_solution")
        self.assertEqual(rows[0]["solution_path"], "")

    def test_invalid_or_missing_terminal_summary_is_rejected(self):
        result, rows = self.run_batch(["duplicate", "infinite_solve", "invalid_post",
                                      "missing_summary", "negative"])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual([row["status"] for row in rows],
                         ["invalid_metrics", "invalid_metrics", "invalid_metrics",
                          "missing_metrics", "invalid_metrics"])

    def test_left_looking_sync_free_and_perturb_forwarding(self):
        result, rows = self.run_batch(["small"], "--update-strategy", "left-looking",
                                     "--scheduling", "synchronization-free", "--perturb")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        row = rows[0]
        self.assertEqual((row["update_strategy"], row["scheduling"], row["perturb"]),
                         ("left-looking", "synchronization-free", "1"))
        self.assertIn("strategy=left-looking scheduling=synchronization-free perturb=True",
                      (self.output / row["log_path"]).read_text())

    def test_failures_are_recorded_and_remaining_matrices_continue(self):
        result, rows = self.run_batch(["failed", "invalid", "missing", "timeout", "z_ok"],
                                     "--timeout", "1")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual([row["status"] for row in rows],
                         ["failed", "invalid_metrics", "missing_metrics", "timeout", "ok"])
        self.assertEqual(rows[0]["exit_code"], "7")
        self.assertEqual(rows[3]["solution_path"], "")
        self.assertTrue(all(row["note"] for row in rows[:-1]))

    def test_sf_diagnostic_options_reach_solver(self):
        result, rows = self.run_batch(["small"], "--update-strategy", "left-looking",
                                     "--scheduling", "synchronization-free",
                                     "--sf-workers", "1", "--progress")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("workers=1 progress=True", (self.output / rows[0]["log_path"]).read_text())

    def test_symbolic_sf_options_reach_solver(self):
        result, rows = self.run_batch(["small"], "--symbolic-backend", "gpu-sf",
            "--symbolic-sf-workers", "3", "--symbolic-pool-entries", "1",
            "--symbolic-max-pool-entries", "100", "--symbolic-block-size", "64",
            "--symbolic-verify", "--symbolic-no-fallback")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(rows[0]["symbolic_backend"], "gpu-sf")
        self.assertIn("symbolic=gpu-sf workers=3 pool=1 maxpool=100 threads=64 verify=True strict=True",
                      (self.output / rows[0]["log_path"]).read_text())

    def test_sf_update_is_forwarded_and_recorded(self):
        result, rows = self.run_batch(["small"], "--update-strategy", "left-looking",
            "--scheduling", "synchronization-free", "--sf-workers", "49", "--sf-update", "wave")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(rows[0]["sf_update"], "wave")
        log = (self.output / rows[0]["log_path"]).read_text()
        self.assertIn("sf_update=wave", log)
        self.assertIn("workers=49", log)

    def test_sf_update_rejects_incompatible_scheduling(self):
        for options in (("--sf-update", "wave", "--scheduling", "level"),
                        ("--sf-update", "wave", "--update-strategy", "right-looking"),
                        ("--sf-update", "wave", "--update-strategy", "right-looking",
                         "--scheduling", "level")):
            result = subprocess.run([sys.executable, str(ROOT / "src/run_batch.py")] + list(options),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("--sf-update wave requires", result.stderr)
            self.assertFalse(self.output.exists())

    def test_auto_accepts_gpu_options(self):
        result, rows = self.run_batch(["small"], "--symbolic-backend", "auto",
            "--symbolic-sf-workers", "3", "--symbolic-pool-entries", "1",
            "--symbolic-max-pool-entries", "100", "--symbolic-verify", "--symbolic-no-fallback")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(rows[0]["symbolic_backend"], "auto")
        self.assertIn("symbolic=auto workers=3 pool=1 maxpool=100 threads=256 verify=True strict=True",
                      (self.output / rows[0]["log_path"]).read_text())

    def test_explicit_cpu_overrides_auto(self):
        result, rows = self.run_batch(["small"], "--symbolic-backend", "cpu")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(rows[0]["symbolic_backend"], "cpu")
        self.assertIn("symbolic=cpu ", (self.output / rows[0]["log_path"]).read_text())

    def test_graph_batch_option_reaches_solver(self):
        result, rows = self.run_batch(["small"], "--symbolic-backend", "gpu",
                                     "--symbolic-batch-cols", "7")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(rows[0]["symbolic_backend"], "gpu")
        self.assertIn("symbolic_batch_cols=7", (self.output / rows[0]["log_path"]).read_text())

    def test_invalid_symbolic_options(self):
        for options in (("--symbolic-backend", "cpu", "--symbolic-sf-workers", "1"),
                        ("--symbolic-backend", "cpu", "--symbolic-verify"),
                        ("--symbolic-backend", "gpu", "--symbolic-sf-workers", "1"),
                        ("--symbolic-batch-cols", "0"), ("--symbolic-batch-cols", "4294967296"),
                        ("--symbolic-backend", "gpu-sf", "--symbolic-pool-entries", "-1"),
                        ("--symbolic-backend", "gpu-sf", "--symbolic-sf-workers", "4294967296"),
                        ("--symbolic-backend", "gpu-sf", "--symbolic-pool-entries", "2",
                         "--symbolic-max-pool-entries", "1")):
            result = subprocess.run([sys.executable, str(ROOT / "src/run_batch.py")] + list(options),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("symbolic", result.stderr)
            self.assertFalse(self.output.exists())

    def test_refinement_options_reach_solver(self):
        result, rows = self.run_batch(["small"], "--residual-tol", "1e-5",
                                     "--max-refine-iters", "7")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("tolerance=1e-05 limit=7", (self.output / rows[0]["log_path"]).read_text())

    def test_default_refinement_and_unconverged_results(self):
        result, rows = self.run_batch(["exit2", "unconverged", "z_ok"])
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual([r["status"] for r in rows],
                         ["residual_not_converged", "residual_not_converged", "ok"])
        self.assertEqual(rows[0]["exit_code"], "2")
        self.assertTrue(rows[0]["solution_path"])
        self.assertIn("tolerance=1e-06 limit=n", (self.output / rows[0]["log_path"]).read_text())

    def test_invalid_refinement_options(self):
        for options in (("--residual-tol", "nan"), ("--residual-tol", "inf"),
                        ("--residual-tol", "0"), ("--residual-tol", "-1"),
                        ("--max-refine-iters", "-1"), ("--max-refine-iters", "1.5"),
                        ("--max-refine-iters", "4294967296")):
            result = subprocess.run([sys.executable, str(ROOT / "src/run_batch.py")] + list(options),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn(options[0], result.stderr)
            self.assertFalse(self.output.exists())

    def test_invalid_worker_options_fail_before_creating_results(self):
        for options in (("--sf-workers", "-1"), ("--sf-workers", "4294967296"),
                        ("--sf-workers", "1", "--scheduling", "level")):
            result = subprocess.run([sys.executable, str(ROOT / "src/run_batch.py")] + list(options),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("--sf-workers", result.stderr)
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
