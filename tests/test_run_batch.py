"""Exercise batch stage timing output through an independent solver subprocess."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOLVER = '''#!/usr/bin/env python3
import argparse
from pathlib import Path
import time

parser = argparse.ArgumentParser()
parser.add_argument('-i', required=True)
parser.add_argument('-u', required=True)
parser.add_argument('-s', required=True)
parser.add_argument('-p', action='store_true')
parser.add_argument('--sf-workers', type=int, default=0)
parser.add_argument('--numeric-sm-padding', action='store_true')
parser.add_argument('--progress', action='store_true')
parser.add_argument('--residual-tol', type=float, default=1e-6)
parser.add_argument('--max-refine-iters', default='n')
parser.add_argument('--symbolic-backend', choices=['auto', 'cpu', 'gpu', 'gpu-sf'], default='auto')
parser.add_argument('--symbolic-sf-workers', type=int, default=0)
parser.add_argument('--symbolic-pool-entries', type=int, default=0)
parser.add_argument('--symbolic-max-pool-entries', type=int, default=0)
parser.add_argument('--symbolic-block-size', type=int, default=256)
parser.add_argument('--symbolic-batch-cols', type=int, default=128)
parser.add_argument('--symbolic-verify', action='store_true')
parser.add_argument('--symbolic-no-fallback', action='store_true')
args = parser.parse_args()
print('symbolic_batch_cols={}'.format(args.symbolic_batch_cols), flush=True)
mode = Path(args.i).stem
print('tolerance={} limit={}'.format(args.residual_tol, args.max_refine_iters), flush=True)
print('strategy={} scheduling={} perturb={}'.format(args.u, args.s, args.p), flush=True)
print('workers={} progress={}'.format(args.sf_workers, args.progress), flush=True)
print('numeric_sm_padding={}'.format(args.numeric_sm_padding), flush=True)
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
print('Task 1 total solve time (T_gpu): {} ms'.format('nan' if mode == 'invalid' else '13.75'))
print('Ax-b (1-norm): 0.000003')
print('Ax-b (2-norm): {}'.format('0.000002' if mode in ('unconverged', 'exit2') else '0.0000002'))
if mode != 'missing':
    print('Ax-b (infinite-norm): 0.000001')
Path('x.dat').write_text('1\\n' * 4)
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
            expected = {"solver.log", "x.dat"} if row["solution_path"] else {"solver.log"}
            self.assertEqual({p.name for p in run_dir.iterdir()}, expected)
        return result, rows

    def test_default_stage_timings_and_residuals(self):
        result, rows = self.run_batch(["small"])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        row = rows[0]
        self.assertEqual(row["symbolic_backend"], "auto")
        self.assertEqual(row["numeric_sm_padding"], "0")
        self.assertIn("numeric_sm_padding=False", (self.output / row["log_path"]).read_text())
        self.assertEqual((row["status"], row["exit_code"]), ("ok", "0"))
        self.assertEqual((row["update_strategy"], row["scheduling"], row["perturb"]),
                         ("right-looking", "level", "0"))
        expected = dict(n=4, nnz=8, preprocessing_time=1.25, symbolic_time=2.5,
                        csr_time=0.25, predictlu_time=0.75, leveling_time=0,
                        symbolic_preparation_time=3.5, numeric_lu_time=8.5,
                        triangular_solve_time=0.5, total_solve_ms=13.75,
                        residual_l1=3e-6, residual_l2=2e-7, residual_linf=1e-6)
        for field, value in expected.items():
            self.assertEqual(float(row[field]), value, field)
        self.assertEqual((self.output / row["solution_path"]).read_text(), "1\n" * 4)

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

    def test_numeric_sm_padding_is_forwarded_and_recorded(self):
        result, rows = self.run_batch(["small"], "--update-strategy", "left-looking",
            "--scheduling", "synchronization-free", "--sf-workers", "49", "--numeric-sm-padding")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(rows[0]["numeric_sm_padding"], "1")
        log = (self.output / rows[0]["log_path"]).read_text()
        self.assertIn("numeric_sm_padding=True", log)
        self.assertIn("workers=49", log)

    def test_numeric_sm_padding_rejects_incompatible_scheduling(self):
        for options in (("--numeric-sm-padding",),
                        ("--numeric-sm-padding", "--update-strategy", "left-looking"),
                        ("--numeric-sm-padding", "--scheduling", "synchronization-free")):
            result = subprocess.run([sys.executable, str(ROOT / "src/run_batch.py")] + list(options),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("--numeric-sm-padding requires", result.stderr)
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
                        ("--sf-workers", "1")):
            result = subprocess.run([sys.executable, str(ROOT / "src/run_batch.py")] + list(options),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    universal_newlines=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("--sf-workers", result.stderr)
            self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
